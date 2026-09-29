"""The feedback of Bowman et al., https://dx.doi.org/10.1088/0953-4075/48/11/115303."""

from __future__ import annotations

import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from numpy.typing import NDArray

from .abstract import CameraFeedbackData, FeedbackCorrectorBase
from ..phase_retrieval import PhaseRetrievalData

from ...analysis.error_metrics import (
    DEFAULT_INTENSITY_METRICS,
    IntensityMetric,
    evaluate_metrics,
    normalize,
)
from ...datasets import CaptureStore
from ...hardware.camera import Background, background_at
from ...profiles.amplitude import gaussian_blur
from ...utils import ProgressBar, gpu_to_numpy


# TODO: Reference missing
class SimpleFeedbackCorrector(FeedbackCorrectorBase):
    """Correct a light potential against the camera, one measurement at a time.

    Each iteration measures the potential the current hologram produces, takes its
    discrepancy from the target that was actually wanted, and adds that discrepancy to
    the target the retriever is given next. Where there was too little light in the 
    measurement, the target intensity is raised to cancel the error and vice versa.

    The discrepancy is always measured against the original target.
    """

    def run(
        self,
        retriever_iterations: Sequence[int] = (20,) * 5,
        gain: float | Sequence[float] = 1.0,
        exposure: float | None = None,
        autoexpose: bool = True,
        averages: int = 10,
        background: Background | None = None,
        blur: float = 0.0,
        metrics: Sequence[IntensityMetric] = DEFAULT_INTENSITY_METRICS,
        step_stride: int | None = None,
        step_directory: str | os.PathLike | None = None,
        dataset_path: str | os.PathLike | None = None,
        retrieve_options: dict | None = None,
        name: str = "",
        verbose: bool = True,
    ) -> CameraFeedbackData:
        """Run the feedback loop.

        Args:
            retriever_iterations: Optimizer iterations to spend on each feedback
                iteration, one entry per iteration. Its length is how many feedback
                iterations run. Later corrections are smaller than the first, so a 
                decreasing sequence often converges in less total time.
            gain: How much of the discrepancy to apply, per iteration or once for all.
                One is the textbook choice. Lower it if the loop oscillates.
            exposure: Camera exposure in seconds while ``autoexpose`` is off, or None to
                leave it as set.
            autoexpose: Autoexpose on the signal region of every hologram before it is
                measured. The potential's peak moves as the corrections change it, so
                each frame is metered on its own. The exposures are recorded in the
                result's metadata.
            averages: Camera frames to average per measurement. Each frame is
                captured on its own. A measurement counts as overexposed when a pixel
                of the signal region reaches full scale in any of its frames
                (:attr:`~hologradpy.hardware.camera.Camera.overexposed`).
            background: The counts subtracted from every measurement before it is
                compared with the target. It is one level for every pixel, a
                whole-sensor frame, or a function of the exposure in seconds
                returning either. A function is evaluated at the exposure of each
                measurement. None, the default, subtracts nothing.
            blur: Gaussian blur applied to the corrected target, in pixels. Smooths the
                correction so the retriever is not asked to reproduce measurement noise
                smaller than the diffraction limit.
            metrics: How the measured potential is compared against the target each
                iteration. Each one gets
                its own series in the result and its own convergence panel. Defaults to
                rmse and psnr.
            step_stride: Save the SLM phase every nth optimizer iteration of
                every search. None, the default, saves nothing.
            step_directory: Where those files go. Each feedback iteration gets its
                own subdirectory, so the searches do not overwrite each other.
            dataset_path: Write the full sensor frame and the SLM phase of every
                iteration into a sample store. None, the default, writes nothing.
            retrieve_options: Extra keyword arguments for the retriever's
                ``retrieve_phase``, such as ``method``. These options differ between
                retrievers and are not retriever state.
            name: Label stored on the record.
            verbose: Show a progress bar carrying the error metrics.

        Returns:
            CameraFeedbackData: The corrected hologram, and the whole run behind it.
            Its metadata holds the exposure of every measurement under ``"exposures"``.
            The iterations with an overexposed measurement inside the signal region are
            listed under ``"overexposed_iterations"``. The measured images are raw, and
            the background subtracted from each is kept in ``background_images``.
        """
        self._check_grids_match()
        self.apply_camera_mapping(verbose=verbose)
        self.place_target()

        retriever_steps = self._retriever_steps(retriever_iterations)
        iterations = len(retriever_steps)
        gains = self._per_iteration(gain, iterations, "gain")
        options = retrieve_options or {}

        signal_region = gpu_to_numpy(self.signal_region)
        target = normalize(gpu_to_numpy(self.target), signal_region)
        discrepancy = np.zeros_like(target)
        corrected = target

        initial_guess = self.phase_retriever.predicted_intensity()

        signal_roi = self._signal_roi
        region_mask = signal_region.astype(bool)

        history: dict[str, list[float]] = {}
        retrievals: list[PhaseRetrievalData] = []
        corrected_targets: list[NDArray] = []
        measured_frames: list[NDArray] = []
        background_images: list[NDArray] = []
        full_frames: list[NDArray] = []
        exposures: list[float] = []
        overexposed_iterations: list[int] = []
        final_camera_image: NDArray | None = None

        retriever_bar = ProgressBar(
            total=max(retriever_steps),
            description="Phase retrieval",
            verbose=verbose,
            position=1,
            leave=False,
        )
        # The whole sensor is read out, since the model predicts all of it. The
        # camera's exposure and region of interest are put back when the loop ends.
        with (
            self.camera.preserve_exposure_and_roi(full_sensor=True),
            ProgressBar(
                total=iterations,
                description="Camera feedback",
                verbose=verbose,
                position=0,
            ) as bar,
            retriever_bar,
        ):
            for iteration in range(iterations):
                corrected = self._corrected_target_for(
                    corrected, discrepancy, gains[iteration], signal_region, blur
                )
                self.update_target(
                    torch.as_tensor(
                        corrected, dtype=self.target.dtype, device=self.target.device
                    )
                )

                # Warm started using the previous hologram.
                retrieval = self.phase_retriever.retrieve(
                    retriever_steps[iteration],
                    name=f"feedback iteration {iteration + 1}",
                    metrics=metrics,
                    step_stride=step_stride,
                    step_directory=self._step_subdirectory(
                        step_directory, iteration
                    ),
                    progress_bar=retriever_bar,
                    **options,
                )

                # .lean() removes target and signal region
                retrievals.append(retrieval.lean())
                self.slm.set_phase(retrieval.phase)

                # Exposed on the signal region of every hologram.
                if autoexpose:
                    exposure = self.camera.autoexpose(
                        roi=signal_roi,
                        mask=signal_roi.crop(region_mask),
                        raise_on_rail=False,
                    )

                measured, overexposed = self._capture_mean_and_overexposure(
                    exposure, averages, region_mask
                )
                exposures.append(float(self.camera.get_exposure()))
                if overexposed:
                    overexposed_iterations.append(iteration)

                # The background carries no signal, so it is taken off before the frame
                # is compared with the target.
                signal = measured
                if background is not None:
                    background_counts = background_at(
                        background, exposures[-1], measured.shape
                    )
                    signal = measured - background_counts
                    background_images.append(
                        signal_roi.crop(
                            np.broadcast_to(background_counts, measured.shape)
                        ).copy()
                    )
                measured_normalized = normalize(signal, signal_region)

                # Against the original target, not the corrected one.
                discrepancy = target - measured_normalized

                corrected_targets.append(signal_roi.crop(corrected))
                measured_frames.append(signal_roi.crop(measured))
                full_frames.append(measured)
                final_camera_image = measured
                evaluate_metrics(metrics, signal_region, target, signal, history)

                bar.update(**{label: values[-1] for label, values in history.items()})

        if overexposed_iterations:
            warnings.warn(
                f"The measurements of iterations {overexposed_iterations} are "
                "overexposed inside the signal region. An overexposed pixel reads "
                "below the light it receives, which the loop takes for too little "
                "light. Autoexpose, or lower the exposure.",
                stacklevel=2,
            )

        data = CameraFeedbackData(
            timestamp=datetime.now(),
            name=name or type(self).__name__,
            target=target,
            signal_region=signal_region,
            corrected_targets=corrected_targets,
            measured_images=measured_frames,
            background_images=background_images if background is not None else None,
            final_camera_image=final_camera_image,
            initial_guess=initial_guess,
            camera_mapping=self._mapping.lean(),
            retrievals=retrievals,
            metrics=history,
            lower_is_better={
                metric.name: metric.lower_is_better for metric in metrics
            },
            metadata={
                "exposures": exposures,
                "overexposed_iterations": overexposed_iterations,
            },
        )
        if dataset_path is not None:
            bitdepth = self.slm.bitdepth
            patterns = [
                self.slm.phase_to_levels(retrieval.phase) for retrieval in retrievals
            ]

            CaptureStore.write(
                dataset_path,
                data,
                camera_images=full_frames,
                phase_bitdepth=bitdepth,
                slm_levels=patterns,
            )
        return data

    def _capture_mean_and_overexposure(
        self, exposure: float | None, averages: int, region: NDArray[np.bool_]
    ) -> tuple[NDArray[np.float64], bool]:
        """Capture ``averages`` single frames, and return the mean and the overexposure.

        The frames are captured one at a time, and each frame is checked for
        overexposure inside ``region``
        (:attr:`~hologradpy.hardware.camera.Camera.overexposed`). The check catches a
        few overexposed pixels in one frame, even when the mean of the frames stays
        below full scale.

        Args:
            exposure: The exposure in seconds to set first, or None to leave it as set.
            averages: The number of frames.
            region: True at the pixels to check for overexposure.

        Returns:
            tuple[NDArray[np.float64], bool]: The mean frame, and whether any frame was
            overexposed inside ``region``.
        """
        if exposure is not None:
            self.camera.set_exposure(exposure)
        frame_count = max(1, int(averages))
        total = np.asarray(self.camera.get_image(mask=region), dtype=np.float64)
        overexposed = self.camera.overexposed
        for _ in range(frame_count - 1):
            total += np.asarray(self.camera.get_image(mask=region), dtype=np.float64)
            overexposed = overexposed or self.camera.overexposed
        return total / frame_count, overexposed

    @staticmethod
    def _step_subdirectory(
        directory: str | os.PathLike | None, iteration: int
    ) -> Path | None:
        """Each feedback iteration writes into its own subdirectory."""
        if directory is None:
            return None
        return Path(directory) / f"iteration_{iteration}"

    @staticmethod
    def _corrected_target_for(
        corrected: np.ndarray,
        discrepancy: np.ndarray,
        gain: float,
        signal_region: np.ndarray,
        blur: float,
    ) -> np.ndarray:
        """Updates the target using the previous one plus the discrepancy."""
        updated = signal_region * (corrected + gain * discrepancy)
        updated = np.clip(updated, 0.0, None)
        if blur > 0:
            updated = gpu_to_numpy(
                gaussian_blur(torch.as_tensor(updated).float(), blur)
            )
        return updated

    @staticmethod
    def _retriever_steps(retriever_iterations: Sequence[int]) -> list[int]:
        """The per-iteration optimizer budgets, validated."""
        if isinstance(retriever_iterations, (int, np.integer)):
            raise TypeError(
                "retriever_iterations is one entry per feedback iteration, not a "
                f"single count. Pass [{retriever_iterations}] * n to run n iterations "
                f"of {retriever_iterations} optimizer steps each."
            )

        steps = [int(value) for value in retriever_iterations]
        if not steps:
            raise ValueError(
                "retriever_iterations is empty, so there are no feedback iterations "
                "to run."
            )
        return steps

    @staticmethod
    def _per_iteration(
        value: float | Sequence[float], iterations: int, name: str
    ) -> list:
        """One value per iteration, from either a scalar or a sequence."""
        if np.isscalar(value):
            return [value] * iterations

        values = list(value)
        if len(values) != iterations:
            raise ValueError(
                f"{name} has {len(values)} entries but there are {iterations} feedback "
                "iterations. Pass one value per iteration, or a single value for all."
            )
        return values
