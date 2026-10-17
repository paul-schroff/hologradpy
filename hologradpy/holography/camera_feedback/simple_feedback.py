"""The feedback of Bruce et al., https://dx.doi.org/10.1088/0953-4075/48/11/115303."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .abstract import (
    CameraFeedbackData,
    FeedbackCorrectorBase,
    corrected_target_intensity,
)
from ..phase_retrieval import PhaseRetrievalData

from ...analysis.error_metrics import (
    DEFAULT_INTENSITY_METRICS,
    IntensityMetric,
    evaluate_metrics,
)
from ...datasets import CaptureStore
from ...hardware.camera import Background, background_at
from ...utils import gpu_to_numpy


# TODO: Reference missing
class SimpleFeedbackCorrector(FeedbackCorrectorBase):
    """Correct a light potential against the camera, one measurement at a time.

    Each iteration measures the potential of the current hologram and takes its
    discrepancy from the original target. The discrepancy is added to the target the
    retriever is given next
    (:func:`~hologradpy.holography.camera_feedback.abstract.corrected_target_intensity`).
    Where the measurement holds too little light, the target intensity is raised to
    cancel the error, and vice versa.

    Every frame of a measurement is moved onto the target by its own fitted x and y
    shift (:func:`~hologradpy.analysis.fitting.fit_translation`) before the frames are
    averaged. The loop therefore corrects the shape of the potential and leaves it where
    it landed. An offset of the pattern from the target is not taken for an error at its
    edges, whether it comes from the camera mapping or a drift.
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
                each frame is autoexposed on its own.
            averages: Camera frames to average per measurement. A measurement counts as
                overexposed when a pixel of the signal region reaches full scale in any
                of its frames (:attr:`~hologradpy.hardware.camera.Camera.overexposed`).
            background: The counts subtracted from every measurement before it is
                compared with the target. It is one level for every pixel, a
                whole-sensor frame, or a function of the equivalent exposure in
                seconds returning either. A function is evaluated at the equivalent
                exposure of each measurement
                (:meth:`~hologradpy.hardware.camera.Camera.get_equivalent_exposure`).
                None, the default, subtracts nothing.
            blur: Gaussian blur applied to the corrected target, in pixels. Smooths the
                correction so the retriever is not asked to reproduce measurement noise
                smaller than the diffraction limit.
            metrics: How the measured potential is compared against the target each
                iteration. Each one gets its own series in the result and its own
                convergence panel. Defaults to rmse and psnr.
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
            Its ``measured_images`` hold the frames moved onto the target and divided
            by the mean of their bright pixels. Its metrics compare these frames with
            the target. Its metadata holds the exposure of every measurement under
            ``"exposures"``, and the gain of the camera in dB under ``"camera_gains"``.
            The iterations with an overexposed measurement inside the signal region are
            listed under ``"overexposed_iterations"``. The offset of the pattern from
            the target is kept under ``"pattern_offsets"``, as ``(row, column)`` in
            sensor pixels averaged over the frames of a measurement.
        """
        window = self._prepare_run(verbose)
        retriever_steps = self._retriever_steps(retriever_iterations)
        iterations = len(retriever_steps)
        gains = self._per_iteration(gain, iterations, "gain")
        options = retrieve_options or {}

        initial_guess = self.phase_retriever.predicted_intensity()

        signal_roi = window.roi
        sensor_region = window.sensor_region
        target = window.target
        region = window.region
        # The first update finds no error, so the first retrieval is given the target.
        aligned = target
        corrected = target

        history: dict[str, list[float]] = {}
        retrievals: list[PhaseRetrievalData] = []
        corrected_targets: list[NDArray] = []
        measured_images: list[NDArray] = []
        background_images: list[NDArray] = []
        full_frames: list[NDArray] = []
        exposures: list[float] = []
        camera_gains: list[float] = []
        overexposed_iterations: list[int] = []
        pattern_offsets: list[tuple[float, float]] = []
        final_camera_image: NDArray | None = None

        with self._retrieval_loop(iterations, retriever_steps, verbose) as (
            bar,
            retriever_bar,
        ):
            for iteration in range(iterations):
                corrected = corrected_target_intensity(
                    corrected, aligned, target, region, gains[iteration], blur=blur
                )
                self.update_target(corrected)

                # Warm started using the previous hologram.
                retrieval = self._retrieve_and_display(
                    retriever_steps[iteration],
                    iteration,
                    metrics,
                    retriever_bar,
                    step_stride=step_stride,
                    step_directory=self._step_subdirectory(step_directory, iteration),
                    **options,
                )
                # .lean() removes target and signal region
                retrievals.append(retrieval.lean())

                # Exposed on the signal region of every hologram.
                camera_exposure, camera_gain = self._expose(
                    autoexpose, exposure, sensor_region
                )
                exposures.append(camera_exposure)
                camera_gains.append(camera_gain)

                background_counts = None
                if background is not None:
                    background_counts = background_at(
                        background,
                        self.camera.get_equivalent_exposure(),
                        sensor_region.shape,
                    )
                    background_images.append(
                        signal_roi.crop(
                            np.broadcast_to(background_counts, sensor_region.shape)
                        ).copy()
                    )
                aligned, fits, overexposed, measured = self._aligned_frame(
                    averages,
                    target,
                    region,
                    sensor_region,
                    destination=(0.0, 0.0),
                    background=background_counts,
                )
                if overexposed:
                    overexposed_iterations.append(iteration)
                pattern_offsets.append(
                    tuple(np.mean([fit.shift for fit in fits], axis=0).tolist())
                )

                corrected_targets.append(gpu_to_numpy(corrected))
                measured_images.append(gpu_to_numpy(aligned))
                full_frames.append(measured)
                final_camera_image = measured
                evaluate_metrics(
                    metrics,
                    gpu_to_numpy(region).astype(bool),
                    gpu_to_numpy(target),
                    gpu_to_numpy(aligned),
                    history,
                )

                bar.update(**{label: values[-1] for label, values in history.items()})

        if overexposed_iterations:
            self._warn_overexposed(f"iterations {overexposed_iterations}")

        data = self._record(
            name,
            gpu_to_numpy(window.sensor_target),
            gpu_to_numpy(self.signal_region),
            metrics,
            history,
            corrected_targets=corrected_targets,
            measured_images=measured_images,
            background_images=background_images if background is not None else None,
            final_camera_image=final_camera_image,
            initial_guess=initial_guess,
            retrievals=retrievals,
            metadata={
                "exposures": exposures,
                "camera_gains": camera_gains,
                "overexposed_iterations": overexposed_iterations,
                "pattern_offsets": pattern_offsets,
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

    @staticmethod
    def _step_subdirectory(
        directory: str | os.PathLike | None, iteration: int
    ) -> Path | None:
        """Each feedback iteration writes into its own subdirectory."""
        if directory is None:
            return None
        return Path(directory) / f"iteration_{iteration}"
