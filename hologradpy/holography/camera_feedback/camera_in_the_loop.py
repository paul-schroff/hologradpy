"""Camera-in-the-loop optimization of the hologram, after Peng et al. (2020)."""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

import torch
from numpy.typing import NDArray
from torch import Tensor

from .abstract import CameraFeedbackData, FeedbackCorrectorBase
from ..phase_retrieval import PhaseRetrievalData

from ...analysis.fourier_filter import fourier_filter
from ...analysis.error_metrics import (
    DEFAULT_INTENSITY_METRICS,
    IntensityMetric,
    evaluate_metrics,
    normalize,
)
from ...utils import ProgressBar, as_image, gpu_to_numpy

# The Adam step size on the levels, in fractions of full scale (one level of an 8-bit
# SLM is 1/256). On a simulated bench, after two iterations of the learning corrector
# from a calibrated start, 1e-4 reached the floor of 0.25 % within 25 steps.
DEFAULT_LEARNING_RATE = 1e-4


class CameraInTheLoopCorrector(FeedbackCorrectorBase):
    """Correct a light potential by gradient steps on the hologram, with the error
    measured on the camera and its gradient taken through the model.

    This is the camera-in-the-loop optimization of Peng et al., "Neural holography with
    camera-in-the-loop training", ACM Trans. Graph. 39, 185 (2020), Sec. 4, Eq. (4),
    https://doi.org/10.1145/3414685.3417802. Every step displays the hologram and
    captures one frame. The frame is moved onto the model's prediction by its fitted
    shift. The target is then moved onto the frame by its own fitted shift, so the
    error holds the shape of the pattern alone. The error is the frame minus the
    target, Fourier-filtered to the band of the optics. It is back-propagated through
    the model into the SLM's levels. These are rounded to whole levels with a
    straight-through gradient, and Adam takes one step on them.

    The model stays fixed, so it has to be an accurate one, such as a calibrated model
    or one learned by a
    :class:`~hologradpy.holography.camera_feedback.ModelLearningFeedbackCorrector`. The
    run starts from the model's hologram, such as the one that corrector leaves for its
    next retrieval. On a simulated bench, the error stopped falling near 0.25 %, a
    floor set by the noise of single frames and the pointing jitter.
    """

    def run(
        self,
        steps: int = 25,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        averages: int = 20,
        exposure: float | None = None,
        autoexpose: bool = True,
        metrics: Sequence[IntensityMetric] = DEFAULT_INTENSITY_METRICS,
        name: str = "",
        verbose: bool = True,
    ) -> CameraFeedbackData:
        """Run the camera-in-the-loop steps.

        Args:
            steps: Gradient steps, each with one displayed hologram and one frame.
            learning_rate: Adam step size on the levels, in fractions of full scale.
            averages: Camera frames of each measurement of the first and the last
                hologram, which the record keeps.
            exposure: Camera exposure in seconds while ``autoexpose`` is off, or None
                to leave it as set.
            autoexpose: Autoexpose on the signal region once, on the first hologram.
            metrics: How the measured potential is compared against the target.
            name: Label stored on the record.
            verbose: Show a progress bar carrying the error of every step's frame.

        Returns:
            CameraFeedbackData: Two entries, the first hologram and the last. Each entry
            holds its frame, moved onto the prediction and divided by the mean of its
            bright pixels. Its metrics compare the frame with the target moved onto it.
            The metadata holds the error of every step under ``"step_errors"``. Each is
            the RMS of the frame minus the target in the band of the optics, over the
            bright part of the target. The metadata also holds the exposures and the
            camera gains of the two measurements, the steps and the measurements with a
            frame overexposed inside the signal region, ``steps`` and ``learning_rate``.
        """
        window = self._prepare_run(verbose)
        model = self.slm_camera_model
        virtual_slm = model.virtual_slm
        region = window.region
        target = window.target
        region_float = region.to(target.dtype)
        sensor_region = window.sensor_region
        initial_guess = self.phase_retriever.predicted_intensity()

        history: dict[str, list[float]] = {}
        measured_images: list[NDArray] = []
        retrievals: list[PhaseRetrievalData] = []
        exposures: list[float] = []
        camera_gains: list[float] = []
        overexposed_steps: list[int] = []
        overexposed_measurements: list[str] = []
        step_errors: list[float] = []
        final_camera_image: NDArray | None = None

        def differentiable_prediction() -> Tensor:
            """The model's intensity of the hologram it holds, divided by the mean of
            its bright pixels in the region, with its graph through the levels.
            """
            return normalize(as_image(model().intensity).to(target.dtype), region_float)

        def measure(label: str) -> NDArray:
            """Measure the displayed hologram with ``averages`` frames and record it."""
            frame, _, overexposed, captured = self._aligned_frame(
                averages,
                self._normalized_prediction(region),
                region,
                sensor_region,
                (0.0, 0.0),
            )
            if overexposed:
                overexposed_measurements.append(label)
            placed, _ = self._place_target_on(frame, target, region)
            evaluate_metrics(
                metrics,
                gpu_to_numpy(region).astype(bool),
                gpu_to_numpy(normalize(placed, region_float)),
                gpu_to_numpy(frame),
                history,
            )
            measured_images.append(gpu_to_numpy(frame))
            retrievals.append(
                PhaseRetrievalData(
                    timestamp=datetime.now(),
                    name=f"camera in the loop, {label}",
                    phase=gpu_to_numpy(virtual_slm.get_phase().detach()),
                )
            )
            exposures.append(float(self.camera.get_exposure()))
            camera_gains.append(float(self.camera.get_gain()))
            return captured

        def display() -> None:
            self.slm.set_phase(gpu_to_numpy(virtual_slm.get_phase().detach()))

        held_quantize = virtual_slm.quantize
        held_gradients = [
            (parameter, parameter.requires_grad) for parameter in model.parameters()
        ]
        for parameter, _ in held_gradients:
            parameter.requires_grad_(False)
        virtual_slm.quantize = True
        virtual_slm.levels.requires_grad_(True)
        optimizer = torch.optim.Adam([virtual_slm.levels], lr=learning_rate)
        try:
            with (
                self.camera.preserve_exposure_gain_and_roi(full_sensor=True),
                model.restrict_output_to(window.roi),
                ProgressBar(
                    total=steps, description="Camera in the loop", verbose=verbose
                ) as bar,
            ):
                display()
                self._expose(autoexpose, exposure, sensor_region)
                measure("first hologram")

                for step in range(steps):
                    expected = differentiable_prediction()
                    frame, _, overexposed, _ = self._aligned_frame(
                        1, expected.detach(), region, sensor_region, (0.0, 0.0)
                    )
                    if overexposed:
                        overexposed_steps.append(step)
                    placed, _ = self._place_target_on(frame, target, region)
                    placed = normalize(placed, region_float)
                    error = fourier_filter(
                        (frame - placed) * region_float, window.band_mask
                    )
                    step_errors.append(float(error[placed > 0.5].pow(2).mean().sqrt()))

                    # The camera's error in the forward pass, the model's gradient in
                    # the backward pass.
                    optimizer.zero_grad()
                    (expected * error).sum().backward()
                    optimizer.step()
                    display()
                    bar.update(error=step_errors[-1])

                final_camera_image = measure("last hologram")
        finally:
            virtual_slm.levels.grad = None
            for parameter, requires_grad in held_gradients:
                parameter.requires_grad_(requires_grad)
            virtual_slm.quantize = held_quantize

        overexposed = [f"step {step}" for step in overexposed_steps]
        overexposed += [f"the {label}" for label in overexposed_measurements]
        if overexposed:
            self._warn_overexposed(", ".join(overexposed))

        return self._record(
            name,
            gpu_to_numpy(window.sensor_target),
            sensor_region,
            metrics,
            history,
            measured_images=measured_images,
            final_camera_image=final_camera_image,
            initial_guess=initial_guess,
            retrievals=retrievals,
            metadata={
                "steps": int(steps),
                "learning_rate": float(learning_rate),
                "step_errors": step_errors,
                "exposures": exposures,
                "camera_gains": camera_gains,
                "overexposed_steps": overexposed_steps,
                "overexposed_measurements": overexposed_measurements,
            },
        )
