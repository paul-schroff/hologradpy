"""A camera feedback loop that learns its model from every frame."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader, Sampler

from .abstract import (
    CameraFeedbackData,
    FeedbackCorrectorBase,
    SignalWindow,
    corrected_target_intensity,
)
from ..phase_retrieval import PhaseRetrievalData

from ...analysis.fourier_filter import filtered_noise_rms, fourier_filter
from ...analysis.error_metrics import (
    DEFAULT_INTENSITY_METRICS,
    IntensityMetric,
    evaluate_metrics,
    normalize,
)
from ...analysis.fitting import TranslationFit, fit_translation
from ...calibration.model_fit import ModelFitter
from ...datasets import HologramFrames
from ...fourier_transforms import translate_real
from ...grids import get_frequency_grid
from ...optics.modules.slm_fields import PixelwiseSLMField
from ...profiles.phase import defocus_phase, random_diversity_phase, tilt_to_angle
from ...utils import as_image, gpu_to_numpy

# The Adam step size, for a kernel at unit peak.
DEFAULT_LEARNING_RATE = 1e-2

# The CG steps of each retrieval, for a calibrated model.
DEFAULT_RETRIEVER_ITERATIONS = (300, 50, 50, 50, 50)

# The fraction of the target's peak above which the target is its plateau.
# top_hat_gaussian_shoulders defines the width of its plateau at this threshold.
PLATEAU_FRACTION = 1 - np.exp(-4.0)


class ModelLearningFeedbackCorrector(FeedbackCorrectorBase):
    """Correct a light potential against the camera, learning the model from every
    frame.

    Each iteration displays a hologram and runs three steps.

    1. **Measure.** The translation of each frame from the model's prediction is fitted
       (:func:`~hologradpy.analysis.fitting.fit_translation`). In the first iteration,
       the frames are moved to their mean translation, so the pointing jitter between
       them is removed. The learning then fits the pattern's displacement. In later
       iterations, each frame is moved onto the prediction, so the beam's pointing drift
       is removed as well. The frames are averaged and divided by the mean of their
       bright pixels. With ``defocus_distances``, defocused copies of the hologram
       (:func:`~hologradpy.profiles.phase.defocus_phase`) are measured in the same way
       and moved to the same place as the frames in focus. With ``random_diversity``,
       every defocused copy also carries a new smooth random phase
       (:func:`~hologradpy.profiles.phase.random_diversity_phase`).
    2. **Learn.** The parameters named by ``learned_parameters`` are refitted to the
       frames, which are stored in one
       :class:`~hologradpy.datasets.HologramFrames`. By default,
       Levenberg-Marquardt fits every stored hologram on the residual of the PSF
       speckle calibration
       (:meth:`~hologradpy.calibration.model_fit.ModelFitter.fit_levenberg_marquardt`).
       With ``learning_method="adam"``, every Adam step fits the newest iteration's
       holograms and frames, and as many older ones drawn at random. Its cost is that
       of the wavefront calibration
       (:class:`~hologradpy.loss_functions.MaskedIntensityMSE`).
    3. **Step.** The target is moved onto the measured pattern, and the error is the
       target minus the frame. The error is Fourier-filtered to the band of the
       optics, so it holds the shape of the pattern alone. The next retrieval starts
       from the displayed hologram plus a linear phase. This phase moves the pattern
       back by the learned model's offset from the target. The retrieval is given the
       learned model's intensity of the moved hologram plus the error, moved with the
       pattern.

    The frames of a few holograms have to determine the SLM field, so the learning
    needs a field with few parameters, such as a
    :class:`~hologradpy.optics.modules.slm_fields.psf.PSFSLMField` seeded by
    ``from_calibration_data``.
    """

    def run(
        self,
        retriever_iterations: Sequence[int] = DEFAULT_RETRIEVER_ITERATIONS,
        gain: float | Sequence[float] = 1.0,
        learning_method: str = "levenberg_marquardt",
        learning_steps: int | None = None,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        learned_parameters: Sequence[str] = ("slm_field",),
        defocus_distances: Sequence[float] = (),
        random_diversity: bool = False,
        exposure: float | None = None,
        autoexpose: bool = True,
        averages: int = 10,
        metrics: Sequence[IntensityMetric] = DEFAULT_INTENSITY_METRICS,
        retrieve_options: dict | None = None,
        name: str = "",
        verbose: bool = True,
    ) -> CameraFeedbackData:
        """Run the feedback loop.

        Args:
            retriever_iterations: Optimizer iterations of each retrieval, one entry per
                feedback iteration. Its length is how many iterations run. The default,
                :data:`DEFAULT_RETRIEVER_ITERATIONS`, suits a calibrated model. A model
                wrong by about 1 rad gives a first frame near 30 % and needs 1000 steps
                and then 200.
            gain: The fraction of the measured error each step corrects, per iteration
                or once for all. At 1, a step removes the in-phase error to first order.
                A lower gain averages the noise of several frames.
            learning_method: How the learning refits the model,
                ``"levenberg_marquardt"`` or ``"adam"`` (the class description, step
                2). On a simulated bench from a model wrong by 1 rad,
                Levenberg-Marquardt reached 0.5 % two holograms before Adam. From a
                calibrated model, the two did the same. The time of a
                Levenberg-Marquardt learning grows with the stored holograms, by 1.3 s
                per iteration on an M6000 for a 1024 x 1272 SLM.
            learning_steps: Steps of the chosen method after every measurement. None
                takes 5 Levenberg-Marquardt steps or 50 Adam steps. A model wrong by
                about 1 rad needs 200 Adam steps. Zero learns nothing.
            learning_rate: Adam step size of the learning. Levenberg-Marquardt has
                none.
            learned_parameters: The model parameters the learning fits, by name or by
                the name of their module, such as the default ``"slm_field"``. The
                focal-plane partial affine and the hologram are never fitted.
            defocus_distances: Distances along the optical axis, in metres, by which
                every hologram is also displayed defocused after its measurement in
                focus (:func:`~hologradpy.profiles.phase.defocus_phase`). A positive
                distance moves the focus away from the SLM. The learning fits these
                frames as well. The step uses the frame in focus. A start is
                half the Rayleigh range of the focal spot on either side,
                ``(-z_R / 2, z_R / 2)``. Empty measures in focus only.
            random_diversity: Add a new smooth random phase to every defocused
                hologram (:func:`~hologradpy.profiles.phase.random_diversity_phase`).
                The phase is drawn from PyTorch's global generator, so
                ``torch.manual_seed`` repeats a run. Its intensity-weighted RMS over the
                model's beam is that of the largest defocus. Its band is the disc of SLM
                spatial frequency ``R / (lambda f)``. ``R`` is the smallest image-plane
                distance between the target's plateau and the edge of the signal ROI,
                so the light deflected from the plateau stays inside the ROI. A wider
                margin of the signal region around the plateau therefore widens the
                band. On a simulated bench from a model wrong by 1 rad, it learned the
                field to 0.023 to 0.029 rad on a margin of 6 spot radii around the
                plateau. The defocus pair alone learned the field to 0.045 to 0.052 rad.
                Both ranges are over three draws. On a margin of 4 spot radii, its
                improvement depended on the draw. Needs ``defocus_distances``.
            exposure: Camera exposure in seconds while ``autoexpose`` is off, or None
                to leave it as set.
            autoexpose: Autoexpose on the signal region before every measurement in
                focus. The defocused frames are measured at the exposure of their
                frames in focus. A defocused frame overexposed inside the signal region
                is autoexposed and measured again.
            averages: Camera frames per measurement.
            metrics: How the measured potential is compared against the target each
                iteration.
            retrieve_options: Extra keyword arguments for the retriever's
                ``retrieve_phase``, such as ``method``.
            name: Label stored on the record.
            verbose: Show progress bars.

        Returns:
            CameraFeedbackData: The run. ``measured_images`` hold the aligned frames,
            each divided by the mean of its bright pixels as the target is.
            ``corrected_targets`` hold the intensity each retrieval was given. The
            metrics compare each frame with the target moved onto it, so they measure
            the shape. Per iteration, the metadata holds the exposures, the camera
            gains, the fitted translation of every frame against the model, where the
            pattern landed relative to the target, the noise floor, the innovation, the
            in-band residual after learning, the ratio of the measured to the predicted
            change of the step, the final cost of the learning, and the PSF kernel when
            the field has one. The translations, under ``"translations"``, and where
            the patterns landed, under ``"pattern_offsets"``, are ``(columns, rows)`` in
            sensor pixels. With defocused frames, it also holds the distances, the
            innovation of each defocused frame per iteration, and the defocused frames
            measured again for overexposure as ``(iteration, distance index)`` pairs.
            ``random_diversity`` is recorded. With it set, the RMS of the random phase
            in radians and ``R`` in metres are recorded as well.

        Raises:
            ValueError: ``random_diversity`` is set without ``defocus_distances``, or
                the target's plateau leaves the random phase no band.
        """
        if random_diversity and len(defocus_distances) == 0:
            raise ValueError(
                "random_diversity adds its phase to the defocused holograms and takes "
                "its RMS from the defocus, so it needs defocus_distances."
            )
        window = self._prepare_run(verbose)
        retriever_steps = self._retriever_steps(retriever_iterations)
        iterations = len(retriever_steps)
        gains = self._per_iteration(gain, iterations, "gain")
        options = retrieve_options or {}
        defocus_phases = self._defocus_phases(defocus_distances)
        diversity = (
            self._random_diversity(window.target, defocus_phases)
            if random_diversity
            else None
        )
        learning = self._learning(
            learning_method,
            learning_steps,
            learning_rate,
            learned_parameters,
            1 + len(defocus_phases),
        )
        initial_guess = self.phase_retriever.predicted_intensity()
        region = window.region
        band_mask = window.band_mask

        hologram_frames = HologramFrames(
            region,
            bitdepth=self.slm.bitdepth,
            band_mask=band_mask,
            dtype=window.target.dtype,
        )
        records: list[_IterationRecord] = []
        retrievals: list[PhaseRetrievalData] = []
        history: dict[str, list[float]] = {}
        served_target = window.target
        last_step: _Step | None = None
        final_camera_image: NDArray | None = None

        with self._retrieval_loop(iterations, retriever_steps, verbose) as (
            bar,
            retriever_bar,
        ):
            for iteration in range(iterations):
                retrieval = self._retrieve_and_display(
                    retriever_steps[iteration],
                    iteration,
                    metrics,
                    retriever_bar,
                    **options,
                )
                retrievals.append(retrieval.lean())
                first_iteration = iteration == 0
                measurement = self._measure(
                    retrieval.phase,
                    first_iteration,
                    defocus_phases,
                    diversity,
                    averages,
                    exposure,
                    autoexpose,
                    window,
                )
                final_camera_image = measurement.captured
                frame = measurement.frame
                hologram_frames.append(
                    self.slm.phase_to_levels(retrieval.phase), frame
                )
                for defocused in measurement.defocused:
                    hologram_frames.append(defocused.levels, defocused.frame)

                response_ratio = (
                    None
                    if last_step is None
                    else _response_ratio(
                        fourier_filter(
                            (frame - last_step.frame) * region, band_mask
                        ),
                        measurement.prediction - last_step.prediction,
                        region,
                    )
                )
                learning_cost = (
                    self._learn(hologram_frames, learning)
                    if learning.parameters
                    else None
                )
                # The learned model's intensity of the displayed hologram.
                learned_prediction = self._normalized_prediction(region)
                step = self._step(
                    retrieval.phase,
                    measurement,
                    first_iteration,
                    learned_prediction,
                    window,
                    gains[iteration],
                )

                records.append(
                    _IterationRecord(
                        exposure=measurement.exposure,
                        camera_gain=measurement.camera_gain,
                        overexposed=measurement.overexposed,
                        translations=[fit.shift for fit in measurement.fits],
                        pattern_offset=step.pattern_offset,
                        noise_floor=filtered_noise_rms(frame, region, band_mask),
                        innovation=_region_rms(frame - measurement.prediction, region),
                        defocused_innovations=[
                            _region_rms(defocused.frame - defocused.prediction, region)
                            for defocused in measurement.defocused
                        ],
                        reexposed_defocused=[
                            index
                            for index, defocused in enumerate(measurement.defocused)
                            if defocused.reexposed
                        ],
                        residual=_region_rms(
                            fourier_filter(
                                (frame - learned_prediction) * region, band_mask
                            ),
                            region,
                        ),
                        response_ratio=response_ratio,
                        learning_cost=learning_cost,
                        psf_kernel=self._psf_kernel_copy(),
                        corrected_target=gpu_to_numpy(served_target),
                        measured_image=gpu_to_numpy(frame),
                    )
                )
                served_target = step.next_target
                last_step = step

                evaluate_metrics(
                    metrics,
                    gpu_to_numpy(region).astype(bool),
                    gpu_to_numpy(step.placed_target),
                    gpu_to_numpy(frame),
                    history,
                )
                bar.update(
                    **{label: values[-1] for label, values in history.items()}
                )

        overexposed_iterations = [
            index for index, record in enumerate(records) if record.overexposed
        ]
        if overexposed_iterations:
            self._warn_overexposed(f"iterations {overexposed_iterations}")

        return self._record(
            name,
            gpu_to_numpy(window.sensor_target),
            window.sensor_region,
            metrics,
            history,
            corrected_targets=[record.corrected_target for record in records],
            measured_images=[record.measured_image for record in records],
            final_camera_image=final_camera_image,
            initial_guess=initial_guess,
            retrievals=retrievals,
            metadata=_metadata(
                records,
                gains,
                learning_method,
                learning.steps,
                defocus_distances,
                diversity,
            ),
        )

    def _defocus_phases(self, distances: Sequence[float]) -> list[NDArray]:
        """The phase each defocus distance adds to a hologram
        (:func:`~hologradpy.profiles.phase.defocus_phase`).
        """
        slm_grid = self.slm.get_spatial_grid()
        return [
            gpu_to_numpy(
                defocus_phase(
                    *slm_grid,
                    distance,
                    self.slm_camera_model.focal_length,
                    2 * np.pi / self.slm.wavelength,
                )
            )
            for distance in distances
        ]

    def _random_diversity(
        self, target: Tensor, defocus_phases: Sequence[NDArray]
    ) -> _RandomDiversity:
        """The random phase every defocused hologram carries, with the RMS and the band
        of ``random_diversity`` in :meth:`run`.

        A phase ripple of frequency ``nu`` deflects light by ``lambda f nu`` in the
        image plane. To first order in the phase, the band therefore keeps the light
        deflected from the plateau inside the window. The model computes the window
        alone.

        Args:
            target: The target on the bounding box of the signal region.
            defocus_phases: The phase each defocus distance adds to a hologram.

        Raises:
            ValueError: The band holds no spatial frequency beyond the piston.
        """
        with torch.no_grad():
            beam = self.slm_camera_model.slm_field.get_wavefront().detach()
        weights = (beam.abs() ** 2).to(device="cpu", dtype=torch.float64)
        weights = weights.reshape(defocus_phases[0].shape)
        rms = max(
            _weighted_rms(torch.as_tensor(phase, dtype=torch.float64), weights)
            for phase in defocus_phases
        )
        band_radius = _plateau_gap(target, self._mapping.sensor_pixels_per_metre)
        frequency_x, frequency_y = get_frequency_grid(
            tuple(self.slm.resolution),
            torch.tensor(self.slm.pixel_size, dtype=torch.float64),
        )
        band_frequency = (2 * np.pi / self.slm.wavelength) * tilt_to_angle(
            band_radius, focal_length=self.slm_camera_model.focal_length
        )
        band_mask = torch.hypot(frequency_x, frequency_y) <= band_frequency
        if int(band_mask.sum()) <= 1:
            raise ValueError(
                f"The target's plateau lies {band_radius * 1e6:.1f} um from the edge "
                "of the signal region in the image plane. The random phase then has "
                "no spatial frequency beyond the piston. Give the signal region a "
                "wider margin around the plateau."
            )
        return _RandomDiversity(band_mask, weights, rms, band_radius)

    def _learning(
        self,
        learning_method: str,
        learning_steps: int | None,
        learning_rate: float,
        learned_parameters: Sequence[str],
        holograms_per_iteration: int,
    ) -> _Learning:
        """The learning of ``learning_method``, with its default steps when
        ``learning_steps`` is None. A learning of zero steps fits no parameters.

        Raises:
            ValueError: The method is not one of the learning methods.
        """
        if learning_method not in _LEARNING_TYPES:
            raise ValueError(
                f"learning_method is one of {', '.join(_LEARNING_TYPES)}, not "
                f"{learning_method!r}."
            )
        learning_type = _LEARNING_TYPES[learning_method]
        steps = (
            learning_type.default_steps
            if learning_steps is None
            else int(learning_steps)
        )
        parameters = (
            self._learned_parameter_list(learned_parameters) if steps > 0 else []
        )
        return learning_type(
            self, parameters, steps, holograms_per_iteration, learning_rate
        )

    def _measure(
        self,
        phase: NDArray,
        first_iteration: bool,
        defocus_phases: Sequence[NDArray],
        diversity: _RandomDiversity | None,
        averages: int,
        exposure: float | None,
        autoexpose: bool,
        window: SignalWindow,
    ) -> _Measurement:
        """Measure the displayed hologram in focus and defocused, with the model's
        prediction before the learning.

        The frames in focus are moved as in step 1 of the class description. Each
        defocused hologram is displayed with its own pointing, and its frames are moved
        to the place of the frames in focus. One displacement then remains in the first
        iteration for the learning to take up. Every defocused measurement starts from
        the exposure and the gain of the frames in focus. The SLM is left showing the
        hologram measured in focus.
        """
        region = window.region
        sensor_region = window.sensor_region
        prediction = self._normalized_prediction(region)
        camera_exposure, camera_gain = self._expose(autoexpose, exposure, sensor_region)
        frame, fits, overexposed, captured = self._aligned_frame(
            averages,
            prediction,
            region,
            sensor_region,
            None if first_iteration else (0.0, 0.0),
        )
        landed_shift = tuple(np.mean([fit.shift for fit in fits], axis=0).tolist())
        destination = landed_shift if first_iteration else (0.0, 0.0)
        defocused_frames: list[_DefocusedFrame] = []
        for added_phase in defocus_phases:
            if diversity is not None:
                added_phase = added_phase + diversity.draw()
            with self.camera.preserve_exposure_gain_and_roi():
                defocused, defocused_overexposed = self._defocused_frame(
                    phase + added_phase,
                    autoexpose,
                    averages,
                    region,
                    sensor_region,
                    destination,
                )
            defocused_frames.append(defocused)
            overexposed = overexposed or defocused_overexposed
        if defocus_phases:
            self.slm.set_phase(phase)
        return _Measurement(
            prediction=prediction,
            frame=frame,
            fits=fits,
            landed_shift=landed_shift,
            captured=captured,
            overexposed=overexposed,
            exposure=camera_exposure,
            camera_gain=camera_gain,
            defocused=defocused_frames,
        )

    def _step(
        self,
        phase: NDArray,
        measurement: _Measurement,
        first_iteration: bool,
        learned_prediction: Tensor,
        window: SignalWindow,
        gain: float,
    ) -> _Step:
        """Give the retriever its next target, and start the next retrieval from the
        displayed hologram moved back onto the target (step 3 of the class
        description).

        The translation that moves the target onto the measured pattern is recorded as
        where the pattern landed. After the first iteration, the frame sits on the
        prediction, so the mean translation of its frames is added back. The next
        pattern is moved back by the offset of the learned prediction from the target,
        which holds no pointing jitter of the frames. The error moves with the pattern.
        """
        target = window.target
        region = window.region
        frame = measurement.frame
        placed_target, pattern_offset = self._place_target_on(frame, target, region)
        if not first_iteration:
            pattern_offset = (
                pattern_offset[0] + measurement.landed_shift[0],
                pattern_offset[1] + measurement.landed_shift[1],
            )
        model_offset = fit_translation(learned_prediction, target, region).shift
        back = (-model_offset[0], -model_offset[1])
        self._start_from_moved_hologram(phase, back)
        moved_prediction = self._normalized_prediction(region)
        moved_frame = translate_real(frame, back)
        next_target = corrected_target_intensity(
            moved_prediction,
            moved_frame,
            translate_real(placed_target, back),
            region,
            gain,
            band_mask=window.band_mask,
        )
        self.update_target(next_target.to(target.dtype))
        return _Step(
            placed_target=placed_target,
            pattern_offset=pattern_offset,
            next_target=next_target,
            prediction=moved_prediction,
            frame=moved_frame,
        )

    def _learn(self, frames: HologramFrames, learning: _Learning) -> float:
        """Refit the learned parameters on the holograms and frames so far
        (:class:`_Learning`), and return the final cost.

        The retriever's hologram is put back afterwards. The parameters are left
        without gradients, so the retrieval holds the field they describe.
        """
        for parameter in self.slm_camera_model.parameters():
            parameter.requires_grad_(False)
        try:
            return learning.learn(frames)
        finally:
            for parameter in learning.parameters:
                parameter.requires_grad_(False)

    def _predict_intensity(self, levels: NDArray) -> Tensor:
        """The model's intensity of the hologram ``levels``, on the target's device and
        in its dtype. The retriever's hologram is put back afterwards.
        """
        virtual_slm = self.slm_camera_model.virtual_slm
        with virtual_slm.preserve_levels():
            virtual_slm.set_levels(levels, self.slm.bitdepth)
            with torch.no_grad():
                intensity = as_image(self.slm_camera_model().intensity)
        return intensity.to(dtype=self.target.dtype, device=self.target.device)

    def _defocused_frame(
        self,
        phase: NDArray,
        autoexpose: bool,
        averages: int,
        region: Tensor,
        sensor_region: NDArray[np.bool_],
        destination: tuple[float, float] | None = None,
    ) -> tuple[_DefocusedFrame, bool]:
        """Display a defocused hologram and measure it like the hologram in focus, at
        the camera's current exposure.

        Args:
            phase: The hologram plus its defocus, in radians.
            autoexpose: Autoexpose on the signal region and measure again when a frame
                is overexposed inside the region.
            averages: The number of frames.
            region: True inside the signal region, on the bounding box of the region.
            sensor_region: The signal region on the whole sensor.
            destination: Where every frame is moved relative to the model's
                prediction (:meth:`_aligned_frame`).

        Returns:
            tuple[_DefocusedFrame, bool]: The measurement, and whether any frame of the
            last measurement was overexposed inside the region.
        """
        self.slm.set_phase(phase)
        levels = self.slm.phase_to_levels(phase)
        intensity = self._predict_intensity(levels)
        prediction = normalize(intensity, region.to(intensity.dtype))
        frame, _, overexposed, _ = self._aligned_frame(
            averages, prediction, region, sensor_region, destination
        )
        reexposed = autoexpose and overexposed
        if reexposed:
            self._autoexpose(sensor_region)
            frame, _, overexposed, _ = self._aligned_frame(
                averages, prediction, region, sensor_region, destination
            )
        return _DefocusedFrame(levels, prediction, frame, reexposed), overexposed

    def _learned_parameter_list(
        self, names: Sequence[str]
    ) -> list[torch.nn.Parameter]:
        """The model parameters named by ``names``, without the partial affine and
        the hologram.

        Raises:
            ValueError: Nothing matches, or the names select a pixelwise SLM field.
        """
        model = self.slm_camera_model
        excluded = {id(model.virtual_slm.levels)}
        partial_affine = model.focal_plane_partial_affine
        if partial_affine is not None:
            excluded.update(id(parameter) for parameter in partial_affine.parameters())

        chosen = [
            (label, parameter)
            for label, parameter in model.named_parameters()
            if id(parameter) not in excluded
            and any(label == name or label.startswith(name + ".") for name in names)
        ]
        if not chosen:
            available = ", ".join(label for label, _ in model.named_parameters())
            raise ValueError(
                f"No parameter of the model matches {list(names)}. Its parameters are "
                f"{available}."
            )
        if isinstance(model.slm_field, PixelwiseSLMField) and any(
            label.startswith("slm_field.") for label, _ in chosen
        ):
            raise ValueError(
                "The model's SLM field is a PixelwiseSLMField, a free value per pixel, "
                "which the frames of a few holograms cannot determine. Build the model "
                "with PSFSLMField.from_calibration_data. Its kernel band limits what "
                "is learned."
            )
        return [parameter for _, parameter in chosen]

    def _start_from_moved_hologram(
        self, phase: NDArray, shift: tuple[float, float]
    ) -> None:
        """Set the model's hologram to ``phase`` plus the linear phase that moves its
        pattern by ``shift``, so the next retrieval starts from it. ``shift`` is
        ``(columns, rows)`` in sensor pixels.

        The inverse of the camera mapping's linear part
        (:attr:`~hologradpy.calibration.camera_mapping.mapping.CameraMapping.sensor_pixels_per_metre`)
        turns the shift on the sensor into one in the image plane.
        :meth:`~hologradpy.optics.systems.SLMFourierLensModel.steering_phase` gives the
        tilt that moves the pattern by it.
        """
        shift_x, shift_y = (
            float(value)
            for value in np.linalg.solve(self._mapping.sensor_pixels_per_metre, shift)
        )
        tilt = self.slm_camera_model.steering_phase((shift_x, shift_y))
        self.slm_camera_model.virtual_slm.set_phase(
            np.asarray(phase) + gpu_to_numpy(tilt)
        )

    def _psf_kernel_copy(self) -> NDArray | None:
        """The SLM field's PSF kernel, or None for a field without one."""
        kernel = getattr(self.slm_camera_model.slm_field, "get_psf_kernel", None)
        if kernel is None:
            return None
        return gpu_to_numpy(kernel().detach()).copy()


class _Learning(ABC):
    """How the corrector refits its model to the holograms and frames so far."""

    #: The steps of a learning when :meth:`ModelLearningFeedbackCorrector.run` is given
    #: none.
    default_steps: int

    def __init__(
        self,
        corrector: ModelLearningFeedbackCorrector,
        parameters: list[torch.nn.Parameter],
        steps: int,
        holograms_per_iteration: int,
        learning_rate: float,
    ) -> None:
        self.corrector = corrector
        self.parameters = parameters
        self.steps = steps
        self.holograms_per_iteration = holograms_per_iteration
        self.learning_rate = learning_rate

    @abstractmethod
    def learn(self, frames: HologramFrames) -> float:
        """Refit the parameters on the holograms and frames so far, and return the
        final cost.
        """


class _LevenbergMarquardtLearning(_Learning):
    """Levenberg-Marquardt over every stored hologram in one chunk
    (:meth:`~hologradpy.calibration.model_fit.ModelFitter.fit_levenberg_marquardt`),
    with the noise floor of the newest frame in focus.
    """

    default_steps = 5

    def learn(self, frames: HologramFrames) -> float:
        fitter = ModelFitter(self.corrector.slm_camera_model, frames, self.parameters)
        costs = fitter.fit_levenberg_marquardt(
            self.steps,
            chunk_size=None,
            noise_frames=[len(frames) - self.holograms_per_iteration],
            verbose=False,
        )
        return costs[-1]


class _AdamLearning(_Learning):
    """Adam steps on the masked intensity cost of the wavefront calibration
    (:meth:`~hologradpy.calibration.model_fit.ModelFitter.fit_batches`).

    Every step fits the newest iteration's holograms and as many older ones, drawn at
    random for each step (:class:`_NewestAndOlderHolograms`). A step then costs the
    same however many holograms are stored. One fitter, and so one optimizer, serves
    every learning of a run.
    """

    default_steps = 50
    fitter: ModelFitter | None = None

    def learn(self, frames: HologramFrames) -> float:
        if self.fitter is None:
            self.fitter = ModelFitter(
                self.corrector.slm_camera_model,
                frames,
                self.parameters,
                learning_rate=self.learning_rate,
            )
        batches = DataLoader(
            frames,
            batch_sampler=_NewestAndOlderHolograms(
                len(frames), self.holograms_per_iteration, self.steps
            ),
        )
        return self.fitter.fit_batches(batches)[-1]


class _NewestAndOlderHolograms(Sampler[list[int]]):
    """``steps`` batches of positions, each the newest iteration's holograms and as
    many older ones, drawn at random for each batch.
    """

    def __init__(
        self, number_of_holograms: int, holograms_per_iteration: int, steps: int
    ) -> None:
        self.number_of_holograms = number_of_holograms
        self.holograms_per_iteration = holograms_per_iteration
        self.steps = steps

    def __iter__(self):
        older = self.number_of_holograms - self.holograms_per_iteration
        newest = list(range(older, self.number_of_holograms))
        for _ in range(self.steps):
            drawn = torch.randperm(older)[: self.holograms_per_iteration]
            yield newest + drawn.tolist()

    def __len__(self) -> int:
        return self.steps


# The learnings, keyed by the values of learning_method in
# ModelLearningFeedbackCorrector.run.
_LEARNING_TYPES: dict[str, type[_Learning]] = {
    "levenberg_marquardt": _LevenbergMarquardtLearning,
    "adam": _AdamLearning,
}


@dataclass
class _DefocusedFrame:
    """A defocused copy of an iteration's hologram, measured like the hologram in
    focus.
    """

    #: The displayed levels.
    levels: NDArray
    #: The model's intensity of the levels, divided by the mean of its bright pixels.
    prediction: Tensor
    #: The aligned frame, divided by the mean of its bright pixels.
    frame: Tensor
    #: Whether the frame was overexposed at the exposure of the frames in focus, and
    #: was autoexposed and measured again.
    reexposed: bool


@dataclass
class _RandomDiversity:
    """The random phase every defocused hologram carries, drawn anew for each
    (:func:`~hologradpy.profiles.phase.random_diversity_phase`).
    """

    #: The band of the phase, on the centred frequency grid of the SLM.
    band_mask: Tensor
    #: The intensity of the model's beam. It weights the RMS, the piston and the tilt.
    weights: Tensor
    #: The RMS of the phase over the beam, in radians.
    rms: float
    #: The image-plane distance in metres that sets the band. The band is the disc of
    #: SLM spatial frequency ``band_radius / (lambda f)``.
    band_radius: float

    def draw(self) -> NDArray:
        """A new phase in radians, from PyTorch's global generator."""
        return gpu_to_numpy(
            random_diversity_phase(self.band_mask, self.weights, self.rms)
        )


@dataclass
class _Measurement:
    """The frames of one iteration's hologram, in focus and defocused."""

    #: The model's intensity of the hologram before it learns from the frames.
    prediction: Tensor
    #: The aligned frame in focus, divided by the mean of its bright pixels.
    frame: Tensor
    #: The translation fit of every frame in focus.
    fits: list[TranslationFit]
    #: The mean translation of the frames in focus from the prediction,
    #: ``(columns, rows)`` in sensor pixels.
    landed_shift: tuple[float, float]
    #: The mean of the frames in focus as captured, on the whole sensor.
    captured: NDArray
    #: Whether any frame, in focus or defocused, was overexposed inside the region.
    overexposed: bool
    #: The exposure in seconds the frames in focus were measured at.
    exposure: float
    #: The camera gain in dB the frames in focus were measured at.
    camera_gain: float
    #: The defocused copies.
    defocused: list[_DefocusedFrame]


@dataclass
class _Step:
    """The step of one iteration toward the target."""

    #: The target moved onto the measured pattern.
    placed_target: Tensor
    #: Where the pattern landed relative to the target, ``(columns, rows)`` in sensor
    #: pixels.
    pattern_offset: tuple[float, float]
    #: The intensity the next retrieval is given.
    next_target: Tensor
    #: The learned model's intensity of the moved hologram.
    prediction: Tensor
    #: The frame in focus, moved with the hologram.
    frame: Tensor


@dataclass
class _IterationRecord:
    """What one iteration of the loop records."""

    exposure: float
    camera_gain: float
    overexposed: bool
    translations: list[tuple[float, float]]
    pattern_offset: tuple[float, float]
    noise_floor: float | None
    innovation: float
    defocused_innovations: list[float]
    reexposed_defocused: list[int]
    residual: float
    response_ratio: float | None
    learning_cost: float | None
    psf_kernel: NDArray | None
    corrected_target: NDArray
    measured_image: NDArray


def _metadata(
    records: Sequence[_IterationRecord],
    gains: Sequence[float],
    learning_method: str,
    learning_steps: int,
    defocus_distances: Sequence[float],
    diversity: _RandomDiversity | None,
) -> dict:
    """The metadata of a run, one entry per iteration in each list."""
    metadata = {
        "exposures": [record.exposure for record in records],
        "camera_gains": [record.camera_gain for record in records],
        "overexposed_iterations": [
            index for index, record in enumerate(records) if record.overexposed
        ],
        "gains": [float(value) for value in gains],
        "learning_method": learning_method,
        "learning_steps": int(learning_steps),
        "translations": [record.translations for record in records],
        "pattern_offsets": [record.pattern_offset for record in records],
        "noise_floor": [record.noise_floor for record in records],
        "innovation": [record.innovation for record in records],
        "residual": [record.residual for record in records],
        "response_ratio": [record.response_ratio for record in records],
        "learning_cost": [record.learning_cost for record in records],
    }
    if len(defocus_distances) > 0:
        metadata["defocus_distances"] = [
            float(distance) for distance in defocus_distances
        ]
        metadata["defocused_innovation"] = [
            record.defocused_innovations for record in records
        ]
        metadata["reexposed_defocused_frames"] = [
            (iteration, index)
            for iteration, record in enumerate(records)
            for index in record.reexposed_defocused
        ]
    metadata["random_diversity"] = diversity is not None
    if diversity is not None:
        metadata["random_diversity_rms"] = diversity.rms
        metadata["random_diversity_band_radius"] = diversity.band_radius
    kernels = [record.psf_kernel for record in records if record.psf_kernel is not None]
    if kernels:
        metadata["psf_kernels"] = kernels
    return metadata


def _region_rms(image: Tensor, region: Tensor) -> float:
    """The RMS of ``image`` over the region."""
    return float(torch.sqrt((image[region] ** 2).mean()))


def _weighted_rms(phase: Tensor, weights: Tensor) -> float:
    """The RMS of ``phase`` about its mean, both weighted by ``weights``."""
    mean = (weights * phase).sum() / weights.sum()
    return float(torch.sqrt((weights * (phase - mean) ** 2).sum() / weights.sum()))


def _plateau_gap(target: Tensor, sensor_pixels_per_metre: NDArray) -> float:
    """The smallest image-plane distance in metres between the plateau of ``target``
    and the edge of its window.

    The plateau is the target above :data:`PLATEAU_FRACTION` of its peak. Each gap is
    counted in pixels between the plateau's outermost pixel and the window's outermost
    pixel, along a row or a column. The inverse of the camera mapping's linear part
    turns a gap across an edge of the window into the distance across the image of
    that edge, so a rotated or sheared camera is measured in the image plane.

    Args:
        target: The target on its window, the bounding box of the signal region.
        sensor_pixels_per_metre: The linear part of the camera mapping
            (:attr:`~hologradpy.calibration.camera_mapping.mapping.CameraMapping.sensor_pixels_per_metre`).
    """
    plateau = gpu_to_numpy(target >= PLATEAU_FRACTION * target.max())
    rows = np.flatnonzero(plateau.any(axis=1))
    columns = np.flatnonzero(plateau.any(axis=0))
    height, width = plateau.shape
    row_gap = min(rows[0], height - 1 - rows[-1])
    column_gap = min(columns[0], width - 1 - columns[-1])
    # From a sensor (column, row) step to the image-plane (x, y) step in metres.
    metres_per_pixel = np.linalg.inv(sensor_pixels_per_metre)
    area = abs(np.linalg.det(metres_per_pixel))
    # A gap across rows meets an edge along the columns, and the reverse.
    return float(
        min(
            row_gap * area / np.linalg.norm(metres_per_pixel[:, 0]),
            column_gap * area / np.linalg.norm(metres_per_pixel[:, 1]),
        )
    )


def _response_ratio(
    measured_change: Tensor, predicted_change: Tensor, region: Tensor
) -> float | None:
    """The projection of the measured change onto the predicted one, over the
    region. One when the model predicted the step's effect, and None when it predicted
    no change.
    """
    predicted = predicted_change[region]
    power = float((predicted**2).sum())
    if power <= 0.0:
        return None
    return float((measured_change[region] * predicted).sum()) / power
