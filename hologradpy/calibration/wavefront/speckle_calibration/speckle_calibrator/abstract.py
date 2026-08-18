from __future__ import annotations
from typing import ClassVar, Sequence

from abc import abstractmethod
from datetime import datetime

import numpy as np

import torch

from ..wavefront_fitter import WavefrontFitter
from .....calibration.speckle import SpeckleCaptureData
from .....calibration.speckle.calibrator import FitSettings, SpeckleCalibrator

from .....datasets import CaptureStore
from ..visualizer import SpeckleVisualizationData

from ...abstract import WavefrontCalibrationData

from .....analysis.error_metrics import (
    DEFAULT_WAVEFRONT_METRICS,
    WavefrontMetric,
    evaluate_wavefront_metrics,
)
from .....analysis.fitting import remove_tilt



from .....optics.complex_amplitude import ComplexAmplitude
from .....optics.modules.slm_fields import SLMField

BEAM_MASK_THRESHOLD_DEFAULT = float(np.exp(-4.0))  # ~0.0183

class WavefrontSpeckleCalibrator(SpeckleCalibrator):
    """Recover the SLM-plane beam from captured speckle."""

    fitter_type: ClassVar[type[WavefrontFitter]] = WavefrontFitter
    slm_field_type: ClassVar[type[SLMField]] = SLMField
    visualization_data_type: ClassVar[type[SpeckleVisualizationData]] = (
        SpeckleVisualizationData
    )

    def _prepare_model(self) -> None:
        """Give the model the parameters this calibrator fits, and turn them on."""
        self._prepare_slm_field()
        for parameter in self.slm_camera_model.slm_field.parameters():
            parameter.requires_grad_(True)

    def _prepare_slm_field(self) -> None:
        """Generate the SLM-plane field this calibrator fits."""
        slm_field = self.slm_camera_model.slm_field
        if isinstance(slm_field, self.slm_field_type):
            return

        print(
            f"{type(self).__name__} fits a {self.slm_field_type.__name__}. Replacing "
            f"the model's {type(slm_field).__name__}."
        )

        self.slm_camera_model.slm_field = self._build_slm_field()

    @abstractmethod
    def _build_slm_field(self) -> SLMField:
        """Build the field from the camera mapping. Called only when the model does not 
        already have one of :attr:`slm_field_type`.
        """

    @abstractmethod
    def _fit_settings(self, mask: torch.Tensor) -> FitSettings:
        """The cost and step size this parameterization is fitted with.

        Args:
            mask: The region of interest, cropped to its bounding box and already in the
                model's dtype and on its device.
        """

    def _visualization_extras(self) -> dict:
        """Payload fields beyond the shared panels. Most parameterizations add none."""
        return {}

    def fit_wavefront(self, *args, **kwargs) -> list[float]:
        """Fit the SLM-plane field to a captured dataset. The wavefront name for
        :meth:`~hologradpy.calibration.speckle.calibrator.SpeckleCalibrator.fit`.
        """
        return self.fit(*args, **kwargs)

    def _injected_field(self) -> np.ndarray | None:
        return getattr(self.camera, "static_slm_field", None)

    def _comparison_mask(
        self, recovered_mask: np.ndarray, beam_mask_threshold: float
    ) -> np.ndarray:
        """The region that covers a comparison against the truth which is taken from the
        injected intensity when available. On real hardware, this falls back to the 
        recovered mask.
        """
        injected = self._injected_field()
        if injected is None:
            return recovered_mask

        intensity = np.abs(injected) ** 2
        return intensity > beam_mask_threshold * intensity.max()

    def dataset_visualization_data(
        self,
        capture_data: SpeckleCaptureData | None = None,
        sample_index: int = 0,
    ) -> SpeckleVisualizationData:
        """The captured dataset before any fitting.

        Args:
            capture_data: The dataset to show. Defaults to the one from the last
                :meth:`fit_wavefront`, so a fitted calibrator needs no argument.
            sample_index: Which captured pattern to show.

        Returns:
            SpeckleVisualizationData: Carrying only what a dataset holds.

        Raises:
            RuntimeError: If no dataset has been captured or loaded yet.
        """
        if capture_data is not None:
            self.capture_data = capture_data

        if self.capture_data is None:
            raise RuntimeError(
                "No dataset to show. Capture one with "
                "calibrator.dataset_generator.generate_dataset(...) and pass it as "
                "capture_data."
            )

        with CaptureStore.open(self.dataset_path) as store:
            sample = store.read(sample_index)
        return self.visualization_data_type(
            camera_image=sample["camera_image"],
            roi_mask=self.capture_data.roi_mask,
            slm_pattern=sample["slm_levels"],
        )

    def _build_visualization_data(
        self,
        recovered_amplitude: np.ndarray,
        recovered_phase: np.ndarray,
        beam_mask: np.ndarray,
    ) -> SpeckleVisualizationData | None:
        """Collect the panels' inputs, or None if anything needed is missing.

        Visualization must never be the reason a calibration fails, so a problem
        here degrades to no figure rather than losing the recovered field.
        """
        try:
            measured_roi, predicted_roi = self.fitter.measured_and_predicted_roi(0)
            with CaptureStore.open(self.dataset_path) as store:
                sample = store.read(0)
            return self.visualization_data_type(
                camera_image=sample["camera_image"],
                roi_mask=self.capture_data.roi_mask,
                slm_pattern=sample["slm_levels"],
                measured_roi=measured_roi,
                predicted_roi=predicted_roi,
                recovered_amplitude=recovered_amplitude,
                recovered_phase=recovered_phase,
                loss_history=list(self.loss_history),
                loss_component_history={
                    label: list(values)
                    for label, values in self.loss_component_history.items()
                },
                injected_field=self._injected_field(),
                beam_mask=beam_mask,
                **self._visualization_extras(),
            )
        except (RuntimeError, OSError, KeyError, IndexError) as error:
            print(f"Could not collect visualization data ({error}).")
            return None

    def _residual_metrics(
        self,
        recovered_phase: np.ndarray,
        beam_mask: np.ndarray,
        metrics: Sequence[WavefrontMetric] | None = None,
    ) -> dict:
        """How close the recovered wavefront came, when the ground truth is known from
        simulated hardware. Uses named :class:`WavefrontMetric` callables.

        Args:
            recovered_phase: The wavefront that was recovered.
            beam_mask: Binary mask in which to evaluate the metrics.
            metrics: Which metric to use. Defaults to
                :data:`~hologradpy.analysis.error_metrics.DEFAULT_WAVEFRONT_METRICS`.

        Returns:
            dict: ``{name: value}``, empty when there is no ground truth to compare to.
        """
        injected = self._injected_field()
        if injected is None:
            return {}

        return evaluate_wavefront_metrics(
            DEFAULT_WAVEFRONT_METRICS if metrics is None else metrics,
            recovered_phase,
            np.angle(injected),
            beam_mask,
        )

    def generate_slm_beam_calibration(
        self, beam_mask_threshold: float = BEAM_MASK_THRESHOLD_DEFAULT
    ) -> WavefrontCalibrationData:
        """The recovered SLM-plane field, as a saveable calibration.

        Args:
            beam_mask_threshold: Fraction of the peak intensity above which a pixel
                counts as illuminated. Default is 1/e^4 (~0.0183).
        """
        

        field = self.fitter.get_wavefront().detach().cpu().numpy()

        intensity = np.abs(field) ** 2
        intensity /= np.max(intensity)
        try:
            beam_radius, shift_x, shift_y = self.fit_gaussian_beam(intensity)
        except (RuntimeError, ValueError) as error:
            print(f"Gaussian beam fit failed ({error}). Leaving beam metadata unset.")
            beam_radius = shift_x = shift_y = None

        slm_mask = intensity > beam_mask_threshold
        comparison_mask = self._comparison_mask(slm_mask, beam_mask_threshold)

        phase = np.angle(field)
        phase_no_tilt = remove_tilt(phase, mask=slm_mask)

        amplitude = np.sqrt(intensity)
        calibrated_field = amplitude * np.exp(1j * phase_no_tilt)

        complex_amplitude = ComplexAmplitude(
            torch.tensor(calibrated_field, dtype=torch.complex64),
            wavelength=torch.tensor(self.slm.wavelength),
            pixel_size=torch.tensor(tuple(self.slm.pixel_size)),
        )

        return WavefrontCalibrationData(
            timestamp=datetime.now(),
            name="speckle",
            complex_amplitude=complex_amplitude,
            metadata={
                "beam_radius": beam_radius,
                "shift_x": shift_x,
                "shift_y": shift_y,
                "focal_length": self.focal_length,
                "number_of_random_patterns": self.number_of_random_patterns,
                "beam_mask_threshold": beam_mask_threshold,
                **self._residual_metrics(phase_no_tilt, comparison_mask),
            },
            visualization_data=self._build_visualization_data(
                amplitude, phase_no_tilt, comparison_mask
            ),
        )

    def calibrate(
        self,
        speckle_pattern_extent: tuple[float, float] | None = None,
        number_of_epochs: int = 50,
        batch_size: int = 5,
        subset_indices: Sequence[int] | None = None,
        benchmark_calibration: WavefrontCalibrationData | None = None,
        seed: int | None = None,
        verbose: bool = True,
        beam_mask_threshold: float = BEAM_MASK_THRESHOLD_DEFAULT,
    ) -> WavefrontCalibrationData:
        """Capture a dataset, fit the model to it, and return the wavefront.

        The :class:`WavefrontCalibratorBase` entry point, so a caller can treat this and
        :class:`RasterCalibrator` interchangeably. To capture once and refit several
        times, call the two phases yourself: ``dataset_generator.generate_dataset(...)``
        followed by :meth:`fit_wavefront`.

        Args:
            speckle_pattern_extent: Full width ``(y, x)`` of the speckle at the camera,
                in metres, setting both the pattern band limit and the region of
                interest. Defaults to the largest speckle that fits on the
                sensor, measured from the camera mapping, so an off-axis camera is
                limited by whichever sensor edge the zeroth order sits closest to.
            number_of_epochs: Passes over the dataset.
            batch_size: Patterns per optimizer step.
            subset_indices: Fit only these patterns of the dataset. Defaults to all.
            benchmark_calibration: An existing calibration to add to every pattern, for
                measuring the residual of a previous fit.
            seed: Seed for the pattern noise.
            verbose: Print the loss as the fit runs.
            beam_mask_threshold: Fraction of the peak intensity above which a pixel
                counts as illuminated, passed to
                :meth:`generate_slm_beam_calibration`. Defaults to 1/e^4 (~0.0183).

        Returns:
            WavefrontCalibrationData: The fitted SLM-plane complex amplitude.
        """
        # A previous fit holds the file mapped, which blocks writing over it.
        self.release_dataset()

        capture_data = self.dataset_generator.generate_dataset(
            speckle_pattern_extent,
            benchmark_calibration=benchmark_calibration,
            seed=seed,
        )

        self.fit_wavefront(
            number_of_epochs=number_of_epochs,
            batch_size=batch_size,
            subset_indices=subset_indices,
            verbose=verbose,
            capture_data=capture_data,
        )

        return self.generate_slm_beam_calibration(
            beam_mask_threshold=beam_mask_threshold
        )
