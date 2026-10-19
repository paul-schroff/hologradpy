from __future__ import annotations
from typing import Any, ClassVar, Sequence

from abc import abstractmethod
from datetime import datetime

import numpy as np

import torch

from ..wavefront_fitter import WavefrontFitter
from .....calibration.speckle import SpeckleCaptureData
from .....calibration.speckle.calibrator import SpeckleCalibrator

from .....hardware.camera import Background
from ..visualizer import SpeckleVisualizationData

from ...abstract import WavefrontCalibrationData, WavefrontCalibratorBase

from .....analysis.error_metrics import (
    DEFAULT_WAVEFRONT_METRICS,
    WavefrontMetric,
    evaluate_wavefront_metrics,
)
from .....analysis.fitting import remove_tilt



from .....optics.complex_amplitude import ComplexAmplitude
from .....optics.modules.slm_fields import SLMField

BEAM_MASK_THRESHOLD_DEFAULT = float(np.exp(-4.0))  # ~0.0183

class WavefrontSpeckleCalibrator(SpeckleCalibrator, WavefrontCalibratorBase):
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

    def _fit_visualization_extras(self) -> dict:
        """Payload fields of the fitted field beyond the shared panels. Most
        parameterizations add none.
        """
        return {}

    def _calibration_metadata_extras(self) -> dict:
        """Metadata of the calibration record beyond the shared entries. Most
        parameterizations add none.
        """
        return {}

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

        :meth:`fit_beam` fits the calibration's ``beam`` to the field. If that fit
        fails, the beam is None.

        Args:
            beam_mask_threshold: Fraction of the peak intensity above which a pixel
                counts as illuminated. Default is 1/e^4 (~0.0183).
        """
        field = self.fitter.get_wavefront().detach().cpu().numpy()

        intensity = np.abs(field) ** 2
        intensity /= np.max(intensity)

        slm_mask = intensity > beam_mask_threshold
        comparison_mask = self._comparison_mask(slm_mask, beam_mask_threshold)

        phase = np.angle(field)
        phase_no_tilt = remove_tilt(phase, mask=slm_mask)

        amplitude = np.sqrt(intensity)
        calibrated_field = amplitude * np.exp(1j * phase_no_tilt)
        try:
            beam = self.fit_beam(calibrated_field)
        except (RuntimeError, ValueError) as error:
            print(f"Beam fit failed ({error}). Leaving the beam unset.")
            beam = None

        complex_amplitude = ComplexAmplitude(
            torch.tensor(calibrated_field, dtype=torch.complex64),
            wavelength=torch.tensor(self.slm.wavelength),
            pixel_size=torch.tensor(tuple(self.slm.pixel_size)),
        )

        return WavefrontCalibrationData(
            timestamp=datetime.now(),
            name="speckle",
            complex_amplitude=complex_amplitude,
            beam=beam,
            metadata={
                "focal_length": self.focal_length,
                "number_of_random_patterns": self.number_of_random_patterns,
                "beam_mask_threshold": beam_mask_threshold,
                "background_level": self.fitter.frames.background_level,
                **self._residual_metrics(phase_no_tilt, comparison_mask),
                **self._calibration_metadata_extras(),
            },
            visualization_data=self._build_visualization_data(
                recovered_amplitude=amplitude,
                recovered_phase=phase_no_tilt,
                injected_field=self._injected_field(),
                beam_mask=comparison_mask,
                **self._fit_visualization_extras(),
            ),
        )

    def calibrate(
        self,
        speckle_pattern_extent: tuple[float, float] | None = None,
        speckle_pattern_tilt: tuple[float, float] | None = None,
        subset_indices: Sequence[int] | None = None,
        benchmark_calibration: WavefrontCalibrationData | None = None,
        seed: int | None = None,
        verbose: bool = True,
        beam_mask_threshold: float = BEAM_MASK_THRESHOLD_DEFAULT,
        background: Background | None = None,
        **fit_settings: Any,
    ) -> WavefrontCalibrationData:
        """Capture a dataset, fit the model to it, and return the wavefront.

        This is the :class:`WavefrontCalibratorBase` entry point, so a caller can treat
        this calibrator and :class:`RasterCalibrator` interchangeably. A dataset
        captured once can be refitted several times with :meth:`fit`.

        Args:
            speckle_pattern_extent: Full width ``(y, x)`` of the speckle in the image
                plane, in metres, setting both the pattern band limit and the region of
                interest. Defaults to the largest speckle whose image fits on the
                sensor around the speckle centre, measured through the camera mapping.
            speckle_pattern_tilt: Image-plane position ``(x, y)`` of the speckle centre
                in metres, measured from the zeroth order. None steers the speckle to
                the sensor centre, and ``(0.0, 0.0)`` keeps it on the zeroth order.
            subset_indices: Fit only these patterns of the dataset. Defaults to all.
            benchmark_calibration: An existing calibration. Its correction, the
                negative of its phase, is added to every pattern, to measure the
                residual of a previous fit.
            seed: Seed for the pattern noise.
            verbose: Print the loss as the fit runs.
            beam_mask_threshold: Fraction of the peak intensity above which a pixel
                counts as illuminated, passed to :meth:`generate_slm_beam_calibration`.
            background: The counts subtracted from every frame before the fit, in any
                form :meth:`fit` takes. None subtracts nothing.
            **fit_settings: The settings of the parameterization's fit (:meth:`fit`).

        Returns:
            WavefrontCalibrationData: The fitted SLM-plane complex amplitude.
        """
        capture_data = self._capture_dataset(
            speckle_pattern_extent,
            speckle_pattern_tilt,
            benchmark_calibration,
            seed,
        )

        self.fit(
            capture_data=capture_data,
            background=background,
            subset_indices=subset_indices,
            verbose=verbose,
            **fit_settings,
        )

        return self.generate_slm_beam_calibration(
            beam_mask_threshold=beam_mask_threshold
        )

    def _capture_dataset(
        self,
        speckle_pattern_extent: tuple[float, float] | None,
        speckle_pattern_tilt: tuple[float, float] | None,
        benchmark_calibration: WavefrontCalibrationData | None,
        seed: int | None,
    ) -> SpeckleCaptureData:
        """Capture a dataset with :attr:`dataset_generator`. The arguments are those of
        :meth:`calibrate`.
        """
        return self.dataset_generator.generate_dataset(
            speckle_pattern_extent,
            tilt=speckle_pattern_tilt,
            benchmark_calibration=benchmark_calibration,
            seed=seed,
        )
