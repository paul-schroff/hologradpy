"""Speckle wavefront calibration of a field stored directly, one value per SLM pixel."""

from __future__ import annotations

import torch

from .abstract import WavefrontSpeckleCalibrator
from ....speckle.calibrator import FitSettings

from .....loss_functions import (
    AmplitudeSmoothness,
    MaskedIntensityMSE,
    PhaseSmoothness,
)

from .....optics.modules.slm_fields import PixelwiseSLMField


class PixelwiseSpeckleCalibrator(WavefrontSpeckleCalibrator):
    """Recover the SLM-plane field stored directly, one complex value per SLM pixel."""

    slm_field_type = PixelwiseSLMField

    phase_smoothness: float = 1e-4
    amplitude_smoothness: float = 1e-4

    def _build_slm_field(self) -> PixelwiseSLMField:
        """An unmodulated field, which the fit then learns pixel by pixel."""
        return PixelwiseSLMField()

    def _fit_settings(self, mask: torch.Tensor) -> FitSettings:
        # One free value per SLM pixel, so it is unconstrained and can fit the speckle
        # with a rough, unphysical field. The smoothness terms suppress this.
        slm_field = self.slm_camera_model.slm_field
        return FitSettings(
            loss=(
                MaskedIntensityMSE(mask)
                + PhaseSmoothness(slm_field, self.phase_smoothness)
                + AmplitudeSmoothness(slm_field, self.amplitude_smoothness)
            ),
            learning_rate=1e-2,
        )
