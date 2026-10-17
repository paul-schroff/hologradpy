"""Speckle wavefront calibration of a field stored directly, one value per SLM pixel."""

from __future__ import annotations

from typing import Sequence

from .abstract import WavefrontSpeckleCalibrator
from ....model_fit import ModelFitter

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

    def _fit(
        self,
        fitter: ModelFitter,
        subset_indices: Sequence[int] | None,
        verbose: bool,
        number_of_epochs: int = 50,
        batch_size: int = 5,
    ) -> list[float]:
        """Adam steps at a step size of 1e-2 on the masked intensity cost with two
        smoothness terms, and the mean loss of each epoch.

        The field holds one free value per SLM pixel, so the fit can match the speckle
        with a rough, unphysical field. The smoothness terms suppress this roughness.

        Args:
            fitter: The fitter of the dataset.
            subset_indices: Which dataset samples to fit against, or None for all.
            verbose: Show a progress bar.
            number_of_epochs: Passes over the dataset.
            batch_size: Phase patterns per step.
        """
        slm_field = self.slm_camera_model.slm_field
        loss = (
            MaskedIntensityMSE(fitter.frames.region.to(fitter.frames.dtype))
            + PhaseSmoothness(slm_field, self.phase_smoothness)
            + AmplitudeSmoothness(slm_field, self.amplitude_smoothness)
        )
        return self._fit_by_adam(
            fitter,
            loss,
            1e-2,
            subset_indices,
            verbose,
            number_of_epochs,
            batch_size,
        )
