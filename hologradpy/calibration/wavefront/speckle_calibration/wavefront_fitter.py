"""Fit an SLM-plane field to a captured dataset of phase patterns and camera frames."""

from __future__ import annotations

from typing import Iterable

import torch

from ....calibration.speckle.fitter import SpeckleFitter, region_of_interest

__all__ = ["WavefrontFitter", "region_of_interest"]


class WavefrontFitter(SpeckleFitter):
    """Recover the SLM-plane field by fitting it to captured camera frames."""

    description = "Fitting wavefront"

    def trainable_parameters(self) -> Iterable[torch.nn.Parameter]:
        """The SLM-plane field, plus anything the model already had enabled."""
        for parameter in self.slm_camera_model.slm_field.parameters():
            parameter.requires_grad_(True)
        return self.enabled_parameters()

    def get_wavefront(self) -> torch.Tensor:
        """The recovered SLM-plane complex field, whatever parameterized it.

        Delegated to the field module, which is the thing that knows: stored directly by
        a ``PixelwiseSLMField``, mapped from the fitted kernel by a ``PSFSLMField``.
        """
        return self.slm_camera_model.slm_field.get_wavefront()
