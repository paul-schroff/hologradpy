"""Fit an SLM-plane field to a captured dataset of phase patterns and camera frames."""

from __future__ import annotations

from typing import Iterable

import torch

from ....calibration.speckle.fitter import SpeckleFitter

__all__ = ["WavefrontFitter"]


class WavefrontFitter(SpeckleFitter):
    """Recover the SLM-plane field by fitting it to captured camera frames."""

    description = "Fitting wavefront"

    def trainable_parameters(self) -> Iterable[torch.nn.Parameter]:
        """The SLM-plane field and the parameters already enabled on the model, except
        the focal-plane partial affine. The affine is frozen, so it keeps the values
        calibrated from the camera mapping.
        """
        for parameter in self.slm_camera_model.slm_field.parameters():
            parameter.requires_grad_(True)
        partial_affine = self.slm_camera_model.focal_plane_partial_affine
        if partial_affine is not None:
            partial_affine.requires_grad_(False)
        return self.enabled_parameters()

    def get_wavefront(self) -> torch.Tensor:
        """The recovered SLM-plane complex field, whatever parameterized it.

        Delegated to the field module, which is the thing that knows: stored directly by
        a ``PixelwiseSLMField``, mapped from the fitted kernel by a ``PSFSLMField``.
        """
        return self.slm_camera_model.slm_field.get_wavefront()
