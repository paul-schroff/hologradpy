"""Fit a pixel-crosstalk model to captured speckle."""

from __future__ import annotations

from typing import Iterable

import torch

from ...speckle.fitter import SpeckleFitter
from ....optics.modules.pixel_crosstalk import PixelCrosstalk


class CrosstalkFitter(SpeckleFitter):
    """Recover the fringing-field kernel by fitting it to captured camera frames."""

    description = "Fitting pixel crosstalk"

    def trainable_parameters(self) -> Iterable[torch.nn.Parameter]:
        """The crosstalk model and the parameters already enabled on the model, except
        the SLM-plane beam and the focal-plane partial affine. The affine is frozen, so
        it keeps the values calibrated from the camera mapping.
        """
        model = self.slm_camera_model

        crosstalk = model.virtual_slm.pixel_crosstalk
        if crosstalk is None:
            raise RuntimeError(
                "The model's SLM stage carries no pixel crosstalk, so this fit has "
                "nothing to recover. Build the model with a VirtualSLM that carries "
                "the crosstalk model to fit."
            )

        for parameter in model.slm_field.parameters():
            parameter.requires_grad_(False)
        for parameter in crosstalk.parameters():
            parameter.requires_grad_(True)

        partial_affine = model.focal_plane_partial_affine
        if partial_affine is not None:
            partial_affine.requires_grad_(False)

        return self.enabled_parameters()

    def get_crosstalk(self) -> PixelCrosstalk | None:
        """The crosstalk model that was fitted."""
        return self.slm_camera_model.virtual_slm.pixel_crosstalk
