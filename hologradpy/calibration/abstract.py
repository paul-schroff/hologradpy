from __future__ import annotations

from abc import ABC

import numpy as np
from numpy.typing import NDArray

import torch
from torch import Tensor

from ..hardware import Camera, SLM, as_camera, as_slm
from ..grids import get_spatial_grid
from ..profiles.gaussian import Gaussian, GaussianBeam
from ..utils import resolve_device


class CalibratorBase(ABC):
    """Devices normalized to the native interfaces, and the SLM-plane grid."""

    def __init__(
        self, slm: SLM, camera: Camera, device: torch.device | None = None
    ) -> None:
        """
        Args:
            slm: The SLM being driven, coerced to the native interface.
            camera: The camera watching it, likewise.
            device: Torch device the calculations run on, the CPU for None.
        """
        self.camera: Camera = as_camera(camera)
        self.slm: SLM = as_slm(slm)
        self.device: torch.device = resolve_device(device)

    @property
    def spatial_grid_slm(self) -> tuple[Tensor, Tensor]:
        """The ``(x, y)`` coordinates of the SLM pixels, in metres."""
        return get_spatial_grid(
            self.slm.resolution, self.slm.pixel_size, device=self.device
        )

    def fit_gaussian_beam(
        self, measured_intensity: NDArray[np.float64]
    ) -> GaussianBeam:
        """Fit a Gaussian beam to a measured SLM-plane intensity.

        Returns:
            GaussianBeam: The fitted beam, in metres on the SLM.
        """
        beam_radius_guess = min(self.slm.aperture_extent) / 2
        gaussian = Gaussian(self.slm.resolution, self.slm.pixel_size)
        return gaussian.fit(measured_intensity, beam_radius_guess, blur_sigma=10).beam
