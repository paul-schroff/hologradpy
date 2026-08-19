from __future__ import annotations

from abc import ABC

import numpy as np
from numpy.typing import NDArray

import torch
from torch import Tensor

from ..hardware import Camera, SLM, as_camera, as_slm
from ..grids import get_spatial_grid
from ..analysis.fitting import fit_gaussian_beam_intensity


class CalibratorBase(ABC):
    """Devices normalized to the native interfaces, and the SLM-plane grid."""

    def __init__(self, slm: SLM, camera: Camera, device: torch.device = "cpu") -> None:
        """
        Args:
            slm: The SLM being driven, coerced to the native interface.
            camera: The camera watching it, likewise.
            device: Torch device the calculations run on.
        """
        self.camera: Camera = as_camera(camera)
        self.slm: SLM = as_slm(slm)
        self.device: torch.device = device

    @property
    def spatial_grid_slm(self) -> tuple[Tensor, Tensor]:
        """The ``(x, y)`` coordinates of the SLM pixels, in metres."""
        return get_spatial_grid(
            self.slm.resolution, self.slm.pixel_size, device=self.device
        )

    def fit_gaussian_beam(
        self, measured_intensity: NDArray[np.float64]
    ) -> tuple[float, float, float]:
        """Fit a Gaussian beam to a measured SLM-plane intensity.

        Returns:
            The fitted beam radius and the ``(x, y)`` shifts of its center, in metres.
        """
        beam_radius_guess = min(self.slm.aperture_extent) / 2

        # The fit runs on the CPU with numpy, so bring the coordinate grid over from the
        # (possibly CUDA) device.
        grid_x, grid_y = self.spatial_grid_slm
        popt, _ = fit_gaussian_beam_intensity(
            grid_x.cpu(),
            grid_y.cpu(),
            measured_intensity,
            beam_radius_guess,
            blur_sigma=10,
        )
        return popt[0], popt[1], popt[2]
