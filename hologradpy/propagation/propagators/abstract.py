from __future__ import annotations

import numpy as np

import torch
import torch.nn as nn
from torch._prims_common import (
    corresponding_complex_dtype,
    corresponding_real_dtype,
)

from ..utils.fourier_utils import get_spatial_grid


# %% Propagator base class
class PropagatorBase(nn.Module):
    dtype = torch.float32
    wavelength: float

    def __init__(
        self: PropagatorBase,
        resolution_in: tuple[int, int],
        pixel_size_in: tuple[float, float],
        device: str = "cpu",
    ) -> None:
        super().__init__()

        self.resolution_in = resolution_in
        self.pixel_size_in = pixel_size_in
        self.device = device

        self.spatial_extent_in = tuple(
            self.resolution_in[i] * self.pixel_size_in[i] for i in range(2)
        )

    @property
    def dtype_r(self: PropagatorBase) -> torch.dtype:
        if self.dtype.is_complex:
            return corresponding_real_dtype(self.dtype)
        else:
            return self.dtype

    @property
    def dtype_c(self: PropagatorBase) -> torch.dtype:
        if self.dtype.is_complex:
            return self.dtype
        else:
            return corresponding_complex_dtype(self.dtype)
        
    @property
    def eps(self: PropagatorBase) -> float:
        return torch.finfo(self.dtype_r).eps

    @property
    def wavenumber(self: PropagatorBase) -> float:
        return 2 * np.pi / self.wavelength

    @property
    def pixel_size_out(self: PropagatorBase) -> tuple[float, float]:
        return self.pixel_size_in

    @property
    def resolution_out(self: PropagatorBase) -> tuple[int, int]:
        return self.resolution_in

    @property
    def spatial_extent_out(self: PropagatorBase) -> tuple[float, float]:
        return tuple(
            self.resolution_out[i] * self.pixel_size_out[i] for i in range(2)
        )

    def get_spatial_grid_input(
        self: PropagatorBase,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return get_spatial_grid(
            self.resolution_in, self.pixel_size_in, self.device
        )

    def get_spatial_grid_output(
        self: PropagatorBase,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return get_spatial_grid(
            self.resolution_out, self.pixel_size_out, self.device
        )

    def forward(self: PropagatorBase) -> torch.Tensor:
        """Simulates the propagation of light between two planes."""
        raise NotImplementedError(
            "Each subclass should implements its own forward method."
        )

    def inverse(self: PropagatorBase) -> torch.Tensor:
        """This method should implement the inverse propagation of light.
        This is especially useful for iterative Fourier transform algorithms.
        """
        raise NotImplementedError(
            "Each subclass should implements its own inverse method."
        )
