from __future__ import annotations

import numpy as np

import torch
from torch._prims_common import corresponding_complex_dtype

from ..utils.tensor_utils import (
    pad_to_shape_2D,
    unsqueeze_to,
)
from ..utils.fourier_utils import get_frequency_grid, fft_2d, ifft_2d

from .abstract import PropagatorBase


class AngularSpectrumMethod(PropagatorBase):
    def __init__(
        self: AngularSpectrumMethod,
        propagation_distance: float,
        resolution_in: tuple[int, int],
        pixel_pitch_in: float,
        padded_resolution: tuple[int, int] | None = None,
        fft_kwargs: dict = {},
        device: str = "cpu",
    ) -> None:
        self.propagation_distance = propagation_distance

        if padded_resolution is None:
            padded_resolution = tuple(2 * resolution_in[i] for i in range(2))
        self.padded_resolution = padded_resolution

        self.fft_kwargs = fft_kwargs

        super().__init__(
            resolution_in=resolution_in,
            pixel_size_in=(pixel_pitch_in, pixel_pitch_in),
            device=device,
        )

        self.frequency_grid = get_frequency_grid(
            self.padded_resolution, self.pixel_size_in, device=self.device
        )

        self.phase_factor = (
            self.get_phase_factor(self.propagation_distance).to(
                corresponding_complex_dtype(self.dtype)
            )
        )

    @property
    def pixel_size_out(self: AngularSpectrumMethod) -> tuple[float, float]:
        return self.pixel_size_in

    @property
    def resolution_out(self: AngularSpectrumMethod) -> tuple[int, int]:
        return self.padded_resolution

    def get_phase_factor(
        self: AngularSpectrumMethod, propagation_distance: float
    ) -> torch.Tensor:
        return torch.exp(
            1j * propagation_distance * 
            torch.sqrt(
                self.wavenumber ** 2 - self.frequency_grid[0] ** 2 
                 - self.frequency_grid[1] ** 2 + 0j
            )
        )

    def forward(
        self: AngularSpectrumMethod, input_field: torch.Tensor
    ) -> torch.Tensor:
        input_field = unsqueeze_to(input_field, 3)
        phase_factor = unsqueeze_to(self.phase_factor, 3)

        padded_field = pad_to_shape_2D(input_field, self.padded_resolution)
        angular_spectrum = fft_2d(padded_field, **self.fft_kwargs)
        return ifft_2d(angular_spectrum * phase_factor, **self.fft_kwargs)
