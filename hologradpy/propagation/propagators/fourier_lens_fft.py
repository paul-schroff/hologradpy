from __future__ import annotations

import torch

from ..utils.tensor_utils import pad_to_shape_2D
from ..utils.fourier_utils import fft_2d

from .abstract import PropagatorBase

class FourierLensFft(PropagatorBase):
    def __init__(
        self: FourierLensFft,
        focal_length: float,
        wavelength: float,
        resolution_in: tuple[int, int],
        pixel_pitch_in: float,
        padded_resolution: tuple[int, int] = None,
        fft_kwargs: dict = {},
        device: str = "cpu",
    ) -> None:
        self.focal_length = focal_length
        self.wavelength = wavelength
        self.fft_kwargs = fft_kwargs

        if padded_resolution is None:
            padded_resolution = tuple(2 * resolution_in[i] for i in range(2))
        self.padded_resolution = padded_resolution
        self.padded_spatial_extent = tuple(
            self.padded_resolution[i] * pixel_pitch_in for i in range(2)
        )

        super().__init__(
            resolution_in=resolution_in,
            pixel_size_in=(pixel_pitch_in, pixel_pitch_in),
            device=device,
        )

    @property
    def pixel_size_out(self: FourierLensFft) -> tuple[float, float]:
        return tuple(
            self.wavelength * self.focal_length / self.padded_spatial_extent[i]
            for i in range(2)
        )

    @property
    def resolution_out(self: FourierLensFft) -> tuple[int, int]:
        return self.padded_resolution

    def forward(self: FourierLensFft, input_field: torch.Tensor) -> torch.Tensor:
        e_in = pad_to_shape_2D(input_field, self.padded_resolution)
        return fft_2d(e_in, **self.fft_kwargs)
