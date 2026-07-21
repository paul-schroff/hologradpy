from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor
from torch.fft import fftn, ifftn, fftshift, ifftshift

from .abstract import FourierBase
from ...grids import get_zoom_frequency_grid


def fft_2d(
    input: Tensor,
    norm: Literal["backward", "forward", "ortho"] = "backward",
    fft_shift: bool = True,
) -> Tensor:
    if fft_shift is True:
        return fftshift(
            fftn(ifftshift(input, dim=(-1, -2)), norm=norm, dim=(-1, -2)),
            dim=(-1, -2),
        )
    else:
        return fftn(input, norm=norm, dim=(-1, -2))


def ifft_2d(
    input: Tensor,
    norm: Literal["backward", "forward", "ortho"] = "backward",
    fft_shift: bool = True,
) -> Tensor:
    if fft_shift is True:
        return ifftshift(
            ifftn(fftshift(input, dim=(-1, -2)), norm=norm, dim=(-1, -2)),
            dim=(-1, -2),
        )
    else:
        return ifftn(input, norm=norm, dim=(-1, -2))


class FastFourierTransform(FourierBase):
    """The full DFT on the native, resolution-preserving bin grid (``is_gridded``).

    ``forward`` is :func:`fft_2d`, ``adjoint`` is :func:`ifft_2d`; with
    ``norm="ortho"`` they are exact conjugate transposes.
    """

    def __init__(
        self,
        resolution: tuple[int, int],
        device: torch.device = "cpu",
        norm: Literal["backward", "forward", "ortho"] = "backward",
        fft_shift: bool = True,
    ) -> None:
        omega_x, omega_y = get_zoom_frequency_grid(
            resolution, resolution, (1.0, 1.0), (0.0, 0.0), device
        )
        grid_x, grid_y = torch.meshgrid(omega_x, omega_y, indexing="xy")
        frequencies = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=0)
        super().__init__(
            resolution,
            frequencies=frequencies,
            is_gridded=True,
            resolution_out=resolution,
            device=device,
        )
        self.norm: Literal["backward", "forward", "ortho"] = norm
        self.fft_shift: bool = fft_shift

    def forward(self, input: Tensor) -> Tensor:
        return fft_2d(input, norm=self.norm, fft_shift=self.fft_shift)

    def adjoint(self, input: Tensor) -> Tensor:
        return ifft_2d(input, norm=self.norm, fft_shift=self.fft_shift)
