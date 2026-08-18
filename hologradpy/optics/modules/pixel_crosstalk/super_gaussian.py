"""Super-Gaussian crosstalk kernels, built from their spatial-frequency profile."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Parameter

from .convolutional import ConvolutionalCrosstalk, normalized


FREQUENCY_FLOOR = 1e-12


class SuperGaussianCrosstalk(ConvolutionalCrosstalk):
    """The kernel is a super-Gaussian in spatial frequency.

    ``K = |F^-1{exp(-(|kx| ** q + |ky| ** q) / sigma ** q)}|``.

    ``width`` is in cycles per SLM pixel and ``order`` is dimensionless, so neither
    depends on the pixel pitch.
    """

    def __init__(
        self: SuperGaussianCrosstalk,
        upscale_factor: int = 3,
        extent: int = 3,
        order: float = 1.20,
        width: float = 2.03,
        learnable: bool = True,
    ) -> None:
        """
        Args:
            upscale_factor: Sub-pixels across one SLM pixel.
            extent: Reach of the fringing field, in SLM pixels.
            order: Super-Gaussian order ``q``. Two is a Gaussian. Defaults to the
                fitted value of the paper.
            width: Super-Gaussian width ``sigma``, in cycles per SLM pixel. Defaults
                to the fitted value of the paper.
            learnable: Whether the two take gradients.
        """
        super().__init__(upscale_factor=upscale_factor, extent=extent)

        self.order = Parameter(torch.tensor(float(order)), requires_grad=learnable)
        self.width = Parameter(torch.tensor(float(width)), requires_grad=learnable)
        self.register_buffer("frequency", _frequency_grid(self.kernel_size, extent))

    def kernel(self: SuperGaussianCrosstalk) -> Tensor:
        rows, columns = self.frequency
        profile = _super_gaussian(rows, self.order, self.width) * _super_gaussian(
            columns, self.order, self.width
        )
        return normalized(_to_real_space(profile))


class PiecewiseSuperGaussianCrosstalk(ConvolutionalCrosstalk):
    """A super-Gaussian with its own order and width per half-axis.

    Four kernels are built, one per quadrant, from the order and width belonging to the
    two half-axes that quadrant spans, and each is masked to its quadrant.
    """

    def __init__(
        self: PiecewiseSuperGaussianCrosstalk,
        upscale_factor: int = 3,
        extent: int = 3,
        order_y: tuple[float, float] = (2, 2),
        order_x: tuple[float, float] = (2, 2),
        width_y: tuple[float, float] = (1, 1),
        width_x: tuple[float, float] = (1, 1),
        learnable: bool = True,
    ) -> None:
        """
        Args:
            upscale_factor: Sub-pixels across one SLM pixel.
            extent: Reach of the fringing field, in SLM pixels.
            order_y: Super-Gaussian order for the vertical axis, ``(negative,
                positive)``.
            order_x: Order for the horizontal axis, ``(negative, positive)``.
            width_y: Width for the vertical axis in cycles per SLM pixel,
                ``(negative, positive)``.
            width_x: Width for the horizontal axis, ``(negative, positive)``.
            learnable: Whether the eight parameters take gradients.
        """
        super().__init__(upscale_factor=upscale_factor, extent=extent)

        self.orders = Parameter(
            torch.tensor([list(order_y), list(order_x)], dtype=torch.float32),
            requires_grad=learnable,
        )
        self.widths = Parameter(
            torch.tensor([list(width_y), list(width_x)], dtype=torch.float32),
            requires_grad=learnable,
        )
        self.register_buffer("frequency", _frequency_grid(self.kernel_size, extent))
        self.register_buffer("quadrants", _quadrant_masks(self.kernel_size))

    def kernel(self: PiecewiseSuperGaussianCrosstalk) -> Tensor:
        rows, columns = self.frequency

        combined = torch.zeros_like(rows)
        for row_side in (0, 1):
            for column_side in (0, 1):
                profile = _super_gaussian(
                    rows, self.orders[0, row_side], self.widths[0, row_side]
                ) * _super_gaussian(
                    columns, self.orders[1, column_side], self.widths[1, column_side]
                )
                mask = self.quadrants[row_side * 2 + column_side]
                combined = combined + _to_real_space(profile) * mask

        return normalized(combined)


def _frequency_grid(size: int, extent: int) -> Tensor:
    """The ``(rows, columns)`` spatial frequencies of a kernel, in cycles per SLM pixel.
    """
    line = torch.arange(-size / 2, size / 2, 1.0) / extent
    return torch.stack(torch.meshgrid(line, line, indexing="ij"))


def _super_gaussian(frequency: Tensor, order: Tensor, width: Tensor) -> Tensor:
    magnitude = frequency.abs().clamp(min=FREQUENCY_FLOOR)
    return torch.exp(-((magnitude / width) ** order))


def _to_real_space(profile: Tensor) -> Tensor:
    """The magnitude of a centered frequency profile, transformed and re-centered."""
    return torch.fft.fftshift(torch.fft.ifft2(profile)).abs()


def _quadrant_masks(size: int) -> Tensor:
    """Four masks splitting a kernel into quadrants, in the order used by the loop."""
    half = size // 2 + 1
    masks = torch.zeros(4, size, size)
    masks[0, :half, :half] = 1.0
    masks[1, :half, half:] = 1.0
    masks[2, half:, :half] = 1.0
    masks[3, half:, half:] = 1.0
    return masks
