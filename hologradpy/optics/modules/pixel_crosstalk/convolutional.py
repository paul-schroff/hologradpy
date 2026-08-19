"""Crosstalk as a convolution of the upscaled phase with a kernel."""

from __future__ import annotations

from abc import abstractmethod

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn import functional

from .abstract import PixelCrosstalk, pack_planes, replicate_pad, unpack_planes


class ConvolutionalCrosstalk(PixelCrosstalk):
    """The upscaled phase convolved with a kernel.

    The phase is spread over its sub-pixel blocks by nearest neighbour, then smoothed
    across the block edges by a kernel spanning ``extent`` SLM pixels, so
    ``extent * P`` samples across. Subclasses differ only in how the kernel is
    parameterized.

    An odd ``upscale_factor`` gives an odd kernel, whose center sample sits on the
    center of an SLM pixel.
    """

    @property
    def kernel_size(self) -> int:
        """Samples across the kernel, ``extent * P``."""
        return self.extent * self.upscale_factor

    @abstractmethod
    def kernel(self: ConvolutionalCrosstalk) -> Tensor:
        """The convolution kernel ``(kernel_size, kernel_size)``, summing to one."""

    def forward(self: ConvolutionalCrosstalk, phase: Tensor) -> Tensor:
        kernel = self.kernel()
        upscaled = self.repeat_pixels(phase)

        # Replicating the border and convolving with no further padding keeps the
        # result the size of the input.
        size = self.kernel_size
        padded = replicate_pad(upscaled, size // 2, size - 1 - size // 2)
        flat, leading = pack_planes(padded)
        smeared = functional.conv2d(flat, kernel[None, None])
        return unpack_planes(smeared, leading)


def normalized(kernel: Tensor) -> Tensor:
    """``kernel`` scaled to sum to one."""
    return kernel / kernel.sum()


class FreeKernelCrosstalk(ConvolutionalCrosstalk):
    """Every kernel sample learned, with nothing tying them together. Starts as a delta
    at the center, which is the identity.
    """

    def __init__(
        self: FreeKernelCrosstalk,
        upscale_factor: int = 3,
        extent: int = 3,
        init_kernel: Tensor | None = None,
        learnable: bool = True,
    ) -> None:
        """
        Args:
            upscale_factor: Sub-pixels across one SLM pixel.
            extent: Reach of the fringing field, in SLM pixels.
            init_kernel: Kernel to start from, ``(extent * P, extent * P)``. Defaults
                to a center delta.
            learnable: Whether the kernel takes gradients.
        """
        super().__init__(upscale_factor=upscale_factor, extent=extent)

        size = self.kernel_size
        if init_kernel is None:
            init_kernel = torch.zeros(size, size)
            init_kernel[size // 2, size // 2] = 1.0
        elif tuple(init_kernel.shape) != (size, size):
            raise ValueError(
                f"init_kernel must be {(size, size)} for upscale_factor="
                f"{upscale_factor} and extent={extent}, got "
                f"{tuple(init_kernel.shape)}."
            )

        self.weights = Parameter(init_kernel.clone().float(), requires_grad=learnable)

    @classmethod
    def from_parametric(
        cls: type[FreeKernelCrosstalk],
        source: ConvolutionalCrosstalk,
        learnable: bool = True,
    ) -> FreeKernelCrosstalk:
        """A free kernel seeded from a fitted parametric one: the warm start a fit
        wants.
        """
        return cls(
            upscale_factor=source.upscale_factor,
            extent=source.extent,
            init_kernel=source.kernel().detach(),
            learnable=learnable,
        )

    def kernel(self: FreeKernelCrosstalk) -> Tensor:
        return normalized(self.weights)
