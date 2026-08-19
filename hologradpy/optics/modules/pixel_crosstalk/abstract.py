from __future__ import annotations

from abc import abstractmethod

import torch
from einops import rearrange, repeat
from torch import Tensor, nn


class PixelCrosstalk(nn.Module):
    """Fringing fields between neighbouring liquid-crystal pixels. Modeling this needs a
    grid finer than the SLM: every model here maps the displayed phase ``(..., N, N)``
    onto ``(..., P * N, P * N)``, with ``P`` sub-pixels across each SLM pixel.
    """

    def __init__(
        self: PixelCrosstalk,
        upscale_factor: int = 3,
        extent: int = 3,
    ) -> None:
        """
        Args:
            upscale_factor: Sub-pixels across one SLM pixel, ``P``. Every plane after
                the SLM grows by this factor, so the cost of everything downstream
                goes with its square.
            extent: How far the crosstalk reaches, in SLM pixels. Odd, so the reach is 
                the same in both directions.
        """
        super().__init__()

        if int(upscale_factor) < 1:
            raise ValueError(
                f"upscale_factor must be a positive integer, got {upscale_factor}."
            )
        if int(extent) < 1 or int(extent) % 2 == 0:
            raise ValueError(f"extent must be a positive odd integer, got {extent}.")

        self.upscale_factor: int = int(upscale_factor)
        self.extent: int = int(extent)

    @property
    def reach(self) -> int:
        """How far the fringing field reaches either side of a pixel, in SLM pixels."""
        return self.extent // 2

    def repeat_pixels(self: PixelCrosstalk, phase: Tensor) -> Tensor:
        """Nearest neighbor interpolation of one physical SLM pixel into ``P x P``
        sub-pixels. All ``P x P`` sub-pixels will have the same phase value as the 
        physical pixel.
        """
        factor = self.upscale_factor
        if factor == 1:
            return phase
        return repeat(
            phase,
            "... h w -> ... (h p1) (w p2)",
            p1=factor,
            p2=factor,
        )

    @abstractmethod
    def forward(self: PixelCrosstalk, phase: Tensor) -> Tensor:
        """The phase after applying the crosstalk model on the sub-pixel grid.

        Args:
            phase: Displayed phase ``(..., N, N)``, at SLM resolution.

        Returns:
            Tensor: ``(..., P * N, P * N)``.
        """


def replicate_pad(phase: Tensor, before: int, after: int | None = None) -> Tensor:
    """``phase`` with its border repeated outwards, ``before`` and ``after`` samples."""
    after = before if after is None else after
    if before == 0 and after == 0:
        return phase
    flat, leading = pack_planes(phase)
    padded = torch.nn.functional.pad(
        flat, (before, after, before, after), mode="replicate"
    )
    return unpack_planes(padded, leading)


def pack_planes(phase: Tensor) -> tuple[Tensor, tuple[int, ...]]:
    """Collapse every leading axis into one, giving ``(B, 1, H, W)``."""
    leading = tuple(phase.shape[:-2])
    flat = rearrange(phase.reshape(-1, *phase.shape[-2:]), "b h w -> b 1 h w")
    return flat, leading


def unpack_planes(flat: Tensor, leading: tuple[int, ...]) -> Tensor:
    """Undo :func:`pack_planes`, restoring the leading axes."""
    plane = rearrange(flat, "b 1 h w -> b h w")
    return plane.reshape(*leading, *plane.shape[-2:])
