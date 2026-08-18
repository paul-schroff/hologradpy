"""Crosstalk as a weighted sum of the differences to the neighbouring pixels."""

from __future__ import annotations

import itertools

import torch
from einops import repeat
from torch import Tensor
from torch.nn import Parameter

from .abstract import PixelCrosstalk, replicate_pad


class NeighbourDifferenceCrosstalk(PixelCrosstalk):
    """Each sub-pixel relaxes towards its neighbours.

    ``Theta[m', n'] = theta[m, n] + sum_i T_i[s, t] * (theta[neighbour i] -
    theta[m, n])``, with ``m = floor(m' / P)`` the SLM pixel a sub-pixel belongs to and
    ``s = m' mod P`` where in that pixel it sits. Each ``T_i`` is a learnable ``P x P``
    matrix saying how strongly each position inside a pixel is pulled towards neighbour
    ``i``, giving ``(extent ** 2 - 1) * P ** 2`` parameters.

    Starts with every weight at zero.
    """

    def __init__(
        self: NeighbourDifferenceCrosstalk,
        upscale_factor: int = 3,
        extent: int = 3,
        init_transitions: Tensor | None = None,
        learnable: bool = True,
    ) -> None:
        """
        Args:
            upscale_factor: Sub-pixels across one SLM pixel.
            extent: Width of the neighbourhood, in SLM pixels.
            init_transitions: Weights to start from, ``(extent ** 2 - 1, P, P)`` in the
                order of :attr:`neighbour_offsets`. Defaults to zeros.
            learnable: Whether the weights take gradients.
        """
        super().__init__(upscale_factor=upscale_factor, extent=extent)

        shape = (
            self.number_of_neighbours,
            self.upscale_factor,
            self.upscale_factor,
        )
        if init_transitions is None:
            init_transitions = torch.zeros(shape)
        elif tuple(init_transitions.shape) != shape:
            raise ValueError(
                f"init_transitions must be {shape} for upscale_factor="
                f"{upscale_factor} and extent={extent}, got "
                f"{tuple(init_transitions.shape)}."
            )

        self.transitions = Parameter(
            init_transitions.clone().float(), requires_grad=learnable
        )

    @property
    def neighbour_offsets(self) -> tuple[tuple[int, int], ...]:
        """The ``(rows, columns)`` offset of each neighbour, row-major from the top
        left, with the pixel itself left out. Indexes the first axis of
        :attr:`transitions`."""
        span = range(-self.reach, self.reach + 1)
        return tuple(
            offset for offset in itertools.product(span, span) if offset != (0, 0)
        )

    @property
    def number_of_neighbours(self) -> int:
        return self.extent**2 - 1

    def _tiled(
        self: NeighbourDifferenceCrosstalk,
        index: int,
        resolution: tuple[int, int],
    ) -> Tensor:
        """Transition matrix ``index`` repeated across every SLM pixel."""
        return repeat(
            self.transitions[index],
            "p1 p2 -> (h p1) (w p2)",
            h=resolution[0],
            w=resolution[1],
        )

    def forward(self: NeighbourDifferenceCrosstalk, phase: Tensor) -> Tensor:
        reach = self.reach
        height, width = phase.shape[-2], phase.shape[-1]
        # Replicating the border makes an edge pixel its own missing neighbour, so the
        # difference vanishes there and a flat phase stays flat.
        padded = replicate_pad(phase, reach)

        crosstalk = self.repeat_pixels(phase)
        for index, (rows, columns) in enumerate(self.neighbour_offsets):
            neighbour = padded[
                ...,
                reach + rows : reach + rows + height,
                reach + columns : reach + columns + width,
            ]
            difference = self.repeat_pixels(neighbour - phase)
            crosstalk = crosstalk + self._tiled(index, (height, width)) * difference

        return crosstalk
