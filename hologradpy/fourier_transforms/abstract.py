from __future__ import annotations

import torch
from torch import Tensor
import torch.nn as nn


class FourierBase(nn.Module):
    """Base class for 2D Fourier transforms."""

    def __init__(
        self,
        resolution: tuple[int, int],
        frequencies: Tensor,
        is_gridded: bool,
        resolution_out: tuple[int, int] | None = None,
        device: torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.resolution = resolution
        self.resolution_out = resolution if resolution_out is None else resolution_out
        self._frequencies = frequencies
        self.is_gridded = is_gridded
        self.device = device

    @property
    def frequencies(self) -> Tensor:
        """The sample points, in rad/sample."""
        if self._frequencies is None:
            self._frequencies = self._build_frequencies()
        return self._frequencies

    def _build_frequencies(self) -> Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} deferred its frequencies but does not build them."
        )

    def forward(self, input: Tensor) -> Tensor:
        raise NotImplementedError(
            "Subclasses of FourierBase must implement forward()."
        )

    def adjoint(self, input: Tensor) -> Tensor:
        raise NotImplementedError(
            "Subclasses of FourierBase must implement adjoint()."
        )
