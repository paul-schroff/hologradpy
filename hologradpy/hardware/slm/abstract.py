"""The native SLM template: the ``SLM`` base class a device subclasses, together with
the ``SLMData`` snapshot record.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import pickle

import numpy as np
import torch
from numpy.typing import NDArray

from ...grids import get_spatial_grid as _spatial_grid


class SLM(ABC):
    """A HoloGradPy-native SLM: SI units and ``(y, x)`` geometry.

    A device implements the geometry and ``set_phase`` abstract members. The
    ``get_spatial_grid`` template method is provided here. Third-party devices subclass
    this base (or register a wrapper with
    :func:`hologradpy.hardware.as_native.as_slm`).
    """

    @property
    @abstractmethod
    def pixel_size(self) -> NDArray[np.float64]:
        """Pixel pitch ``(y, x)`` in metres."""

    @property
    @abstractmethod
    def resolution(self) -> tuple[int, int]:
        """SLM resolution ``(height, width)`` in pixels."""

    @property
    @abstractmethod
    def wavelength(self) -> float:
        """Design wavelength in metres."""

    @abstractmethod
    def set_phase(self, phase) -> None:
        """Display the desired optical phase (a ``(height, width)`` array)."""

    def get_spatial_grid(
        self, device: torch.device = torch.device("cpu")
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The SLM-plane ``(x, y)`` coordinate meshgrid, in metres."""
        return _spatial_grid(self.resolution, self.pixel_size, device=device)


@dataclass(frozen=True, unsafe_hash=True)
class SLMData:
    """A native snapshot of an SLM's geometry and modulation settings."""

    name: str
    resolution: tuple[int, int]
    pixel_size: tuple[float, float]
    wavelength: float
    phase_scaling: float
    settle_time_s: float

    @classmethod
    def from_slm(cls, slm: SLM) -> SLMData:
        return cls(
            name=getattr(slm, "name", ""),
            resolution=slm.resolution,
            pixel_size=tuple(float(v) for v in slm.pixel_size),
            wavelength=slm.wavelength,
            phase_scaling=getattr(slm, "phase_scaling", 1.0),
            settle_time_s=getattr(slm, "settle_time_s", 0.0),
        )

    def save(self, filename: str):
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename: str) -> SLMData:
        with open(filename, "rb") as file:
            slm_data: SLMData = pickle.load(file)
        return slm_data
