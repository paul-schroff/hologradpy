"""The native SLM template: the ``SLM`` base class a device subclasses, together with
the ``SLMData`` snapshot record.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from ... import phase_levels
from ...grids import get_spatial_grid as _spatial_grid
from ...serialization import SaveableRecord, record_type


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
        """Display the desired optical phase in radians (a ``(height, width)`` array).

        Quantised to the panel's grey levels. Use :meth:`set_levels` to display levels
        directly.
        """

    def set_levels(self, levels) -> None:
        """Display grey levels directly, without going through a phase."""
        raise NotImplementedError(
            f"{type(self).__name__} cannot be given grey levels directly."
        )

    @property
    def bitdepth(self) -> int | None:
        return None

    @property
    def phase_response(self) -> phase_levels.PhaseResponse | None:
        """Graylevel to phase response of the SLM."""
        if self.bitdepth is None:
            return None
        return phase_levels.LinearResponse(
            bitdepth=self.bitdepth, phase_scaling=getattr(self, "phase_scaling", 1.0)
        )

    def phase_to_levels(self, phase) -> NDArray:
        """Convert a target phase to levels the device would display.

        Raises:
            ValueError: The device reports no bit depth, so there are no levels to
                convert to.
        """
        if self.bitdepth is None:
            raise ValueError(
                f"{type(self).__name__} reports no bitdepth, so its phase cannot be "
                "expressed as display levels."
            )
        return self.phase_response.display_levels(np.asarray(phase))

    def get_spatial_grid(
        self, device: torch.device = torch.device("cpu")
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The SLM-plane ``(x, y)`` coordinate meshgrid, in metres."""
        return _spatial_grid(self.resolution, self.pixel_size, device=device)


@record_type("slm_data")
@dataclass(frozen=True, unsafe_hash=True)
class SLMData(SaveableRecord):
    """A native snapshot of an SLM's geometry and modulation settings."""

    name: str
    resolution: tuple[int, int]
    pixel_size: tuple[float, float]
    wavelength: float
    phase_scaling: float
    settle_time_s: float
    bitdepth: int | None = None
    phase_response: phase_levels.PhaseResponse | None = None

    @classmethod
    def from_slm(cls, slm: SLM) -> SLMData:
        return cls(
            name=getattr(slm, "name", ""),
            resolution=slm.resolution,
            pixel_size=tuple(float(v) for v in slm.pixel_size),
            wavelength=slm.wavelength,
            phase_scaling=getattr(slm, "phase_scaling", 1.0),
            settle_time_s=getattr(slm, "settle_time_s", 0.0),
            bitdepth=slm.bitdepth,
            phase_response=slm.phase_response,
        )
