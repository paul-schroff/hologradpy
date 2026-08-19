"""The native SLM template: the ``SLM`` base class a device subclasses, together with
the ``SLMData`` snapshot record.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import torch
from numpy.typing import NDArray

from ... import phase_levels
from ...grids import get_spatial_grid as _spatial_grid
from ...serialization import SaveableRecord, record_type

if TYPE_CHECKING:
    # Imported for type annotations only.
    from ...calibration.wavefront.abstract import WavefrontCalibrationData
    from ...optics.complex_amplitude import ComplexAmplitude

    WavefrontSource: TypeAlias = (
        WavefrontCalibrationData | ComplexAmplitude | NDArray | torch.Tensor
    )


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

    _phase_correction: NDArray | None = None
    _vendor_correction: NDArray | None = None

    def set_phase(
        self,
        phase: NDArray | torch.Tensor,
        apply_phase_correction: bool = False,
        apply_vendor_correction: bool = False,
    ) -> None:
        """Display the desired optical phase in radians (a ``(height, width)`` array).

        Quantized to the SLM's gray levels. Use :meth:`set_levels` to display levels
        directly.

        Each correction is added where it was calibrated: a measured wavefront in
        radians, before the response converts, and a vendor correction in gray levels,
        after.

        Args:
            phase: The desired optical phase in radians.
            apply_phase_correction: Add :attr:`phase_correction` before converting.
            apply_vendor_correction: Add :attr:`vendor_correction` after converting.

        Raises:
            TypeError: ``phase`` holds integers, which almost always means gray levels.
            ValueError: A correction was asked for that has not been loaded.
        """
        phase = self._checked_phase(phase)
        if apply_phase_correction:
            phase = phase + self._required_correction("phase")

        levels = self.phase_to_levels(phase)
        if apply_vendor_correction:
            wide = np.asarray(levels, dtype=np.int64) + np.asarray(
                self._required_correction("vendor"), dtype=np.int64
            )
            levels = (wide % self.phase_response.number_of_levels).astype(
                phase_levels.level_dtype(self.bitdepth)
            )

        self.set_levels(levels)

    def set_levels(self, levels: NDArray | torch.Tensor) -> None:
        """Display gray levels directly, without going through a phase."""
        raise NotImplementedError(
            f"{type(self).__name__} cannot be given gray levels directly."
        )

    @staticmethod
    def _checked_phase(phase: NDArray | torch.Tensor) -> NDArray:
        """A phase array, refusing the integers that almost always mean levels."""
        phase = np.asarray(phase.detach().cpu() if torch.is_tensor(phase) else phase)
        if np.issubdtype(phase.dtype, np.integer):
            raise TypeError(
                "set_phase takes an optical phase in radians, and an integer array "
                "almost always means gray levels. Pass those to set_levels, or cast to "
                "float if radians was meant."
            )
        return phase

    @property
    def phase_correction(self) -> NDArray | None:
        """A per-pixel phase in radians, added to a desired phase when asked for.

        The correction, not the aberration: already negated, so displaying it cancels
        what was measured.
        """
        return self._phase_correction

    @property
    def vendor_correction(self) -> NDArray | None:
        """A per-pixel correction in gray levels, as a vendor ships it."""
        return self._vendor_correction

    def load_phase_correction(self, correction: WavefrontSource) -> None:
        """Take a per-pixel phase correction, in the sense it will be displayed in.

        Used as it stands. Pass a measurement to :meth:`load_measured_wavefront`
        instead, which negates it for you: the sign is chosen by which method you call,
        never inferred from what you pass, because getting it backwards doubles the
        aberration and looks plausible either way.

        Only the phase is kept: a phase-only SLM cannot fix an amplitude, and the
        amplitude half of a measurement belongs in the model's ``slm_field``.

        Args:
            correction: The correction, as a ``WavefrontCalibrationData``, a
                :class:`~hologradpy.optics.complex_amplitude.ComplexAmplitude`, or an
                array of radians.
        """
        self._phase_correction = self._checked_correction(
            _phase_of(correction), "phase"
        )

    def load_measured_wavefront(self, measurement: WavefrontSource) -> None:
        """Take a measured wavefront and hold the correction that cancels it.

        A measurement says what aberration is present, so the correction is its
        negative, and that negation happens here.

        Args:
            measurement: The measured wavefront, as a ``WavefrontCalibrationData``, a
                :class:`~hologradpy.optics.complex_amplitude.ComplexAmplitude`, or an
                array of radians.
        """
        self._phase_correction = self._checked_correction(
            -_phase_of(measurement), "phase"
        )

    def load_vendor_correction(self, levels: NDArray | torch.Tensor) -> None:
        """Take a vendor's per-pixel correction, in the gray levels it ships as."""
        if torch.is_tensor(levels):
            levels = levels.detach().cpu()
        levels = np.asarray(levels)
        self._vendor_correction = self._checked_correction(levels, "vendor")

    def _checked_correction(self, correction: NDArray, kind: str) -> NDArray:
        if tuple(correction.shape) != tuple(self.resolution):
            raise ValueError(
                f"The {kind} correction is {tuple(correction.shape)} but the SLM is "
                f"{tuple(self.resolution)}. A correction is per pixel, so it has to be "
                "the SLM's own shape."
            )
        return correction

    def _required_correction(self, kind: str) -> NDArray:
        correction = (
            self._phase_correction if kind == "phase" else self._vendor_correction
        )
        if correction is None:
            raise ValueError(
                f"A {kind} correction was asked for but none has been loaded. Call "
                f"load_{kind}_correction first, or leave apply_{kind}_correction off."
            )
        return correction

    @property
    def bitdepth(self) -> int | None:
        return None

    @property
    def phase_response(self) -> phase_levels.PhaseResponse | None:
        """Graylevel to phase response of the SLM.

        A straight line over the whole cycle unless a device says otherwise. One that
        reaches less, or that has been measured, overrides this.
        """
        if self.bitdepth is None:
            return None
        return phase_levels.LinearResponse(bitdepth=self.bitdepth)

    @property
    def phase_scaling(self) -> float:
        """The reachable phase range in cycles, read from the response."""
        response = self.phase_response
        return 1.0 if response is None else response.phase_scaling

    def phase_to_levels(self, phase: NDArray | torch.Tensor) -> NDArray:
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
        if torch.is_tensor(phase):
            phase = phase.detach().cpu()
        return self.phase_response.display_levels(np.asarray(phase))

    @property
    def aperture_extent(self) -> tuple[float, float]:
        """The SLM's physical size as ``(height, width)`` in metres."""
        return tuple(
            float(count) * float(pitch)
            for count, pitch in zip(self.resolution, self.pixel_size)
        )

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
    settle_time_s: float
    phase_response: phase_levels.PhaseResponse | None = None
    phase_correction: NDArray | None = field(default=None, compare=False, hash=False)
    vendor_correction: NDArray | None = field(default=None, compare=False, hash=False)

    @property
    def bitdepth(self) -> int | None:
        """Bits per pixel, from the response that gives the levels their meaning."""
        return None if self.phase_response is None else self.phase_response.bitdepth

    @property
    def phase_scaling(self) -> float:
        """The reachable phase range in cycles."""
        return 1.0 if self.phase_response is None else self.phase_response.phase_scaling

    @classmethod
    def from_slm(cls, slm: SLM) -> SLMData:
        return cls(
            name=getattr(slm, "name", ""),
            resolution=slm.resolution,
            pixel_size=tuple(float(v) for v in slm.pixel_size),
            wavelength=slm.wavelength,
            settle_time_s=getattr(slm, "settle_time_s", 0.0),
            phase_response=slm.phase_response,
            phase_correction=getattr(slm, "phase_correction", None),
            vendor_correction=getattr(slm, "vendor_correction", None),
        )


def _phase_of(source: WavefrontSource) -> NDArray:
    """Extracts the per-pixel phase in radians inside ``source``."""
    phase = getattr(source, "complex_amplitude", source)
    if hasattr(phase, "as_tensor"):
        phase = phase.as_tensor()
    if torch.is_tensor(phase):
        phase = phase.detach().cpu()
        if phase.is_complex():
            phase = torch.angle(phase)
    return np.asarray(phase)
