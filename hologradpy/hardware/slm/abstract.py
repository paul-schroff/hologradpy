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
from ...utils import as_image

if TYPE_CHECKING:
    # Imported for type annotations only.
    from ...calibration.wavefront.abstract import WavefrontCalibrationData
    from ...optics.complex_amplitude import ComplexAmplitude, FieldGeometry

    WavefrontSource: TypeAlias = (
        WavefrontCalibrationData | ComplexAmplitude | NDArray | torch.Tensor
    )


class SLM(ABC):
    """A HoloGradPy-native SLM: SI units and ``(y, x)`` geometry.

    A device implements the abstract geometry members ``pixel_size``, ``resolution``
    and ``wavelength``. It reports ``bitdepth`` and implements ``set_levels`` to
    display a pattern. ``set_phase`` converts a phase to gray levels through
    ``phase_response`` and hands them to ``set_levels``. A device that converts phases
    itself overrides ``set_phase``. The ``get_spatial_grid`` template method is
    provided here. Third-party devices subclass this base (or register a wrapper with
    :func:`hologradpy.hardware.as_native.as_slm`).

    Attributes:
        settle_time: The settling time of the SLM in seconds. ``set_levels`` waits
            this long after each write. Zero means no wait.
        display: The displayed gray levels, or None when the device does not report
            them.
    """

    settle_time: float = 0.0
    display: NDArray | None = None

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
        """Operating wavelength in metres.

        The displayed phase is meant for this wavelength.
        """

    _phase_correction: NDArray | None = None
    _vendor_correction: NDArray | None = None
    _phase_response: phase_levels.PhaseResponse | None = None

    def set_phase(
        self,
        phase: NDArray | torch.Tensor,
        apply_phase_correction: bool = False,
        apply_vendor_correction: bool | None = None,
    ) -> None:
        """Display the desired optical phase in radians (a ``(height, width)`` array).

        The phase is quantized to the SLM's gray levels through :attr:`phase_response`.
        Use :meth:`set_levels` to display levels directly.

        Both corrections are added to the phase before the response converts it. The
        vendor correction is read on the nominal scale, where ``2**bitdepth`` levels
        make one cycle. It therefore wraps at the SLM's own full cycle.

        Args:
            phase: The desired optical phase in radians.
            apply_phase_correction: Add :attr:`phase_correction` before converting.
            apply_vendor_correction: Add :attr:`vendor_correction` before converting.
                None adds it when one is loaded, and True raises when none is.

        Raises:
            TypeError: ``phase`` holds integers or complex numbers. Integers almost
                always mean gray levels.
            ValueError: ``phase`` has NaN or infinite values, or a correction was asked
                for but has not been loaded.
        """
        phase = self._checked_phase(phase)
        if apply_phase_correction:
            phase = phase + self._required_correction("phase")
        if apply_vendor_correction is None:
            apply_vendor_correction = self._vendor_correction is not None
        if apply_vendor_correction:
            phase = phase + self._vendor_phase

        self.set_levels(self.phase_to_levels(phase))

    def set_levels(self, levels: NDArray | torch.Tensor) -> None:
        """Display gray levels as given, without either correction.

        A device returns once the SLM has settled for :attr:`settle_time`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot be given gray levels directly."
        )

    def _checked_levels(self, levels: NDArray | torch.Tensor) -> NDArray:
        """Displayable levels at the SLM's resolution, in its level dtype.

        A tensor is moved to the CPU first. With a bit depth, the levels are returned as
        a new array, so the caller's array never becomes the display.

        Args:
            levels: Whole gray levels at the SLM resolution.

        Returns:
            NDArray: The levels in ``level_dtype(bitdepth)``, or the integer array as
            given when the SLM reports no bit depth.

        Raises:
            TypeError: ``levels`` are not integers.
            ValueError: ``levels`` are not the SLM's shape, or lie outside the range
                from 0 to ``2**bitdepth - 1``.
        """
        if torch.is_tensor(levels):
            levels = levels.detach().cpu()
        levels = np.asarray(levels)
        if not np.issubdtype(levels.dtype, np.integer):
            raise TypeError(
                f"set_levels takes integer gray levels, got {levels.dtype}. An SLM "
                "displays whole levels. For a phase in radians, call set_phase."
            )
        if tuple(levels.shape) != tuple(self.resolution):
            raise ValueError(
                f"The levels are {tuple(levels.shape)} but the SLM is "
                f"{tuple(self.resolution)}. Levels are per pixel, so they have to be "
                "the SLM's own shape."
            )

        bitdepth = self.bitdepth
        if bitdepth is None:
            return levels
        number_of_levels = 2 ** int(bitdepth)
        # Python integers, so the comparison is exact under NEP 50 for any dtype.
        lowest, highest = int(levels.min()), int(levels.max())
        if lowest < 0 or highest >= number_of_levels:
            raise ValueError(
                f"The levels run from {lowest} to {highest}, but a {bitdepth}-bit SLM "
                f"shows levels from 0 to {number_of_levels - 1}."
            )
        return levels.astype(phase_levels.level_dtype(bitdepth))

    @staticmethod
    def _checked_phase(phase: NDArray | torch.Tensor) -> NDArray:
        """The phase as a real, finite array.

        An integer array is refused, since it almost always means gray levels.
        """
        phase = np.asarray(phase.detach().cpu() if torch.is_tensor(phase) else phase)
        if np.issubdtype(phase.dtype, np.integer):
            raise TypeError(
                "set_phase takes an optical phase in radians, and an integer array "
                "almost always means gray levels. Pass those to set_levels, or cast to "
                "float if radians was meant."
            )
        if np.iscomplexobj(phase):
            raise TypeError(
                "set_phase takes a real phase in radians. For a field, pass its angle."
            )
        _reject_non_finite(phase, "phase")
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
        """A per-pixel correction in gray levels, as a vendor ships it.

        It is read on the nominal scale, where ``2**bitdepth`` levels make one cycle.
        :meth:`set_phase` adds it to every phase while it is loaded.
        """
        return self._vendor_correction

    @property
    def _vendor_phase(self) -> NDArray:
        """The vendor correction as a phase in radians, ``-2 pi V / 2**bitdepth`` for
        ``V`` levels.

        It is read on the nominal scale, where ``2**bitdepth`` levels make one cycle.
        A higher level gives a more negative phase, as in a phase response.

        Raises:
            ValueError: No vendor correction is loaded, or the SLM reports no bit depth.
        """
        levels = np.asarray(self._required_correction("vendor"), dtype=np.float64)
        return -phase_levels.TWO_PI * levels / 2**self._required_bitdepth

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
        """Take a per-pixel correction in gray levels, as a vendor ships it.

        The levels are read on the nominal scale, where ``2**bitdepth`` levels make one
        cycle. :meth:`set_phase` adds them as a phase before the response converts it.

        Args:
            levels: Whole gray levels at the SLM resolution, such as the pixels of a
                vendor's correction bitmap.

        Raises:
            TypeError: ``levels`` are not integers.
            ValueError: ``levels`` are not the SLM's shape.
        """
        if torch.is_tensor(levels):
            levels = levels.detach().cpu()
        levels = np.asarray(levels)
        if not np.issubdtype(levels.dtype, np.integer):
            raise TypeError(
                f"A vendor correction is in whole gray levels, got {levels.dtype}. "
                "Load a phase in radians with load_phase_correction."
            )
        self._vendor_correction = self._checked_correction(levels, "vendor")

    def _checked_correction(self, correction: NDArray, kind: str) -> NDArray:
        if tuple(correction.shape) != tuple(self.resolution):
            raise ValueError(
                f"The {kind} correction is {tuple(correction.shape)} but the SLM is "
                f"{tuple(self.resolution)}. A correction is per pixel, so it has to be "
                "the SLM's own shape."
            )
        if kind == "phase":
            _reject_non_finite(correction, "phase correction")
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
    def _required_bitdepth(self) -> int:
        """The bit depth, for a conversion that needs one.

        Raises:
            ValueError: The device reports no bit depth, so there are no levels to
                convert to.
        """
        bitdepth = self.bitdepth
        if bitdepth is None:
            raise ValueError(
                f"{type(self).__name__} reports no bitdepth, so its phase cannot be "
                "expressed as display levels."
            )
        return int(bitdepth)

    @property
    def phase_response(self) -> phase_levels.PhaseResponse | None:
        """Gray level to phase response of the SLM.

        The response loaded with :meth:`load_phase_response`, or else the nominal
        response of the device. The nominal response of the base class is a straight
        line over one cycle, or None for a device without a bit depth.
        """
        if self._phase_response is not None:
            return self._phase_response
        return self._nominal_phase_response

    @property
    def _nominal_phase_response(self) -> phase_levels.PhaseResponse | None:
        """The device's nominal response, which applies until a response is loaded."""
        if self.bitdepth is None:
            return None
        return phase_levels.LinearResponse(bitdepth=self.bitdepth)

    def load_phase_response(self, response: phase_levels.PhaseResponse | None) -> None:
        """Take the gray level to phase response of the SLM.

        It converts every phase from then on. Building a model or record copies the
        response, so the response is loaded before any model is built.

        Args:
            response: The response at the SLM's bit depth. A measured curve is a
                :class:`~hologradpy.phase_levels.LookupResponse`. A straight line
                reaching one cycle below full scale is a
                :class:`~hologradpy.phase_levels.LinearResponse` with
                ``full_scale_cycles`` above one. None returns to the nominal response.

        Raises:
            TypeError: ``response`` is not a ``PhaseResponse``.
            ValueError: The SLM reports no bit depth, or ``response`` is at another one.
        """
        if response is not None:
            self._check_phase_response(response)
        self._phase_response = response

    def _check_phase_response(self, response: phase_levels.PhaseResponse) -> None:
        """Check that ``response`` is a ``PhaseResponse`` at the SLM's bit depth.

        Raises:
            TypeError: ``response`` is not a ``PhaseResponse``.
            ValueError: The SLM reports no bit depth, or ``response`` is at another one.
        """
        if not isinstance(response, phase_levels.PhaseResponse):
            raise TypeError(
                f"A phase response is a PhaseResponse, got {type(response).__name__}. "
                "For a PhaseResponseModule, pass its response."
            )
        bitdepth = self._required_bitdepth
        if int(response.bitdepth) != bitdepth:
            raise ValueError(
                f"The response is {response.bitdepth}-bit but {type(self).__name__} "
                f"is {bitdepth}-bit, so its levels do not mean the same phase."
            )

    @property
    def full_scale_cycles(self) -> float:
        """The phase delay at full scale in cycles, read from the response."""
        response = self.phase_response
        return 1.0 if response is None else response.full_scale_cycles

    def phase_to_levels(self, phase: NDArray | torch.Tensor) -> NDArray:
        """Convert a target phase to gray levels through :attr:`phase_response`.

        Neither correction is added here (see :meth:`set_phase`).

        Raises:
            ValueError: The device reports no bit depth, so there are no levels to
                convert to.
        """
        bitdepth = self._required_bitdepth
        if torch.is_tensor(phase):
            phase = phase.detach().cpu()
        levels = self.phase_response.display_levels(np.asarray(phase))
        return levels.astype(phase_levels.level_dtype(bitdepth), copy=False)

    @property
    def aperture_extent(self) -> tuple[float, float]:
        """The SLM's physical size as ``(height, width)`` in metres."""
        return tuple(
            float(count) * float(pitch)
            for count, pitch in zip(self.resolution, self.pixel_size)
        )

    def get_spatial_grid(
        self, device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The SLM-plane ``(x, y)`` coordinate meshgrid, in metres.

        Args:
            device: The torch device for the grid, or None for the CPU.
        """
        return _spatial_grid(self.resolution, self.pixel_size, device=device)

    def to_field_geometry(
        self, device: torch.device | None = None, dtype: torch.dtype = torch.float32
    ) -> FieldGeometry:
        """The geometry of a field on the SLM.

        Args:
            device: The torch device for the geometry, or None for the CPU.
            dtype: The dtype of the pixel size and the wavelength.
        """
        from ...optics.complex_amplitude import FieldGeometry

        return FieldGeometry(
            resolution=tuple(self.resolution),
            pixel_size=torch.tensor(self.pixel_size, dtype=dtype, device=device),
            wavelength=torch.tensor(self.wavelength, dtype=dtype, device=device),
        )


@record_type("slm_data")
@dataclass(frozen=True, unsafe_hash=True)
class SLMData(SaveableRecord):
    """A native snapshot of an SLM's geometry and modulation settings."""

    name: str
    resolution: tuple[int, int]
    pixel_size: tuple[float, float]
    wavelength: float
    settle_time: float
    phase_response: phase_levels.PhaseResponse | None = None
    phase_correction: NDArray | None = field(default=None, compare=False, hash=False)
    vendor_correction: NDArray | None = field(default=None, compare=False, hash=False)

    @property
    def bitdepth(self) -> int | None:
        """Bits per pixel, from the response that gives the levels their meaning."""
        return None if self.phase_response is None else self.phase_response.bitdepth

    @property
    def full_scale_cycles(self) -> float:
        """The phase delay at full scale in cycles, read from the response."""
        if self.phase_response is None:
            return 1.0
        return self.phase_response.full_scale_cycles

    @classmethod
    def from_slm(cls, slm: SLM) -> SLMData:
        """Record an SLM as it stands.

        Args:
            slm: The SLM, as :func:`~hologradpy.hardware.factory.open_slm` or
                :func:`~hologradpy.hardware.as_native.as_slm` returns it.
        """
        return cls(
            name=getattr(slm, "name", ""),
            resolution=slm.resolution,
            pixel_size=tuple(float(v) for v in slm.pixel_size),
            wavelength=slm.wavelength,
            settle_time=float(slm.settle_time),
            phase_response=slm.phase_response,
            phase_correction=slm.phase_correction,
            vendor_correction=slm.vendor_correction,
        )


def _phase_of(source: WavefrontSource) -> NDArray:
    """Extracts the per-pixel phase in radians inside ``source``."""
    phase = getattr(source, "complex_amplitude", source)
    if hasattr(phase, "as_tensor"):
        phase = as_image(phase)
    if torch.is_tensor(phase):
        phase = phase.detach().cpu()
        if phase.is_complex():
            phase = torch.angle(phase)
    phase = np.asarray(phase)
    if np.iscomplexobj(phase):
        phase = np.angle(phase)
    return phase


def _reject_non_finite(values: NDArray, kind: str) -> None:
    """Raise when any pixel of ``values`` is NaN or infinite.

    Args:
        values: A per-pixel phase in radians.
        kind: A name for ``values`` in the error message.

    Raises:
        ValueError: A pixel is NaN or infinite.
    """
    count = int(np.count_nonzero(~np.isfinite(values)))
    if count:
        raise ValueError(
            f"The {kind} is NaN or infinite at {count} of its {values.size} pixels, "
            "and an SLM shows only a finite phase."
        )
