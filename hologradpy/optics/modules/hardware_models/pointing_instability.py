from __future__ import annotations

import torch
from torch import Tensor

from ..abstract import OpticsModule
from ...complex_amplitude import ComplexAmplitude, broadcast_wavelength_operand
from ....grids import get_spatial_grid
from ....profiles.phase import linear_phase, tilt_to_angle


class PointingInstability(OpticsModule):
    """Random beam-pointing jitter applied as a phase tilt (sampling-preserving).

    Each :meth:`forward` draws a beam tilt [rad] from a zero-mean Gaussian and
    applies it as a linear phase ramp (:func:`linear_phase`), modelling a random
    change in beam angle which shifts the focal spot downstream. Intended to sit
    just after ``PixelwiseSLMField`` in the SLM-plane chain. Output geometry equals
    input geometry.

    Args:
        tilt_std: Standard deviation of the (zero-mean Gaussian) beam angle [rad]
            -- a scalar (same for x and y) or ``(std_x, std_y)``. Each forward
            draws ``angle_x ~ N(0, std_x)`` and ``angle_y ~ N(0, std_y)``; ``x``
            is the width axis, ``y`` the height axis.
        seed: Optional integer seed. When given, an internal
            :class:`torch.Generator` (on the field's device) is seeded with it for
            reproducible sampling; otherwise the global RNG is used.
    """

    def __init__(
        self,
        tilt_std: float | tuple[float, float],
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.tilt_std = self._as_pair(tilt_std)
        self.seed = seed
        # Built lazily on the field's device in lazy_init (None -> global RNG).
        self._generator: torch.Generator | None = None
        # Last sampled tilt (and the angles), so adjoint() can invert the most
        # recent forward() and recordables() / tests can inspect the sampled angles.
        self._last_tilt: Tensor | None = None
        self._last_angle: tuple[Tensor, Tensor] | None = None

    @classmethod
    def from_focal_shift(
        cls,
        focal_shift_std: float | tuple[float, float],
        focal_length: float,
        *,
        seed: int | None = None,
    ) -> PointingInstability:
        """Build with the tilt std given as a focal-spot displacement [m] instead
        of an angle.

        Converts paraxially with ``angle = focal_shift / focal_length`` (matching
        ``linear_phase(tilt_units="metres")``), so ``focal_length`` is the focal
        length of the downstream Fourier lens.
        """
        std_x, std_y = cls._as_pair(focal_shift_std)
        return cls(
            (
                tilt_to_angle(std_x, "metres", focal_length=focal_length),
                tilt_to_angle(std_y, "metres", focal_length=focal_length),
            ),
            seed=seed,
        )

    @staticmethod
    def _as_pair(value: float | tuple[float, float]) -> tuple[float, float]:
        if isinstance(value, (tuple, list)):
            return (float(value[0]), float(value[1]))
        return (float(value), float(value))

    @property
    def last_angle(self) -> tuple[Tensor, Tensor] | None:
        """The most recently sampled beam tilt ``(angle_x, angle_y)`` [rad], or
        ``None`` before the first :meth:`forward`. Handy for recording the realized
        per-frame jitter (e.g. to verify a downstream pointing tracker)."""
        return self._last_angle

    def recordables(self) -> dict[str, Tensor]:
        """Record the sampled beam tilt each forward as ``{"angle": (angle_x,
        angle_y)}`` (see
        :class:`~hologradpy.optics.modules.recording.RecordingMixin`);
        empty before the first :meth:`forward`."""
        if self._last_angle is None:
            return {}
        angle_x, angle_y = self._last_angle
        return {"angle": torch.stack((angle_x, angle_y))}

    @property
    def angle_history(self) -> Tensor:
        """The beam tilts recorded while :meth:`record` was on, as an ``(n, 2)``
        tensor ``[angle_x, angle_y]`` per :meth:`forward` (empty ``(0, 2)`` if
        none). Convenience alias for ``history["angle"]``."""
        return self.history.get("angle", torch.empty((0, 2)))

    def lazy_init(
        self: PointingInstability, complex_amplitude: ComplexAmplitude
    ) -> None:
        # Sampling-preserving (output geometry == input); build the spatial grids.
        grid_x, grid_y = get_spatial_grid(
            self.resolution_in,
            tuple(self.pixel_size_in[0].tolist()),
            complex_amplitude.device,
        )
        self.register_buffer("grid_x", grid_x)
        self.register_buffer("grid_y", grid_y)
        if self.seed is not None:
            self._generator = torch.Generator(device=complex_amplitude.device)
            self._generator.manual_seed(self.seed)

    def _normal(self, std: float, device: torch.device) -> Tensor:
        """A 0-d sample from ``N(0, std)``."""
        return torch.randn((), device=device, generator=self._generator) * std

    def _sampled_tilt(
        self, complex_amplitude: ComplexAmplitude, angle_x: Tensor, angle_y: Tensor
    ) -> Tensor:
        # Per-wavelength linear phase ramp exp(i * k * (angle_x*x + angle_y*y)).
        wavenumber = complex_amplitude.wavenumber.reshape(-1, 1, 1)
        tilt_phase = linear_phase(
            self.grid_x.unsqueeze(0),
            self.grid_y.unsqueeze(0),
            angle_x,
            angle_y,
            tilt_units="radians",
            wavenumber=wavenumber,
        )
        tilt = torch.exp(1j * tilt_phase)
        return broadcast_wavelength_operand(tilt, complex_amplitude.ndim)

    def forward(
        self: PointingInstability, complex_amplitude: ComplexAmplitude
    ) -> ComplexAmplitude:
        device = complex_amplitude.device
        angle_x = self._normal(self.tilt_std[0], device)
        angle_y = self._normal(self.tilt_std[1], device)

        tilt = self._sampled_tilt(complex_amplitude, angle_x, angle_y)
        self._last_angle = (angle_x, angle_y)
        self._last_tilt = tilt

        out = complex_amplitude * tilt
        return out.with_geometry(
            wavelength=complex_amplitude.wavelength,
            pixel_size=self.pixel_size_out,
        )

    def adjoint(
        self: PointingInstability, complex_amplitude: ComplexAmplitude
    ) -> ComplexAmplitude:
        """Conjugate transpose of the most recent :meth:`forward` (the unitary
        inverse of the last sampled tilt): undo the tilt."""
        if self._last_tilt is None:
            raise RuntimeError(
                "PointingInstability.adjoint() needs a prior forward() -- it "
                "inverts the last sampled tilt."
            )
        untilted = complex_amplitude * self._last_tilt.conj()
        return untilted.with_geometry(
            wavelength=complex_amplitude.wavelength,
            pixel_size=self.pixel_size_in,
        )
