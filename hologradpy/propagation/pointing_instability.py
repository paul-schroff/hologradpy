"""Random beam-pointing instability: a sampled phase tilt (beam-angle jitter)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import Tensor

from .optics_module import OpticsModule
from .complex_amplitude import ComplexAmplitude, broadcast_wavelength_operand
from .fourier import get_spatial_grid
from .phase_profiles import linear_phase


class PointingInstability(OpticsModule):
    """Random beam-pointing jitter applied as a phase tilt (sampling-preserving).

    Each :meth:`forward` draws a beam tilt [rad] from a zero-mean Gaussian and
    applies it as a linear phase ramp (:func:`linear_phase`), modelling a random
    change in beam angle which shifts the focal spot downstream. Intended to sit
    just after ``StaticSLMField`` in the SLM-plane chain. Output geometry equals
    input geometry.

    Args:
        tilt_std: Standard deviation of the (zero-mean Gaussian) beam angle [rad]
            -- a scalar (same for x and y) or ``(std_x, std_y)``. Each forward
            draws ``angle_x ~ N(0, std_x)`` and ``angle_y ~ N(0, std_y)``; ``x``
            is the width axis, ``y`` the height axis.
        generator: Optional :class:`torch.Generator` for reproducible sampling.
    """

    def __init__(
        self,
        tilt_std: float | tuple[float, float],
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        self.tilt_std = self._as_pair(tilt_std)
        self.generator = generator
        # Last sampled tilt (and the angles), so adjoint() can invert the most
        # recent forward() and tests can inspect the sampled angles.
        self._last_tilt: Tensor | None = None
        self._last_angle: tuple[Tensor, Tensor] | None = None
        # Optional recording of every sampled tilt (see record / record_samples).
        self._recording: bool = False
        self._angle_history: list[Tensor] = []

    @classmethod
    def from_focal_shift(
        cls,
        focal_shift_std: float | tuple[float, float],
        focal_length: float,
        *,
        generator: torch.Generator | None = None,
    ) -> PointingInstability:
        """Build with the tilt std given as a focal-spot displacement [m] instead
        of an angle.

        Converts paraxially with ``angle = focal_shift / focal_length`` (matching
        ``linear_phase(tilt_units="metres")``), so ``focal_length`` is the focal
        length of the downstream Fourier lens.
        """
        std_x, std_y = cls._as_pair(focal_shift_std)
        return cls(
            (std_x / focal_length, std_y / focal_length),
            generator=generator,
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

    @property
    def angle_history(self) -> Tensor:
        """The beam tilts sampled while recording was on, as an ``(n, 2)`` tensor
        ``[angle_x, angle_y]`` per :meth:`forward` (empty ``(0, 2)`` if none was
        recorded). Populated by :meth:`record` / :meth:`record_samples`."""
        if not self._angle_history:
            return torch.empty((0, 2))
        return torch.stack(self._angle_history)

    def record(self, enabled: bool = True) -> None:
        """Toggle recording of the per-:meth:`forward` sampled beam tilt.

        Enabling clears any previously recorded history, so each recording starts
        fresh; disabling keeps it. Read the result from :attr:`angle_history`.
        """
        self._recording = enabled
        if enabled:
            self._angle_history = []

    @contextmanager
    def record_samples(self) -> Iterator[PointingInstability]:
        """Record sampled tilts for the duration of the ``with`` block (recording
        is turned off again on exit). Read them from :attr:`angle_history`."""
        self.record(True)
        try:
            yield self
        finally:
            self.record(False)

    def lazy_init(
        self: PointingInstability, complex_amplitude: ComplexAmplitude
    ) -> None:
        # Sampling-preserving: output geometry == input geometry.
        super().lazy_init(complex_amplitude)
        grid_x, grid_y = get_spatial_grid(
            self.resolution_in,
            tuple(self.pixel_size_in[0].tolist()),
            complex_amplitude.device,
        )
        self.register_buffer("grid_x", grid_x)
        self.register_buffer("grid_y", grid_y)

    def _normal(self, std: float, device: torch.device) -> Tensor:
        """A 0-d sample from ``N(0, std)``."""
        return torch.randn((), device=device, generator=self.generator) * std

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
        if self._recording:
            self._angle_history.append(torch.stack((angle_x, angle_y)).detach())

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
        self._ensure_initialized()
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
