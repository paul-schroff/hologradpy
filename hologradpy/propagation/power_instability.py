from __future__ import annotations

import torch
from torch import Tensor

from .optics_module import OpticsModule
from .complex_amplitude import ComplexAmplitude


class PowerInstability(OpticsModule):
    """Random optical-power fluctuation (a fluctuating laser).

    Each :meth:`forward` draws a relative power factor from a Gaussian centred on 1
    and scales the field power by it (the amplitude by its square root), modelling a
    laser whose output power drifts from frame to frame. Intended to sit just after
    ``StaticSLMField`` in the SLM-plane chain, so the (static) SLM-plane field carries
    a freshly sampled power on every forward pass. Output geometry equals input
    geometry.

    Args:
        power_std: Standard deviation of the (mean-1) Gaussian relative power factor.
            For example 0.05 is a 5 percent RMS power fluctuation. The drawn factor is
            clamped at 0 so the power stays non-negative.
        seed: Optional integer seed. When given, an internal
            :class:`torch.Generator` (on the field's device) is seeded with it for
            reproducible sampling; otherwise the global RNG is used.
    """

    def __init__(self, power_std: float, *, seed: int | None = None) -> None:
        super().__init__()
        self.power_std = float(power_std)
        self.seed = seed
        # Built lazily on the field's device in lazy_init (None -> global RNG).
        self._generator: torch.Generator | None = None
        # Last sampled factor, so adjoint() can re-apply it and recordables() / tests
        # can inspect the realized per-frame power.
        self._last_power_factor: Tensor | None = None

    @property
    def last_power_factor(self) -> Tensor | None:
        """The most recently sampled relative power factor, or ``None`` before the
        first :meth:`forward`. Handy for recording the realized per-frame power (e.g.
        to verify a downstream power-normalization)."""
        return self._last_power_factor

    def recordables(self) -> dict[str, Tensor]:
        """Record the sampled relative power factor each forward as
        ``{"power_factor": factor}`` (see
        :class:`~hologradpy.propagation.recording.RecordingMixin`); empty before the
        first :meth:`forward`."""
        if self._last_power_factor is None:
            return {}
        return {"power_factor": self._last_power_factor}

    @property
    def power_factor_history(self) -> Tensor:
        """The relative power factors recorded while :meth:`record` was on, as an
        ``(n,)`` tensor (empty ``(0,)`` if none). Convenience alias for
        ``history["power_factor"]``."""
        return self.history.get("power_factor", torch.empty((0,)))

    def lazy_init(self, complex_amplitude: ComplexAmplitude) -> None:
        # Sampling-preserving (output geometry == input); only the RNG is built here,
        # on the field's device.
        if self.seed is not None:
            self._generator = torch.Generator(device=complex_amplitude.device)
            self._generator.manual_seed(self.seed)

    def _sample_factor(self, device: torch.device) -> Tensor:
        """A 0-d relative power factor ``~ N(1, power_std)``, clamped at 0."""
        factor = 1.0 + self.power_std * torch.randn(
            (), device=device, generator=self._generator
        )
        return factor.clamp_min(0.0)

    def forward(self, complex_amplitude: ComplexAmplitude) -> ComplexAmplitude:
        factor = self._sample_factor(complex_amplitude.device)
        self._last_power_factor = factor

        # Power scales by factor, so the amplitude scales by its square root.
        scale = torch.sqrt(factor).to(complex_amplitude.dtype_r)
        scaled = complex_amplitude * scale
        return scaled.with_geometry(
            wavelength=complex_amplitude.wavelength,
            pixel_size=self.pixel_size_out,
        )

    def adjoint(self, complex_amplitude: ComplexAmplitude) -> ComplexAmplitude:
        """Conjugate transpose of the most recent :meth:`forward`. A real scalar
        scaling is self-adjoint, so re-apply the last sampled square-root factor."""
        if self._last_power_factor is None:
            raise RuntimeError(
                "PowerInstability.adjoint() needs a prior forward() -- it re-applies "
                "the last sampled power factor."
            )
        scale = torch.sqrt(self._last_power_factor).to(complex_amplitude.dtype_r)
        scaled = complex_amplitude * scale
        return scaled.with_geometry(
            wavelength=complex_amplitude.wavelength,
            pixel_size=self.pixel_size_in,
        )
