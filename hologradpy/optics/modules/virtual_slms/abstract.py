from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

import torch
import torch.nn as nn

from ....utils import unsqueeze_to
from ..abstract import OpticsModule
from ...complex_amplitude import ComplexAmplitude

from slmsuite.hardware.slms.slm import SLM

if TYPE_CHECKING:
    from ....hardware.slm import SLMData


class VirtualSLM(OpticsModule):
    """Differentiable phase-only SLM module.

    Sign convention: ``phase`` holds the *desired* optical phase. The field picks up
    ``exp(1j * phase)`` (wrapped to the modulation range), matching the argument of
    ``slmsuite.hardware.slms.slm.SLM.set_phase``. The value the hardware actually
    displays is the negative of it since slmsuite negates before grayscale conversion
    (see :meth:`get_displayed_phase`).

    The phase may be a single pattern ``(H, W)`` or a batch ``(N, H, W)``. A batch
    produces a field of rank ``(N, n_wavelengths, H, W)`` from a single forward pass, so
    a whole set of patterns propagates at once.
    """

    def __init__(
        self: VirtualSLM,
        phase_scaling: float,
        init_phase: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        self.phase_scaling: float = phase_scaling
        self.init_phase: torch.Tensor | None = init_phase

    def lazy_init(self, complex_amplitude: ComplexAmplitude) -> None:
        if self.init_phase is None:
            self.init_phase = torch.zeros(
                self.resolution_in,
                device=complex_amplitude.device,
                dtype=complex_amplitude.dtype_r,
            )

        self.phase = nn.Parameter(self.init_phase, requires_grad=False)

    @classmethod
    def from_slm(
        cls: type[VirtualSLM],
        slm: SLM,
        init_phase: torch.Tensor | None = None,
    ) -> VirtualSLM:
        if slm.pixel_size[0] != slm.pixel_size[1]:
            raise ValueError("Non-square pixel pitch is not supported.")
        return cls(
            phase_scaling=slm.phase_scaling,
            init_phase=init_phase,
        )

    @classmethod
    def from_slm_data(
        cls: type[VirtualSLM],
        slm_data: SLMData,
        init_phase: torch.Tensor | None = None,
    ) -> VirtualSLM:
        if slm_data.pixel_size[0] != slm_data.pixel_size[1]:
            raise ValueError("Non-square pixel pitch is not supported.")
        return cls(
            phase_scaling=slm_data.phase_scaling,
            init_phase=init_phase,
        )

    def set_phase(self, phase: torch.Tensor | NDArray) -> None:
        """Set the desired optical phase (same argument convention as
        ``slmsuite.SLM.set_phase``).

        Accepts either a single pattern ``(H, W)`` or a batch ``(N, H, W)``. A batch is
        imprinted in one forward pass, which is far cheaper than looping: the whole
        chain then runs once on a batched field instead of once per pattern.
        """
        if isinstance(phase, np.ndarray):
            phase = torch.as_tensor(phase)
        phase = phase.to(dtype=self.phase.dtype, device=self.phase.device)

        if phase.ndim not in (2, 3):
            raise ValueError(
                f"Phase must be a single (H, W) pattern or a batch of them "
                f"(N, H, W), got shape {tuple(phase.shape)}."
            )
        if tuple(phase.shape[-2:]) != tuple(self.resolution_in):
            raise ValueError(
                f"Phase resolution {tuple(phase.shape[-2:])} does not match the "
                f"SLM resolution {tuple(self.resolution_in)}."
            )

        self.phase.data = phase

    def get_phase(self) -> torch.Tensor:
        """The desired optical phase imprinted on the field (before the modulation-range
        wrap)."""
        return self.phase

    def get_displayed_phase(self) -> torch.Tensor:
        """The phase pattern as displayed on the SLM before grayscale conversion: the 
        hardware displays the negative of the desired phase.
        """
        return (-self.get_phase()).remainder(self.phase_scaling * 2 * torch.pi)

    # TODO: Add discretization and pixel crosstallk here
    def apply_phase_transforms(self: VirtualSLM, phase: torch.Tensor) -> torch.Tensor:
        """Hook for subclasses to transform the applied phase; identity by default."""
        return phase

    def forward(
        self: VirtualSLM, complex_amplitude: ComplexAmplitude
    ) -> ComplexAmplitude:
        # Wrap to the modulation range like the hardware, then imprint the
        # desired phase.
        phase = self.get_phase().remainder(self.phase_scaling * 2 * torch.pi)
        phase = self.apply_phase_transforms(phase)

        if phase.ndim >= 3:
            # A batch of patterns (N, H, W). Insert the wavelength axis to get
            # (N, 1, H, W).
            transformed_phase = phase.unsqueeze(-3)
        else:
            transformed_phase = unsqueeze_to(phase, complex_amplitude.ndim)

        # Avoid in-place modification so each forward pass builds an
        # independent autograd graph for repeated optimizer closure calls.
        complex_amplitude = complex_amplitude * torch.exp(1j * transformed_phase)

        return complex_amplitude.with_geometry(
            wavelength=complex_amplitude.wavelength,
            pixel_size=self.pixel_size_out,
        )
