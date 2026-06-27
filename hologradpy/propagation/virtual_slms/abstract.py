from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import torch
import torch.nn as nn

from ..utils.tensor_utils import unsqueeze_to
from ..optics_module import OpticsModule
from ..complex_amplitude import ComplexAmplitude

from ...hardware.slm_data import SLMData

from slmsuite.hardware.slms.slm import SLM


class VirtualSLM(OpticsModule):
    def __init__(
        self: VirtualSLM,
        phase_scaling: float,
        init_phase: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        self.phase_scaling: float = phase_scaling
        self.init_phase: torch.Tensor | None = init_phase

    def lazy_init(self, complex_amplitude: ComplexAmplitude) -> None:
        super().lazy_init(complex_amplitude)

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
        if slm.pitch_um[0] != slm.pitch_um[1]:
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
        if slm_data.pitch_um[0] != slm_data.pitch_um[1]:
            raise ValueError("Non-square pixel pitch is not supported.")
        return cls(
            phase_scaling=slm_data.phase_scaling,
            init_phase=init_phase,
        )

    def set_phase(self, phase: torch.Tensor | NDArray) -> None:
        if isinstance(phase, np.ndarray):
            self.phase.data = torch.tensor(
                phase, dtype=self.phase.dtype, device=self.phase.device
            )
        else:
            self.phase.data = phase.to(dtype=self.phase.dtype, device=self.phase.device)

    def get_displayed_phase(self) -> torch.Tensor:
        """Returns the phase pattern as displayed on the SLM before grayscale
        conversion.
        """
        return -self.phase.remainder(self.phase_scaling * 2 * torch.pi)

    def apply_phase_transforms(self: VirtualSLM, phase: torch.Tensor) -> torch.Tensor:
        return self.get_displayed_phase()

    def forward(
        self: VirtualSLM, complex_amplitude: ComplexAmplitude
    ) -> ComplexAmplitude:
        phase = self.get_displayed_phase()

        transformed_phase = unsqueeze_to(
            self.apply_phase_transforms(phase), complex_amplitude.ndim
        )

        # Avoid in-place modification so each forward pass builds an
        # independent autograd graph for repeated optimizer closure calls.
        complex_amplitude = complex_amplitude * torch.exp(1j * transformed_phase)

        return complex_amplitude.with_geometry(
            wavelength=complex_amplitude.wavelength,
            pixel_size=self.pixel_size_out,
        )
