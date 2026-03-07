from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import torch
import torch.nn as nn

from ..utils.tensor_utils import unsqueeze_to
from ..propagators.abstract import PropagatorBase

from ...hardware.utils import SLMData

from slmsuite.hardware.slms.slm import SLM


class VirtualSLM(PropagatorBase):
    def __init__(
        self: VirtualSLM,
        resolution: tuple[int, int],
        pixel_size: tuple[float, float],
        wavelength: float,
        phase_scaling: float,
        init_phase: torch.Tensor | None = None,
        device: str = "cpu",
    ) -> None:
        self.phase_scaling: float = phase_scaling

        super().__init__(resolution, pixel_size, device=device)

        self.wavelength: float = wavelength

        if init_phase is None:
            init_phase = torch.zeros(
                self.resolution_in, dtype=self.dtype, device=self.device
            )

        self.phase = nn.Parameter(
            torch.tensor(init_phase, dtype=self.dtype, device=self.device),
            requires_grad=True,
        )
    
    @classmethod
    def from_slm(
        cls: type[VirtualSLM],
        slm: SLM,
        init_phase: torch.Tensor | None = None,
        device: str = "cpu",
    ) -> VirtualSLM:
        return cls(
            resolution=slm.shape,
            pixel_size=tuple(slm.pitch_um[i] * 1e-6 for i in range(2)),
            wavelength=slm.wav_um * 1e-6,
            phase_scaling=slm.phase_scaling,
            init_phase=init_phase,
            device=device,
        )
    
    @classmethod
    def from_slm_data(
        cls: type[VirtualSLM],
        slm_data: SLMData,
        init_phase: torch.Tensor | None = None,
        device: str = "cpu",
    ) -> VirtualSLM:
        return cls(
            resolution=slm_data.shape,
            pixel_size=tuple(slm_data.pitch_um[i] * 1e-6 for i in range(2)),
            wavelength=slm_data.wav_um * 1e-6,
            phase_scaling=slm_data.phase_scaling,
            init_phase=init_phase,
            device=device,
        )

    def set_phase(self, phase: torch.Tensor | NDArray) -> None:
        if phase.shape != self.resolution_in:
            raise ValueError(
                f"Phase shape {phase.shape} does not match SLM shape "
                + f"{self.resolution_in}."
            )

        if isinstance(phase, np.ndarray):
            self.phase.data = torch.tensor(
                phase, dtype=self.phase.dtype, device=self.phase.device
            )
        else:
            self.phase.data = phase.to(
                dtype=self.phase.dtype, device=self.phase.device
            )

    def get_displayed_phase(self) -> torch.Tensor:
        """Returns the phase pattern as displayed on the SLM before grayscale
        conversion.
        """
        return -self.phase.remainder(self.phase_scaling * 2 * torch.pi)

    def apply_phase_transforms(
            self: VirtualSLM,
            phase: torch.Tensor
        ) -> torch.Tensor:
        return unsqueeze_to(self.get_displayed_phase(), 3)

    def forward(
        self: VirtualSLM, input_field: torch.Tensor | None = None
    ) -> torch.Tensor:
        phase = self.get_displayed_phase()
        transformed_phase = self.apply_phase_transforms(phase)

        if input_field is None:
            return torch.exp(1j * transformed_phase).squeeze()
        else:
            input_field = unsqueeze_to(input_field, 3)
            return input_field * torch.exp(1j * transformed_phase).squeeze()
