from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import torch
import torch.nn as nn

from ..utils.tensor_utils import unsqueeze_to
from ..propagators.abstract import PropagatorBase

from slmsuite.hardware.slms.slm import SLM


class VirtualSLM(PropagatorBase):
    def __init__(
        self: VirtualSLM,
        slm: SLM,
        init_phase: torch.Tensor | None = None,
        device: str = "cpu",
    ) -> None:
        self.slm: SLM = slm

        super().__init__(
            self.slm.shape,
            (self.slm.pitch_um[0] * 1e-6, self.slm.pitch_um[1] * 1e-6),
            device=device,
        )

        if init_phase is None:
            init_phase = torch.zeros(
                slm.shape, dtype=self.dtype, device=self.device
            )

        self.phase = nn.Parameter(
            torch.tensor(init_phase, dtype=self.dtype, device=self.device),
            requires_grad=True,
        )

    def set_phase(self, phase: torch.Tensor | NDArray) -> None:
        if phase.shape != self.slm.shape:
            raise ValueError(
                f"Phase shape {phase.shape} does not match SLM shape "
                + f"{self.slm.shape}."
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
        conversion."""
        return -self.phase.remainder(self.slm.phase_scaling * 2 * torch.pi)

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
