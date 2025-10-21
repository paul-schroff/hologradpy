from __future__ import annotations

import torch

from ..elements import Zernike

from slmsuite.hardware.slms.slm import SLM

from .abstract import VirtualSLM

class ZernikeSLM(VirtualSLM):
    def __init__(
        self: ZernikeSLM,
        slm: SLM,
        number_of_orders: int = 5,
        initial_coefficients: torch.Tensor | None = None,
        device: str = "cpu",
    ) -> None:
        self.slm: SLM = slm

        super(VirtualSLM, self).__init__(
            self.slm.shape,
            (self.slm.pitch_um[0] * 1e-6, self.slm.pitch_um[1] * 1e-6),
            device=device,
        )
        self.zernike = Zernike(
            self.resolution_in,
            self.pixel_size_in,
            number_of_orders=number_of_orders,
            initial_coefficients=initial_coefficients,
            device=self.device,
            norm='noll',
        )
    
    def set_phase(self, phase: torch.Tensor) -> None:
        raise NotImplementedError(
            "ZernikeSLM does not support setting the phase directly."
        )

    def get_displayed_phase(self):
        phase = self.zernike.get_phase()
        return phase.remainder(self.slm.phase_scaling * 2 * torch.pi)
