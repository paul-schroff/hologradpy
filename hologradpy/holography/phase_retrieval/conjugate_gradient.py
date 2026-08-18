from __future__ import annotations

import torch

from .abstract import PhaseRetrieverBase

from ...loss_functions import INTENSITY_MSE_SCALE


from ...optics.systems import SLMFourierLensModel


# TODO: Add convergence error metrics
class CGPhaseRetriever(PhaseRetrieverBase):
    def __init__(
        self,
        slm_camera_model: SLMFourierLensModel,
        target: torch.Tensor | None = None,
        signal_region: torch.Tensor | None = None,
        init_slm_phase: torch.Tensor | None = None,
        loss_scale: float = INTENSITY_MSE_SCALE,
    ) -> None:
        super().__init__(slm_camera_model, target, signal_region, loss_scale)

        if init_slm_phase is not None:
            self.slm_camera_model.virtual_slm.set_phase(init_slm_phase.detach())

    # What this search optimizes, and with which method. The run itself is the
    # base class template.
    PARAMETER_NAME = "virtual_slm.levels"
    METHOD = "cg"
