from __future__ import annotations

import torch


from ...optics.systems import SLMFourierLensModel

from .abstract import PhaseRetrieverBase

from ...loss_functions import INTENSITY_MSE_SCALE


# TODO: Add error metrics
# TODO: Add saving functionality
class ZernikePhaseRetriever(PhaseRetrieverBase):
    def __init__(
        self,
        slm_camera_model: SLMFourierLensModel,
        target: torch.Tensor | None = None,
        signal_region: torch.Tensor | None = None,
        loss_scale: float = INTENSITY_MSE_SCALE,
    ) -> None:
        super().__init__(slm_camera_model, target, signal_region, loss_scale)

    def optimizer_options(self) -> dict:
        """l-bfgs on Zernike coefficients converges long after the default tolerances
        would stop it, so they are tightened well below them."""
        return {"tol": 1e-20, "options": {"gtol": 1e-20, "xtol": 1e-20}}

    # What this search optimises, and with which method. The run itself is the
    # base class template.
    PARAMETER_NAME = "virtual_slm.zernike.zernike_coefficients"
    METHOD = "l-bfgs"
