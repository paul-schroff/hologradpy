from __future__ import annotations

import torch

from .abstract import GradientPhaseRetriever

from ...loss_functions import INTENSITY_MSE_SCALE


from ...optics.systems import SLMFourierLensModel


# TODO: Add convergence error metrics
class PixelwisePhaseRetriever(GradientPhaseRetriever):
    """Search the SLM phase pixel by pixel, following the gradient of a cost.

    The phase of every pixel is a free parameter, and the model is differentiable end
    to end, so the search is a local minimization over all pixels. Which minimizer runs
    is an argument to :meth:`~PhaseRetrieverBase.retrieve`. Options are ``"cg"`` for 
    conjugate gradient, ``"l-bfgs"``, and ``"adam"``.
    """

    def __init__(
        self,
        slm_camera_model: SLMFourierLensModel,
        target: torch.Tensor | None = None,
        signal_region: torch.Tensor | None = None,
        init_slm_phase: torch.Tensor | None = None,
        loss_scale: float = INTENSITY_MSE_SCALE,
    ) -> None:
        """
        Args:
            slm_camera_model: The differentiable model to optimize.
            target: Target intensity on the model's output grid.
            signal_region: Region the target is optimized in.
            init_slm_phase: Phase to start from. The starting guess decides which
                minimum is reached. A guess should spread light roughly where the target
                is so the optimization converges.
            loss_scale: Weight of the default intensity cost.
        """
        super().__init__(slm_camera_model, target, signal_region, loss_scale)

        if init_slm_phase is not None:
            self.slm_camera_model.virtual_slm.set_phase(init_slm_phase.detach())

    PARAMETER_NAME = "virtual_slm.levels"
