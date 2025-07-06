from __future__ import annotations

import torch
import torchmin

from ..utils import Timer

from ...propagation.optical_systems import SLMCameraModel

from .abstract import PhaseRetrievalBase
from ..loss_functions import LossFunctionIntensityMSE


# TODO: Add error metrics
# TODO: Add saving functionality
class CGPhaseRetrieval(PhaseRetrievalBase):
    def __init__(
        self,
        slm_camera_model: SLMCameraModel,
        target: torch.Tensor,
        signal_region: torch.Tensor,
        init_slm_phase: torch.Tensor,
        device: str = "cpu",
    ) -> None:
        super().__init__(slm_camera_model, device)
        self.target: torch.Tensor = target
        self.signal_region: torch.Tensor = signal_region
        self.device: str = device
        use_cuda = "cuda" in device

        self.slm_camera_model.virtual_slm.set_phase(init_slm_phase)

        for name, parameter in self.slm_camera_model.named_parameters():
            if name == "slm.phase":
                parameter.requires_grad = True
            else:
                parameter.requires_grad = False
            print(name, parameter.requires_grad)

        self.loss_function = LossFunctionIntensityMSE(
            target_intensity=self.target, signal_mask=self.signal_region
        )

        self.timer = Timer(use_cuda=use_cuda, verbose=True)

        self.iteration: int = 0

    def set_optimizer(self, number_of_iterations: int, **kwargs):
        self.set_gradient_requirements()
        self.optimizer = torchmin.Minimizer(
            self.slm_camera_model.parameters(),
            method="cg",
            max_iter=number_of_iterations,
            disp=1,
            callback=self.callback,
            **kwargs,
        )

    def set_gradient_requirements(self) -> None:
        for name, parameter in self.slm_camera_model.named_parameters():
            if name == "virtual_slm.phase":
                parameter.requires_grad = True
            else:
                parameter.requires_grad = False

    def callback(self, _):
        print(f"Iteration {self.iteration}.")
        self.iteration += 1

    def closure(self):
        self.optimizer.zero_grad()
        loss = self.loss_function.loss(self.slm_camera_model())
        print(f"Loss: {loss.item()}")
        return loss

    def retrieve_phase(
        self: CGPhaseRetrieval,
        number_of_iterations: int = 10,
    ) -> torch.Tensor:
        self.timer.start()
        self.set_optimizer(number_of_iterations)
        self.optimizer.step(self.closure)
        self.timer.stop()

        if "cuda" in self.device:
            torch.cuda.empty_cache()
        return self.slm_camera_model.virtual_slm.phase.detach()