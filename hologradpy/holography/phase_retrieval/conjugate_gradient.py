from __future__ import annotations

import torch
import torchmin

from .abstract import PhaseRetrieverBase

from ..utils import Timer
from ..loss_functions import LossIntensityMSE

from ...propagation.optical_systems import SLMCameraModel

# TODO: Add convergence error metrics
# TODO: Add saving functionality
class CGPhaseRetriever(PhaseRetrieverBase):
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

        self.loss_function = LossIntensityMSE(
            target_intensity=self.target, signal_mask=self.signal_region
        )

        self.timer = Timer(use_cuda=use_cuda, verbose=True)

        self.iteration: int = 0

    def set_optimizer(
            self,
            number_of_iterations: int,
            method: str = "cg"
        ) -> None:
        self.optimizer = torchmin.Minimizer(
            self.slm_camera_model.parameters(),
            method=method,
            max_iter=number_of_iterations,
            disp=1,
            callback=self.callback,
        )

    def callback(self, _):
        print(f"Iteration {self.iteration}.")
        self.iteration += 1

    def closure(self):
        self.optimizer.zero_grad()
        electric_field = self.slm_camera_model()
        loss = self.loss_function.loss(electric_field)
        print(f"Loss: {loss.item()}")
        return loss

    def retrieve_phase(
        self: CGPhaseRetriever,
        number_of_iterations: int = 10,
        parameter_name: str = "virtual_slm.phase",
        method: str = "cg",
    ) -> torch.Tensor:
        self.timer.start()
        self.set_gradient_requirements(parameter_name)
        self.set_optimizer(number_of_iterations, method=method)
        self.optimizer.step(self.closure)
        self.timer.stop()

        if "cuda" in self.device:
            torch.cuda.empty_cache()
        return self.slm_camera_model.virtual_slm.get_displayed_phase()