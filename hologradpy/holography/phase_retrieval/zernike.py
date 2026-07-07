from __future__ import annotations

import torch
import torchmin

from ...utils import Timer

from ...propagation.optical_systems import SLMFourierLensModel

from .abstract import PhaseRetrieverBase
from ..loss_functions import LossIntensityMSE


# TODO: Add error metrics
# TODO: Add saving functionality
class ZernikePhaseRetriever(PhaseRetrieverBase):
    def __init__(
        self,
        slm_camera_model: SLMFourierLensModel,
        target: torch.Tensor,
        signal_region: torch.Tensor,
    ) -> None:
        super().__init__(slm_camera_model)
        self.target: torch.Tensor = target
        self.signal_region: torch.Tensor = signal_region

        self.loss_function = LossIntensityMSE(
            target_intensity=self.target,
            signal_mask=self.signal_region,
            steepness=1e12,
        )

        self.timer = Timer(use_cuda=self.device.type == "cuda", verbose=True)

        self.iteration: int = 0

    def set_optimizer(self, number_of_iterations: int, method: str = "l-bfgs") -> None:
        self.optimizer = torchmin.Minimizer(
            self.slm_camera_model.parameters(),
            method=method,
            max_iter=number_of_iterations,
            disp=1,
            callback=self.callback,
            tol=1e-20,
            options={"gtol": 1e-20, "xtol": 1e-20},
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
        self: ZernikePhaseRetriever,
        number_of_iterations: int = 10,
        parameter_name: str = "virtual_slm.zernike.zernike_coefficients",
        method: str = "l-bfgs",
    ) -> torch.Tensor:
        self.timer.start()
        self.set_gradient_requirements(parameter_name)
        self.set_optimizer(number_of_iterations, method=method)

        # for _ in range(number_of_iterations):
        self.optimizer.step(self.closure)
        # self.callback(_)

        self.timer.stop()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return self.slm_camera_model.virtual_slm.get_phase().detach()
