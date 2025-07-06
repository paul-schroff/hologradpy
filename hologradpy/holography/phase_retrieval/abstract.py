from __future__ import annotations
from typing import Callable

from numpy.typing import NDArray

from ...propagation.optical_systems import SLMCameraModel

class PhaseRetrievalBase:
    def __init__(
        self,
        slm_camera_model: SLMCameraModel,
        device: str = "cpu",
    ) -> None:
        self.slm_camera_model: SLMCameraModel = slm_camera_model
        self.device: str = device

    def set_optimizer(self):
        pass

    def set_gradient_requirements(self) -> None:
        for name, parameter in self.slm_camera_model.named_parameters():
            if name == "virtual_slm.phase":
                parameter.requires_grad = True
            else:
                parameter.requires_grad = False

    def set_target(self, target: NDArray) -> None:
        pass

    def set_loss_function(self, loss_function: Callable) -> None:
        pass

    def callback(self) -> None:
        pass

    def retrieve_phase(
        self,
        number_of_iterations: int = 10,
    ) -> NDArray:
        pass

    def save_results(self) -> None:
        pass
