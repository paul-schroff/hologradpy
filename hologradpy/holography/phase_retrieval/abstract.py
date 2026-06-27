from __future__ import annotations
from typing import Callable

import torch
from numpy.typing import NDArray

from ...propagation.optical_systems import SLMFourierLensModel


# TODO: Add saving functionality
class PhaseRetrieverBase:
    def __init__(
        self,
        slm_camera_model: SLMFourierLensModel,
    ) -> None:
        self.slm_camera_model: SLMFourierLensModel = slm_camera_model
        # The device is determined by the optical model rather than passed in.
        self.device: torch.device = slm_camera_model.device

    def set_optimizer(self):
        pass

    def set_gradient_requirements(
        self, parameter_name: str = "virtual_slm.phase"
    ) -> None:
        named_parameters = dict(self.slm_camera_model.named_parameters())

        if parameter_name not in named_parameters:
            _ = self.slm_camera_model()
            named_parameters = dict(self.slm_camera_model.named_parameters())

        if parameter_name not in named_parameters:
            available_parameters = ", ".join(named_parameters.keys())
            raise ValueError(
                f"Parameter '{parameter_name}' not found in "
                f"slm_camera_model.named_parameters(). "
                f"Available parameters: {available_parameters}"
            )

        for name, parameter in named_parameters.items():
            parameter.requires_grad = name == parameter_name

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
