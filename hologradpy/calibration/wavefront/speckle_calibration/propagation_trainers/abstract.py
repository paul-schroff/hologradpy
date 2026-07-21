from __future__ import annotations
from typing import Callable

import numpy as np
from numpy.typing import NDArray

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..calibration_dataset import DatasetDescriptor

from ....camera_mapping import CameraMapping

from .....roi import ROI
from .....optics.systems import SLMFourierLensModel


# TODO (PS): A save and load method for the propagator would be nice.
class PropagationTrainer:
    cplx = torch.complex64
    real = torch.float32

    def __init__(
        self,
        dataset_descriptor: DatasetDescriptor,
        load_path: str,
        slm_camera_model: SLMFourierLensModel,
        roi_mask: NDArray[np.bool_] | None = None,
        device: str = "cpu",
    ) -> None:
        self.dataset_descriptor: DatasetDescriptor = dataset_descriptor
        self.load_path: str = load_path
        self.slm_data = self.dataset_descriptor.slm_data
        self.camera_data = self.dataset_descriptor.camera_data
        self.camera_mapping: CameraMapping = self.dataset_descriptor.camera_mapping
        self.device: str = device

        if roi_mask is None:
            roi_mask = self.dataset_descriptor.roi_masks[0]

        self.roi: ROI = ROI.detect(roi_mask, pad=0)
        self.roi_mask: NDArray[np.float_] = self.roi.crop(roi_mask)
        self.roi_size: tuple[int, ...] = self.roi_mask.shape

        self.slm_camera_model: SLMFourierLensModel = slm_camera_model

        self.mse_loss_function: nn.Module = nn.MSELoss(reduction="sum")

    def initialize_optimization(
        self,
    ) -> tuple[DataLoader, torch.optim.Optimizer]:
        raise NotImplementedError(
            f"""
            The initialize_optimization() method has not been
            implemented for {type(self).__name__}.
            """
        )

    def run_optimization(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        | None = None,
        number_of_epochs: int = 100,
        verbose: bool = True,
    ) -> None:
        # Custom loss function can be passed in as an argument
        # to the run_optimization() method. If not, the default loss function
        # is used.
        if loss_function is None:
            loss_function = self.get_loss_function()

        number_of_batches = len(dataloader)
        number_of_iterations = number_of_epochs * number_of_batches
        print(
            f"Running {number_of_epochs} epochs with {number_of_batches} "
            + f"batches each. The batch size is {dataloader.batch_size}."
            + f"\nTotal number of iterations: {number_of_iterations}."
        )

        for i in range(number_of_epochs):
            for j, sample in enumerate(dataloader):
                camera_image = sample["camera_image"]
                phase_pattern = sample["phase_pattern"]

                optimizer.zero_grad()

                output_field = self.slm_camera_model(phase_pattern)

                loss = loss_function(output_field, camera_image)

                loss.backward(retain_graph=True)

                optimizer.step()

                if verbose:
                    print(
                        f"\nEpoch {i + 1} of {number_of_epochs}, "
                        + f"sample {j + 1} of {len(dataloader)}."
                    )

        print(f"Finished {number_of_epochs} epochs.")

    def get_loss_function(
        self,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        raise NotImplementedError(
            f"""
            The get_loss_function() method has not been
            implemented for {type(self).__name__}.
            """
        )

    def get_wavefront(self) -> torch.Tensor:
        raise NotImplementedError(
            f"""
            The get_wavefront() method has not been
            implemented for {type(self).__name__}.
            """
        )

    # Gaussian-beam fitting lives on WavefrontCalibratorBase.fit_gaussian_beam, and
    # piston/tilt removal on analysis.fitting.remove_tilt; both were duplicated here
    # and have been removed.
