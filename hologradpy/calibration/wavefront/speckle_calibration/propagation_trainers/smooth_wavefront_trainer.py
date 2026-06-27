from __future__ import annotations
from typing import Tuple, Callable
from collections import OrderedDict

import numpy as np
from numpy.typing import NDArray

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose

from .abstract import PropagationTrainer

from ..calibration_dataset import (
    DatasetDescriptor,
    CalibrationDataset,
)
from ..dataset_transforms import (
    BackgroundSubtraction,
    CropToRoi,
    Normalize,
    TransformToTensor,
)

from .....propagation.virtual_slms import VirtualSLM
from .....propagation.optical_systems import SLMNUFFTAffine
from .....propagation.utils.tensor_utils import unsqueeze_to


class SmoothWavefrontTrainer(PropagationTrainer):
    def __init__(
        self,
        dataset_descriptor: DatasetDescriptor,
        load_path: str | None = None,
        roi_mask: NDArray[np.bool_] | None = None,
        shift_nufft: tuple[float, float] = (0.0, 0.0),
        device: str = "cpu",
    ) -> None:
        self.shift_nufft = shift_nufft

        super().__init__(
            dataset_descriptor,
            load_path,
            roi_mask=roi_mask,
            device=device,
        )

        self.shift = self.camera_mapping.zeroth_order_position  # TODO: fix this

    def initialize_optimization(
        self: SmoothWavefrontTrainer,
        subset_indices: NDArray[np.int_] | None = None,
        batch_size: int = 3,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> tuple[DataLoader, torch.optim.Optimizer]:
        self.roi_mask_torch = torch.tensor(
            self.roi_mask, dtype=self.real, device=self.device
        )

        background_image = self.dataset_descriptor.camera_background_image

        dataset = CalibrationDataset(
            dataset_descriptor=self.dataset_descriptor,
            load_path=self.load_path,
            transform=Compose(
                [
                    BackgroundSubtraction(background_image),
                    CropToRoi(self.roi),
                    Normalize(self.roi_mask),
                    TransformToTensor(self.device, self.real),
                ]
            ),
        )

        if subset_indices is None:
            subset_indices = np.arange(len(dataset))

        subset_dataset = Subset(dataset, subset_indices)

        dataloader = DataLoader(
            subset_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
            pin_memory=False,
            num_workers=num_workers,
        )

        optimizer = torch.optim.Adam(
            self.slm_camera_model.parameters(), lr=0.01, amsgrad=True
        )

        return dataloader, optimizer

    def gradient_loss_function(
        self,
        output_field: torch.Tensor,
        target: torch.Tensor,
        loss_scale: float,
        smooth_phase_scale: float,
        smooth_amplitude_scale: float,
        verbose=True,
    ) -> torch.Tensor:
        number_of_targets = target.shape[-3]
        intensity_output: torch.Tensor = output_field.abs() ** 2 * self.roi_mask_torch
        target *= self.roi_mask_torch

        # TODO (PS): Make type hints work for the two lines below
        amplitude: torch.Tensor = self.slm_camera_model.constant_field.amplitude
        phase: torch.Tensor = self.slm_camera_model.constant_field.phase

        amplitude = (amplitude / amplitude.mean()).abs()

        loss_smooth_phase = gradient_loss(phase)
        loss_smooth_amplitude = gradient_loss(amplitude / amplitude.mean())

        intensity_normalized = intensity_output / unsqueeze_to(
            intensity_output.sum(dim=(1, 2)), 3, -1
        )

        target_normalized = target / unsqueeze_to(target.sum(dim=(1, 2)), 3, -1)

        loss_mse = (
            self.mse_loss_function(intensity_normalized, target_normalized)
            / number_of_targets
        )

        loss = loss_scale * (
            loss_mse
            + smooth_phase_scale * loss_smooth_phase
            + smooth_amplitude_scale * loss_smooth_amplitude
        )

        if verbose:
            print(f"MSE: {loss_mse.item():.4E}")
            print(f"Smoothness amplitude: {loss_smooth_amplitude.item():.4E}")
            print(f"Smoothness phase: {loss_smooth_phase.item():.4E}")
            print(f"Loss: {loss.item():.4E}")
        return loss

    def get_loss_function(
        self,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Wrapper function for the loss function."""

        def loss_function(
            output_field: torch.Tensor,
            camera_image: torch.Tensor,
        ) -> torch.Tensor:
            return self.gradient_loss_function(
                output_field,
                camera_image,
                loss_scale=1e14,
                smooth_phase_scale=1e-2,
                smooth_amplitude_scale=1e-2,
                verbose=True,
            )

        return loss_function


# Loss function utitlies
def forward_difference(
    input: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grad_img_x = torch.diff(input, dim=-2)
    grad_img_y = torch.diff(input, dim=-1)
    return grad_img_x, grad_img_y


def gradient_loss(input):
    gradient_x, gradient_y = forward_difference(input)
    return torch.mean(gradient_x**2) + torch.mean(gradient_y**2)


def mean_curvature(input: torch.Tensor, pixel_pitch: float = 1.0) -> torch.Tensor:
    """Calculate the mean curvature of a 2D image using finite differences."""
    gradient_x, gradient_y = torch.gradient(
        input, dim=[-2, -1], spacing=pixel_pitch, edge_order=2
    )
    curvature_xx, curvature_xy = torch.gradient(
        gradient_x, dim=[-2, -1], spacing=1, edge_order=2
    )

    # Note: curvature_yx is not needed in this calculation
    curvature_yx, curvature_yy = torch.gradient(
        gradient_y, dim=[-2, -1], spacing=1, edge_order=2
    )

    mean_curvature = (
        0.5
        * (
            (1 + gradient_x**2) * curvature_yy
            + (1 + gradient_y**2) * curvature_xx
            - 2 * gradient_x * gradient_y * curvature_xy
        )
        / ((1 + gradient_x**2 + gradient_y**2) ** (3 / 2))
    )
    return mean_curvature
