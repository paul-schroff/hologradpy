from __future__ import annotations
from typing import Callable

import numpy as np
from numpy.typing import NDArray

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..calibration_dataset import DatasetDescriptor

from ....camera_mapping import CameraMapping

from .....propagation.utils.tensor_utils import crop_to_roi, find_roi
from .....propagation.utils.optics_utils import gaussian_beam_intensity
from .....propagation.optical_systems import SLMFourierLensModel
from .....analysis.fitting import curve_fit_2d


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
        self.dataset_descriptor: DatasetDescriptor = (
            dataset_descriptor
        )
        self.load_path: str = load_path
        self.slm_data: SLMDisplayData = self.dataset_descriptor.slm_data
        self.camera_data: CameraData = self.dataset_descriptor.camera_data
        self.camera_mapping: CameraMapping = (
            self.dataset_descriptor.camera_mapping
        )
        self.device: str = device

        if roi_mask is None:
            roi_mask = self.dataset_descriptor.roi_masks[0]

        self.roi: tuple[int, int, int, int] = find_roi(roi_mask, pad=0)
        self.roi_mask: NDArray[np.float_] = crop_to_roi(roi_mask, self.roi)
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

    def fit_gaussian_beam(
        self, intensity: NDArray[np.complex_]
    ) -> tuple[float, float, float, float]:
        """
        Fits a Gaussian beam to the measured intensity and returns the beam
        parameters: center_x, center_y, waist_x, waist_y.

        Args:
            None
        Returns:
            center_x (float): x-coordinate of the beam center.
            center_y (float): y-coordinate of the beam center.
            waist_x (float): x-coordinate of the beam waist.
            waist_y (float): y-coordinate of the beam waist.
        """

        max_intensity = np.max(intensity)
        max_index = np.unravel_index(np.argmax(intensity), intensity.shape)

        beam_waist_estimate = np.min(self.slm_data.size) / 4

        fit_guess = [
            beam_waist_estimate,
            self.slm_data.spatial_grid_x[max_index[0], max_index[1]],
            self.slm_data.spatial_grid_y[max_index[0], max_index[1]],
            max_intensity,
            0.0,
        ]

        optmized_parameters, _ = curve_fit_2d(
            self.slm_data.spatial_grid_x,
            self.slm_data.spatial_grid_y,
            intensity,
            gaussian_beam_intensity,
            *fit_guess,
        )

        center_x = optmized_parameters[0]
        center_y = optmized_parameters[1]
        waist_x = optmized_parameters[3]
        waist_y = optmized_parameters[4]

        return center_x, center_y, waist_x, waist_y

    def remove_phase_tilt(
        self,
        phase: NDArray[np.complex_],
        mask: NDArray[np.bool_] | None = None,
    ) -> NDArray[np.complex_]:
        if mask is None:
            mask = np.ones_like(phase, dtype=np.bool_)

        def tilt(x, y, slope_x, slope_y, offset):
            return slope_x * x + slope_y * y + offset

        optimized_parameters, _ = curve_fit_2d(
            self.slm_data.spatial_grid_x[mask],
            self.slm_data.spatial_grid_y[mask],
            phase[mask],
            tilt,
            0.0,  # Initial guess for slope_x
            0.0,  # Initial guess for slope_y
            0.0,  # Initial guess for offset
        )

        fitted_tilt = tilt(
            self.slm_data.spatial_grid_x,
            self.slm_data.spatial_grid_y,
            *optimized_parameters,
        )

        return phase - fitted_tilt
