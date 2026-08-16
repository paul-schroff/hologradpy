"""Fit an SLM-plane field to a captured dataset of phase patterns and camera frames."""

from __future__ import annotations
from typing import Sequence

import operator
import os
from functools import reduce
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import torch
from torch.utils.data import DataLoader, Subset

from .records import SpeckleCaptureData
from ....datasets import CaptureStore, SampleDataset
from .dataset_transforms import PrepareSample

from ....loss_functions import LossFunction, MaskedIntensityMSE
from ....roi import ROI
from ....optics.systems import SLMFourierLensModel
from ....utils import ProgressBar


def region_of_interest(
    capture_data: SpeckleCaptureData,
    slm_camera_model: SLMFourierLensModel,
    roi_mask: NDArray[np.bool_] | None = None,
) -> tuple[ROI, torch.Tensor]:
    """The region the fit scores over, as a crop and a mask.

    Shared by :class:`WavefrontFitter`, which crops its predictions to the ``ROI``, and
    by whoever builds the loss, which weights by the mask. Deriving both here keeps the
    two from drifting apart.

    The mask is returned in the model's dtype and on its device, since it is only ever
    multiplied against what the model returns.

    Args:
        capture_data: The captured dataset, carrying the full-frame region mask.
        slm_camera_model: The model being fitted, which fixes the dtype and device.
        roi_mask: A region mask to use instead of the capture's own.

    Returns:
        The bounding-box ``ROI`` and the mask cropped to it.
    """
    if roi_mask is None:
        roi_mask = capture_data.roi_mask

    roi = ROI.detect(roi_mask, pad=0)
    mask = torch.as_tensor(
        roi.crop(roi_mask),
        dtype=slm_camera_model.init_field.dtype_r,
        device=slm_camera_model.device,
    )
    return roi, mask


class WavefrontFitter:
    """Recover an SLM-plane field by fitting a model to captured camera frames.

    For each captured ``(slm pattern, camera_image)`` the model predicts the camera
    speckle, and whatever parameterises its ``slm_field`` is stepped to match. Nothing
    here is specific to a parameterisation: the fit optimises the parameters that field
    module registers, and the cost is supplied by the caller.

    Args:
        capture_data: The captured dataset.
        slm_camera_model: The differentiable model to fit. Its ``slm_field`` holds the
            parameters that are recovered, and it also fixes the device and dtype.
        dataset_path: The dataset file holding the samples.
        loss: The cost, taking ``(predicted_field, camera_image)``. Defaults to
            :class:`~hologradpy.loss_functions.MaskedIntensityMSE` alone, which is the
            whole cost for a parameterisation that is band limited by construction. Add
            a prior for one that is not, e.g.
            ``MaskedIntensityMSE(mask) + PhaseSmoothness(field)``. A sum reports its
            terms separately in :attr:`component_history`.
        learning_rate: Adam step size. The parameterisation sets the gradient scale, so
            this belongs to the caller: one kernel pixel moves the whole SLM plane while
            one field pixel moves one pixel.
        roi_mask: A region mask to use instead of the capture's own.
    """

    def __init__(
        self,
        capture_data: SpeckleCaptureData,
        slm_camera_model: SLMFourierLensModel,
        dataset_path: str | os.PathLike,
        loss: LossFunction | None = None,
        learning_rate: float = 1e-2,
        roi_mask: NDArray[np.bool_] | None = None,
    ) -> None:
        self.capture_data: SpeckleCaptureData = capture_data
        self.dataset_path: Path = Path(dataset_path)
        self.slm_camera_model: SLMFourierLensModel = slm_camera_model
        self.learning_rate: float = learning_rate

        # Both taken from the model rather than passed in: the mask and the camera
        # images are only ever compared against what it returns, so they have to sit
        # where it sits and carry what it carries.
        self.device: torch.device = slm_camera_model.device
        self.dtype: torch.dtype = slm_camera_model.init_field.dtype_r

        self.roi, self.roi_mask_torch = region_of_interest(
            capture_data, slm_camera_model, roi_mask
        )
        self.roi_mask: NDArray[np.bool_] = self.roi.crop(
            capture_data.roi_mask if roi_mask is None else roi_mask
        )

        self.loss: LossFunction = (
            MaskedIntensityMSE(self.roi_mask_torch) if loss is None else loss
        )

        self.dataset: SampleDataset | None = None
        self.phase_bitdepth: int | None = None
        self.component_history: dict[str, list[float]] = {}

    def fit(
        self,
        number_of_epochs: int = 100,
        batch_size: int = 3,
        subset_indices: Sequence[int] | NDArray[np.int_] | None = None,
        shuffle: bool = True,
        verbose: bool = True,
    ) -> list[float]:
        """Fit the model to the dataset and return the mean loss per epoch.

        Args:
            number_of_epochs: Passes over the dataset.
            batch_size: Phase patterns per iteration.
            subset_indices: Which dataset samples to fit against. Defaults to all.
            shuffle: Reshuffle the samples between epochs.
            verbose: Show a progress bar. 

        Returns:
            list[float]: The mean loss of each epoch, so a caller can plot the
            convergence or check that it actually went down.
        """
        dataloader = self._build_dataloader(subset_indices, batch_size, shuffle)
        optimizer = self._build_optimizer()

        number_of_batches = len(dataloader)
        if verbose:
            print(
                f"Running {number_of_epochs} epochs of {number_of_batches} "
                f"batches (batch size {dataloader.batch_size}), "
                f"{number_of_epochs * number_of_batches} iterations total."
            )

        history: list[float] = []
        self.component_history = {}
        epochs = ProgressBar(
            total=number_of_epochs, description="Fitting wavefront", verbose=verbose
        )
        with epochs:
            for _ in range(number_of_epochs):
                epoch_loss = 0.0
                epoch_components: dict[str, float] = {}
                for sample in dataloader:
                    optimizer.zero_grad()
                    output_field = self._predict_roi_fields(sample["slm_levels"])
                    components = self.loss.components(
                        output_field, sample["camera_image"]
                    )
                    loss = reduce(operator.add, components.values())
                    loss.backward()
                    optimizer.step()

                    epoch_loss += float(loss)
                    for label, value in components.items():
                        epoch_components[label] = epoch_components.get(
                            label, 0.0
                        ) + float(value)

                history.append(epoch_loss / max(number_of_batches, 1))
                for label, total in epoch_components.items():
                    self.component_history.setdefault(label, []).append(
                        total / max(number_of_batches, 1)
                    )

                epochs.update(loss=history[-1])

        return history

    def _build_dataloader(
        self,
        subset_indices: Sequence[int] | NDArray[np.int_] | None,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        # Kept on the fitter so the same transform chain can be replayed afterwards,
        # e.g. by measured_and_predicted_roi for the visualization.
        store = CaptureStore.open(self.dataset_path)
        self.phase_bitdepth: int | None = store.phase_bitdepth
        self.dataset = SampleDataset(
            store,
            transform=PrepareSample(
                self.roi, self.roi_mask, self.device, self.dtype
            ),
        )

        if subset_indices is None:
            subset_indices = np.arange(len(self.dataset))

        subset_dataset = Subset(self.dataset, subset_indices)

        if len(subset_dataset) < batch_size:
            raise ValueError(
                f"batch_size ({batch_size}) exceeds the number of samples "
                f"({len(subset_dataset)}), which would leave no batches to fit "
                "on. Lower batch_size or capture more phase patterns."
            )

        return DataLoader(
            subset_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
            pin_memory=False,
            num_workers=0,
        )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        for parameter in self.slm_camera_model.slm_field.parameters():
            parameter.requires_grad_(True)

        return torch.optim.Adam(
            (p for p in self.slm_camera_model.parameters() if p.requires_grad),
            lr=self.learning_rate,
            amsgrad=True,
        )

    def _predict_roi_fields(self, patterns: torch.Tensor) -> torch.Tensor:
        """Predict the camera-plane field for a batch of SLM phase patterns, cropped to
        the ROI.

        The whole batch is imprinted at once (:meth:`VirtualSLM.set_phase` takes ``(N,
        H, W)``) and the model runs a single forward pass, giving a field of rank ``(N,
        n_wavelengths, H, W)``.

        The SLM-plane field being recovered lives in the model's ``slm_field`` and is
        shared across the batch.
        """
        self.slm_camera_model.virtual_slm.set_levels(patterns, self.phase_bitdepth)
        field = self.slm_camera_model().as_tensor()

        if field.ndim == 4:
            # Single-wavelength setup: drop the length-1 wavelength axis.
            field = field[:, 0]
        elif field.ndim == 3:
            # A single pattern was passed in, so there is no batch axis yet.
            field = field[0].unsqueeze(0)

        return self.roi.crop(field)

    def measured_and_predicted_roi(
        self, sample_index: int = 0
    ) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
        """Measured and predicted ROI intensity for one dataset sample.

        Replays the transform chain, so the two images are directly comparable: exactly
        what the loss sees. Comparing them is the quickest way to tell a genuinely
        converging fit from a model whose light does not land in the region of interest
        at all.

        Requires :meth:`fit` to have run.
        """
        if self.dataset is None:
            raise RuntimeError("No dataset yet. Call fit() first.")

        sample = self.dataset[sample_index]
        pattern = sample["slm_levels"].unsqueeze(0)
        with torch.no_grad():
            predicted = self._predict_roi_fields(pattern)
            predicted_intensity = (predicted.abs() ** 2) * self.roi_mask_torch
            measured = sample["camera_image"] * self.roi_mask_torch

        return (
            measured.squeeze().detach().cpu().numpy(),
            predicted_intensity.squeeze().detach().cpu().numpy(),
        )

    def get_wavefront(self) -> torch.Tensor:
        """The recovered SLM-plane complex field, whatever parameterised it.

        Delegated to the field module, which is the thing that knows: stored directly by
        a ``PixelwiseSLMField``, mapped from the fitted kernel by a ``PSFSLMField``.
        """
        return self.slm_camera_model.slm_field.get_wavefront()
