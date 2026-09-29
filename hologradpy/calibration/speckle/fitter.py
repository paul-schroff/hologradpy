from __future__ import annotations
from typing import Iterable, Sequence

import operator
import os
from abc import ABC, abstractmethod
from functools import reduce
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import torch
from jaxtyping import Complex, Shaped
from torch.utils.data import DataLoader, Subset

from .records import SpeckleCaptureData
from .dataset_transforms import PrepareSample

from ...datasets import CaptureStore, SampleDataset
from ...hardware.camera import Background, background_at
from ...loss_functions import LossFunction, MaskedIntensityMSE
from ...roi import ROI
from ...optics.systems import SLMFourierLensModel
from ...utils import ProgressBar


class SpeckleFitter(ABC):
    """Fit a model to captured speckle patterns using gradient descent.

    For each captured ``(slm pattern, camera_image)`` the model predicts the camera
    speckle pattern and the parameters named by :meth:`trainable_parameters` are stepped
    to match.
    """

    # Shown on the progress bar.
    description: str = "Fitting"

    def __init__(
        self,
        capture_data: SpeckleCaptureData,
        slm_camera_model: SLMFourierLensModel,
        dataset_path: str | os.PathLike,
        loss: LossFunction | None = None,
        learning_rate: float = 1e-2,
        roi_mask: NDArray[np.bool_] | None = None,
        background: Background | None = None,
    ) -> None:
        """
        Args:
            capture_data: The captured dataset.
            slm_camera_model: The differentiable model to fit.
            dataset_path: The dataset file holding the samples.
            loss: The cost, taking ``(predicted_field, camera_image)``. Defaults to
                :class:`~hologradpy.loss_functions.MaskedIntensityMSE`.
            learning_rate: Adam step size.
            roi_mask: A region mask that replaces the capture's own.
            background: The counts subtracted from every frame before it is
                normalized.
        """
        self.capture_data: SpeckleCaptureData = capture_data
        self.dataset_path: Path = Path(dataset_path)
        self.slm_camera_model: SLMFourierLensModel = slm_camera_model
        self.learning_rate: float = learning_rate
        self.background: Background | None = background
        # The mean of the subtracted background over the region of the fit, or None.
        self.background_level: float | None = None

        self.device: torch.device = slm_camera_model.device
        self.dtype: torch.dtype = slm_camera_model.init_field.dtype_r

        # The frames and predictions are cropped to the bounding box of the region
        # mask, and the mask is cropped with them.
        region_mask = capture_data.roi_mask if roi_mask is None else roi_mask
        self.roi: ROI = ROI.detect(region_mask, pad=0)
        self.roi_mask: torch.Tensor = torch.as_tensor(
            self.roi.crop(region_mask), dtype=self.dtype, device=self.device
        )

        self.loss: LossFunction = (
            MaskedIntensityMSE(self.roi_mask) if loss is None else loss
        )

        self.store: CaptureStore | None = None
        self.dataset: SampleDataset | None = None
        self.phase_bitdepth: int | None = None
        self.component_history: dict[str, list[float]] = {}

    def close(self) -> None:
        """Let go of the dataset file."""
        if self.store is not None:
            self.store.close()
            self.store = None
        self.dataset = None

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
            subset_indices: Which dataset samples to use. Defaults to all.
            shuffle: Reshuffle the samples between epochs.
            verbose: Show a progress bar.

        Returns:
            list[float]: The mean loss of each epoch.
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
            total=number_of_epochs, description=self.description, verbose=verbose
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
        # Closed first, so refitting does not leave the previous mapping behind.
        self.close()
        store = CaptureStore.open(self.dataset_path)
        self.store = store
        self.phase_bitdepth: int | None = store.phase_bitdepth
        background = self._background_for(store)
        self.dataset = SampleDataset(
            store,
            transform=PrepareSample(
                self.roi,
                self.roi_mask,
                self.device,
                self.dtype,
                background=background,
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

    def _background_for(self, store: CaptureStore) -> float | NDArray | None:
        """The background to subtract from the stored frames. Its mean over the region
        of the fit is kept in :attr:`background_level`.

        The frames are stored raw. A background function is evaluated at the recorded
        exposure of the dataset.
        """
        self.background_level = None
        if self.background is None:
            return None
        background = background_at(
            self.background,
            self.capture_data.metadata.get("exposure_time"),
            np.shape(store.read(0)["camera_image"]),
        )
        if np.ndim(background) == 0:
            self.background_level = float(background)
        else:
            region = self.roi_mask.detach().cpu().numpy() > 0
            self.background_level = float(np.mean(self.roi.crop(background)[region]))
        return background

    @abstractmethod
    def trainable_parameters(self) -> Iterable[torch.nn.Parameter]:
        """The parameters this fit steps.

        An implementation switches ``requires_grad`` on for the parameters to be fitted
        and returns :meth:`enabled_parameters`.
        """

    def enabled_parameters(self) -> Iterable[torch.nn.Parameter]:
        """Every model parameter left requiring a gradient."""
        return (p for p in self.slm_camera_model.parameters() if p.requires_grad)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.trainable_parameters(), lr=self.learning_rate, amsgrad=True
        )

    def _predict_roi_fields(
        self, patterns: Shaped[torch.Tensor, "... H_slm W_slm"]
    ) -> Complex[torch.Tensor, "N H_roi W_roi"]:
        """Predict the camera-plane field for a batch of SLM phase patterns, cropped to
        the ROI.

        The whole batch is imprinted at once (:meth:`VirtualSLM.set_phase` takes ``(N,
        H, W)``) and the model runs a single forward pass, giving a field of rank ``(N,
        n_wavelengths, H, W)``. A single pattern gives ``N = 1``.
        """
        self.slm_camera_model.virtual_slm.set_levels(patterns, self.phase_bitdepth)
        field = self.slm_camera_model()
        if field.number_of_wavelengths != 1:
            raise ValueError(
                "The speckle fit compares one camera image per pattern, so the model "
                f"must carry a single wavelength, not {field.number_of_wavelengths}."
            )
        flat, _ = field.flatten_batch()  # (N, 1, H, W)
        return self.roi.crop(flat[:, 0])

    def measured_and_predicted_roi(
        self, sample_index: int = 0
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Measured and predicted ROI intensity for one dataset sample. Requires
        :meth:`fit` to have run.
        """
        if self.dataset is None:
            raise RuntimeError("No dataset yet. Call fit() first.")

        sample = self.dataset[sample_index]
        pattern = sample["slm_levels"].unsqueeze(0)
        with torch.no_grad():
            predicted = self._predict_roi_fields(pattern)
            predicted_intensity = (predicted.abs() ** 2) * self.roi_mask
            measured = sample["camera_image"] * self.roi_mask

        return (
            measured.squeeze().detach().cpu().numpy(),
            predicted_intensity.squeeze().detach().cpu().numpy(),
        )
