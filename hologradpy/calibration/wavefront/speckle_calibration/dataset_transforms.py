from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import torch

from .calibration_dataset import TrainingSample
from ....loss_functions import TINY


class TransformToTensor:
    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype

    def __call__(self, sample: TrainingSample) -> TrainingSample:
        return {
            key: torch.as_tensor(value, dtype=self.dtype, device=self.device)
            for key, value in sample.items()
        }


class BackgroundSubtraction:
    def __init__(self, background_camera_image: NDArray[np.float_] | torch.Tensor):
        self.background_camera_image = torch.as_tensor(background_camera_image)

    def __call__(self, sample: TrainingSample) -> TrainingSample:
        camera_image = sample["camera_image"]
        # ``.to`` matches the sample's dtype and device, and is a no-op once it has.
        background = self.background_camera_image.to(camera_image)
        return {**sample, "camera_image": (camera_image - background).clamp_min(0.0)}


class CropToRoi:
    def __init__(self, roi):
        self.roi = roi

    def __call__(self, sample: TrainingSample) -> TrainingSample:
        return {**sample, "camera_image": self.roi.crop(sample["camera_image"])}


class Normalize:
    def __init__(self, roi_mask: NDArray[np.bool_] | torch.Tensor):
        self.roi_mask = torch.as_tensor(roi_mask)

    def __call__(self, sample: TrainingSample) -> TrainingSample:
        camera_image = sample["camera_image"] * self.roi_mask.to(sample["camera_image"])
        camera_image = camera_image / camera_image.sum().clamp_min(TINY)
        return {**sample, "camera_image": camera_image}


class PrepareSample:
    """The full chain from a raw captured sample to training tensors.

    Convert to torch, background-subtract, crop to the region of interest, then
    normalise. The four steps stay separate classes above, since each is useful and
    testable on its own, but they only ever run in this order, so the order lives here
    once rather than at the call site.

    The crop precedes the normalisation because ``roi_mask`` is already cropped to the
    region's bounding box, which is also the shape the loss compares over.
    """

    def __init__(
        self,
        background_camera_image: NDArray[np.float_],
        roi,
        roi_mask,
        device,
        dtype,
    ) -> None:
        self.transforms = (
            TransformToTensor(device, dtype),
            BackgroundSubtraction(background_camera_image),
            CropToRoi(roi),
            Normalize(roi_mask),
        )

    def __call__(self, sample: TrainingSample) -> TrainingSample:
        for transform in self.transforms:
            sample = transform(sample)
        return sample
