from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import torch

from ...datasets import CapturedSample
from ...loss_functions import smallest_divisor
from ...roi import ROI


class TransformToTensor:
    def __init__(self, device: torch.device, dtype: torch.dtype) -> None:
        self.device = device
        self.dtype = dtype

    def __call__(self, sample: CapturedSample) -> CapturedSample:
        return {
            key: torch.as_tensor(value, dtype=self.dtype, device=self.device)
            for key, value in sample.items()
        }


class CropToRoi:
    def __init__(self, roi: ROI) -> None:
        self.roi = roi

    def __call__(self, sample: CapturedSample) -> CapturedSample:
        return {**sample, "camera_image": self.roi.crop(sample["camera_image"])}


class Normalize:
    def __init__(self, roi_mask: NDArray[np.bool_] | torch.Tensor) -> None:
        self.roi_mask = torch.as_tensor(roi_mask)

    def __call__(self, sample: CapturedSample) -> CapturedSample:
        camera_image = sample["camera_image"] * self.roi_mask.to(sample["camera_image"])
        total = camera_image.sum()
        camera_image = camera_image / total.clamp_min(smallest_divisor(total))
        return {**sample, "camera_image": camera_image}


class PrepareSample:
    """The full chain from a raw captured sample to training tensors. Convert to torch,
    crop to the region of interest, then normalize.
    """

    def __init__(
        self,
        roi: ROI,
        roi_mask: NDArray[np.bool_] | torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.transforms = (
            TransformToTensor(device, dtype),
            CropToRoi(roi),
            Normalize(roi_mask),
        )

    def __call__(self, sample: CapturedSample) -> CapturedSample:
        for transform in self.transforms:
            sample = transform(sample)
        return sample
