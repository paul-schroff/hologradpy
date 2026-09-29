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


class SubtractBackground:
    """Subtract a background from the camera image of a sample.

    The difference is not clipped at zero, so the read noise about the background stays
    unbiased.

    Args:
        background: The counts to subtract, as one level for every pixel or a frame of
            the camera image's shape.
        device: The device of the camera images.
        dtype: The dtype of the camera images.
    """

    def __init__(
        self, background: float | NDArray, device: torch.device, dtype: torch.dtype
    ) -> None:
        self.background = torch.as_tensor(background, dtype=dtype, device=device)

    def __call__(self, sample: CapturedSample) -> CapturedSample:
        return {**sample, "camera_image": sample["camera_image"] - self.background}


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
    subtract the background when there is one, crop to the region of interest, then
    normalize.

    Args:
        roi: The camera image is cropped to this region of interest.
        roi_mask: True at the pixels to keep, in the shape of the cropped region.
        device: The device of the training tensors.
        dtype: The dtype of the training tensors.
        background: The counts subtracted before the image is normalized, as one level
            for every pixel or a whole-sensor frame, or None to subtract nothing.
    """

    def __init__(
        self,
        roi: ROI,
        roi_mask: NDArray[np.bool_] | torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        background: float | NDArray | None = None,
    ) -> None:
        subtraction = []
        if background is not None:
            subtraction.append(SubtractBackground(background, device, dtype))
        self.transforms = (
            TransformToTensor(device, dtype),
            *subtraction,
            CropToRoi(roi),
            Normalize(roi_mask),
        )

    def __call__(self, sample: CapturedSample) -> CapturedSample:
        for transform in self.transforms:
            sample = transform(sample)
        return sample
