from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import torch

from .calibration_dataset import TrainingSample


# Pytorch Dataset transform classes for the speckle calibration dataset
# TODO: Check if they should be subclassed from a pytorch base class
class TransformToTensor(object):
    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype

    def __call__(self, sample: TrainingSample) -> TrainingSample:
        camera_image, phase_pattern = (
            sample["camera_image"],
            sample["phase_pattern"],
        )
        return {
            "camera_image": torch.tensor(
                camera_image, dtype=self.dtype, device=self.device
            ),
            "phase_pattern": torch.tensor(
                phase_pattern, dtype=self.dtype, device=self.device
            ),
        }


class Normalize(object):
    def __init__(self, roi_mask):
        self.roi_mask = roi_mask

    def __call__(self, sample: TrainingSample) -> TrainingSample:
        camera_image = sample["camera_image"]
        camera_image *= self.roi_mask
        camera_image /= np.sum(camera_image)
        sample["camera_image"] = camera_image
        return sample


class CropToRoi(object):
    def __init__(self, roi):
        self.roi = roi

    def __call__(self, sample: TrainingSample) -> TrainingSample:
        camera_image = sample["camera_image"]
        sample["camera_image"] = self.roi.crop(camera_image)
        return sample


class BackgroundSubtraction(object):
    def __init__(self, background_camera_image: NDArray[np.float_]):
        self.background_camera_image = background_camera_image

    def __call__(self, sample: TrainingSample) -> TrainingSample:
        camera_image = sample["camera_image"]
        camera_image -= self.background_camera_image
        camera_image[camera_image < 0] = 0
        sample["camera_image"] = camera_image
        return sample
