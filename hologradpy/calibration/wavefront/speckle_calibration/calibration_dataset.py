from __future__ import annotations
from typing import TypedDict

import os
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import torch
from torch.utils.data import Dataset

from ..abstract import WavefrontCalibrationData

from ...camera_mapping import CameraMapping

from ....hardware.camera import CameraData
from ....hardware.slm import SLMData
from ....serialization import SaveableRecord

# A captured dataset is a directory: the per-sample ``.npy`` files plus this
# manifest, which is a saved DatasetDescriptor.
DATASET_MANIFEST_NAME = "dataset_descriptor.pkl"


class CalibrationDataset(Dataset):
    """Torch dataset over the ``.npy`` sample files of a captured dataset.

    ``dataset_directory`` is where the sample files live. It is supplied here rather
    than read off the descriptor, so a dataset that has been moved or copied still
    loads (see :class:`DatasetDescriptor`).
    """

    def __init__(
        self,
        dataset_descriptor: DatasetDescriptor,
        dataset_directory: str | os.PathLike,
        transform=None,
        cache: bool = True,
    ):
        self.dataset_descriptor: DatasetDescriptor = dataset_descriptor
        self.transform = transform
        self.dataset_directory: Path = Path(dataset_directory)
        self.cache = cache
        self._cached: dict[int, dict] = {}

    def __len__(self):
        return self.dataset_descriptor.number_of_patterns

    def __getitem__(self, sample_index: int):
        if torch.is_tensor(sample_index):
            sample_index = sample_index.tolist()

        if self.cache and sample_index in self._cached:
            return self._cached[sample_index]

        sample = self.dataset_descriptor.load_training_sample(
            sample_index, self.dataset_directory
        )

        if self.transform:
            sample = self.transform(sample)

        if self.cache:
            self._cached[sample_index] = sample
        return sample


@dataclass(frozen=True)
class DatasetDescriptor(SaveableRecord):
    """Everything about a captured speckle dataset except the bulk sample arrays.

    Records how the patterns were made and what the setup looked like, and names the
    per-sample files relative to the directory they live in.

    Callers pass the directory to :meth:`load_training_sample` (or to
    :class:`CalibrationDataset`), which keeps a moved or copied dataset readable.
    """

    timestamp: datetime
    phase_pattern_type: str
    number_of_patterns: int
    slm_data: SLMData
    camera_data: CameraData
    camera_mapping: CameraMapping
    roi_mask: NDArray[np.bool_]
    data_filenames: list[TrainingSampleFilenames]
    camera_background_image: NDArray[np.float_]
    benchmark_calibration: WavefrontCalibrationData | None
    metadata: dict

    # save / load come from SaveableRecord.

    def load_training_sample(
        self, sample_index: int, dataset_directory: str | os.PathLike
    ) -> TrainingSample:
        """Load one ``(camera_image, phase_pattern)`` pair from the dataset."""
        sample_file_names: TrainingSampleFilenames = self.data_filenames[sample_index]
        root = Path(dataset_directory)

        return {
            "camera_image": np.load(root / sample_file_names["camera_image"]),
            "phase_pattern": np.load(root / sample_file_names["phase_pattern"]),
        }


class TrainingSampleFilenames(TypedDict, total=False):
    camera_image: str
    phase_pattern: str


class TrainingSample(TypedDict):
    camera_image: NDArray[np.float_]
    phase_pattern: NDArray[np.float_]
