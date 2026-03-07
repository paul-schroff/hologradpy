from __future__ import annotations
from typing import TypedDict

import pickle
from datetime import datetime
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

import torch
from torch.utils.data import Dataset

from ..abstract import WavefrontCalibrationData

from ...camera_mapping import CameraMapping


class CalibrationDataset(Dataset):
    def __init__(
        self,
        dataset_descriptor: DatasetDescriptor,
        load_path: str | None = None,
        transform=None,
    ):
        self.dataset_descriptor: DatasetDescriptor = (
            dataset_descriptor
        )
        self.transform = transform

        if load_path is None:
            load_path = dataset_descriptor.directory
        self.load_path: str = load_path

    def __len__(self):
        return self.dataset_descriptor.number_of_patterns

    def __getitem__(self, sample_index: int):
        if torch.is_tensor(sample_index):
            sample_index = sample_index.tolist()

        sample = self.dataset_descriptor.load_training_sample(sample_index)

        if self.transform:
            sample = self.transform(sample)
        return sample


@dataclass(frozen=True, unsafe_hash=True)
class DatasetDescriptor:
    timestamp: datetime
    phase_pattern_type: str
    directory: str
    number_of_patterns: int
    slm_data: dict
    camera_data: dict
    camera_mapping: CameraMapping
    roi_mask: NDArray[np.bool_]
    data_filenames: list[TrainingSampleFilenames]
    camera_background_image: NDArray[np.float_]
    benchmark_calibration: WavefrontCalibrationData | None
    metadata: dict

    def save(self, filename: str):
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename: str) -> DatasetDescriptor:
        with open(filename, "rb") as file:
            training_data: DatasetDescriptor = pickle.load(file)
        return training_data

    def load_training_sample(self, sample_index: int) -> TrainingSample:
        """Load a training sample from the dataset."""
        sample_file_names: TrainingSampleFilenames = self.data_filenames[
            sample_index
        ]

        camera_image = np.load(
            self.directory + sample_file_names["camera_image"]
        )

        phase_pattern = np.load(
            self.directory + sample_file_names["phase_pattern"]
        )

        sample: TrainingSample = {
            "camera_image": camera_image,
            "phase_pattern": phase_pattern,
        }
        return sample


class TrainingSampleFilenames(TypedDict, total=False):
    camera_image: str
    phase_pattern: str


class TrainingSample(TypedDict):
    camera_image: NDArray[np.float_]
    phase_pattern: NDArray[np.float_]
