from __future__ import annotations
from dataclasses import dataclass
from typing import TypeVar
import pickle 

import torch
from numpy.typing import NDArray
from datetime import datetime

from slmsuite.hardware.slms.slm import SLM
from slmsuite.hardware.cameras.camera import Camera
from ...propagation import SLMCameraModel

ArrayLike = TypeVar("ArrayLike", torch.Tensor, NDArray)

@dataclass
class CameraMapping:
    timestamp: datetime
    name: str
    transform: ArrayLike
    inverse_transform: ArrayLike
    detected_points: list[tuple[float, float]]
    calculated_points: list[tuple[float, float]]
    camera_images: list[ArrayLike]
    simulated_images: list[ArrayLike]
    zeroth_order_position: tuple[float, float]
    focal_spot_radius: float
    metadata = []

    def save(self, filename: str):
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename: str) -> CameraMapping:
        with open(filename, "rb") as file:
            camera_mapping: CameraMapping = pickle.load(file)
        return camera_mapping


# TODO: Add saving functionality
class CameraMapper:
    """A class to determine the coordinate transform between the camera pixels 
    and the pixels of the simulated image.
    """
    def __init__(
        self,
        slm: SLM,
        camera: Camera,
        slm_camera_model: SLMCameraModel,
    ):
        self.slm = slm
        self.camera = camera
        self.slm_camera_model = slm_camera_model
        self.detected_points = []
        self.calculated_points = []

    def map_camera(self) -> CameraMapping:
        raise NotImplementedError(
            "Each subclass should implement its own map_camera() method."
            )
    
    def calculate_reprojection_error(self) -> float:
        raise NotImplementedError(
            "Each subclass should implement its own "
            "calculate_reprojection_error() method."
        )

