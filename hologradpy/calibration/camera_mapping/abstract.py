from __future__ import annotations
from dataclasses import dataclass
from typing import TypeVar
import pickle

import numpy as np
import torch
from numpy.typing import NDArray
from datetime import datetime

from slmsuite.hardware.slms.slm import SLM
from slmsuite.hardware.cameras.camera import Camera
from ...propagation import SLMFourierLensModel
from ...visualizer import VisualizationData
from ...hardware.camera_data import CameraData

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
    reprojection_errors: ArrayLike | None = None
    reprojection_rms: float | None = None
    spot_fit_parameters: list[ArrayLike] | None = None
    spot_fit_covariances: list[ArrayLike] | None = None
    average_waist: float | None = None
    average_waist_uncertainty: float | None = None
    zeroth_order_mask: ArrayLike | None = None
    excluded_points: list[tuple[float, float]] | None = None
    excluded_reasons: list[str] | None = None
    visualization_data: VisualizationData | None = None
    camera_data: CameraData | None = None
    suggested_orientation: dict | None = None
    residual_transform: ArrayLike | None = None
    metadata = []

    @property
    def is_mirrored(self) -> bool:
        """True if the camera view is mirrored (the transform flips handedness)."""
        linear = np.asarray(self.transform, dtype=np.float64)[:, :2]
        return bool(np.linalg.det(linear) < 0)

    @property
    def rotation_degrees(self) -> float:
        """Rotation of the camera axes relative to the model plane in degrees, from the 
        polar decomposition of the transform (reflection factored out for a mirrored 
        camera)."""
        linear = np.asarray(self.transform, dtype=np.float64)[:, :2]
        u, _, vt = np.linalg.svd(linear)
        if np.linalg.det(u @ vt) < 0:
            u[:, -1] *= -1
        rotation = u @ vt
        return float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))

    @property
    def scales(self) -> tuple[float, float]:
        """Scale factors of the transform (singular values, major then minor)."""
        linear = np.asarray(self.transform, dtype=np.float64)[:, :2]
        singular_values = np.linalg.svd(linear, compute_uv=False)
        return (float(singular_values[0]), float(singular_values[1]))

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
        slm_camera_model: SLMFourierLensModel,
    ):
        self.slm = slm
        self.camera = camera
        self.slm_camera_model = slm_camera_model
        self.detected_points = []
        self.calculated_points = []

        self.slm_camera_model()

    def map_camera(self) -> CameraMapping:
        raise NotImplementedError(
            "Each subclass should implement its own map_camera() method."
        )

    @staticmethod
    def calculate_reprojection_error(
        detected_points,
        calculated_points,
        transform,
    ) -> tuple[NDArray, float]:
        """Residuals of mapping the detected points through the affine transform.

        Returns the per-point residual vectors ``mapped - calculated`` (in
        simulated-plane pixels, shape ``(N, 2)``) and their root-mean-square
        length.
        """
        detected = np.asarray(detected_points, dtype=np.float64)
        calculated = np.asarray(calculated_points, dtype=np.float64)
        transform = np.asarray(transform, dtype=np.float64)
        mapped = detected @ transform[:, :2].T + transform[:, 2]
        errors = mapped - calculated
        rms = float(np.sqrt(np.mean(np.sum(errors**2, axis=1))))
        return errors, rms
