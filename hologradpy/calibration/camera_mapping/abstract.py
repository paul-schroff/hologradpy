from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ...hardware import Camera, SLM, as_camera, as_slm
from ...optics import SLMFourierLensModel
from ...geometry import AffineTransform

from .mapping import CameraMapping


class CameraMapper:
    """A class to determine the coordinate transform between the camera pixels
    and the pixels of the simulated image.
    """

    def __init__(
        self,
        slm: SLM,
        camera: Camera,
        slm_camera_model: SLMFourierLensModel,
    ) -> None:
        self.slm = as_slm(slm)
        self.camera = as_camera(camera)
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
        detected_points: ArrayLike,
        calculated_points: ArrayLike,
        transform: ArrayLike,
    ) -> tuple[NDArray, float]:
        """Residuals of mapping the detected points through the affine transform.

        Returns the per-point residual vectors ``mapped - calculated`` (in
        simulated-plane pixels, shape ``(N, 2)``) and their root-mean-square
        length.
        """
        affine = AffineTransform.from_matrix(np.asarray(transform, dtype=np.float64))
        return affine.reprojection_error(detected_points, calculated_points)
