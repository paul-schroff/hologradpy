from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from numpy.typing import ArrayLike

from ...geometry import AffineTransform
from ...hardware import Camera, SLM, as_camera, as_slm
from ...hardware.camera import CameraData
from ...optics import SLMFourierLensModel
from ...visualizer import VisualizationData

from .mapping import CameraMapping, FocalSpotFit, MappingFit, OrientationSuggestion


class CameraMapper:
    """A class to determine the coordinate transform between the camera pixels
    and the pixels of the simulated image.
    """

    # The name of the mappings this mapper measures.
    mapping_name: ClassVar[str]

    def __init__(
        self,
        slm: SLM,
        camera: Camera,
        slm_camera_model: SLMFourierLensModel,
    ) -> None:
        self.slm = as_slm(slm)
        self.camera = as_camera(camera)
        self.slm_camera_model = slm_camera_model.build()

    def map_camera(self) -> CameraMapping:
        raise NotImplementedError(
            "Each subclass should implement its own map_camera() method."
        )

    def _record(
        self,
        affine: AffineTransform,
        detected_points: ArrayLike,
        calculated_points: ArrayLike,
        spot_fit: FocalSpotFit,
        visualization_data: VisualizationData,
        excluded_points: list[tuple[float, float]] | None = None,
        orientation: OrientationSuggestion | None = None,
    ) -> CameraMapping:
        """The mapping of a measurement, with its reprojection error, the time, the
        camera and the output plane of the model.

        Args:
            affine: The fitted camera -> model transform.
            detected_points: The ``(column, row)`` camera points that the transform was
                fitted to.
            calculated_points: The ``(column, row)`` model points that they correspond
                to.
            spot_fit: The fitted focal spot.
            visualization_data: The frames of the measurement.
            excluded_points: The camera detections left out of the fit.
            orientation: The camera orientation suggested by the measurement.
        """
        reprojection_errors, reprojection_rms = affine.reprojection_error(
            detected_points, calculated_points
        )
        pixel_size_out, resolution_out = self.slm_camera_model.output_plane
        return CameraMapping(
            timestamp=datetime.now(),
            name=self.mapping_name,
            transform=affine.as_matrix(homogeneous=False),
            detected_points=detected_points,
            calculated_points=calculated_points,
            spot_fit=spot_fit,
            fit=MappingFit(
                reprojection_errors=reprojection_errors,
                reprojection_rms=reprojection_rms,
                excluded_points=excluded_points,
            ),
            orientation=orientation,
            camera_data=CameraData.from_camera(self.camera),
            output_pixel_size=pixel_size_out,
            output_resolution=resolution_out,
            visualization_data=visualization_data,
        )
