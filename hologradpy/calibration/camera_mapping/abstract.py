from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import ClassVar

import torch
from numpy.typing import ArrayLike

from ...geometry import AffineTransform
from ...grids import pixel_frame_vector
from ...hardware import Camera, SLM, as_camera, as_slm
from ...hardware.camera import CameraData
from ...optics import SLMFourierLensModel
from ...visualizer import VisualizationData

from .mapping import CameraMapping, FocalSpotFit, MappingFit


class CameraMapper:
    """Measures a :class:`~.CameraMapping`, the map from camera pixels to the pixels of
    the output plane of a model with its focal-plane partial affine held at identity.
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

    @contextmanager
    def _model_window_centered_on(
        self, point: tuple[float, float]
    ) -> Iterator[tuple[float, float]]:
        """Centre the model's output window on a focal-plane point for the duration of
        a ``with`` block.

        The block holds the focal-plane partial affine at identity
        (:meth:`~hologradpy.optics.systems.SLMFourierLensModel.bypass_partial_affine`).
        Subtracting ``point`` from its shift moves the image, so ``point`` renders in
        the middle of the window. The partial affine is restored on exit.

        - If the model has no focal-plane partial affine, the window stays on the zeroth
          order. A plain FFT model has none, and its window spans the whole plane.

        Args:
            point: The ``(x, y)`` focal-plane position in metres, from the zeroth order.

        Yields:
            tuple[float, float]: The ``(column, row)`` output pixels to add to a
            position in the window. The sum is the position in the plane's own frame,
            centred on the zeroth order.
        """
        with self.slm_camera_model.build().bypass_partial_affine():
            partial_affine = self.slm_camera_model.focal_plane_partial_affine
            if partial_affine is None:
                yield (0.0, 0.0)
                return
            # The shift is (x, y) in output pixels, and the pitch is (y, x).
            pixel_size_out, _ = self.slm_camera_model.output_plane
            offset = (
                float(point[0]) / pixel_size_out[1],
                float(point[1]) / pixel_size_out[0],
            )
            with torch.no_grad():
                partial_affine.shift.sub_(
                    torch.tensor(
                        offset,
                        dtype=partial_affine.shift.dtype,
                        device=partial_affine.shift.device,
                    )
                )
            yield pixel_frame_vector(offset)

    def _record(
        self,
        affine: AffineTransform,
        detected_points: ArrayLike,
        calculated_points: ArrayLike,
        spot_fit: FocalSpotFit,
        visualization_data: VisualizationData,
        excluded_points: list[tuple[float, float]] | None = None,
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
            camera_data=CameraData.from_camera(self.camera),
            output_pixel_size=pixel_size_out,
            output_resolution=resolution_out,
            visualization_data=visualization_data,
        )
