from __future__ import annotations
from dataclasses import dataclass, field, replace
from datetime import datetime

import numpy as np
from numpy.typing import ArrayLike, NDArray
from torch import Tensor

from ...geometry import AffineTransform, PartialAffineTransform
from ...grids import plane_center
from ...profiles.masks import disc_mask
from ...serialization import SaveableRecord, record_type
from ...visualizer import VisualizationData
from ...hardware import as_camera
from ...hardware.camera import Camera, CameraData, CameraOrientation

@record_type("focal_spot_fit")
@dataclass(frozen=True)
class FocalSpotFit:
    waist: float
    waist_uncertainty: float | None = None
    parameters: list[NDArray | Tensor] | None = None
    covariances: list[NDArray | Tensor] | None = None


@record_type("mapping_fit")
@dataclass(frozen=True)
class MappingFit:
    reprojection_errors: NDArray | Tensor = field(compare=False, hash=False)
    reprojection_rms: float
    excluded_points: list[tuple[float, float]] | None = None


@record_type("orientation_suggestion")
@dataclass(frozen=True)
class OrientationSuggestion:
    """The camera orientation that would align the sensor with the model plane.

    ``residual_transform`` is the mapping that would remain after adopting it, via
    :meth:`hologradpy.hardware.camera.Camera.set_orientation`.
    """

    suggested: CameraOrientation
    residual_transform: NDArray | Tensor = field(compare=False, hash=False)


@record_type("camera_mapping")
@dataclass
class CameraMapping(SaveableRecord):
    """The coordinate mapping between camera pixels and the output plane of the model
    without its focal-plane partial affine. The camera mappers measure in this frame
    (see :meth:`~hologradpy.optics.systems.SLMFourierLensModel.bypass_partial_affine`).
    """

    timestamp: datetime
    name: str
    transform: NDArray | Tensor
    detected_points: list[tuple[float, float]]
    calculated_points: list[tuple[float, float]]
    spot_fit: FocalSpotFit
    fit: MappingFit | None = None
    orientation: OrientationSuggestion | None = None
    camera_data: CameraData | None = None
    output_pixel_size: tuple[float, float] | None = None
    output_resolution: tuple[int, int] | None = None
    metadata: dict = field(default_factory=dict)
    visualization_data: VisualizationData | None = None

    @property
    def affine(self) -> AffineTransform:
        """The camera -> model transform as an :class:`AffineTransform` value object
        (the single home for the rotation / mirror / scale decomposition).
        """
        return AffineTransform.from_matrix(
            np.asarray(self.transform, dtype=np.float64)
        )

    @property
    def inverse_transform(self) -> NDArray:
        """The model -> camera transform."""
        return self.affine.inverse().as_matrix(homogeneous=False)

    @property
    def partial_affine(self) -> PartialAffineTransform:
        """The camera -> model mapping refit as a similarity (uniform scale +
        rotation + translation, no shear or mirror), from the same detected /
        calculated point pairs.

        This is the ``(scale, angle, shift)`` parameterization the differentiable
        Fourier lenses and the field warp calibrate against, so it is fit directly
        from the correspondences rather than reduced from the 6-DOF
        :attr:`affine` (which would discard shear inconsistently).
        """
        return PartialAffineTransform.fit(
            self.detected_points, self.calculated_points
        )

    @property
    def placement_affine(self) -> AffineTransform:
        """The camera -> model transform that :meth:`image_plane_to_sensor`,
        :meth:`sensor_to_image_plane` and :attr:`zeroth_order_position` run through.

        It is :attr:`partial_affine`, which a model is calibrated from. The mapping
        places a point where the calibrated model puts it. A partial affine describes no
        mirror, so a mirrored mapping places points through :attr:`affine`. No model is
        calibrated from a mirrored mapping.
        """
        return self.affine if self.is_mirrored else self.partial_affine

    @property
    def zeroth_order_position(self) -> tuple[float, float]:
        """Where a model calibrated from this mapping puts the undiffracted spot on the
        sensor, as ``(row, column)``. It is the image-plane origin carried onto the
        sensor by :meth:`image_plane_to_sensor`.

        Raises:
            ValueError: The mapping records no output plane.
        """
        row, column = self.image_plane_to_sensor([(0.0, 0.0)])[0]
        return (float(row), float(column))

    @property
    def zeroth_order_xy(self) -> tuple[float, float]:
        """The zeroth order as ``(x, y)`` camera pixels. :attr:`zeroth_order_position`
        is ``(row, column)``, which most of the geometry wants the other way round.
        """
        row, column = self.zeroth_order_position
        return (column, row)

    def zeroth_order_mask(
        self, shape: tuple[int, int], radius: float
    ) -> NDArray[np.bool_]:
        """True inside a disc around the zeroth order.

        Args:
            shape: The ``(height, width)`` of the whole sensor.
            radius: The radius of the disc in sensor pixels.

        Raises:
            ValueError: The mapping records no output plane.
        """
        return disc_mask(shape, self.zeroth_order_xy, radius)

    def tilt_to_sensor_center(self, camera: Camera) -> tuple[float, float]:
        """The tilt that steers a spot to the centre of the sensor of ``camera``. It is
        the image-plane position of the sensor centre, measured from the zeroth order
        (:meth:`sensor_to_image_plane`).

        Args:
            camera: The camera, for its sensor resolution, or a driver that
                :func:`~hologradpy.hardware.as_native.as_camera` wraps. The tilt aims at
                the middle of the whole sensor, whatever region of interest is set.

        Returns:
            tuple[float, float]: ``(tilt_x, tilt_y)`` in metres, ready for
            :meth:`~hologradpy.optics.systems.SLMFourierLensModel.steering_phase`.

        Raises:
            ValueError: The mapping records no output plane, or was measured on a
                camera with another sensor, pitch or orientation.
        """
        camera = as_camera(camera)
        self.check_camera(camera)
        sensor_height, sensor_width = tuple(camera.sensor_resolution)
        tilt_x, tilt_y = self.sensor_to_image_plane(
            [(sensor_height / 2.0, sensor_width / 2.0)]
        )[0]
        return (float(tilt_x), float(tilt_y))

    def image_plane_to_sensor(self, points: ArrayLike) -> NDArray[np.float64]:
        """Image-plane positions from the zeroth order, as sensor pixels.

        A position lands ``position / output_pixel_size`` output pixels from the centre
        of the output plane. The inverse of :attr:`placement_affine` carries that point
        onto the sensor, as in a model calibrated from this mapping.

        Args:
            points: The ``(N, 2)`` or ``(2,)`` positions ``(x, y)`` in metres.

        Returns:
            NDArray: The ``(N, 2)`` sensor positions ``(row, column)``.

        Raises:
            ValueError: The mapping records no output plane.
        """
        (pitch_y, pitch_x), center = self._output_plane()
        points = np.atleast_2d(np.asarray(points, dtype=np.float64))
        model_pixels = center + np.stack(
            [points[:, 0] / pitch_x, points[:, 1] / pitch_y], axis=1
        )
        sensor = self.placement_affine.inverse().transform_points(model_pixels)
        return np.stack([sensor[:, 1], sensor[:, 0]], axis=1)

    def sensor_to_image_plane(self, pixels: ArrayLike) -> NDArray[np.float64]:
        """Sensor ``(row, column)`` pixels as image-plane positions ``(x, y)`` in metres
        from the zeroth order, the inverse of :meth:`image_plane_to_sensor`.

        Args:
            pixels: The ``(N, 2)`` or ``(2,)`` sensor positions ``(row, column)``.

        Returns:
            NDArray: The ``(N, 2)`` positions ``(x, y)`` in metres.

        Raises:
            ValueError: The mapping records no output plane.
        """
        (pitch_y, pitch_x), center = self._output_plane()
        pixels = np.atleast_2d(np.asarray(pixels, dtype=np.float64))
        model_pixels = self.placement_affine.transform_points(
            np.stack([pixels[:, 1], pixels[:, 0]], axis=1)
        )
        offsets = model_pixels - center
        return np.stack([offsets[:, 0] * pitch_x, offsets[:, 1] * pitch_y], axis=1)

    @property
    def sensor_pixels_per_metre(self) -> NDArray[np.float64]:
        """The linear part of :meth:`image_plane_to_sensor`, in sensor pixels per
        image-plane metre.

        Returns:
            NDArray: The ``(2, 2)`` matrix. Its columns are the change of the sensor
            ``(row, column)`` along the image-plane x and y.

        Raises:
            ValueError: The mapping records no output plane.
        """
        origin, along_x, along_y = self.image_plane_to_sensor(
            [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        )
        return np.stack([along_x - origin, along_y - origin], axis=1)

    def _output_plane(self) -> tuple[tuple[float, float], NDArray[np.float64]]:
        """The recorded pitch ``(y, x)`` in metres and the ``(x, y)`` centre of the
        output plane.

        Raises:
            ValueError: The mapping records no output plane.
        """
        if self.output_pixel_size is None or self.output_resolution is None:
            raise ValueError(
                "The camera mapping records no output plane, so its model pixels "
                "cannot be converted to metres. Map the camera again to record it."
            )
        pitch_y, pitch_x = (float(size) for size in self.output_pixel_size)
        resolution = tuple(int(size) for size in self.output_resolution)
        return (pitch_y, pitch_x), np.asarray(plane_center(resolution), dtype=float)

    @property
    def is_mirrored(self) -> bool:
        """True if the camera view is mirrored (the transform flips handedness)."""
        return self.affine.is_mirrored

    @property
    def rotation_degrees(self) -> float:
        """Rotation of the camera axes relative to the model plane in degrees, from the
        polar decomposition of the transform (reflection factored out for a mirrored
        camera).
        """
        return self.affine.rotation_degrees

    @property
    def scales(self) -> tuple[float, float]:
        """Scale factors of the transform (singular values, major then minor)."""
        return self.affine.scales

    def lean(self) -> CameraMapping:
        """A copy without the frames it was fit from, ready to embed in another
        record.
        """
        return replace(self, visualization_data=None)

    def check_camera(self, camera: Camera) -> None:
        """Raise unless ``camera`` has the sensor this mapping was measured with.

        The mapping holds whole-sensor pixel positions in the displayed frame, so the
        sensor resolution, the pixel pitch and the orientation of ``camera`` are
        compared with :attr:`camera_data`. The region of interest, the exposure and the
        name are left out, since every consumer reads out the whole sensor. A mapping
        without :attr:`camera_data`, such as one built by hand, is not checked.

        Args:
            camera: The camera the mapping is about to be used with.

        Raises:
            ValueError: ``camera`` differs from the recorded camera in its sensor
                resolution, pixel pitch or orientation. The message names each
                difference.
        """
        recorded = self.camera_data
        if recorded is None:
            return
        differences = []
        recorded_sensor = tuple(int(size) for size in recorded.sensor_resolution)
        sensor = tuple(int(size) for size in camera.sensor_resolution)
        if recorded_sensor != sensor:
            differences.append(
                f"The mapping was measured on a {recorded_sensor[0]} x "
                f"{recorded_sensor[1]} sensor, and the camera reads out {sensor[0]} x "
                f"{sensor[1]}."
            )
        recorded_pitch = np.asarray(recorded.pixel_size, dtype=np.float64)
        pitch = np.asarray(camera.pixel_size, dtype=np.float64)
        if not np.allclose(recorded_pitch, pitch, rtol=1e-6, atol=0.0):
            differences.append(
                "The mapping was measured at a pitch of "
                f"{recorded_pitch[0] * 1e6:.3f} x {recorded_pitch[1] * 1e6:.3f} um, "
                f"and the camera has {pitch[0] * 1e6:.3f} x {pitch[1] * 1e6:.3f} um."
            )
        recorded_orientation = np.asarray(recorded.orientation, dtype=np.float64)
        if not np.allclose(recorded_orientation, camera.orientation_matrix()):
            differences.append(
                "The mapping was measured with the sensor mounted as "
                f"{recorded.orientation_flags}, and the camera is mounted as "
                f"{camera.orientation}."
            )
        if differences:
            raise ValueError(" ".join(differences) + " Map the camera again.")

    def check_output_plane(
        self, pixel_size: tuple[float, float], resolution: tuple[int, int]
    ) -> None:
        """Raise unless the mapping was measured against an output plane of
        ``pixel_size`` and ``resolution``.

        The point pairs of a mapping are pixels of the output plane it was measured
        against. On another plane, they correspond to other positions. A mapping that
        records no plane, such as one built by hand, is not checked.

        Args:
            pixel_size: The pitch ``(y, x)`` of the output plane in metres.
            resolution: The ``(height, width)`` of the output plane.

        Raises:
            ValueError: The recorded pitch or resolution differs.
        """
        if self.output_pixel_size is None:
            return
        resolution = tuple(int(size) for size in resolution)
        recorded_pixel_size = tuple(float(size) for size in self.output_pixel_size)
        recorded_resolution = (
            resolution
            if self.output_resolution is None
            else tuple(int(size) for size in self.output_resolution)
        )
        same_pitch = np.allclose(recorded_pixel_size, pixel_size, rtol=1e-6, atol=0.0)
        if same_pitch and recorded_resolution == resolution:
            return
        raise ValueError(
            f"The mapping was measured against a {recorded_resolution[0]} x "
            f"{recorded_resolution[1]} output plane at a pitch of "
            f"{recorded_pixel_size[0] * 1e6:.3f} x {recorded_pixel_size[1] * 1e6:.3f} "
            f"um, and the model's output plane is {resolution[0]} x {resolution[1]} "
            f"at {pixel_size[0] * 1e6:.3f} x {pixel_size[1] * 1e6:.3f} um. Measure the "
            "mapping on this model."
        )
