from __future__ import annotations
from dataclasses import dataclass, field, replace
from datetime import datetime

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import affine_transform
from torch import Tensor

from ...profiles.gaussian import GaussianFit
from ...geometry import AffineTransform, PartialAffineTransform
from ...grids import metres_to_pixel, pixel_frame_angle, pixel_to_metres, plane_center
from ...profiles.masks import disc_mask
from ...serialization import SaveableRecord, record_type
from ...visualizer import VisualizationData
from ...hardware import as_camera
from ...hardware.camera import Camera, CameraData, CameraOrientation

@record_type("focal_spot_fit")
@dataclass(frozen=True)
class FocalSpotFit:
    """The focal spot's mean radius on the camera, and the fit of each spot it was
    measured on.

    ``waist`` and ``waist_uncertainty`` are camera-plane metres, at the camera's pitch.
    In the focal plane the spot's radius is ``waist`` divided by
    :attr:`CameraMapping.camera_zoom`.
    """

    waist: float
    waist_uncertainty: float | None = None
    fits: list[GaussianFit] | None = None


@record_type("mapping_fit")
@dataclass(frozen=True)
class MappingFit:
    """How well :attr:`CameraMapping.transform` fits its point pairs.

    ``reprojection_errors`` are the camera points carried to the model minus the model
    points, ``(N, 2)`` in model ``(column, row)`` pixels, and ``reprojection_rms`` is
    their root mean square. ``excluded_points`` are the camera ``(column, row)``
    detections left out of the fit.
    """

    reprojection_errors: NDArray | Tensor = field(compare=False, hash=False)
    reprojection_rms: float
    excluded_points: list[tuple[float, float]] | None = None


@record_type("orientation_suggestion")
@dataclass(frozen=True)
class OrientationSuggestion:
    """The mounting that aligns the sensor with the output plane, as
    :meth:`~.CoarseMapper.map_camera` suggests it with ``find_camera_orientation=True``.

    ``suggested`` is the whole mounting, the correction composed onto the mounting the
    camera was mapped in, ready for
    :meth:`hologradpy.hardware.camera.Camera.set_orientation`. Mounted so, the camera
    shows the focal plane unmirrored and turned by at most 45 degrees.
    ``residual_transform`` is the ``(2, 3)`` camera -> model affine on ``(column, row)``
    pixels that remains once the camera is mounted as ``suggested``.
    """

    suggested: CameraOrientation
    residual_transform: NDArray | Tensor = field(compare=False, hash=False)


@record_type("camera_mapping", version=2)
@dataclass
class CameraMapping(SaveableRecord):
    """The map from camera pixels to the pixels of a model's output plane, measured
    with the model's focal-plane partial affine held at identity.

    The camera mappers measure in this frame
    (:meth:`~.SLMFourierLensModel.bypass_partial_affine`), so a mapping describes the
    camera alone. The stored fields run from the camera to the model, in the pixel
    frame. :attr:`camera_angle` and :attr:`camera_shift` give the same camera from the
    model to the camera, in the physical frame, with the meaning and the sign of the
    ``camera_angle`` and ``camera_shift`` that the camera-aware optical systems take
    (:class:`~.SLMCZT`). :attr:`camera_zoom` gives the zoom that their focal-plane
    partial affine holds as ``scale_factor``. The conventions give the direction, the
    frame and the units of every property (:ref:`Lenses and camera mappings`).
    """

    timestamp: datetime
    """When the mapping was measured."""
    name: str
    """The name of the mapper that measured it."""
    transform: NDArray | Tensor
    """The ``(2, 3)`` affine fitted from camera ``(column, row)`` pixels to model
    ``(column, row)`` pixels."""
    detected_points: list[tuple[float, float]]
    """The ``(column, row)`` camera pixels of the measured spots."""
    calculated_points: list[tuple[float, float]]
    """The ``(column, row)`` model pixels of the same spots, in the same order."""
    spot_fit: FocalSpotFit
    """The focal spot, fitted on the camera."""
    fit: MappingFit | None = None
    """The residuals of :attr:`transform` at the point pairs."""
    orientation: OrientationSuggestion | None = None
    """The mounting the mapping suggests, when it was asked for."""
    camera_data: CameraData | None = None
    """The camera the mapping was measured with, with its whole sensor."""
    output_pixel_size: tuple[float, float] | None = None
    """The ``(y, x)`` pitch in metres of the output plane the camera was measured
    against."""
    output_resolution: tuple[int, int] | None = None
    """The ``(height, width)`` of the output plane the camera was measured against."""
    metadata: dict = field(default_factory=dict)
    """Values the mapper records beside the fit."""
    visualization_data: VisualizationData | None = None
    """The data the mapper's visualizer draws. :meth:`lean` leaves it out."""

    @property
    def affine(self) -> AffineTransform:
        """:attr:`transform` as an :class:`AffineTransform`, the full fit from camera
        ``(column, row)`` pixels to model ``(column, row)`` pixels.

        It is in the pixel frame. Its ``rotation_degrees`` turns +column towards +row,
        clockwise on the displayed frame, and its ``scales`` are model pixels per camera
        pixel, major then minor.
        """
        return AffineTransform.from_matrix(
            np.asarray(self.transform, dtype=np.float64)
        )

    @property
    def partial_affine(self) -> PartialAffineTransform:
        """The similarity, a uniform zoom, a turn and a shift with no shear or mirror,
        fitted from camera ``(column, row)`` pixels to model ``(column, row)`` pixels.

        :meth:`~.PartialAffineTransform.fit` fits it to :attr:`detected_points` and
        :attr:`calculated_points` by RANSAC, on every access.
        :meth:`~.SLMFourierLensModel.calibrate_from_mapping` inverts it into a model's
        focal-plane partial affine. :attr:`camera_angle`, :attr:`camera_shift` and
        :attr:`camera_zoom` read its inverse in the physical frame.
        """
        return PartialAffineTransform.fit(
            self.detected_points, self.calculated_points
        )

    @property
    def _placement_affine(self) -> AffineTransform:
        """The camera -> model transform that :meth:`image_plane_to_sensor`,
        :meth:`sensor_to_image_plane`, :meth:`sensor_crop_to_model` and
        :attr:`zeroth_order_position` run through.

        It is :attr:`partial_affine`, which a model is calibrated from, so the mapping
        places a point where the calibrated model puts it. A similarity holds no mirror,
        so a mirrored mapping places points through :attr:`affine`, and
        :meth:`~.SLMFourierLensModel.calibrate_from_mapping` refuses it.
        """
        return self.affine if self.is_mirrored else self.partial_affine

    @property
    def zeroth_order_position(self) -> tuple[float, float]:
        """Where a model calibrated from this mapping puts the undiffracted spot on the
        sensor, as ``(column, row)`` camera pixels. It is the focal-plane origin carried
        onto the sensor by :meth:`image_plane_to_sensor`. :attr:`camera_shift` gives the
        same position in metres from the sensor's centre pixel.

        Raises:
            ValueError: The mapping records no output plane.
        """
        column, row = self.image_plane_to_sensor([(0.0, 0.0)])[0]
        return (float(column), float(row))

    def _check_similarity_in_metres(self) -> None:
        """Raise unless :attr:`partial_affine` describes the camera in metres.

        The mapping has to be unmirrored, it has to record its camera and its output
        plane, and the pixels of both planes have to be square. The pitches are
        compared to a relative 1e-6, as :meth:`check_camera` compares them.

        Raises:
            ValueError: The mapping is mirrored, it records no camera or no output
                plane, or the pixels of the sensor or of the output plane are not
                square.
        """
        if self.is_mirrored:
            raise ValueError(
                "The camera shows the focal plane mirrored, and a similarity holds no "
                "mirror, so the mapping gives no camera angle, shift or zoom. Mount "
                "the camera as map_camera(find_camera_orientation=True) suggests, and "
                "map it again."
            )
        if self.camera_data is None:
            raise ValueError(
                "The camera mapping records no camera, so the sensor's pitch and its "
                "centre pixel are unknown. Map the camera again to record it."
            )
        output_pitch, _ = self._output_plane()
        planes = {
            "sensor": tuple(float(size) for size in self.camera_data.pixel_size),
            "output plane": output_pitch,
        }
        for plane, (pitch_y, pitch_x) in planes.items():
            if not np.isclose(pitch_y, pitch_x, rtol=1e-6, atol=0.0):
                raise ValueError(
                    f"The pixels of the {plane} measure {pitch_y * 1e6:.3f} x "
                    f"{pitch_x * 1e6:.3f} um. A similarity fitted in pixels describes "
                    "the camera in metres on square pixels only, so the mapping gives "
                    "no camera angle, shift or zoom."
                )

    @property
    def camera_angle(self) -> float:
        """The turn of the focal plane as the camera shows it, in degrees from +x
        towards +y, between -180 and 180.

        It has the meaning and the sign of the ``camera_angle`` of the camera-aware
        optical systems, such as :class:`~.SLMCZT`, so a camera simulated with
        ``camera_angle=15`` reads 15. A positive angle turns the displayed image
        counterclockwise. It is the angle of the inverse of :attr:`partial_affine`, the
        model -> camera similarity, converted to the physical frame. The inverse and the
        change of frame each negate the angle, so ``partial_affine.rotation_degrees``
        holds the same number in the pixel frame.

        The similarity is a RANSAC fit to the point pairs. For a camera with an
        anisotropic zoom, RANSAC can keep a subset of the pairs, so the angle depends
        on the points the mapper measured.

        Raises:
            ValueError: The mapping is mirrored (:attr:`is_mirrored`), it records no
                camera or no output plane, or the pixels of the sensor or of the output
                plane are not square.
        """
        self._check_similarity_in_metres()
        model_to_camera = self.partial_affine.inverse()
        return float(pixel_frame_angle(model_to_camera.rotation_degrees))

    @property
    def camera_shift(self) -> tuple[float, float]:
        """Where a model calibrated from this mapping puts the zeroth order on the
        camera, as ``(x, y)`` metres from the sensor's centre pixel,
        :func:`~hologradpy.grids.plane_center` of the whole sensor.

        It has the meaning and the sign of the ``camera_shift`` of a camera-aware
        optical system with the whole sensor as its output plane. A positive y lies
        above the centre pixel on the displayed frame. It is
        :attr:`zeroth_order_position` converted by
        :func:`~hologradpy.grids.pixel_to_metres` at the pitch of :attr:`camera_data`.

        For a camera with an anisotropic zoom, RANSAC can fit the similarity to a
        subset of the point pairs, as few as two of the four pairs of a coarse mapping.
        The shift then moves by the anisotropy times the distance from those pairs to
        the zeroth order, and the calibrated model puts its zeroth order at the same
        place.

        Raises:
            ValueError: As for :attr:`camera_angle`.
        """
        self._check_similarity_in_metres()
        x, y = pixel_to_metres(
            self.zeroth_order_position,
            tuple(float(size) for size in self.camera_data.pixel_size),
            tuple(int(size) for size in self.camera_data.sensor_resolution),
        )
        return (float(x), float(y))

    @property
    def camera_zoom(self) -> float:
        """The zoom of the focal plane as the camera shows it, in camera-plane metres
        per focal-plane metre.

        It is the scale of the inverse of :attr:`partial_affine`, in camera pixels per
        model pixel, times the sensor's pitch over the pitch of the output plane. A
        model with the right focal length therefore gives 1, whatever pitch the
        mapper's model has. When the mapping was measured against a model with the
        whole sensor as its output plane,
        :meth:`~.SLMFourierLensModel.calibrate_from_mapping` writes it into both
        components of that model's ``scale_factor``.

        For a camera with an anisotropic zoom, RANSAC can fit the similarity to the
        pairs along one axis, so the zoom can be the zoom along that axis.

        Raises:
            ValueError: As for :attr:`camera_angle`.
        """
        self._check_similarity_in_metres()
        output_pitch, _ = self._output_plane()
        sensor_pitch = float(self.camera_data.pixel_size[1])
        zoom_in_pixels = self.partial_affine.inverse().scale
        return float(zoom_in_pixels * sensor_pitch / output_pitch[1])

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
        return disc_mask(shape, self.zeroth_order_position, radius)

    def tilt_to_sensor_center(self, camera: Camera) -> tuple[float, float]:
        """The tilt that steers a spot to the centre pixel of the sensor of ``camera``,
        :func:`~hologradpy.grids.plane_center` of its whole sensor. It is the
        focal-plane position of that pixel, measured from the zeroth order
        (:meth:`sensor_to_image_plane`).

        The centre pixel is the origin of :attr:`camera_shift`. For square pixels the
        tilt is therefore minus :attr:`camera_shift`, turned by minus
        :attr:`camera_angle` and divided by :attr:`camera_zoom`.

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
        tilt_x, tilt_y = self.sensor_to_image_plane(
            [plane_center(tuple(camera.sensor_resolution))]
        )[0]
        return (float(tilt_x), float(tilt_y))

    def image_plane_to_sensor(self, points: ArrayLike) -> NDArray[np.float64]:
        """Image-plane positions ``(x, y)`` in metres from the zeroth order, as sensor
        ``(column, row)`` pixels.

        :func:`~hologradpy.grids.metres_to_pixel` puts a position on its output-plane
        pixel. The inverse of :attr:`partial_affine` carries that pixel onto the sensor,
        as a model calibrated from this mapping carries it. A mirrored mapping carries
        it through the inverse of :attr:`affine`.

        Args:
            points: The ``(N, 2)`` or ``(2,)`` positions ``(x, y)`` in metres.

        Returns:
            NDArray: The ``(N, 2)`` sensor positions ``(column, row)``.

        Raises:
            ValueError: The mapping records no output plane.
        """
        pixel_size, resolution = self._output_plane()
        points = np.atleast_2d(np.asarray(points, dtype=np.float64))
        model_pixels = np.stack(
            metres_to_pixel((points[:, 0], points[:, 1]), pixel_size, resolution),
            axis=1,
        )
        return self._placement_affine.inverse().transform_points(model_pixels)

    def sensor_to_image_plane(self, pixels: ArrayLike) -> NDArray[np.float64]:
        """Sensor ``(column, row)`` pixels as image-plane positions ``(x, y)`` in metres
        from the zeroth order, the inverse of :meth:`image_plane_to_sensor`.

        Args:
            pixels: The ``(N, 2)`` or ``(2,)`` sensor positions ``(column, row)``.

        Returns:
            NDArray: The ``(N, 2)`` positions ``(x, y)`` in metres.

        Raises:
            ValueError: The mapping records no output plane.
        """
        pixel_size, resolution = self._output_plane()
        pixels = np.atleast_2d(np.asarray(pixels, dtype=np.float64))
        model_pixels = self._placement_affine.transform_points(pixels)
        return np.stack(
            pixel_to_metres(
                (model_pixels[:, 0], model_pixels[:, 1]), pixel_size, resolution
            ),
            axis=1,
        )

    def sensor_crop_to_model(self, crop: NDArray[np.float64]) -> NDArray[np.float64]:
        """``crop``, cut from the sensor, resampled onto the model's pixels about its
        centre pixel.

        Args:
            crop: The ``(height, width)`` crop, such as
                :func:`~hologradpy.calibration.wavefront.speckle_calibration.capture_focal_spot`
                returns.

        Returns:
            NDArray: The resampled crop, shaped like ``crop``. Pixels that map outside
            ``crop`` are 0.
        """
        linear = self._placement_affine.matrix[:2, :2]
        # affine_transform indexes (row, column), and the mapping (column, row).
        swap = np.array([[0.0, 1.0], [1.0, 0.0]])
        matrix = swap @ np.linalg.inv(linear) @ swap
        center = np.array([crop.shape[0] // 2, crop.shape[1] // 2], dtype=np.float64)
        return affine_transform(crop, matrix, offset=center - matrix @ center, order=1)

    @property
    def sensor_pixels_per_metre(self) -> NDArray[np.float64]:
        """The linear part of :meth:`image_plane_to_sensor`, in sensor pixels per
        image-plane metre.

        Returns:
            NDArray: The ``(2, 2)`` matrix. Its rows are the sensor ``(column, row)``,
            and its columns are the change of each along the image-plane x and y.

        Raises:
            ValueError: The mapping records no output plane.
        """
        origin, along_x, along_y = self.image_plane_to_sensor(
            [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        )
        return np.stack([along_x - origin, along_y - origin], axis=1)

    def _output_plane(self) -> tuple[tuple[float, float], tuple[int, int]]:
        """The recorded pitch ``(y, x)`` in metres and the ``(height, width)`` of the
        output plane.

        Raises:
            ValueError: The mapping records no output plane.
        """
        if self.output_pixel_size is None or self.output_resolution is None:
            raise ValueError(
                "The camera mapping records no output plane, so its model pixels "
                "cannot be converted to metres. Map the camera again to record it."
            )
        pixel_size = tuple(float(size) for size in self.output_pixel_size)
        resolution = tuple(int(size) for size in self.output_resolution)
        return pixel_size, resolution

    @property
    def is_mirrored(self) -> bool:
        """True when the camera shows the focal plane mirrored, a negative determinant
        of :attr:`transform`. A mirrored mapping gives no :attr:`camera_angle`,
        :attr:`camera_shift` or :attr:`camera_zoom`, and calibrates no model.
        """
        return self.affine.is_mirrored

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
