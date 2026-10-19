from __future__ import annotations
import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

import torch

from ....geometry import AffineTransform

from ....hardware import Camera, SLM

from ....optics.systems import SLMFourierLensModel
from ....profiles.phase import binary_phase_grating
from ....grids import (
    metres_to_pixel,
    pixel_frame_angle,
    pixel_frame_vector,
    pixel_to_metres,
)
from ....fourier_optics import get_focal_spot_radius
from ....holography.phase_retrieval import LinearSuperpositionPhaseRetriever
from ....profiles.gaussian import Gaussian
from ....utils import as_image, gpu_to_numpy
from ....roi import ROI
from ....profiles.masks import disc_mask

from ...spot_detection import (
    DETECTION_HALF_WIDTH_RADII,
    brightest_pixel,
    detect_spot,
    get_diffraction_spot_position,
    has_prominent_peak,
    zeroth_order_mask_radius,
)
from ...exposure import expose_until_spot

from ....hardware.camera import CameraOrientation

from ..abstract import CameraMapper
from ..mapping import CameraMapping, FocalSpotFit, OrientationSuggestion
from .visualizer import CoarseVisualizationData


_PROBE_RECTANGLE = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))

# Rotation-safe grid spacing for a W x H sensor is min(W, H / sqrt(2)). At least
# one spot lands on the sensor at any rotation.
_PROBE_SPACING_FRACTION = 1.0 / np.sqrt(2.0)

_MAIN_ORDER_BRIGHTNESS_RATIO = 2.0


@dataclass
class _ProbeMeasurements:
    """Camera and model measurements of the four affine probes."""

    camera_points: list[tuple[float, float]]
    simulated_points: list[tuple[float, float]]
    camera_frames: list[NDArray]
    simulated_frames: list[NDArray]
    focal_spot_radius: float


class CoarseMapper(CameraMapper):
    """Coarse camera mapping from sequential single probe spots.

    Linear-phase tilts across the full SLM aperture are displayed one at a time, and
    each spot is fitted on the camera (:func:`get_diffraction_spot_position`). The
    fitted camera positions are matched with the model's output for the same tilts.

    A zeroth order on the sensor is located first, as the spot a 0/pi grating
    suppresses. It seeds the search for the sensor centre and is masked out of every
    probe fit, so it may be brighter than the probes. Without a zeroth order on the
    sensor, the focal plane is searched along an outward spiral of probe spots until
    one lands on the sensor. The spiral stays within the model's field of view. The
    probe pattern is placed around the tilt of that spot. The mapping then places the
    zeroth order off the sensor (:attr:`CameraMapping.zeroth_order_position`).

    The result gives the position of the sensor relative to the zeroth order and its
    orientation (:attr:`CameraMapping.rotation_degrees`,
    :attr:`CameraMapping.is_mirrored` and :attr:`CameraMapping.scales`). It can seed
    :meth:`SpotArrayMapper.map_camera`, so the fine spot array is placed entirely on
    the sensor.
    """

    mapping_name = "coarse"

    def __init__(
        self,
        slm: SLM,
        camera: Camera,
        slm_camera_model: SLMFourierLensModel,
    ) -> None:
        """
        Args:
            slm: Hardware (or simulated) SLM that displays the probe gratings.
            camera: Camera observing the focal plane.
            slm_camera_model: Ideal SLM -> camera model. The camera is mapped against
                the output plane of the model without its partial affine, and the model
                is called once here to initialize its lazy modules.
        """
        super().__init__(slm, camera, slm_camera_model)

        self._search_array_image: NDArray | None = None
        self._walk_frames: list[NDArray] = []

    def map_camera(
        self,
        exposure_time: float | None = None,
        search_radius: float | None = None,
        search_step: float | None = None,
        beam_diameter: float | None = None,
        initial_tilt: tuple[float, float] | None = None,
        find_camera_orientation: bool = False,
    ) -> CameraMapping:
        """Measure the coarse mapping of the camera from probe spots.

        The mapping is measured with the model's focal-plane partial affine at identity
        (:meth:`~hologradpy.optics.systems.SLMFourierLensModel.bypass_partial_affine`).
        The whole sensor is read out while the camera is mapped, and its exposure, gain
        and region of interest are put back afterwards
        (:meth:`~hologradpy.hardware.camera.Camera.preserve_exposure_gain_and_roi`).

        Args:
            exposure_time: Camera exposure in seconds per probe. If None, exposure is
                calibrated automatically.
            search_radius: How far from the zeroth order to search for the sensor, in
                focal-plane metres. Defaults to the full SLM Nyquist-addressable region.
            search_step: Spacing of the rectangular search spiral in metres. Defaults to
                the rotation-safe grid spacing min(W, H / sqrt(2)) of the camera-sensor
                extents (W smaller, H larger), minus the detection window (2 *
                DETECTION_HALF_WIDTH_RADII focal-spot radii).
            beam_diameter: Beam diameter on the SLM in metres. The initial focal-spot
                radius is estimated from it and sizes the spot detection window and
                thresholds. Defaults to the smaller SLM dimension.
            initial_tilt: An (x, y) tilt in focal-plane metres, known to land a spot
                on the sensor. It seeds the centre search directly, so the spiral
                search is skipped and ``search_radius`` is ignored. A ValueError is
                raised if no spot is detected at this tilt. A zeroth order located on
                the sensor seeds the centre search at zero tilt, and ``initial_tilt``
                is then not used.
            find_camera_orientation: If True, suggest the discrete camera orientation
                that aligns the camera best with the model plane. The suggestion and
                its near-identity residual are recorded on the result as
                ``orientation``. The camera is not modified. Apply the suggestion
                yourself with ``camera.set_orientation(mapping.orientation.suggested)``
                for visually-aligned frames.

        Returns:
            CameraMapping: The mapping named ``"coarse"``, with the affine transform and
            its reprojection residuals.
        """
        with (
            self.camera.preserve_exposure_gain_and_roi(full_sensor=True),
            self.slm_camera_model.bypass_partial_affine(),
        ):
            # Reset the per-stage captures recorded for CoarseMapperVisualizer.
            self._search_array_image = None
            self._walk_frames = []

            pixel_size_out, resolution_out = self.slm_camera_model.output_plane
            camera_pixel_size = np.asarray(
                self.camera.pixel_size, dtype=float
            )  # (y, x) m
            camera_pitch = camera_pixel_size[::-1]  # (x, y) for Cartesian geometry
            camera_shape = tuple(self.camera.resolution)  # (height, width)

            field_of_view = (
                camera_shape[1] * camera_pitch[0],
                camera_shape[0] * camera_pitch[1],
            )

            if beam_diameter is None:
                beam_diameter = min(self.slm.aperture_extent)
            spot_radius = get_focal_spot_radius(
                beam_radius=0.5 * beam_diameter,
                wavelength=self.slm.wavelength,
                focal_length=self.slm_camera_model.focal_length,
            )

            if search_step is None:
                search_step = self._default_search_step(spot_radius, field_of_view)

            probe_shift = max(
                0.1 * min(field_of_view),
                2.0 * DETECTION_HALF_WIDTH_RADII * spot_radius,
            )

            addressable = self.slm_camera_model.addressable_half_extent()
            if search_radius is None:
                # Cover the full SLM Nyquist-addressable rectangle.
                half_extent = addressable
            else:
                # Make sure the search spiral does not exceed the SLM's
                # Nyquist-addressable area.
                half_extent = (
                    min(search_radius, addressable[0]),
                    min(search_radius, addressable[1]),
                )

            # The zeroth order is located first if it hits the sensor
            zeroth_order_position = self._locate_zeroth_order(spot_radius)

            # In auto-exposure mode with the zeroth order off the sensor, a spot array
            # is generated over the entire addressable area, and the camera autoexposes
            # on it to calibrate a fixed exposure upfront.
            if exposure_time is None and zeroth_order_position is None:
                exposure_time = self._calibrate_exposure(
                    half_extent, search_step, spot_radius
                )

            tilt_on_sensor = self._tilt_on_sensor(
                zeroth_order_position,
                initial_tilt,
                half_extent=half_extent,
                search_step=search_step,
                probe_shift=probe_shift,
                exposure_time=exposure_time,
                spot_radius=spot_radius,
            )

            # The spot radius and the centre search run at an exposure and a gain
            # autoexposed on the found spot. The block then puts back the probe
            # exposure and its gain, so the probes run at the gain the probe exposure
            # was calculated for.
            with self.camera.preserve_exposure_gain_and_roi():
                # Measuring the focal spot radius from a Gaussian fit. The center
                # search uses this to scale its probe offset and detection window.
                spot_radius = self._measure_spot_radius(
                    tilt_on_sensor, spot_radius
                )

                # Finding the center of the camera sensor and the local tilt that
                # places the probe spots.
                center_tilt, jacobian = self._center_search(
                    tilt=tilt_on_sensor,
                    camera_shape=camera_shape,
                    spot_radius=spot_radius,
                    zeroth_order_position=zeroth_order_position,
                )

            probe_mask = None
            if zeroth_order_position is not None:
                # The zeroth order is masked out of every probe fit.
                mask_radius = zeroth_order_mask_radius(spot_radius, camera_pixel_size)
                probe_mask = ~disc_mask(
                    camera_shape, zeroth_order_position, mask_radius
                )

            probes = self._measure_probes(
                probe_tilts=self._probe_tilts(
                    center_tilt, jacobian, camera_shape, camera_pitch
                ),
                exposure_time=exposure_time,
                camera_pixel_size=camera_pixel_size,
                camera_shape=camera_shape,
                field_of_view=field_of_view,
                model_window_offset=(
                    center_tilt[0] / pixel_size_out[1],
                    center_tilt[1] / pixel_size_out[0],
                ),  # (x, y) in output pixels
                mask=probe_mask,
            )

            affine = AffineTransform.fit(probes.camera_points, probes.simulated_points)
            self._warn_about_unaddressable_sensor(
                affine, camera_shape, pixel_size_out, resolution_out, addressable
            )

            transform = affine.as_matrix(homogeneous=False)
            # Reduce all four probes to one frame.
            probe_composite = np.maximum.reduce(probes.camera_frames)
            visualization_data = self._build_visualization_data(
                half_extent,
                search_step,
                addressable,
                pixel_size_out,
                resolution_out,
                camera_shape,
                transform,
                probe_composite,
                np.maximum.reduce(probes.simulated_frames),
                np.asarray(probes.camera_points, dtype=np.float64),
                np.asarray(probes.simulated_points, dtype=np.float64),
            )

            orientation = None
            if find_camera_orientation:
                orientation = self._suggest_camera_orientation(transform, camera_shape)

            return self._record(
                affine,
                probes.camera_points,
                probes.simulated_points,
                FocalSpotFit(waist=probes.focal_spot_radius),
                visualization_data,
                orientation=orientation,
            )

    def _tilt_on_sensor(
        self,
        zeroth_order_position: tuple[float, float] | None,
        initial_tilt: tuple[float, float] | None,
        half_extent: tuple[float, float],
        search_step: float,
        probe_shift: float,
        exposure_time: float | None,
        spot_radius: float,
    ) -> tuple[float, float]:
        """A tilt that lands a spot on the sensor, to seed the centre search.

        A zeroth order on the sensor is a spot at zero tilt. Without one, a given
        ``initial_tilt`` is checked for a spot. Without either, the focal plane is
        searched along an outward spiral of tilts (:meth:`_search_spot`).

        Raises:
            ValueError: No spot was found at ``initial_tilt``.
        """
        if zeroth_order_position is not None:
            return (0.0, 0.0)
        if initial_tilt is None:
            return self._search_spot(
                half_extent=half_extent,
                search_step=search_step,
                probe_shift=probe_shift,
                exposure_time=exposure_time,
                spot_radius=spot_radius,
            )
        if self._spot_on_sensor(
            initial_tilt, exposure_time, spot_radius
        ) is None:
            raise ValueError(
                f"No spot was found on the sensor at initial_tilt "
                f"{initial_tilt} (focal-plane metres). Check the tilt and "
                "exposure."
            )
        return initial_tilt

    @staticmethod
    def _probe_tilts(
        center_tilt: tuple[float, float],
        jacobian: NDArray | None,
        camera_shape: tuple[int, int],
        camera_pitch: NDArray,
    ) -> list[tuple[float, float]]:
        """The tilts of the four affine probes, at the corners of a rectangle centred in
        the camera frame. The rectangle spans half the sensor width and height.

        The measured ``jacobian``, the change of the sensor ``(column, row)`` per metre
        of tilt along x and y, places the probes. Without one, a nominal unrotated
        tilt-to-pixel map at unit magnification places them, with +y towards row 0.
        The affine fit still recovers the transform from wherever the probes land.
        """
        if jacobian is None:
            jacobian = np.diag(
                pixel_frame_vector((1.0 / camera_pitch[0], 1.0 / camera_pitch[1]))
            )
        half_extent_px = np.array(
            [camera_shape[1] / 4.0, camera_shape[0] / 4.0]  # (column, row)
        )
        corner_offsets = half_extent_px * np.asarray(_PROBE_RECTANGLE)
        return [
            (center_tilt[0] + float(dt[0]), center_tilt[1] + float(dt[1]))
            for dt in corner_offsets @ np.linalg.inv(jacobian).T
        ]

    def _warn_about_unaddressable_sensor(
        self,
        affine: AffineTransform,
        camera_shape: tuple[int, int],
        pixel_size_out: tuple[float, float],
        resolution_out: tuple[int, int],
        addressable: tuple[float, float],
    ) -> None:
        """Warn about sensor regions that the SLM cannot address due to its limited
        diffraction angle. The sensor is sampled on a grid, mapped to focal-plane
        metres through ``affine`` and compared with the first-order Nyquist deflection,
        ``addressable``.
        """
        rows, columns = np.meshgrid(
            np.linspace(0, camera_shape[0] - 1, 16),
            np.linspace(0, camera_shape[1] - 1, 16),
            indexing="ij",
        )
        pixels = np.column_stack([columns.ravel(), rows.ravel()])
        simulated = affine.transform_points(pixels)
        metres_x, metres_y = pixel_to_metres(
            (simulated[:, 0], simulated[:, 1]), pixel_size_out, resolution_out
        )
        outside = (np.abs(metres_x) > addressable[0]) | (
            np.abs(metres_y) > addressable[1]
        )
        if outside.any():
            warnings.warn(
                f"{100.0 * outside.mean():.0f}% of the camera sensor lies "
                "outside the region the SLM can address (first-order Nyquist "
                f"deflection of +/-({addressable[0] * 1e3:.2f}, "
                f"{addressable[1] * 1e3:.2f}) mm around the zeroth order). "
                "Focal spots cannot be placed there.",
                stacklevel=3,
            )

    @staticmethod
    def _linear_rotation_degrees(linear: NDArray) -> float:
        """Rotation [deg] of a 2x2 pixel-frame linear map (reflection factored out),
        in the physical frame, matching CameraMapping.rotation_degrees.
        """
        return pixel_frame_angle(
            AffineTransform.from_matrix(
                np.column_stack([linear, [0.0, 0.0]])
            ).rotation_degrees
        )

    def _suggest_camera_orientation(
        self, transform: NDArray, camera_shape: tuple[int, int]
    ) -> OrientationSuggestion:
        """The mounting that would align the camera with the model plane, up to a
        residual affine transform.

        Enumerates the 8 dihedral orientations. For each, ``D`` is the orientation's
        pixel-space linear map and the residual is ``L' = L @ inv(D)``, ``t' = t - L' @
        d``. Picks the non-mirrored residual (``det > 0``) with the smallest residual
        rotation.

        Raises:
            ValueError: The camera applies a frame transform outside the eight, leaving
                nothing to compose onto.
        """
        current = self.camera.orientation
        if current is None:
            raise ValueError(
                f"{type(self.camera).__name__} applies a frame transform that is not "
                "one of the eight orientations, so there is no mounting to suggest. "
                "Map it with find_camera_orientation=False."
            )

        matrix = np.asarray(transform, dtype=np.float64)
        linear, offset = matrix[:, :2], matrix[:, 2]

        best = None
        for correction in CameraOrientation.dihedral():
            pixel = correction.matrix(camera_shape)
            residual_linear = linear @ np.linalg.inv(pixel[:, :2])
            if np.linalg.det(residual_linear) <= 0:
                continue  # this orientation leaves a mirror in the residual
            residual_offset = offset - residual_linear @ pixel[:, 2]
            angle = abs(self._linear_rotation_degrees(residual_linear))
            if best is None or angle < best[0]:
                best = (
                    angle,
                    OrientationSuggestion(
                        correction.compose(current),
                        np.column_stack([residual_linear, residual_offset]),
                    ),
                )

        return best[1]

    def _build_visualization_data(
        self,
        half_extent: tuple[float, float],
        search_step: float,
        addressable: tuple[float, float],
        pixel_size_out: tuple[float, float],
        resolution_out: tuple[int, int],
        camera_shape: tuple[int, int],
        transform: NDArray,
        probe_image: NDArray,
        simulated_image: NDArray,
        detected_points: NDArray,
        affine_probe_positions: NDArray,
    ) -> CoarseVisualizationData:
        """Bundle the per-stage captures and output-plane geometry recorded during
        map_camera into a self-contained CoarseVisualizationData for
        CoarseMapperVisualizer. Output-plane pixels are (column, row); pixel_size_out /
        resolution_out are (y, x) / (height, width).
        """
        # Spiral candidate tilts (metres) to output-plane pixels
        tilts = np.asarray(
            self._spiral_tilts(half_extent[0], half_extent[1], search_step),
            dtype=np.float64,
        )
        array_spot_positions = np.column_stack(
            metres_to_pixel((tilts[:, 0], tilts[:, 1]), pixel_size_out, resolution_out)
        )
        nyquist_half_extent_px = (
            addressable[0] / pixel_size_out[1],
            addressable[1] / pixel_size_out[0],
        )
        # Camera-sensor corners (column, row) to output-plane pixels via the transform.
        height, width = camera_shape
        corners = np.array(
            [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64
        )
        sensor_polygon = AffineTransform.from_matrix(transform).transform_points(
            corners
        )
        walk_image = (
            np.maximum.reduce(self._walk_frames) if self._walk_frames else None
        )
        return CoarseVisualizationData(
            camera_image=probe_image,
            simulated_image=simulated_image,
            array_image=self._search_array_image,
            walk_image=walk_image,
            probe_image=probe_image,
            detected_points=detected_points,
            array_spot_positions=array_spot_positions,
            affine_probe_positions=affine_probe_positions,
            nyquist_half_extent_px=nyquist_half_extent_px,
            output_resolution=resolution_out,
            sensor_rectangle=sensor_polygon,
        )

    def _measure_probes(
        self,
        probe_tilts: list[tuple[float, float]],
        exposure_time: float | None,
        camera_pixel_size: NDArray,
        camera_shape: tuple[int, int],
        field_of_view: tuple[float, float],
        model_window_offset: tuple[float, float] = (0.0, 0.0),
        mask: NDArray[np.bool_] | None = None,
    ) -> _ProbeMeasurements:
        """Measure every probe on the camera and in the model. Raises RuntimeError when
        a probe fit fails or lands implausibly.

        Args:
            probe_tilts: The ``(x, y)`` probe tilts in focal-plane metres, displayed
                one at a time on the SLM and rendered in the model.
            exposure_time: Camera exposure in seconds held for every probe. If None,
                the camera autoexposes on each probe.
            camera_pixel_size: Camera pixel size ``(y, x)`` in metres, converting each
                fitted spot position into camera pixels.
            camera_shape: Camera resolution ``(height, width)`` in pixels, used for
                that conversion and to pad every cropped frame back to a full frame.
            field_of_view: Extent ``(width, height)`` of the camera sensor in metres.
            model_window_offset: ``(x, y)`` output pixels to move the model's render
                window by while the probes are measured, so it covers the same region
                the camera does. Removed again from the reported model positions, which
                stay in the plane's own frame, centered on the zeroth order.
            mask: True at the sensor pixels each probe is exposed and fitted on, and
                False over the zeroth order. Every pixel is used when None.
        """
        camera_points: list[tuple[float, float]] = []
        simulated_points: list[tuple[float, float]] = []
        camera_frames: list[NDArray] = []
        simulated_frames: list[NDArray] = []
        focal_spot_radius = 0.0

        # The shift of the partial affine translates the image, so subtracting the
        # offset brings the region the camera images to the middle of the window. The
        # centroid below adds the offset back. The partial affine is restored on exit.
        with self.slm_camera_model.build().bypass_partial_affine():
            partial_affine = self.slm_camera_model.focal_plane_partial_affine
            if partial_affine is None:
                model_window_offset = (0.0, 0.0)
            else:
                with torch.no_grad():
                    partial_affine.shift.sub_(
                        torch.tensor(
                            model_window_offset,
                            dtype=partial_affine.shift.dtype,
                            device=partial_affine.shift.device,
                        )
                    )
            # The centroids are (column, row) pixels, so the offset is added back in the
            # pixel frame.
            column_offset, row_offset = pixel_frame_vector(model_window_offset)

            for index, probe in enumerate(probe_tilts):
                # Camera side: display the tilt on the hardware and fit the spot.
                try:
                    (x, y), radius, cropped, roi = get_diffraction_spot_position(
                        self.slm,
                        self.camera,
                        linear_phase_tilt=probe,
                        slm_camera_model=self.slm_camera_model,
                        exposure_time=exposure_time,
                        units="metres",
                        verbose=False,
                        mask=mask,
                    )
                except (RuntimeError, ValueError) as error:
                    raise RuntimeError(
                        f"Probe {probe} could not be fitted: {error}"
                    ) from error

                camera_points.append(
                    metres_to_pixel((x, y), camera_pixel_size, camera_shape)
                )
                camera_frames.append(roi.pad(cropped, camera_shape))
                if index == 0:
                    focal_spot_radius = float(abs(radius))

                # Model side: render the same tilt and locate the spot.
                self.slm_camera_model.virtual_slm.set_phase(
                    self.slm_camera_model.steering_phase(probe)
                )
                simulated = gpu_to_numpy(as_image(self.slm_camera_model().intensity))
                centroid = self._peak_centroid(simulated)
                simulated_points.append(
                    (centroid[0] + column_offset, centroid[1] + row_offset)
                )
                simulated_frames.append(simulated)

        # The probe pattern must not have collapsed (e.g. every "fit" locked onto the
        # same bright artefact).
        points = np.asarray(camera_points)
        distances = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
        distances[np.diag_indices(len(points))] = np.inf
        if distances.min() < 2.0:
            raise RuntimeError("Probe spots collapsed onto each other.")

        # Affine consistency: the fourth probe tilt is a linear combination of the
        # others (t3 = t0 - t1 + t2), so its camera position must be too. A probe that
        # locked onto a conjugate ghost or an edge artefact breaks this parallelogram.
        expected = points[0] - points[1] + points[2]
        deviation = float(np.linalg.norm(points[3] - expected))
        span = float(np.linalg.norm(points[1] - points[0]))
        if deviation > max(5.0, 0.1 * span):
            raise RuntimeError(
                "The probe pattern is not affine-consistent (deviation "
                f"{deviation:.1f} px); a probe may have locked onto a ghost "
                "order."
            )

        return _ProbeMeasurements(
            camera_points,
            simulated_points,
            camera_frames,
            simulated_frames,
            focal_spot_radius,
        )

    def _calibrate_exposure(
        self,
        half_extent: tuple[float, float],
        search_step: float,
        spot_radius: float,
    ) -> float | None:
        """Calibrate one fixed per-probe exposure before the sequential search, for a
        zeroth order off the sensor.

        A spot array covering the entire addressable area is displayed, and the camera
        autoexposes on it. A phase-only superposition of N spots makes each spot ~1/N
        as bright as a single-spot probe, so the per-probe equivalent exposure is the
        array's divided by N
        (:meth:`~hologradpy.hardware.camera.Camera.get_equivalent_exposure`). It is set
        through :meth:`~hologradpy.hardware.camera.Camera.set_equivalent_exposure`, so a
        gain raised for the dim array returns to 0 dB when the per-probe exposure lies
        below the gain onset.

        Returns the per-probe exposure in seconds, at the gain left set here, or None
        to fall back to the adaptive per-probe ladder.
        """
        # Display the full probe array and autoexpose once.
        tilts = [
            tilt
            for tilt in self._spiral_tilts(
                half_extent[0], half_extent[1], search_step
            )
            if np.hypot(tilt[0], tilt[1]) > 1e-9  # drop the undiffracted DC
        ]
        targets = torch.tensor(
            tilts, device=self.slm_camera_model.device, dtype=torch.float64
        )
        generator = torch.Generator(device=targets.device).manual_seed(0)
        target_phases = (
            torch.rand(targets.shape[0], generator=generator, device=targets.device)
            * 2.0
            * np.pi
        )
        phase = LinearSuperpositionPhaseRetriever(
            self.slm_camera_model, targets, target_phases=target_phases
        ).retrieve_phase()
        self.slm.set_phase(gpu_to_numpy(phase))
        try:
            self.camera.autoexpose(set_fraction=0.5, raise_on_rail=False, verbose=False)
        except RuntimeError:
            return None  # An autoexposure that raises has found no signal.

        # Confirm spot is present.
        array_image = np.asarray(self.camera.get_image())
        self._search_array_image = array_image
        if not has_prominent_peak(array_image, self.camera):
            return None

        probe_exposure = self.camera.get_equivalent_exposure() / targets.shape[0]
        self.camera.set_equivalent_exposure(probe_exposure)
        exposure = float(self.camera.get_exposure())
        # The exposure is held at the hardware minimum, and the frames come out
        # brighter than the probe exposure asks for.
        bounds = self.camera.exposure_bounds
        if (
            bounds is not None
            and exposure <= bounds[0]
            and self.camera.get_equivalent_exposure() > probe_exposure
        ):
            warnings.warn(
                f"The calibrated per-probe exposure ({probe_exposure * 1e6:.2f} us at "
                "0 dB) is below the camera's minimum exposure "
                f"({bounds[0] * 1e6:.2f} us), so the probe spots are overexposed. "
                "Attenuate the beam with less power or a denser ND filter to bring the "
                "exposure into range.",
                stacklevel=2,
            )
        return exposure

    def _locate_zeroth_order(self, spot_radius: float) -> tuple[float, float] | None:
        """The ``(column, row)`` of the zeroth order, or None when it misses the sensor.

        With the zeroth order off the sensor, the camera can still find a spot at zero
        tilt in fixed background, such as stray light or speckle. The frame is
        autoexposed, and the spot is located at its brightest pixel, so an overexposed
        frame cannot misplace it. A 2-pixel-period 0/pi binary grating has no DC term
        (``exp(1j*0) + exp(1j*pi) = 0``), so it strongly suppresses the zeroth order and
        leaves fixed background untouched. The spot is the zeroth order when its peak
        within the zeroth-order mask falls below half under the grating. A spot within
        a detection window of the sensor edge counts as off the sensor, since the centre
        search cannot start from it. The camera's exposure, gain and region of interest
        are put back afterwards.

        Args:
            spot_radius: The focal-spot radius in metres, which sizes the mask the two
                peaks are compared in.
        """
        if self._spot_on_sensor((0.0, 0.0), None, spot_radius) is None:
            return None
        mask_radius = zeroth_order_mask_radius(spot_radius, self.camera.pixel_size)
        half = int(np.ceil(mask_radius))

        with self.camera.preserve_exposure_gain_and_roi():
            self.camera.autoexpose(set_fraction=0.5, raise_on_rail=False, verbose=False)
            autoexposed = np.asarray(self.camera.get_image(), dtype=np.float64)
            column, row = brightest_pixel(autoexposed)
            if not self._clear_of_the_edge(
                autoexposed.shape, (column, row), spot_radius
            ):
                return None
            window = ROI.centered(
                (column, row), (2 * half + 1, 2 * half + 1)
            ).trimmed_to(autoexposed.shape)

            def window_peak(frame: NDArray) -> float:
                return float(window.crop(frame).max())

            peak = window_peak(autoexposed)
            # 2-px-period 0/pi vertical binary grating: diffracts the light into the
            # Nyquist-edge +/-1 orders, minimizing the zeroth order.
            self.slm.set_phase(binary_phase_grating(self.slm.resolution))
            suppressed = np.asarray(self.camera.get_image(), dtype=np.float64)
        if window_peak(suppressed) < 0.5 * peak:
            return (float(column), float(row))
        return None

    def _clear_of_the_edge(
        self, shape: tuple[int, ...], point: tuple[float, float], spot_radius: float
    ) -> bool:
        """Whether a spot peaking at the pixel ``point``, ``(column, row)``, lies at
        least ``DETECTION_HALF_WIDTH_RADII`` focal-spot radii inside a frame of
        ``shape``.

        A spot nearer the edge can be the clipped tail of one off the sensor. The
        centre search cannot measure how such a spot moves. The search step leaves
        room for a spot clear of the edge.
        """
        column, row = point
        pixel_pitch = float(np.min(self.camera.pixel_size))
        margin = DETECTION_HALF_WIDTH_RADII * spot_radius / pixel_pitch
        height, width = shape[:2]
        return (
            margin <= row <= height - 1 - margin
            and margin <= column <= width - 1 - margin
        )

    def _default_search_step(
        self, spot_radius: float, field_of_view: tuple[float, float]
    ) -> float:
        """The rotation-safe grid spacing for the aspect ratio of the sensor, minus the
        detection window.

        Raises:
            ValueError: The smaller extent of the sensor lies below the detection
                window, so no step places a probe spot clear of the edge.
        """
        step_over = min(
            min(field_of_view), _PROBE_SPACING_FRACTION * max(field_of_view),
        )
        detection_window = 2.0 * DETECTION_HALF_WIDTH_RADII * spot_radius
        search_step = step_over - detection_window
        if search_step <= 0.0:
            raise ValueError(
                "The camera sensor's smaller extent "
                f"({min(field_of_view) * 1e3:.2f} mm) is below the focal-spot "
                f"detection window ({detection_window * 1e3:.2f} mm), so probe "
                "spots cannot be reliably placed. Use a larger sensor, a "
                "smaller focal spot, or pass search_step explicitly."
            )
        return search_step

    def _measure_spot_radius(
        self,
        tilt: tuple[float, float],
        spot_radius_guess: float,
    ) -> float:
        """Measure the focal-spot 1/e^2 radius by fitting a Gaussian to the found spot,
        replacing the initial estimate. Falls back to the guess if the fit fails.

        The camera is autoexposed on the spot and left at that exposure.
        :meth:`_center_search` holds this exposure for its captures.
        """
        self._display_tilt(tilt)
        self.camera.autoexpose(
            set_fraction=0.5, raise_on_rail=False, verbose=False
        )
        image = np.asarray(self.camera.get_image())
        row, column = np.unravel_index(int(np.argmax(image)), image.shape)
        spot_radius_px = spot_radius_guess / (min(self.camera.pixel_size))
        half_window = max(int(round(DETECTION_HALF_WIDTH_RADII * spot_radius_px)), 1)
        roi = ROI.centered(
            (column, row), (2 * half_window + 1, 2 * half_window + 1)
        ).trimmed_to(image.shape)
        gaussian = Gaussian(self.camera.resolution, self.camera.pixel_size, roi)
        try:
            spot = gaussian.fit(
                roi.crop(image), beam_radius_guess=spot_radius_guess
            ).beam
        except (RuntimeError, ValueError):
            return spot_radius_guess
        return spot.mean_radius

    def _search_spot(
        self,
        half_extent: tuple[float, float],
        search_step: float,
        probe_shift: float,
        exposure_time: float | None,
        spot_radius: float,
    ) -> tuple[float, float]:
        """Find a tilt whose probe spot lands on the sensor, walking the rectangular
        spiral outward from the zeroth order.

        The search runs once the zeroth order is known to miss the sensor, so zero tilt
        is left out of the spiral.
        """
        for tilt in self._spiral_tilts(
            half_extent[0], half_extent[1], search_step
        ):
            if np.hypot(tilt[0], tilt[1]) <= 1e-9:
                continue
            image = self._spot_on_sensor(
                tilt, exposure_time, spot_radius
            )
            if image is None:
                continue
            if not self._clear_of_the_edge(
                image.shape, brightest_pixel(image), spot_radius
            ):
                continue
            if self._is_static_background(
                tilt, image, probe_shift, spot_radius,
            ):
                continue
            return self._prefer_main_order(tilt)
        raise RuntimeError(
            "Could not find any probe spot on the sensor within the search "
            f"extent (+/-{half_extent[0] * 1e3:.1f}, {half_extent[1] * 1e3:.1f}) "
            "mm. Check the camera images the focal plane."
        )
    
    def _prefer_main_order(self, tilt: tuple[float, float]) -> tuple[float, float]:
        """The found spot can be the much dimmer conjugate ghost of the blazed grating.
        Its main order then sits at the mirrored tilt.

        The camera is autoexposed on the frame of each tilt, and a brighter spot needs a
        shorter equivalent exposure to reach the same peak
        (:meth:`~hologradpy.hardware.camera.Camera.get_equivalent_exposure`). The
        mirrored tilt is preferred when its equivalent exposure is shorter than the
        found spot's by more than ``_MAIN_ORDER_BRIGHTNESS_RATIO``. If both orders land
        on the sensor with similar brightness, either choice is a genuine spot. The
        found tilt is kept when both spots are overexposed even at the shortest
        exposure. The camera's exposure, gain and region of interest are put back
        afterwards.
        """
        mirrored = (-tilt[0], -tilt[1])
        with self.camera.preserve_exposure_gain_and_roi(full_sensor=True):
            self._display_tilt(tilt)
            self.camera.autoexpose(raise_on_rail=False)
            found_exposure = self.camera.get_equivalent_exposure()
            self._display_tilt(mirrored)
            self.camera.autoexpose(raise_on_rail=False)
            mirrored_exposure = self.camera.get_equivalent_exposure()
        if found_exposure > _MAIN_ORDER_BRIGHTNESS_RATIO * mirrored_exposure:
            return mirrored
        return tilt

    def _is_static_background(
        self,
        tilt: tuple[float, float],
        image: NDArray,
        probe_shift: float,
        spot_radius: float,
    ) -> bool:
        """True if the found spot does not move when the tilt changes. A real probe spot
        moves (the tilt is stepped by ``probe_shift``, kept small so the spot stays on 
        the sensor), while static background (stray light) does not. A spot that shifts 
        by less than one focal-spot radius (``spot_radius``, metres) is deemed static 
        background.
        """
        row, column = np.unravel_index(int(np.argmax(image)), image.shape)
        self._display_tilt((tilt[0] + probe_shift, tilt[1]))
        shifted = np.asarray(self.camera.get_image())
        shifted_peak = detect_spot(shifted, spot_radius, self.camera)
        if shifted_peak is None:
            return False
        shifted_column, shifted_row = shifted_peak
        pixel_pitch = self.camera.pixel_size[::-1]  # (x, y) for Cartesian geometry
        shift_x = (shifted_column - column) * pixel_pitch[0]
        shift_y = (shifted_row - row) * pixel_pitch[1]
        return bool(np.hypot(shift_x, shift_y) < spot_radius)

    def _center_search(
        self,
        tilt: tuple[float, float],
        camera_shape: tuple[int, int],
        spot_radius: float,
        zeroth_order_position: tuple[float, float] | None = None,
    ) -> tuple[tuple[float, float], NDArray | None]:
        """Move the found spot to the sensor centre by a local linear fit.

        Returns the centre tilt and ``jacobian``, the measured 2x2 change of the spot's
        ``(column, row)`` pixel per metre of tilt. The ``jacobian`` places the affine
        probes, and is None when an offset spot is not detected or the matrix is
        singular.

        Each column of ``jacobian`` comes from a step of the tilt in the negative
        direction of its axis, or in the positive direction when no spot is found
        there. The spot found at ``tilt`` is masked in these captures, since a zeroth
        order there stays in place when the tilt moves. When the sensor centre lies
        clear of the zeroth order at ``zeroth_order_position``, ``(column, row)``, the
        zeroth order is also masked in the capture that confirms the centre tilt.
        """
        center = np.array([(camera_shape[1] - 1) / 2, (camera_shape[0] - 1) / 2])
        exposure = float(self.camera.get_exposure())
        # Deflect by twice the detection window, so the offset spot lies well clear of
        # the zeroth-order mask applied in the derivative captures. On a sensor that is
        # small against the spot, the step is held to a quarter of its smaller side, so
        # the offset spot stays on the sensor on at least one side.
        pixel_size = np.asarray(self.camera.pixel_size, dtype=float)  # (y, x) metres
        smaller_side = float(np.min(np.asarray(camera_shape) * pixel_size))
        offset = min(
            2.0 * DETECTION_HALF_WIDTH_RADII * spot_radius, 0.25 * smaller_side
        )
        mask_radius = zeroth_order_mask_radius(spot_radius, self.camera.pixel_size)

        def spot_position(
            candidate: tuple[float, float], mask_center: NDArray | None = None
        ) -> NDArray | None:
            self._display_tilt(candidate)
            self.camera.set_exposure(exposure)
            image = np.asarray(self.camera.get_image())
            self._walk_frames.append(image)
            if mask_center is not None:
                # The disc around mask_center is set to the median of the frame, so a
                # stationary and possibly brighter zeroth order cannot take the place
                # of the spot.
                image = image.copy()
                disk = disc_mask(image.shape, mask_center, mask_radius)
                image[disk] = float(np.median(image))
            peak = detect_spot(image, spot_radius, self.camera)
            if peak is None:
                return None
            return np.array(peak, dtype=float)  # (column, row)

        initial_position = spot_position(tilt)
        if initial_position is None:
            return tilt, None

        def axis_column(
            minus: tuple[float, float], plus: tuple[float, float]
        ) -> NDArray | None:
            position = spot_position(minus, mask_center=initial_position)
            if position is not None:
                return (position - initial_position) / (-offset)
            position = spot_position(plus, mask_center=initial_position)
            if position is not None:
                return (position - initial_position) / offset
            return None

        jx = axis_column(
            (tilt[0] - offset, tilt[1]), (tilt[0] + offset, tilt[1])
        )
        jy = axis_column(
            (tilt[0], tilt[1] - offset), (tilt[0], tilt[1] + offset)
        )
        if jx is None or jy is None:
            return tilt, None

        jacobian = np.column_stack([jx, jy])  # (px change) per (metre of tilt)
        try:
            delta = np.linalg.solve(jacobian, center - initial_position)
        except np.linalg.LinAlgError:
            return tilt, None
        center_tilt = (tilt[0] + float(delta[0]), tilt[1] + float(delta[1]))

        # Confirm the extrapolated tilt lands the spot near the sensor center. The spot
        # there and a located zeroth order stay apart when they lie two mask radii
        # apart, and the zeroth order is then masked. Fall back to the found tilt if the
        # linear step overshot off the sensor.
        confirm_mask_center = None
        if zeroth_order_position is not None:
            zeroth = np.asarray(zeroth_order_position, dtype=float)
            if np.linalg.norm(center - zeroth) > 2.0 * mask_radius:
                confirm_mask_center = zeroth
        if spot_position(center_tilt, mask_center=confirm_mask_center) is None:
            return tilt, jacobian
        return center_tilt, jacobian

    @staticmethod
    def _spiral_tilts(
        half_extent_x: float, half_extent_y: float, search_step: float
    ) -> list[tuple[float, float]]:
        """Rectangular spiral of probe tilts over the addressable rectangle.

        A grid spaced at exactly ``search_step`` ordered as expanding square rings
        starting at ``(0, 0)``. The outermost ring sits just inside the addressable 
        half-extent. All lengths are focal-plane metres, and the returned ``(x, y)`` 
        tilts are focal-plane displacements in metres.
        """

        def axis(half_extent: float) -> NDArray:
            # Points at multiples of search_step, stopping just inside the addressable
            # half-extent.
            n = int(np.floor(half_extent / search_step))
            if n == 0:
                return np.array([0.0])
            return np.arange(-n, n + 1) * search_step

        x_coordinates = axis(half_extent_x)
        y_coordinates = axis(half_extent_y)
        grid = [(float(x), float(y)) for y in y_coordinates for x in x_coordinates]

        def chebyshev_spiral_sort(point: tuple[float, float]) -> tuple[float, float]:
            """Sort by Chebyshev distance from the origin, then by angle."""
            return (
                max(np.abs(point[0]), np.abs(point[1])), 
                np.arctan2(point[1], point[0])
            )
        
        grid.sort(key=chebyshev_spiral_sort)
        return grid

    def _display_tilt(self, tilt: tuple[float, float]) -> None:
        """Display a full-frame linear-phase tilt on the hardware SLM."""
        self.slm.set_phase(gpu_to_numpy(self.slm_camera_model.steering_phase(tilt)))

    def _spot_on_sensor(
        self,
        tilt: tuple[float, float],
        exposure_time: float | None,
        spot_radius: float,
    ) -> NDArray | None:
        """Display a probe tilt and check whether a spot lands on the sensor, adapting
        the exposure. Returns the captured image when a spot is present, else None.
        """
        self._display_tilt(tilt)
        if exposure_time is not None:
            self.camera.set_exposure(exposure_time)
            image = np.asarray(self.camera.get_image())
            return image if detect_spot(image, spot_radius, self.camera) else None
        return expose_until_spot(self.camera, spot_radius)

    @staticmethod
    def _peak_centroid(
        image: NDArray, half_window: int = 3
    ) -> tuple[float, float]:
        """Sub-pixel (column, row) position of the brightest spot: intensity-weighted
        centroid of a small window around the global maximum.
        """
        row, column = np.unravel_index(int(np.argmax(image)), image.shape)
        roi = ROI.centered(
            (column, row), (2 * half_window + 1, 2 * half_window + 1)
        ).trimmed_to(image.shape)
        window = roi.crop(image)
        rows, columns = np.indices(window.shape)
        total = float(window.sum())
        return (
            roi.left_column + float((columns * window).sum()) / total,
            roi.top_row + float((rows * window).sum()) / total,
        )
