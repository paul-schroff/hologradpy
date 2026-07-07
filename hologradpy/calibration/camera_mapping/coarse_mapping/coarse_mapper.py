from __future__ import annotations
import warnings
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

import torch

from cv2 import invertAffineTransform

from slmsuite.hardware.slms.slm import SLM
from slmsuite.hardware.cameras.camera import Camera
from slmsuite.holography.analysis import get_orientation_transformation

from ....propagation.optical_systems import SLMFourierLensModel
from ....propagation.phase_profiles import linear_phase
from ....propagation.fourier import get_spatial_grid
from ....propagation.amplitude_profiles import get_focal_spot_radius
from ....holography.phase_retrieval import LinearSuperpositionPhaseRetriever
from ....analysis.fitting import fit_gaussian_beam_intensity
from ....utils import crop_to_roi, gpu_to_numpy, pad_from_roi

from ..utils import (
    _WINDOW_SPOT_RADII,
    addressable_half_extent,
    detect_spot,
    disc_mask,
    get_diffraction_spot_position,
    has_prominent_peak,
    metres_to_pixel,
)

from ....hardware.camera_data import CameraData, probe_orientation

from ..abstract import CameraMapper, CameraMapping
from .visualizer import CoarseVisualizationData


_PROBE_RECTANGLE = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))

# Rotation-safe grid spacing for a W x H sensor is min(W, H / sqrt(2)). At least
# one spot lands on the sensor at any rotation.
_PROBE_SPACING_FRACTION = 1.0 / np.sqrt(2.0)


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

    Displays one full-SLM-aperture linear-phase tilt at a time (via
    :func:`get_diffraction_spot_position`, which autoexposes and Gaussian-fits each
    spot) and matches the fitted camera positions with the model's output of the same
    tilts.

    This works for setups where the zeroth-order is not hitting the camera. If no spot
    is found at zero tilt, the focal plane is searched with probe spots along an outward
    spiral (limited by the model's field of view) until one lands on the sensor, and the
    probe pattern is placed around that tilt. The ``zeroth_order_position`` is then
    extrapolated by the affine transformation.

    The result tells you where the sensor sits with respect to the zeroth order and how
    it is oriented (see :attr:`CameraMapping.rotation_degrees`,
    :attr:`CameraMapping.is_mirrored` and :attr:`CameraMapping.scales`), and can seed
    :meth:`SpotArrayMapper.map_camera` so the fine spot array is placed entirely on the
    sensor.
    """

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
            slm_camera_model: Ideal SLM -> camera model whose output plane the camera is
                registered against (called once here to initialise its lazy modules).
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
        """Measure the coarse camera and estimate the transform from probe spots.

        Args:
            exposure_time: Camera exposure in seconds per probe. If None, exposure is 
                calibrated automatically. Defaults to None.
            search_radius: How far from the zeroth order to search for the sensor, in 
                focal-plane metres. Defaults to the full SLM Nyquist-addressable region.
            search_step: Spacing of the rectangular search spiral in metres. Defaults to
                the rotation-safe grid spacing min(W, H / sqrt(2)) of the camera-sensor 
                extents (W smaller, H larger), minus the detection window (2 * 
                _WINDOW_SPOT_RADII focal-spot radii). When auto-derived, it is 
                recomputed once the first spot's radius is measured.
            beam_diameter: Beam diameter on the SLM in metres, used to estimate the
                initial focal-spot radius that sizes the spot detection window /
                thresholds. Defaults to the smaller SLM dimension.
            initial_tilt: (x, y) tilt in focal-plane metres known to land a spot on
                the sensor. When given, the spiral search is skipped and this tilt
                seeds the centre search directly; a ValueError is raised if no spot
                is detected there. search_radius is then ignored.
            find_camera_orientation: If True, suggest the nearest discrete camera
                orientation (rot/fliplr/flipud) that would align the camera with the
                model plane, recorded on the result as ``suggested_orientation`` with
                the near-identity ``residual_transform`` it would give. The camera is
                not modified (apply it yourself via ``Camera(..., **orientation)`` for
                visually-aligned frames). Defaults to False.

        Returns:
            CameraMapping named ``"coarse"`` with the affine transform and its
            reprojection residuals.
        """
        # Reset the per-stage captures recorded for CoarseMapperVisualizer.
        self._search_array_image = None
        self._walk_frames = []

        output_module = self.slm_camera_model[-1]
        pixel_size_out = output_module.pixel_size_out.tolist()[0]  # (y, x) metres
        resolution_out = tuple(output_module.resolution_out)       # (height, width)
        focal_length = float(self.slm_camera_model.fourier_lens.focal_length)
        camera_pitch = np.asarray(self.camera.pitch_um, dtype=float) * 1e-6  # (x, y)
        camera_shape = tuple(self.camera.shape)  # (height, width)

        # TODO: The diffefrent x/y, h/w conventions are pretty confusing. Try to 
        # standardize this.
        field_of_view = (
            camera_shape[1] * camera_pitch[0],
            camera_shape[0] * camera_pitch[1],
        )

        if beam_diameter is None:
            beam_diameter = min(
                self.slm.shape[i] * self.slm.pitch_um[i] * 1e-6 for i in range(2)
            )
        spot_radius = get_focal_spot_radius(
            beam_radius=0.5 * beam_diameter,
            wavelength=self.slm.wav_um * 1e-6,
            focal_length=focal_length,
        )

        # The rectangular-spiral spacing (derived from the focal-spot size if None).
        # Recomputed later once the first spot's radius is measured.
        search_step_auto = search_step is None
        if search_step_auto:
            search_step = self._default_search_step(spot_radius, field_of_view)
            if search_step <= 0.0:
                detection_window = 2.0 * _WINDOW_SPOT_RADII * spot_radius
                raise ValueError(
                    "The camera sensor's smaller extent "
                    f"({min(field_of_view) * 1e3:.2f} mm) is below the focal-spot "
                    f"detection window ({detection_window * 1e3:.2f} mm); probe "
                    "spots cannot be reliably placed. Use a larger sensor, a "
                    "smaller focal spot, or pass search_step explicitly."
                )

        probe_shift = max(
            0.1 * min(field_of_view), 2.0 * _WINDOW_SPOT_RADII * spot_radius
        )

        addressable = addressable_half_extent(self.slm, focal_length)
        if search_radius is None:
            # Cover the full SLM Nyquist-addressable rectangle.
            half_extent = addressable
        else:
            # Make sure the search spiral does not exceed the SLM's Nyquist-addressable 
            # area.
            half_extent = (
                min(search_radius, addressable[0]), min(search_radius, addressable[1]),
            )

        # In auto-exposure mode, calibrate a fixed exposure upfront. If the zeroth order
        # is off the sensor, a spot array covering the entire adressable area is
        # generated and auto-exposed.
        if exposure_time is None:
            exposure_time = self._calibrate_exposure(
                focal_length, half_extent, search_step, spot_radius
            )

        if initial_tilt is None:
            # Find a tilt resulting in a spot landing on the sensor. Zero tilt is tried
            # first (zeroth order). Then try tilts in an outward spiral.
            center_tilt = self._search_spot(
                focal_length=focal_length,
                half_extent=half_extent,
                search_step=search_step,
                probe_shift=probe_shift,
                exposure_time=exposure_time,
                spot_radius=spot_radius,
            )
        else:
            # Caller-supplied tilt known to land a spot: skip the spiral and use it
            # directly, confirming a spot is actually present.
            if self._spot_on_sensor(
                initial_tilt, focal_length, exposure_time, spot_radius
            ) is None:
                raise ValueError(
                    f"No spot was found on the sensor at initial_tilt "
                    f"{initial_tilt} (focal-plane metres); check the tilt and "
                    "exposure."
                )
            center_tilt = initial_tilt

        # Measuring the focal spot radius from a Gaussian fit. The centre search uses 
        # this to scale its probe offset and detection window.
        spot_radius = self._measure_spot_radius(
            center_tilt, focal_length, spot_radius
        )

        # Finding the centre of the camera sensor and the local tilt that places the 
        # probe spots.
        center_tilt, jacobian = self._center_search(
            tilt=center_tilt,
            focal_length=focal_length,
            camera_shape=camera_shape,
            spot_radius=spot_radius,
        )
        if jacobian is None:
            # Fall back to a nominal ~1:1, un-rotated tilt to pixel map in the rare case
            # the Jacobian could not be computed. The affine fit still recovers the
            # transform from wherever the probes land.
            jacobian = np.diag([1.0 / camera_pitch[0], 1.0 / camera_pitch[1]])
        inverse_jacobian = np.linalg.inv(jacobian)

        # The four affine probes form a rectangle centred in the camera frame spanning
        # half the sensor width/height.
        half_extent_px = np.array(
            [camera_shape[1] / 4.0, camera_shape[0] / 4.0]  # (x, y)
        )
        corner_offsets = half_extent_px * np.asarray(_PROBE_RECTANGLE)
        probe_tilts = [
            (center_tilt[0] + float(dt[0]), center_tilt[1] + float(dt[1]))
            for dt in corner_offsets @ inverse_jacobian.T
        ]
        probes = self._measure_probes(
            probe_tilts=probe_tilts,
            exposure_time=exposure_time,
            focal_length=focal_length,
            pitch=camera_pitch,
            camera_shape=camera_shape,
            field_of_view=field_of_view,
        )

        detected = np.asarray(probes.camera_points, dtype=np.float64)
        calculated = np.asarray(probes.simulated_points, dtype=np.float64)
        design = np.hstack([detected, np.ones((len(detected), 1))])

        # TODO: Potentially move all affine-transform related logic into its own module
        affine, *_ = np.linalg.lstsq(design, calculated, rcond=None)
        transform = affine.T
        inverse_transform = invertAffineTransform(transform)
        reprojection_errors, reprojection_rms = self.calculate_reprojection_error(
            detected, calculated, transform
        )

        center = (resolution_out[0] // 2, resolution_out[1] // 2)
        zeroth_order_position = (
            inverse_transform[1, 0] * center[0]
            + inverse_transform[1, 1] * center[1]
            + inverse_transform[1, 2],
            inverse_transform[0, 0] * center[0]
            + inverse_transform[0, 1] * center[1]
            + inverse_transform[0, 2],
        )

        # Warn about sensor regions the SLM cannot address (limited diffraction angle):
        # sample the sensor on a grid, map to focal-plane metres and compare with the
        # first-order Nyquist deflection.
        rows, columns = np.meshgrid(
            np.linspace(0, camera_shape[0] - 1, 16),
            np.linspace(0, camera_shape[1] - 1, 16),
            indexing="ij",
        )
        pixels = np.column_stack([columns.ravel(), rows.ravel()])
        simulated = pixels @ transform[:, :2].T + transform[:, 2]
        metres_x = (simulated[:, 0] - resolution_out[1] / 2) * pixel_size_out[1]
        metres_y = (simulated[:, 1] - resolution_out[0] / 2) * pixel_size_out[0]
        outside = (np.abs(metres_x) > addressable[0]) | (
            np.abs(metres_y) > addressable[1]
        )
        if outside.any():
            warnings.warn(
                f"{100.0 * outside.mean():.0f}% of the camera sensor lies "
                "outside the region the SLM can address (first-order Nyquist "
                f"deflection of +/-({addressable[0] * 1e3:.2f}, "
                f"{addressable[1] * 1e3:.2f}) mm around the zeroth order); "
                "focal spots cannot be placed there.",
                stacklevel=2,
            )

        # Reduce all four probes to one frame (reused for both the mapping image and the
        # visualization data).
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
            np.asarray(probes.camera_points, dtype=np.float64),
            np.asarray(probes.simulated_points, dtype=np.float64),
        )

        suggested_orientation = None
        residual_transform = None
        if find_camera_orientation:
            suggested_orientation, residual_transform = (
                self._suggest_camera_orientation(transform, camera_shape)
            )

        return CameraMapping(
            timestamp=datetime.now(),
            name="coarse",
            transform=transform,
            inverse_transform=inverse_transform,
            detected_points=probes.camera_points,
            calculated_points=probes.simulated_points,
            camera_images=[probe_composite],
            simulated_images=[np.maximum.reduce(probes.simulated_frames)],
            zeroth_order_position=zeroth_order_position,
            focal_spot_radius=probes.focal_spot_radius,
            reprojection_errors=reprojection_errors,
            reprojection_rms=reprojection_rms,
            visualization_data=visualization_data,
            camera_data=CameraData.from_camera(self.camera),
            suggested_orientation=suggested_orientation,
            residual_transform=residual_transform,
        )

    @staticmethod
    def _linear_rotation_degrees(linear: NDArray) -> float:
        """Rotation [deg] of a 2x2 linear map (reflection factored out), matching
        CameraMapping.rotation_degrees."""
        u, _, vt = np.linalg.svd(linear)
        if np.linalg.det(u @ vt) < 0:
            u[:, -1] *= -1
        rotation = u @ vt
        return float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))

    def _suggest_camera_orientation(
        self, transform: NDArray, camera_shape: tuple[int, int]
    ) -> tuple[dict, NDArray]:
        """Nearest discrete camera orientation (rot/fliplr/flipud) that would align the 
        camera with the model plane, plus the residual affine it would give.

        Enumerates the 8 dihedral orientations; for each, ``D`` is the orientation's
        pixel-space linear map (from :func:`probe_orientation`) and the residual is ``L'
        = L @ inv(D)``, ``t' = t - L' @ d``. Picks the non-mirrored residual (``det >
        0``) with the smallest residual rotation. Returns ``({"rot", "fliplr",
        "flipud"}, [L' | t'])``; the orientation is relative to the camera's current 
        one."""
        matrix = np.asarray(transform, dtype=np.float64)
        linear, offset = matrix[:, :2], matrix[:, 2]

        best = None
        for rot in ("0", "90", "180", "270"):
            for fliplr in (False, True):
                pixel = probe_orientation(
                    get_orientation_transformation(rot, fliplr, False), camera_shape
                )
                residual_linear = linear @ np.linalg.inv(pixel[:, :2])
                if np.linalg.det(residual_linear) <= 0:
                    continue  # this orientation leaves a mirror in the residual
                residual_offset = offset - residual_linear @ pixel[:, 2]
                angle = abs(self._linear_rotation_degrees(residual_linear))
                if best is None or angle < best[0]:
                    best = (
                        angle,
                        {"rot": rot, "fliplr": fliplr, "flipud": False},
                        np.column_stack([residual_linear, residual_offset]),
                    )

        _, orientation, residual = best
        return orientation, residual

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
        detected_points: NDArray,
        affine_probe_positions: NDArray,
    ) -> CoarseVisualizationData:
        """Bundle the per-stage captures and output-plane geometry recorded during 
        map_camera into a self-contained CoarseVisualizationData for
        CoarseMapperVisualizer. Output-plane pixels are (x, y); pixel_size_out /
        resolution_out are (y, x) / (height, width)."""
        center = np.array([resolution_out[1] / 2.0, resolution_out[0] / 2.0])
        # Spiral candidate tilts (metres) to output-plane pixels
        tilts = np.asarray(
            self._spiral_tilts(half_extent[0], half_extent[1], search_step),
            dtype=np.float64,
        )
        pixel_scale = np.array([pixel_size_out[1], pixel_size_out[0]])  # (x, y)
        array_spot_positions = center + tilts / pixel_scale
        nyquist_half_extent_px = (
            addressable[0] / pixel_size_out[1],
            addressable[1] / pixel_size_out[0],
        )
        # Camera-sensor corners (x, y) to output-plane pixels via the transform.
        height, width = camera_shape
        corners = np.array(
            [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64
        )
        sensor_polygon = corners @ transform[:, :2].T + transform[:, 2]
        walk_image = (
            np.maximum.reduce(self._walk_frames) if self._walk_frames else None
        )
        return CoarseVisualizationData(
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
        focal_length: float,
        pitch: NDArray,
        camera_shape: tuple[int, int],
        field_of_view: tuple[float, float],
    ) -> _ProbeMeasurements:
        """Measure every probe on the camera and in the model. Raises RuntimeError when
        a probe fit fails or lands implausibly."""
        geometry = self.slm_camera_model.input_geometry
        grid = geometry.get_spatial_grid()
        wavenumber = geometry.wavenumber.reshape(())

        camera_points: list[tuple[float, float]] = []
        simulated_points: list[tuple[float, float]] = []
        camera_frames: list[NDArray] = []
        simulated_frames: list[NDArray] = []
        focal_spot_radius = 0.0

        for index, probe in enumerate(probe_tilts):
            # Camera side: display the tilt on the hardware and fit the spot.
            try:
                (x, y), radius, cropped, roi = get_diffraction_spot_position(
                    self.slm,
                    self.camera,
                    linear_phase_tilt=probe,
                    focal_length=focal_length,
                    exposure_time=exposure_time,
                    units="metres",
                    verbose=False,
                )
            except (RuntimeError, ValueError) as error:
                raise RuntimeError(
                    f"Probe {probe} could not be fitted: {error}"
                ) from error

            camera_points.append(metres_to_pixel((x, y), pitch, camera_shape))
            camera_frames.append(pad_from_roi(cropped, roi, camera_shape))
            if index == 0:
                focal_spot_radius = float(abs(radius))

            # Model side: render the same tilt and locate the spot.
            phase = linear_phase(
                *grid,
                probe[0],
                probe[1],
                wavenumber=wavenumber,
                focal_length=focal_length,
            )
            self.slm_camera_model.virtual_slm.set_phase(phase)
            simulated = gpu_to_numpy(self.slm_camera_model().intensity)
            simulated_points.append(self._peak_centroid(simulated))
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
        focal_length: float,
        half_extent: tuple[float, float],
        search_step: float,
        spot_radius: float,
    ) -> float | None:
        """Calibrate one fixed per-probe exposure before the sequential search.

        If the zeroth-order spot cannot be located on the sensor, display a spot array 
        covering the entire adressable area and autoexpose the camera. A phase-only 
        superposition of N spots makes each spot ~1/N as bright as a single-spot probe, 
        so the per-probe exposure is calculated as ``t_array / N``.

        Returns the per-probe exposure in seconds, or None to fall back to the
        adaptive per-probe ladder.
        """
        # Zeroth order on the sensor: trivial case, no array needed. A bright
        # stray-light background (e.g. speckle from a different laser) can make the 
        # adaptive-exposure probe find a spurious spot at zero tilt, so confirm it is 
        # really the zeroth order (dims under a 0/pi grating) before skipping the array.
        zod_on_sensor = self._spot_on_sensor(
            (0.0, 0.0), focal_length, None, spot_radius
        )

        if zod_on_sensor is not None and self._confirm_zeroth_order(
            zod_on_sensor, spot_radius
        ):
            return None

        # Zeroth order off the sensor: display the full probe array and autoexpose once.
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
            array_exposure = self.camera.autoexposure(
                set_fraction=0.5, exposure_bounds_s=(0, 1), verbose=False
            )
        except RuntimeError:
            return None  # Autoexposure railed: no signal.

        # Confirm spot is present.
        array_image = np.asarray(self.camera.get_image())
        self._search_array_image = array_image
        if not has_prominent_peak(array_image, self.camera):
            return None

        exposure = float(array_exposure) / targets.shape[0]
        bounds = self.camera.exposure_bounds_s
        hardware_minimum = bounds[0] if bounds is not None else None
        if hardware_minimum is not None and exposure < hardware_minimum:
            warnings.warn(
                f"The calibrated per-probe exposure ({exposure * 1e6:.2f} us) is "
                "below the camera's minimum hardware exposure "
                f"({hardware_minimum * 1e6:.2f} us); probe spots will be "
                "over-exposed. Attenuate the beam (lower power or a denser ND "
                "filter) to bring the exposure into range.",
                stacklevel=2,
            )
            exposure = hardware_minimum
        return exposure

    def _confirm_zeroth_order(self, image: NDArray, spot_radius: float) -> bool:
        """Whether the bright spot in ``image`` (found at zero tilt) is the true zeroth 
        order and not fixed stray light / speckle.

        A 2-pixel-period 0/pi binary grating has no DC term (``exp(1j*0) + exp(1j*pi) =
        0``), so it strongly suppresses the real zeroth order while leaving fixed
        background (unchanged by the SLM) untouched. The spot is the zeroth order if its
        intensity at the same position drops when the grating is displayed."""
        row, column = np.unravel_index(int(np.argmax(image)), image.shape)
        spot_radius_px = spot_radius / (min(self.camera.pitch_um) * 1e-6)
        half = max(int(round(2.0 * spot_radius_px)), 2)

        def window_peak(frame: NDArray) -> float:
            top, left = max(row - half, 0), max(column - half, 0)
            return float(frame[top:row + half + 1, left:column + half + 1].max())

        peak_before = window_peak(image)
        # 2-px-period 0/pi vertical binary grating: diffracts the light into the
        # Nyquist-edge +/-1 orders, leaving no zeroth order.
        grating = np.zeros(self.slm.shape)
        grating[:, 1::2] = np.pi
        self.slm.set_phase(grating)
        suppressed = np.asarray(self.camera.get_image())
        return window_peak(suppressed) < 0.5 * peak_before

    def _default_search_step(
        self, spot_radius: float, field_of_view: tuple[float, float]
    ) -> float:
        """Aspect-aware rotation-safe grid spacing minus the detection window."""
        step_over = min(
            min(field_of_view), _PROBE_SPACING_FRACTION * max(field_of_view),
        )
        return step_over - 2.0 * _WINDOW_SPOT_RADII * spot_radius

    def _measure_spot_radius(
        self,
        tilt: tuple[float, float],
        focal_length: float,
        spot_radius_guess: float,
    ) -> float:
        """Measure the focal-spot 1/e^2 radius by fitting a Gaussian to the found spot, 
        replacing the initial estimate. Falls back to the guess if the fit fails."""
        self._display_tilt(tilt, focal_length)
        self.camera.autoexposure(
            set_fraction=0.5, exposure_bounds_s=(0, 1), verbose=False
        )
        image = np.asarray(self.camera.get_image())
        row, column = np.unravel_index(int(np.argmax(image)), image.shape)
        spot_radius_px = spot_radius_guess / (min(self.camera.pitch_um) * 1e-6)
        half_window = max(int(round(_WINDOW_SPOT_RADII * spot_radius_px)), 1)
        height, width = image.shape
        roi = (
            max(row - half_window, 0),
            min(row + half_window + 1, height),
            max(column - half_window, 0),
            min(column + half_window + 1, width),
        )
        cropped = crop_to_roi(image, roi)
        grid = get_spatial_grid(self.camera.shape, self.camera.pitch_um * 1e-6)
        cropped_grid = [crop_to_roi(axis, roi) for axis in grid]
        try:
            popt, _ = fit_gaussian_beam_intensity(
                *cropped_grid, cropped, beam_radius_guess=spot_radius_guess
            )
        except (RuntimeError, ValueError):
            return spot_radius_guess
        return float(abs(popt[0]))

    def _search_spot(
        self,
        focal_length: float,
        half_extent: tuple[float, float],
        search_step: float,
        probe_shift: float,
        exposure_time: float | None,
        spot_radius: float,
    ) -> tuple[float, float]:
        """Find a tilt whose probe spot lands on the sensor, walking the rectangular 
        spiral outward from the zeroth order."""
        for tilt in self._spiral_tilts(
            half_extent[0], half_extent[1], search_step
        ):
            image = self._spot_on_sensor(
                tilt, focal_length, exposure_time, spot_radius
            )
            if image is None:
                continue
            if self._is_static_background(
                tilt, image, focal_length, probe_shift, spot_radius,
            ):
                continue
            return self._prefer_main_order(tilt, focal_length)
        raise RuntimeError(
            "Could not find any probe spot on the sensor within the search "
            f"extent (+/-{half_extent[0] * 1e3:.1f}, {half_extent[1] * 1e3:.1f}) "
            "mm. Check the camera images the focal plane."
        )
    
    # TODO: Revisit this once SLM discretization is implemented
    def _prefer_main_order(
        self, tilt: tuple[float, float], focal_length: float
    ) -> tuple[float, float]:
        """The found spot can be the much dimmer conjugate ghost of the blazed grating, 
        with its main order sitting at the mirrored tilt. Check the mirrored tilt at a 
        much shorter exposure. Only a main order stays near saturation there, and is 
        preferred when present. If both orders land on the sensor, either choice is a 
        genuine spot."""
        exposure = float(self.camera.get_exposure())
        self.camera.set_exposure(exposure / 30.0)
        self._display_tilt((-tilt[0], -tilt[1]), focal_length)
        image = np.asarray(self.camera.get_image())
        if float(image.max()) >= 0.5 * float(self.camera.bitresolution):
            return (-tilt[0], -tilt[1])
        self.camera.set_exposure(exposure)
        return tilt

    def _is_static_background(
        self,
        tilt: tuple[float, float],
        image: NDArray,
        focal_length: float,
        probe_shift: float,
        spot_radius: float,
    ) -> bool:
        """True if the found spot does not move when the tilt changes. A real probe spot
        moves (the tilt is stepped by ``probe_shift``, kept small so the spot stays on 
        the sensor), while static background (stray light) does not. A spot that shifts 
        by less than one focal-spot radius (``spot_radius``, metres) is deemed static 
        background."""
        row, column = np.unravel_index(int(np.argmax(image)), image.shape)
        self._display_tilt(
            (tilt[0] + probe_shift, tilt[1]), focal_length
        )
        shifted = np.asarray(self.camera.get_image())
        shifted_peak = detect_spot(shifted, spot_radius, self.camera)
        if shifted_peak is None:
            return False
        shifted_row, shifted_column = shifted_peak
        pixel_pitch = np.asarray(self.camera.pitch_um, dtype=float) * 1e-6  # (x, y)
        shift_x = (shifted_column - column) * pixel_pitch[0]
        shift_y = (shifted_row - row) * pixel_pitch[1]
        return bool(np.hypot(shift_x, shift_y) < spot_radius)

    # TODO: This method could use some tidying up.
    def _center_search(
        self,
        tilt: tuple[float, float],
        focal_length: float,
        camera_shape: tuple[int, int],
        spot_radius: float,
    ) -> tuple[tuple[float, float], NDArray | None]:
        """Move the found spot to the sensor centre by a local linear fit.

        Returns ``(center_tilt, jacobian)`` where ``jacobian`` is the measured 2x2 tilt
        to camera-pixel matrix (used to place the affine probes), or ``None`` when it
        could not be measured (offsets undetected / singular).

        The Jacobian comes from how the detected spot moves when the tilt is perturbed.
        The zeroth order does not move with tilt and can be brighter than the first
        order, so it is masked (a disk around the reference spot) in the derivative
        captures."""
        centre = np.array([(camera_shape[1] - 1) / 2, (camera_shape[0] - 1) / 2])
        exposure = float(self.camera.get_exposure())
        # Deflect by twice the detection window so the offset spot lands clear of
        # the ZOD mask (radius one window) applied in the derivative captures.
        offset = 2.0 * _WINDOW_SPOT_RADII * spot_radius
        mask_radius_px = _WINDOW_SPOT_RADII * spot_radius / (
            min(self.camera.pitch_um) * 1e-6
        )

        def measure(
            candidate: tuple[float, float], mask_center: NDArray | None = None
        ) -> NDArray | None:
            self._display_tilt(candidate, focal_length)
            self.camera.set_exposure(exposure)
            image = np.asarray(self.camera.get_image())
            self._walk_frames.append(image)
            if mask_center is not None:
                # Blank a disk around the ZOD so detect_spot follows the moving
                # first order rather than the stationary, possibly brighter, ZOD.
                image = image.copy()
                disk = disc_mask(image.shape, mask_center, mask_radius_px)
                image[disk] = float(np.median(image))
            peak = detect_spot(image, spot_radius, self.camera)
            if peak is None:
                return None
            row, column = peak
            return np.array([column, row], dtype=float)  # (x, y)

        initial_position = measure(tilt)
        if initial_position is None:
            return tilt, None

        def axis_column(
            minus: tuple[float, float], plus: tuple[float, float]
        ) -> NDArray | None:
            """One-sided pixel-per-metre column: try ``minus`` (left / top, tilt - 
            offset), else ``plus`` (right / bottom, tilt + offset). The ZOD is masked so
            a bright zeroth order cannot hijack the fit."""
            position = measure(minus, mask_center=initial_position)
            if position is not None:
                return (position - initial_position) / (-offset)
            position = measure(plus, mask_center=initial_position)
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
            delta = np.linalg.solve(jacobian, centre - initial_position)
        except np.linalg.LinAlgError:
            return tilt, None
        center_tilt = (tilt[0] + float(delta[0]), tilt[1] + float(delta[1]))

        # Confirm the extrapolated tilt lands the spot near the sensor centre.
        # Fall back to the found tilt if the linear step overshot off the sensor.
        if measure(center_tilt) is None:
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

    def _display_tilt(
        self, tilt: tuple[float, float], focal_length: float
    ) -> None:
        """Display a full-frame linear-phase tilt on the hardware SLM."""
        slm_grid = get_spatial_grid(self.slm.shape, self.slm.pitch_um * 1e-6)
        phase = linear_phase(
            *slm_grid,
            *tilt,
            wavenumber=2 * np.pi / (self.slm.wav_um * 1e-6),
            focal_length=focal_length,
        )
        self.slm.set_phase(gpu_to_numpy(phase))

    def _spot_on_sensor(
        self,
        tilt: tuple[float, float],
        focal_length: float,
        exposure_time: float | None,
        spot_radius: float,
    ) -> NDArray | None:
        """Display a probe tilt and check whether a spot lands on the sensor, adapting 
        the exposure. Returns the captured image when a spot is present, else None."""
        self._display_tilt(tilt, focal_length)
        if exposure_time is not None:
            self.camera.set_exposure(exposure_time)
            image = np.asarray(self.camera.get_image())
            return image if detect_spot(image, spot_radius, self.camera) else None
        return self._find_spot_adaptive_exposure(spot_radius)

    def _find_spot_adaptive_exposure(
        self,
        spot_radius: float,
        max_steps: int = 4,
        dark_threshold_fraction: float = 0.05,
        saturation_step_fraction: float = 0.05,
    ) -> NDArray | None:
        """Capture at the current tilt. If no spot detected, jump the exposure once 
        and retry, up to max_steps times. Returns the frame with a detected spot, or 
        None when the frame is well exposed but without spot."""
        full_scale = float(self.camera.bitresolution)
        bounds = self.camera.exposure_bounds_s
        max_exposure_s = float(bounds[1]) if bounds is not None else 1.0
        exposure = float(self.camera.get_exposure())
        for _ in range(max_steps):
            image = np.asarray(self.camera.get_image())
            if detect_spot(image, spot_radius, self.camera):
                return image
            peak = float(image.max())
            if peak >= full_scale - 1:  # Sensor saturated, decrease exposure
                exposure *= saturation_step_fraction
            elif (
                peak < dark_threshold_fraction * full_scale 
                and exposure < max_exposure_s
            ):
                exposure = min(exposure / saturation_step_fraction, max_exposure_s)
            else:
                return None  # well exposed but no spot found
            self.camera.set_exposure(exposure)
        return None

    @staticmethod
    def _peak_centroid(
        image: NDArray, half_window: int = 3
    ) -> tuple[float, float]:
        """Sub-pixel (x, y) position of the brightest spot: intensity-weighted
        centroid of a small window around the global maximum."""
        row, column = np.unravel_index(int(np.argmax(image)), image.shape)
        top = max(row - half_window, 0)
        left = max(column - half_window, 0)
        window = image[
            top:row + half_window + 1, left:column + half_window + 1
        ]
        rows, columns = np.indices(window.shape)
        total = float(window.sum())
        return (
            left + float((columns * window).sum()) / total,
            top + float((rows * window).sum()) / total,
        )
