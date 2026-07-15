from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
from numpy.typing import NDArray

from slmsuite.hardware.slms.slm import SLM
from slmsuite.hardware.cameras.camera import Camera

from scipy.ndimage import gaussian_filter

from . import SuperpixelSlicer
from .visualizer import RasterVisualizationData

from ..abstract import WavefrontCalibratorBase, WavefrontCalibrationData
from ..utils import inpaint
from ....analysis.unwrapping import unwrap_nonuniform

from ...camera_mapping.utils import (
    get_diffraction_spot_position,
    addressable_half_extent,
    background_noise,
)
from ...camera_mapping.coarse_mapping.coarse_mapper import CoarseMapper
from ...camera_mapping.abstract import CameraMapping
from ....hardware.utils import set_camera_woi
from ....propagation import SLMFourierLensModel
from ....propagation.optical_systems import SLMFFT
from ....propagation.virtual_slms import VirtualSLM
from ....propagation.diagonal_elements import StaticSLMField
from ....propagation.complex_amplitude import ComplexAmplitude, FieldGeometry
from ....propagation.fourier import get_spatial_grid
from ....propagation.phase_profiles import linear_phase, binary_phase_grating

from ....utils import gpu_to_numpy, roi_bounds, Timer

from ....analysis.fitting import (
    interferometric_fringes,
    fit_interferometric_fringes,
    optical_lattice_fringes,
    fit_optical_lattice_fringes,
    gaussian_beam_intensity,
    fit_gaussian_beam_intensity,
)

_LATTICE_FRAME_AVERAGES = 5
_CORNER_MIN_SNR = 5.0
_AUTOEXPOSURE_SET_FRACTION = 0.9


class RasterCalibrator(WavefrontCalibratorBase):
    """
    Class to calibrate the SLM wavefront using a raster scan.
    """

    def __init__(
        self,
        slm: SLM,
        camera: Camera,
        focal_length: float,
        device: str = "cpu",
        autoexposure_timeout_s: float = 5.0,
    ) -> None:
        super().__init__(slm, camera, device)
        self.focal_length = focal_length
        self.autoexposure_timeout_s = autoexposure_timeout_s
        self.camera_mapping: CameraMapping | None = None
        self._slm_camera_model: SLMFourierLensModel | None = None
        self.power_reference: NDArray[np.float_] | None = None

    def _build_slm_camera_model(self) -> SLMFourierLensModel:
        """A minimal ideal SLM -> Fourier-lens model for the coarse mapping.

        A uniform beam is enough: CoarseMapper localises real camera spots and only
        reads focal-plane geometry from the model. The output field of view is
        ``wavelength * focal / pitch`` (the full addressable extent) for any padding,
        so ``padded_resolution`` only sets the sampling. It is chosen so the model
        samples the focal plane near the camera pitch.
        """
        wavelength = self.slm.wav_um * 1e-6
        slm_pitch = (self.slm.pitch_um[0] * 1e-6, self.slm.pitch_um[1] * 1e-6)  # y, x
        camera_pitch = (
            self.camera.pitch_um[0] * 1e-6,
            self.camera.pitch_um[1] * 1e-6,
        )  # y, x
        padded_resolution = tuple(
            int(
                np.clip(
                    round(
                        wavelength
                        * self.focal_length
                        / (camera_pitch[axis] * slm_pitch[axis])
                    ),
                    self.slm.shape[axis],
                    2048,
                )
            )
            for axis in range(2)
        )
        geometry = FieldGeometry(
            resolution=tuple(self.slm.shape),
            pixel_size=torch.tensor(slm_pitch, device=self.device),
            wavelength=torch.tensor(wavelength, device=self.device),
        )
        beam = ComplexAmplitude(
            torch.ones(
                tuple(self.slm.shape), dtype=torch.complex64, device=self.device
            ),
            wavelength=geometry.wavelength,
            pixel_size=geometry.pixel_size,
        )
        return SLMFFT(
            input_geometry=geometry,
            virtual_slm=VirtualSLM(phase_scaling=1.0),
            static_slm_field=StaticSLMField(beam),
            focal_length=self.focal_length,
            padded_resolution=padded_resolution,
        )

    def _ensure_camera_mapping(
        self,
        camera_mapping: CameraMapping | None,
        slm_camera_model: SLMFourierLensModel | None,
    ) -> None:
        """Populate ``self.camera_mapping`` (and ``self._slm_camera_model``),
        running a :class:`CoarseMapper` when no mapping was supplied."""
        if slm_camera_model is not None:
            self._slm_camera_model = slm_camera_model
        if camera_mapping is not None:
            self.camera_mapping = camera_mapping
            return
        if self._slm_camera_model is None:
            self._slm_camera_model = self._build_slm_camera_model()
        self.camera_mapping = CoarseMapper(
            self.slm, self.camera, self._slm_camera_model
        ).map_camera()

    def _orientation_matrix(self) -> NDArray[np.float_]:
        """2x2 map from camera-plane metres to model / focal-plane metres from
        ``self.camera_mapping``.

        With the model geometry available it is the exact
        ``diag(pixel_size_out) @ L @ diag(1 / camera_pitch)`` (rotation, scale and
        mirror); otherwise the orthonormal rotation+mirror part of the transform
        (correct for a camera imaging the focal plane, magnification 1). Used to both
        place the tilts and orient the fringe/lattice fits, so they stay consistent.
        """
        linear = np.asarray(self.camera_mapping.transform, dtype=np.float64)[:, :2]
        camera_pitch = np.asarray(
            [self.camera.pitch_um[1] * 1e-6, self.camera.pitch_um[0] * 1e-6]
        )  # x, y
        if self._slm_camera_model is not None:
            output = self._slm_camera_model[-1]
            pixel_size_out = output.pixel_size_out.tolist()[0]  # y, x
            pixel_size_out = np.asarray([pixel_size_out[1], pixel_size_out[0]])  # x, y
            return np.diag(pixel_size_out) @ linear @ np.diag(1.0 / camera_pitch)
        left, _, right = np.linalg.svd(linear)
        return left @ right  # orthonormal, preserves the mirror

    def _rotation_matrix(self) -> NDArray[np.float_]:
        """Orthonormal rotation+mirror part of the camera->model transform, from the
        SVD of its linear block (no scale). Identity for an aligned camera, so it
        orients the fits without touching an aligned scan."""
        linear = np.asarray(self.camera_mapping.transform, dtype=np.float64)[:, :2]
        left, _, right = np.linalg.svd(linear)
        return left @ right

    def _orient_grid(self, grid: list[NDArray]) -> list[NDArray]:
        """Rotate a camera-plane ``(x, y)`` metre grid into the model / SLM axes via
        the camera mapping, so the fringe/lattice fits (which use SLM-axis
        separations) line up with a rotated/mirrored camera. Identity without a
        mapping."""
        if self.camera_mapping is None:
            return grid
        matrix = self._rotation_matrix()
        grid_x, grid_y = grid
        return [
            matrix[0, 0] * grid_x + matrix[0, 1] * grid_y,
            matrix[1, 0] * grid_x + matrix[1, 1] * grid_y,
        ]

    def _diagonal_direction(self) -> NDArray[np.float_]:
        """Unit 45/135/225/315 deg diagonal pointing from the zeroth order toward the
        sensor centre (away from the DC). Shared by the main and lattice placements
        so both sit at the same angle."""
        zeroth = np.asarray(
            [
                self.camera_mapping.zeroth_order_position[1],
                self.camera_mapping.zeroth_order_position[0],
            ]
        )  # x, y
        height, width = self.camera.shape
        centre = np.array([width / 2.0, height / 2.0])
        signs = np.sign(centre - zeroth)
        signs[signs == 0] = 1.0  # tie (DC on a centre line) -> positive diagonal
        return signs / np.sqrt(2.0)

    def _auto_phase_tilt(
        self,
        camera_roi_size: tuple[int, int],
        clearance: float,
        direction: NDArray[np.float_],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """A linear-phase tilt (focal-plane metres) placing the pattern along
        ``direction`` from the zeroth order, as close to the sensor centre as
        ``clearance`` (min distance to the DC) allows, on the sensor.

        Returns ``(tilt, target_camera_pixel)``.
        """
        mapping = self.camera_mapping
        zeroth = np.asarray(
            [mapping.zeroth_order_position[1], mapping.zeroth_order_position[0]]
        )  # x, y camera px
        height, width = self.camera.shape
        centre = np.array([width / 2.0, height / 2.0])
        camera_pitch = np.asarray(
            [self.camera.pitch_um[1] * 1e-6, self.camera.pitch_um[0] * 1e-6]
        )  # x, y

        # Keep the pattern ROI (plus a spot-radius margin) inside the sensor. A
        # symmetric margin avoids relying on the (height, width) vs (width, height)
        # ordering of camera_roi_size.
        focal_spot_radius = float(abs(mapping.focal_spot_radius))
        margin = max(camera_roi_size) / 2 + 3.0 * focal_spot_radius / camera_pitch.min()
        if width - 1 - margin < margin or height - 1 - margin < margin:
            raise ValueError(
                "The interference-pattern ROI does not fit on the sensor. Reduce "
                "the superpixel size or camera_roi_size."
            )

        # Farthest travel along +direction from the DC while staying inside the inset
        # box. direction has +-1/sqrt(2) components, so there is no zero-division.
        bound_x = width - 1 - margin if direction[0] > 0 else margin
        bound_y = height - 1 - margin if direction[1] > 0 else margin
        t_max = min(
            (bound_x - zeroth[0]) / direction[0],
            (bound_y - zeroth[1]) / direction[1],
        )
        if clearance > t_max:
            raise ValueError(
                f"No diagonal location clears the zeroth order by {clearance:.0f} "
                "px and still fits on the sensor. Use a larger sensor or a smaller "
                "ROI."
            )
        # As close to the sensor centre as the clearance and sensor allow.
        t_center = float(direction @ (centre - zeroth))
        t = float(np.clip(t_center, clearance, t_max))
        target = zeroth + t * direction

        # Camera pixel -> focal-plane tilt (metres) via the mapping orientation.
        tilt = self._orientation_matrix() @ ((target - zeroth) * camera_pitch)

        # Clamp into the SLM's Nyquist-addressable reach (direction preserved).
        limit = 0.9 * np.asarray(addressable_half_extent(self.slm, self.focal_length))
        scale = min(1.0, float(np.min(limit / np.maximum(np.abs(tilt), 1e-12))))
        tilt = tilt * scale
        return (float(tilt[0]), float(tilt[1])), (float(target[0]), float(target[1]))

    def _ensure_and_place_main(
        self,
        camera_roi_size: tuple[int, int],
        camera_mapping: CameraMapping | None,
        slm_camera_model: SLMFourierLensModel | None,
    ) -> tuple[tuple[float, float], tuple[float, float], NDArray[np.float_]]:
        """Ensure a coarse mapping and place the main interference pattern centrally
        along the shared diagonal, clearing the zeroth order by two interference-
        pattern ROI widths (``camera_roi_size``, not the combined lattice+pattern
        window). Returns ``(tilt, target, direction)`` so the lattice can reuse the
        direction and the main target."""
        if self.camera_mapping is None:
            self._ensure_camera_mapping(camera_mapping, slm_camera_model)
        direction = self._diagonal_direction()
        clearance = 2.0 * max(camera_roi_size)
        tilt, target = self._auto_phase_tilt(camera_roi_size, clearance, direction)
        return tilt, target, direction

    def _auto_lattice_tilt(
        self,
        main_target: tuple[float, float] | None,
        camera_roi_size: tuple[int, int],
        lattice_roi_size: tuple[int, int],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Place a spot on the same diagonal from the zeroth order as the main
        pattern but beyond it, so its detection window clears the DC. Used for the
        optical lattice (measure_phase) and the power-reference spot
        (measure_intensity). Returns ``(tilt, target)``.

        ``main_target`` is the main spot's camera pixel, used to sit beyond it;
        ``None`` falls back to one ROI width from the DC.
        """
        if self.camera_mapping is None:
            self._ensure_camera_mapping(None, None)
        direction = self._diagonal_direction()
        zeroth = np.asarray(
            [
                self.camera_mapping.zeroth_order_position[1],
                self.camera_mapping.zeroth_order_position[0],
            ]
        )
        main_distance = (
            float(np.linalg.norm(np.asarray(main_target) - zeroth))
            if main_target is not None
            else 1.0 * max(camera_roi_size)
        )
        # Beyond the main ROI, and far enough that the corner-steering detection
        # window (~4 * the lattice ROI, so ~2 * its half-width) clears the DC.
        lattice_clearance = (
            main_distance + max(camera_roi_size) + 2.0 * max(lattice_roi_size)
        )
        return self._auto_phase_tilt(
            lattice_roi_size, lattice_clearance, direction
        )

    def get_blazed_grating(
        self,
        linear_phase_tilt: tuple[float, float],
    ) -> NDArray[np.float_]:
        return linear_phase(
            *self.spatial_grid_slm,
            *linear_phase_tilt,
            tilt_units="metres",
            focal_length=self.focal_length,
            wavenumber=2 * np.pi / (self.slm.wav_um * 1e-6),
        )

    def _capture_averaged(
        self,
        exposure_time: float,
        frame_averages: int,
    ) -> NDArray[np.float_]:
        """Mean of ``frame_averages`` camera frames at ``exposure_time``.

        The simulated (and any real) camera draws fresh read and shot noise per
        frame, so the mean has ``frame_averages`` times lower noise variance. This
        is used where the signal is dim (single corner spots, the lattice baseline)
        and the extra frames are cheap because they are taken only a handful of
        times, not once per scanned superpixel.

        The camera's own ``get_image(averaging=...)`` does the batch capture (it
        sums the frames), so this only divides by the count to return the mean.
        """
        frames = max(1, int(frame_averages))
        summed = np.asarray(
            self.camera.get_image(exposure_time, averaging=frames), dtype=float
        )
        return summed / frames

    def calibrate_lattice_corner_tilts(
        self,
        corner_slices: list[tuple[slice, slice]],
        lattice_phase_tilt: tuple[float, float],
        roi_size: tuple[int, int],
        spot_center_pixels: tuple[int, int],
        exposure_time: float | None = None,
        frame_averages: int = _LATTICE_FRAME_AVERAGES,
    ) -> list[tuple[float, float]]:
        """Per-corner linear phase tilts that steer all four corner superpixels
        onto the same camera spot.

        Local SLM wavefront aberration gives each corner a slightly different effective
        tilt, so with a single shared grating the four corner beams land at slightly
        different camera positions and do not interfere cleanly. Each corner is
        displayed on its own and its diffraction spot is located, then the tilt that
        brings it onto the common spot is returned.

        The detection window is centred on ``spot_center_pixels`` (the camera spot the
        full SLM produces for ``lattice_phase_tilt``, i.e. the real, aberrated lattice
        position) and is four times the lattice ROI, so a single corner's dim, offset
        spot is captured. A single corner against the otherwise flat SLM throws a
        bright zeroth-order spot at the sensor centre; the exposure for each corner is
        set with ``autoexposure(window=...)`` restricted to the detection window so it
        exposes the (much dimmer) corner spot rather than that zeroth-order spot.

        Steering stays robust to camera noise: each corner frame is averaged over
        ``frame_averages`` exposures, its noise pedestal (the window median) is
        subtracted before the Gaussian fit, and a fit whose centroid leaves the
        window or whose peak fails to clear the residual noise is discarded in favour
        of the shared lattice tilt (no per-corner steering) for that corner.
        """
        base_grating = self.get_blazed_grating(lattice_phase_tilt)


        center_x, center_y = int(spot_center_pixels[0]), int(spot_center_pixels[1])
        sensor_height, sensor_width = self.camera.shape
        window_width = min(4 * roi_size[1], sensor_width)
        window_height = min(4 * roi_size[0], sensor_height)
        window_x0 = int(
            np.clip(center_x - window_width // 2, 0, sensor_width - window_width)
        )
        window_y0 = int(
            np.clip(center_y - window_height // 2, 0, sensor_height - window_height)
        )
        self.camera.set_woi([window_x0, window_width, window_y0, window_height])

        autoexposure_window = [center_x, window_width, center_y, window_height]

        # Grid referenced to the spot centre (0 = spot centre), so the fitted
        # centre is directly the offset from it even when the window was clamped.
        pitch_x = self.camera.pitch_um[1] * 1e-6
        pitch_y = self.camera.pitch_um[0] * 1e-6
        grid_x, grid_y = np.meshgrid(
            (np.arange(window_width) - (center_x - window_x0)) * pitch_x,
            (np.arange(window_height) - (center_y - window_y0)) * pitch_y,
        )
        grid = [grid_x, grid_y]

        corner_size = corner_slices[0][0].stop - corner_slices[0][0].start
        aperture_radius = corner_size * self.slm.pitch_um[0] * 1e-6 / 2
        spot_radius_guess = (
            (self.slm.wav_um * 1e-6) * self.focal_length
            / (np.pi * aperture_radius)
        )

        corner_tilts = []
        for corner_slice in corner_slices:
            corner_phase = np.zeros(self.slm.shape)
            corner_phase[corner_slice] = base_grating[corner_slice]
            self.slm.set_phase(corner_phase)

            corner_exposure = exposure_time
            if corner_exposure is None:
                bounds = self.camera.exposure_bounds_s
                try:
                    corner_exposure = self.camera.autoexposure(
                        set_fraction=_AUTOEXPOSURE_SET_FRACTION,
                        exposure_bounds_s=bounds,
                        window=autoexposure_window,
                        timeout_s=self.autoexposure_timeout_s,
                    )
                except RuntimeError:
                    corner_exposure = bounds[1] if bounds is not None else 1.0
                    self.camera.set_exposure(corner_exposure)
            camera_image = self._capture_averaged(corner_exposure, frame_averages)
            
            noise_floor = float(np.median(camera_image))
            noise_sigma = background_noise(camera_image)
            denoised = np.clip(camera_image - noise_floor, 0.0, None)
            try:
                popt, _ = fit_gaussian_beam_intensity(
                    *grid, denoised, beam_radius_guess=spot_radius_guess
                )
                # Spot offset from the lattice spot. Steer it back by that much.
                offset_x, offset_y = float(popt[1]), float(popt[2])
                peak = float(popt[3])
                inside_window = (
                    grid_x.min() <= offset_x <= grid_x.max()
                    and grid_y.min() <= offset_y <= grid_y.max()
                )
                if not inside_window or peak < _CORNER_MIN_SNR * noise_sigma:
                    # Safeguard against bad fit
                    offset_x, offset_y = 0.0, 0.0
            except (RuntimeError, ValueError):
                # Use nominal tilt if fit fails rather than aborting the whole 
                # calibration.
                offset_x, offset_y = 0.0, 0.0
            corner_tilts.append(
                (
                    lattice_phase_tilt[0] - offset_x,
                    lattice_phase_tilt[1] - offset_y,
                )
            )
        return corner_tilts

    def get_number_of_superpixels(
        self,
        target_superpixel_width: int = 32,
        target_superpixel_height: int = 32,
    ) -> tuple[int, int]:
        """Return (n_superpixels_x, n_superpixels_y) for the superpixel size
        closest to the target, using divisors of the SLM dimensions."""
        factors_x = [
            i for i in range(1, self.slm.shape[1] + 1) if self.slm.shape[1] % i == 0
        ]
        factors_y = [
            i for i in range(1, self.slm.shape[0] + 1) if self.slm.shape[0] % i == 0
        ]
        superpixel_width = min(
            factors_x, key=lambda x: abs(x - target_superpixel_width)
        )
        superpixel_height = min(
            factors_y, key=lambda x: abs(x - target_superpixel_height)
        )
        return (
            self.slm.shape[1] // superpixel_width,
            self.slm.shape[0] // superpixel_height,
        )

    def get_superpixel_size(
        self,
        target_superpixels_x: int = 16,
        target_superpixels_y: int = 16,
    ) -> tuple[int, int]:
        """Return (superpixel_width, superpixel_height) using the closest divisors of
        the SLM dimensions to slm_size / target_superpixels. """
        factors_x = [
            i for i in range(1, self.slm.shape[1] + 1) if self.slm.shape[1] % i == 0
        ]
        factors_y = [
            i for i in range(1, self.slm.shape[0] + 1) if self.slm.shape[0] % i == 0
        ]
        superpixel_width = min(
            factors_x,
            key=lambda x: abs(x - self.slm.shape[1] / target_superpixels_x),
        )
        superpixel_height = min(
            factors_y,
            key=lambda x: abs(x - self.slm.shape[0] / target_superpixels_y),
        )
        return superpixel_width, superpixel_height

    def get_roi_size(
        self,
        aperture_width: int,
        aperture_height: int,
    ) -> tuple[int, int]:
        """Return the full null-to-null width of the sinc-squared diffraction
        pattern in the Fourier plane for a rectangular aperture, in camera pixels.
        """
        wavelength = self.slm.wav_um * 1e-6

        pitch_x = self.slm.pitch_um[1] * 1e-6
        pitch_y = self.slm.pitch_um[0] * 1e-6

        camera_pitch_x = self.camera.pitch_um[1] * 1e-6
        camera_pitch_y = self.camera.pitch_um[0] * 1e-6

        roi_width = int(
            2
            * wavelength
            * self.focal_length
            / (aperture_width * pitch_x * camera_pitch_x)
        )

        roi_height = int(
            2
            * wavelength
            * self.focal_length
            / (aperture_height * pitch_y * camera_pitch_y)
        )
        return roi_width, roi_height

    # TODO: Add power tracking during measurement
    def measure_intensity(
        self,
        number_of_superpixels_x: int,
        number_of_superpixels_y: int,
        superpixel_width: int,
        superpixel_height: int,
        linear_phase_tilt: tuple[float, float] | None = None,
        camera_roi_size: tuple[int, int] | None = None,
        camera_mapping: CameraMapping | None = None,
        slm_camera_model: SLMFourierLensModel | None = None,
        normalize_power: bool = False,
        verbose: bool = True,
    ) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
        """
        This function measures the intensity profile of the laser beam incident onto the
        SLM by displaying a sequence of rectangular phase masks on the SLM. The phase
        mask contains a linear phase which creates a diffraction spot on the camera. The
        position of the phase mask is varied across the entire area of the SLM and the
        intensity of each diffraction spot is measured using the camera. Read the SI of
        https://doi.org/10.1038/s41598-023-30296-6 for details.

        Args:
            number_of_superpixels_x : int
                Number of superpixels along x.
            number_of_superpixels_y : int
                Number of superpixels along y.
            superpixel_width : int
                Width of superpixels [px].
            superpixel_height : int
                Height of superpixels [px].
            linear_phase_tilt : tuple[float, float]
                x and y gradient of the linear phase in units of the resulting spot
                displacement in the Fourier plane in metres.
            camera_roi_size : tuple[int, int] | None
                Height and width of the region of interest on the camera around each
                diffraction spot. If None, it is sized automatically from the
                superpixel's sinc^2 spot (see below).
            normalize_power : bool
                If True, correct laser-power drift over the scan. The central
                superpixel is held on a second linear phase that places a fixed
                reference spot opposite the main pattern (across the zeroth order),
                like a one-spot optical lattice. Each frame's spot power is divided by
                that reference power, so the map becomes intensity relative to the beam
                centre and is immune to laser drift. Needs a camera mapping. The
                per-frame reference powers are kept on self.power_reference.
        Returns:
            superpixel_intensity : NDArray
                Intensity of the superpixels.
        """
        timer = Timer(verbose=verbose)
        timer.start()

        # Size the camera ROI from the superpixel's sinc^2 diffraction spot when not
        # provided.
        if camera_roi_size is None:
            roi_width, roi_height = self.get_roi_size(
                superpixel_width, superpixel_height
            )
            camera_roi_size = (2 * roi_height, 2 * roi_width)
            if verbose:
                print(
                    "Auto camera ROI size (height, width): "
                    f"{camera_roi_size} px."
                )

        # Auto-place the diffraction spot from a coarse mapping when no tilt is given.
        if linear_phase_tilt is None:
            linear_phase_tilt, _, _ = self._ensure_and_place_main(
                camera_roi_size, camera_mapping, slm_camera_model
            )
            if verbose:
                print(f"Auto linear_phase_tilt (m): {linear_phase_tilt}")

        spot_center, _, _, _ = get_diffraction_spot_position(
            self.slm,
            self.camera,
            linear_phase_tilt,
            focal_length=self.focal_length,
            exposure_time=0.1,
            units="pixels",
            verbose=verbose,
        )
        # Optional laser-power reference: hold the central superpixel on a second
        # (lattice-style) linear phase so a fixed reference spot sits elsewhere on the
        # sensor, and bound both spots in one window. reference_grating is applied to
        # base_phase below, once the slicer gives the central superpixel.
        reference_grating = None
        main_box = None
        reference_box = None
        if normalize_power:
            if self.camera_mapping is None:
                self._ensure_camera_mapping(camera_mapping, slm_camera_model)
            # A second linear phase places a bright fixed reference spot on the
            # opposite diagonal from the main pattern (across the zeroth order), one
            # ROI clear of the DC. The main sits two ROI out on its diagonal, so the
            # reference stays well clear of it while fitting closer to the centre on
            # the opposite side.
            reference_tilt, _ = self._auto_phase_tilt(
                camera_roi_size,
                1.0 * max(camera_roi_size),
                -self._diagonal_direction(),
            )
            reference_grating = self.get_blazed_grating(reference_tilt)
            reference_center, _, _, _ = get_diffraction_spot_position(
                self.slm,
                self.camera,
                reference_tilt,
                focal_length=self.focal_length,
                exposure_time=0.1,
                units="pixels",
                verbose=verbose,
            )
            main_x0, main_x1, main_y0, main_y1 = roi_bounds(
                spot_center, camera_roi_size
            )
            ref_x0, ref_x1, ref_y0, ref_y1 = roi_bounds(
                reference_center, camera_roi_size
            )
            woi_x0 = min(main_x0, ref_x0)
            woi_y0 = min(main_y0, ref_y0)
            woi_width = max(main_x1, ref_x1) - woi_x0
            woi_height = max(main_y1, ref_y1) - woi_y0
            if (
                woi_x0 < 0
                or woi_y0 < 0
                or woi_x0 + woi_width > self.camera.shape[1]
                or woi_y0 + woi_height > self.camera.shape[0]
            ):
                raise ValueError(
                    "The main and reference spots do not both fit on the sensor; "
                    "reduce camera_roi_size."
                )
            self.camera.set_woi([woi_x0, woi_width, woi_y0, woi_height])
            main_box = (
                slice(main_y0 - woi_y0, main_y0 - woi_y0 + camera_roi_size[0]),
                slice(main_x0 - woi_x0, main_x0 - woi_x0 + camera_roi_size[1]),
            )
            reference_box = (
                slice(ref_y0 - woi_y0, ref_y0 - woi_y0 + camera_roi_size[0]),
                slice(ref_x0 - woi_x0, ref_x0 - woi_x0 + camera_roi_size[1]),
            )
        else:
            set_camera_woi(self.camera, spot_center, camera_roi_size)

        slicer = SuperpixelSlicer(
            self.slm.shape,
            number_of_superpixels_x,
            number_of_superpixels_y,
            superpixel_width,
            superpixel_height,
            start_index_x=0,
            start_index_y=0,
            end_index_x=self.slm.shape[1],
            end_index_y=self.slm.shape[0],
        )

        linear_slm_phase = self.get_blazed_grating(linear_phase_tilt)

        # Background: a vertical binary 0/pi grating (every other pixel column) instead
        # of a flat zero phase, so the unmodulated SLM area diffracts into higher
        # diffraction orders instead of the bright central zeroth-order.
        base_phase = binary_phase_grating(self.slm.shape)
        if normalize_power:
            # Central superpixel holds the reference grating on every frame.
            base_phase[slicer.central_slice] = reference_grating[slicer.central_slice]

        # Weights array to handle overlapping superpixels
        weights = np.zeros(self.slm.shape)

        # Display central sub-aperture on SLM and check if camera is
        # over-exposed.
        slm_phase_central_superpixel = np.copy(base_phase)
        slm_phase_central_superpixel[slicer.central_slice] = linear_slm_phase[
            slicer.central_slice
        ]

        self.slm.set_phase(slm_phase_central_superpixel)

        # Find camera exposure time.
        exposure_time = self.camera.autoexposure(
            set_fraction=_AUTOEXPOSURE_SET_FRACTION,
            exposure_bounds_s=(0, 1),
            window=[
                spot_center[0], camera_roi_size[1],
                spot_center[1], camera_roi_size[0],
            ],
            timeout_s=self.autoexposure_timeout_s,
        )

        camera_images = np.zeros(
            (
                slicer.number_of_superpixels,
                *camera_roi_size,
            )
        )
        superpixel_power = np.zeros(self.slm.shape)
        power_reference = [] if normalize_power else None

        # Take camera images
        for i, superpixel_slice in enumerate(slicer.slices):
            superpixel_slice = slicer.get_slice(i)
            is_reference = (
                normalize_power and superpixel_slice == slicer.central_slice
            )

            masked_phase = np.copy(base_phase)
            if not is_reference:
                masked_phase[superpixel_slice] = linear_slm_phase[superpixel_slice]
            if normalize_power:
                # Keep the reference spot intact even if this superpixel overlaps it.
                masked_phase[slicer.central_slice] = reference_grating[
                    slicer.central_slice
                ]

            self.slm.set_phase(masked_phase)
            image = self.camera.get_image(exposure_time)

            weights[superpixel_slice] += 1

            if normalize_power:
                camera_images[i, ...] = image[main_box]
                reference_power = float(np.sum(image[reference_box]))
                power_reference.append(reference_power)
                if is_reference:
                    # The centre is its own reference: relative intensity is 1.
                    superpixel_power[superpixel_slice] += 1.0
                else:
                    main_power = float(np.sum(image[main_box]))
                    superpixel_power[superpixel_slice] += main_power / max(
                        reference_power, np.finfo(float).eps
                    )
            else:
                camera_images[i, ...] = image
                superpixel_power[superpixel_slice] += np.sum(image) / (
                    np.size(image) * exposure_time
                )
            print(
                f"Superpixel {i + 1}/{slicer.number_of_superpixels} "
                f"({100 * (i + 1) / slicer.number_of_superpixels:.2f}%)"
            )

        # Per-frame reference powers, kept for laser-drift diagnostics.
        self.power_reference = (
            np.asarray(power_reference) if normalize_power else None
        )

        # Find SLM intensity profile
        weights[weights == 0] = 1
        superpixel_intensity = superpixel_power / weights

        blur_kernel_size = max(slicer.superpixel_separation) / 2
        superpixel_intensity = gaussian_filter(
            superpixel_intensity, sigma=blur_kernel_size
        )

        timer.stop()
        return superpixel_intensity, camera_images

    def measure_phase(
        self,
        number_of_superpixels_x: int,
        number_of_superpixels_y: int,
        superpixel_width: int,
        superpixel_height: int,
        linear_phase_tilt: tuple[float, float] | None = None,
        camera_roi_size: tuple[int, int] | None = None,
        measured_intensity: NDArray[np.float_] | None = None,
        compensate_pointing: bool = False,
        lattice_phase_tilt: tuple[float, float] | None = None,
        lattice_superpixel_size: int | None = None,
        lattice_roi_size: tuple[int, int] | None = None,
        camera_mapping: CameraMapping | None = None,
        slm_camera_model: SLMFourierLensModel | None = None,
        verbose: bool = True,
        record_displayed_phases: bool = False,
    ) -> tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.float_]]:
        """This function measures the constant phase at the SLM by displaying
        a sequence of rectangular phase masks on the SLM. This scheme was adapted from
        Phillip Zupancic's work (https://doi.org/10.1364/OE.24.013881). For details of
        our implementation, see the supplementary material of
        https://doi.org/10.1038/s41598-023-30296-6.

        Parameters
        ----------
        number_of_superpixels_x : int
            Number of superpixels along x.
        number_of_superpixels_y : int
            Number of superpixels along y.
        superpixel_width : int
            Width of superpixels [px].
        superpixel_height : int
            Height of superpixels [px].
        linear_phase_tilt : tuple[float, float]
            x and y gradient of the linear phase in units of the resulting spot
            displacement in the Fourier plane in metres.
        camera_roi_size : tuple[int, int] | None
            Height and width of the region of interest on the camera around the main
            interference spot. If None, sized automatically to the superpixel's sinc^2
            central lobe (out to the first zero).
        compensate_pointing : bool, optional
            If True, also display four corner superpixels forming a 2D optical lattice
            and use its phase to correct for beam pointing drift. Default is False.
        lattice_phase_tilt : tuple[float, float] | None, optional
            x and y gradient (metres in the Fourier plane) steering the optical lattice
            to a separate camera region. Required if compensate_pointing is True.
        lattice_superpixel_size : int | None, optional
            Side length [px] of the square corner superpixels. If None, sized
            automatically from measured_intensity to match the fringe brightness.
        lattice_roi_size : tuple[int, int] | None, optional
            Height and width of the camera region of interest around the optical
            lattice. If None, sized automatically to the corner superpixel's sinc^2
            central lobe (out to the first zero).
        verbose : bool, optional
            If True, prints the progress of the measurement. Default is True.
        record_displayed_phases : bool, optional
            If True, store the displayed SLM phase (``slm.display``) for every
            scanned superpixel in ``self.displayed_slm_phases`` so the scan can be
            visualised afterwards (see ``RasterCalibratorVisualizer``). Off by
            default as it keeps a full-resolution frame per superpixel.

        Returns
        -------
        superpixel_phase : NDArray
            Phase of the superpixels.
        camera_images : NDArray
            Camera images.
        """
        if (
            compensate_pointing
            and measured_intensity is None
            and lattice_superpixel_size is None
        ):
            raise ValueError(
                "compensate_pointing requires measured_intensity (to size the "
                "corner superpixels) or an explicit lattice_superpixel_size."
            )

        timer = Timer(verbose=verbose)
        timer.start()

        wavenumber = 2 * np.pi / (self.slm.wav_um * 1e-6)

        if measured_intensity is None:
            intensity = np.ones(self.slm.shape)
        else:
            beam_radius, shift_x, shift_y = self.fit_gaussian_beam(measured_intensity)
            intensity = gpu_to_numpy(
                gaussian_beam_intensity(
                    *self.spatial_grid_slm, beam_radius, shift_x, shift_y
                )
            )

        base_phase = binary_phase_grating(self.slm.shape)

        slicer = SuperpixelSlicer(
            self.slm.shape,
            number_of_superpixels_x,
            number_of_superpixels_y,
            superpixel_width,
            superpixel_height,
            start_index_x=0,
            start_index_y=0,
            end_index_x=self.slm.shape[1],
            end_index_y=self.slm.shape[0],
            intensity=intensity,
            max_correction_factor=4,
        )

        # Size the camera ROI (before the lattice setup so the main tilt can be
        # placed and its diagonal shared with the lattice).
        if camera_roi_size is None:
            largest_width = max(
                slice_[1].stop - slice_[1].start for slice_ in slicer.slices
            )
            largest_height = max(
                slice_[0].stop - slice_[0].start for slice_ in slicer.slices
            )
            roi_width, roi_height = self.get_roi_size(largest_width, largest_height)
            camera_roi_size = (roi_height, roi_width)
            if verbose:
                print(
                    "Auto camera ROI size (height, width): "
                    f"{camera_roi_size} px."
                )

        # Auto-place the interference pattern from a coarse mapping when no tilt is
        # given, along a diagonal from the zeroth order that the lattice reuses.
        main_target = None
        if linear_phase_tilt is None:
            linear_phase_tilt, main_target, _ = self._ensure_and_place_main(
                camera_roi_size, camera_mapping, slm_camera_model
            )
            if verbose:
                print(f"Auto linear_phase_tilt (m): {linear_phase_tilt}")

        # Set up the optical lattice: four constant corner superpixels sharing a
        # steeper grating, sized so the lattice is about as bright as the
        # interference fringes.
        if compensate_pointing:
            corner_size = lattice_superpixel_size or slicer.get_lattice_corner_size()

            if lattice_roi_size is None:
                roi_width, roi_height = self.get_roi_size(corner_size, corner_size)
                lattice_roi_size = (roi_height, roi_width)

            # Auto-place the lattice on the same diagonal as the main pattern but
            # further from the DC (beyond the main ROI), when no tilt is given.
            if lattice_phase_tilt is None:
                if self.camera_mapping is None:
                    self._ensure_camera_mapping(camera_mapping, slm_camera_model)
                lattice_phase_tilt, _ = self._auto_lattice_tilt(
                    main_target, camera_roi_size, lattice_roi_size
                )
                if verbose:
                    print(f"Auto lattice_phase_tilt (m): {lattice_phase_tilt}")

            height, width = self.slm.shape
            corner_slices = [
                (slice(0, corner_size), slice(0, corner_size)),
                (slice(0, corner_size), slice(width - corner_size, width)),
                (slice(height - corner_size, height), slice(0, corner_size)),
                (
                    slice(height - corner_size, height),
                    slice(width - corner_size, width),
                ),
            ]
            # Detected camera spot the full SLM produces for the lattice tilt (the
            # real, aberrated lattice position). The corner-steering detection
            # window and the lattice ROI are both centred on it, so the dim corner
            # beams fall inside the window and the steered lattice sits centred in
            # the ROI.
            lattice_center, _, _, _ = get_diffraction_spot_position(
                self.slm,
                self.camera,
                lattice_phase_tilt,
                focal_length=self.focal_length,
                exposure_time=0.1,
                units="pixels",
                verbose=verbose,
            )

            # Local SLM aberration gives each corner a slightly different
            # effective tilt, so steer each corner individually onto the common
            # lattice spot, so the four beams interfere cleanly.
            corner_tilts = self.calibrate_lattice_corner_tilts(
                corner_slices, lattice_phase_tilt, lattice_roi_size, lattice_center
            )
            self.lattice_corner_tilts = corner_tilts
            for corner_slice, corner_tilt in zip(corner_slices, corner_tilts):
                grating = self.get_blazed_grating(corner_tilt)
                base_phase[corner_slice] = grating[corner_slice]

            lattice_separation_x = (
                (width - corner_size) * self.slm.pitch_um[1] * 1e-6
            )
            lattice_separation_y = (
                (height - corner_size) * self.slm.pitch_um[0] * 1e-6
            )
            k_lattice_x = wavenumber * np.sin(
                np.arctan(lattice_separation_x / self.focal_length)
            )
            k_lattice_y = wavenumber * np.sin(
                np.arctan(lattice_separation_y / self.focal_length)
            )

            # Keep the scanned sample superpixels off the fixed corner regions.
            slicer.remove_overlapping_slices(corner_slices)

        if compensate_pointing and slicer.reference_slice not in slicer.slices:
            raise ValueError(
                "The lattice corner superpixels overlap the reference "
                "(brightest) superpixel. Reduce lattice_superpixel_size."
            )

        # Set the camera window of interest. When compensating, enlarge it to
        # bound both the main interference spot and the offset lattice spot.
        main_center, _, _, _ = get_diffraction_spot_position(
            self.slm,
            self.camera,
            linear_phase_tilt,
            focal_length=self.focal_length,
            exposure_time=0.1,
            units="pixels",
            verbose=verbose,
        )

        main_box = None
        lattice_box = None
        if compensate_pointing:
            # lattice_center was detected above (used to centre the corner-steering
            # detection window); reuse it to place the lattice ROI on the same spot.
            main_x0, main_x1, main_y0, main_y1 = roi_bounds(
                main_center, camera_roi_size
            )
            lat_x0, lat_x1, lat_y0, lat_y1 = roi_bounds(
                lattice_center, lattice_roi_size
            )
            woi_x0 = min(main_x0, lat_x0)
            woi_y0 = min(main_y0, lat_y0)
            woi_width = max(main_x1, lat_x1) - woi_x0
            woi_height = max(main_y1, lat_y1) - woi_y0
            if (
                woi_x0 < 0
                or woi_y0 < 0
                or woi_x0 + woi_width > self.camera.shape[1]
                or woi_y0 + woi_height > self.camera.shape[0]
            ):
                raise ValueError(
                    "The main and lattice spots do not both fit on the sensor; "
                    "reduce lattice_phase_tilt or the ROI sizes."
                )
            self.camera.set_woi([woi_x0, woi_width, woi_y0, woi_height])

            # (row_slice, col_slice) of each sub-image within the captured WOI.
            main_box = (
                slice(main_y0 - woi_y0, main_y0 - woi_y0 + camera_roi_size[0]),
                slice(main_x0 - woi_x0, main_x0 - woi_x0 + camera_roi_size[1]),
            )
            lattice_box = (
                slice(lat_y0 - woi_y0, lat_y0 - woi_y0 + lattice_roi_size[0]),
                slice(lat_x0 - woi_x0, lat_x0 - woi_x0 + lattice_roi_size[1]),
            )
        else:
            set_camera_woi(self.camera, main_center, camera_roi_size)

        linear_slm_phase = self.get_blazed_grating(linear_phase_tilt)

        reference_superpixel_phase = np.copy(base_phase)
        reference_superpixel_phase[slicer.reference_slice] = linear_slm_phase[
            slicer.reference_slice
        ]

        reference_superpixel_center_x = (
            slicer.reference_slice[1].start + slicer.reference_slice[1].stop
        ) / 2
        reference_superpixel_center_y = (
            slicer.reference_slice[0].start + slicer.reference_slice[0].stop
        ) / 2

        # Pick a second superpixel (neighbour of the reference within the kept slices)
        # so the exposure test sees two-beam interference. Indexing the reduced slice
        # list directly avoids running off the end when slices are removed (intensity
        # compensation or corner exclusion).
        reference_position = slicer.slices.index(slicer.reference_slice)
        test_position = (reference_position + 1) % len(slicer.slices)
        test_slice = slicer.slices[test_position]
        exposure_test_phase = np.copy(reference_superpixel_phase)
        exposure_test_phase[test_slice] = linear_slm_phase[test_slice]

        self.slm.set_phase(exposure_test_phase)

        # Find camera exposure time.
        exposure_time = self.camera.autoexposure(
            set_fraction=0.9, exposure_bounds_s=(0, 1),
            window=[
                main_center[0], camera_roi_size[1],
                main_center[1], camera_roi_size[0],
            ],
            timeout_s=self.autoexposure_timeout_s,
        )

        camera_images = np.zeros((len(slicer.slices), *camera_roi_size))
        fitted_images = np.zeros((len(slicer.slices), *camera_roi_size))
        # Grid centred on the main-spot crop.
        main_grid = self._orient_grid([
            gpu_to_numpy(grid)
            for grid in get_spatial_grid(
                camera_roi_size, self.camera.pitch_um * 1e-6
            )
        ])

        # Per-superpixel lattice-drift series (filled only when compensating) and the
        # captured/fitted lattice image buffers, pre-allocated in scan order.
        lattice_shift_x = np.zeros(slicer.number_of_superpixels)
        lattice_shift_y = np.zeros(slicer.number_of_superpixels)
        lattice_shift_x_err = np.zeros(slicer.number_of_superpixels)
        lattice_shift_y_err = np.zeros(slicer.number_of_superpixels)
        lattice_images = None
        fitted_lattice_images = None

        if compensate_pointing:
            lattice_grid = self._orient_grid([
                gpu_to_numpy(grid)
                for grid in get_spatial_grid(
                    lattice_roi_size, self.camera.pitch_um * 1e-6
                )
            ])
            # Captured and fitted lattice ROI images, kept for troubleshooting
            # and plotting (mirrors camera_images / fitted_images).
            lattice_images = np.zeros(
                (slicer.number_of_superpixels, *lattice_roi_size)
            )
            fitted_lattice_images = np.zeros(
                (slicer.number_of_superpixels, *lattice_roi_size)
            )
            # Baseline lattice phase from the displayed reference/exposure pattern
            # (which already shows the constant lattice). This anchors the measured
            # drift to zero at the un-drifted reference state, so it is averaged over
            # several frames to keep the anchor phase robust to camera noise.
            baseline_image = self._capture_averaged(
                exposure_time, _LATTICE_FRAME_AVERAGES
            )[lattice_box]
            popt_lattice, _ = fit_optical_lattice_fringes(
                *lattice_grid,
                baseline_image,
                lattice_separation_x,
                lattice_separation_y,
                wavenumber,
                self.focal_length,
                amplitude_guess=np.max(baseline_image) / 2,
            )
            phase_x0, phase_y0 = popt_lattice[0], popt_lattice[1]
            phase_x_prev, phase_y_prev = phase_x0, phase_y0
            self.lattice_baseline_image = baseline_image

        superpixel_coordinates = np.zeros((2, slicer.number_of_superpixels))
        superpixel_phase = np.zeros(slicer.number_of_superpixels)

        # Optionally keep the displayed SLM phase for each superpixel (the actual
        # grayscale shown, so any drift / quantization is captured) for plotting.
        displayed_slm_phases = [] if record_displayed_phases else None
        full_frame_image = None  # Saved later for diagnostocs/visualizer

        # Take camera images
        fitted_phase = 0
        for i, superpixel_slice in enumerate(slicer.slices):
            masked_phase = np.copy(reference_superpixel_phase)

            masked_phase[superpixel_slice] = linear_slm_phase[superpixel_slice]

            self.slm.set_phase(masked_phase)

            if record_displayed_phases:
                displayed_slm_phases.append(np.asarray(self.slm.display).copy())

            if record_displayed_phases and full_frame_image is None:
                # Lift the scan WOI just for this frame so the snapshot spans the
                # whole sensor
                stored_woi = deepcopy(self.camera.woi)
                self.camera.set_woi(None)
                full_frame_image = np.asarray(self.camera.get_image(exposure_time))
                self.camera.set_woi(stored_woi)

            full_image = self.camera.get_image(exposure_time)

            superpixel_center_x = (
                superpixel_slice[1].start + superpixel_slice[1].stop
            ) / 2
            superpixel_center_y = (
                superpixel_slice[0].start + superpixel_slice[0].stop
            ) / 2

            superpixel_separation_x = (
                (superpixel_center_x - reference_superpixel_center_x)
                * self.slm.pitch_um[1]
                * 1e-6
            )
            superpixel_separation_y = (
                (superpixel_center_y - reference_superpixel_center_y)
                * self.slm.pitch_um[0]
                * 1e-6
            )

            # Measure the camera-plane displacement from beam pointing drift via the
            # optical lattice, then fit the main fringes on shifted coordinates to
            # remove it (shift stays 0 when not compensating).
            shift_x = 0.0
            shift_y = 0.0
            if compensate_pointing:
                lattice_image = full_image[lattice_box]
                try:
                    popt_lattice, pcov_lattice = fit_optical_lattice_fringes(
                        *lattice_grid,
                        lattice_image,
                        lattice_separation_x,
                        lattice_separation_y,
                        wavenumber,
                        self.focal_length,
                        # Unbounded phase warm-started from the previous cumulative
                        # value, so the fit tracks continuous drift across the
                        # (-pi, pi) boundary without wrapping.
                        phase_x_guess=phase_x_prev,
                        phase_y_guess=phase_y_prev,
                        amplitude_guess=np.max(lattice_image) / 2,
                        bound_phase=False,
                    )
                    phase_x_prev = popt_lattice[0]
                    phase_y_prev = popt_lattice[1]
                    # 1-sigma fit uncertainty on the phases, propagated to the shift.
                    phase_err = np.sqrt(np.diag(pcov_lattice))
                    lattice_shift_x_err[i] = phase_err[0] / k_lattice_x
                    lattice_shift_y_err[i] = phase_err[1] / k_lattice_y
                except (RuntimeError, ValueError):
                    # A noisy or dim lattice frame can defeat the fit. Hold the last
                    # good phase so the drift estimate degrades gracefully instead of
                    # aborting the scan, and flag its uncertainty as unknown.
                    popt_lattice = (
                        phase_x_prev,
                        phase_y_prev,
                        np.max(lattice_image) / 2,
                    )
                    lattice_shift_x_err[i] = np.nan
                    lattice_shift_y_err[i] = np.nan
                # A pattern shift by d lowers the fitted phase by k*d, so the
                # camera-plane displacement is (phase0 - phase) / k. The main fringes
                # are then fitted on (grid - displacement).
                shift_x = (phase_x0 - phase_x_prev) / k_lattice_x
                shift_y = (phase_y0 - phase_y_prev) / k_lattice_y
                lattice_shift_x[i] = shift_x
                lattice_shift_y[i] = shift_y
                lattice_images[i, ...] = lattice_image
                fitted_lattice_images[i, ...] = optical_lattice_fringes(
                    *lattice_grid,
                    lattice_separation_x,
                    lattice_separation_y,
                    wavenumber,
                    self.focal_length,
                    *popt_lattice,
                )
                main_image = full_image[main_box]
            else:
                main_image = full_image

            camera_images[i, ...] = main_image

            amplitude_guess = np.max(main_image) / np.sqrt(2)

            popt, _ = fit_interferometric_fringes(
                main_grid[0] - shift_x,
                main_grid[1] - shift_y,
                main_image,
                superpixel_separation_x,
                superpixel_separation_y,
                wavenumber,
                self.focal_length,
                phase_guess=0,
                amplitude_guess=amplitude_guess,
            )

            fitted_phase = popt[0]
            if superpixel_slice == slicer.reference_slice:
                fitted_phase = 0

            fitted_images[i, ...] = interferometric_fringes(
                main_grid[0] - shift_x,
                main_grid[1] - shift_y,
                superpixel_separation_x,
                superpixel_separation_y,
                wavenumber,
                self.focal_length,
                *popt,
            )

            superpixel_coordinates[0, i] = superpixel_center_x
            superpixel_coordinates[1, i] = superpixel_center_y
            superpixel_phase[i] = fitted_phase

            print(
                f"Superpixel {i + 1}/{slicer.number_of_superpixels} "
                f"({100 * (i + 1) / slicer.number_of_superpixels:.2f}%)"
            )

        phase_unwrapped = unwrap_nonuniform(
            superpixel_coordinates[0, :],
            superpixel_coordinates[1, :],
            superpixel_phase,
        )

        # Weights array to handle overlapping superpixels
        weights = np.zeros(self.slm.shape)
        measured_mask = np.zeros(self.slm.shape, dtype=bool)
        phase_slm = np.zeros(self.slm.shape)

        for i, superpixel_slice in enumerate(slicer.slices):
            weights[superpixel_slice] += 1
            measured_mask[superpixel_slice] = True
            phase_slm[superpixel_slice] += phase_unwrapped[i]

        weights[weights == 0] = 1
        phase_slm /= weights

        phase = inpaint(phase_slm, measured_mask)

        blur_kernel_size = max(slicer.superpixel_separation) / 2
        phase = gaussian_filter(phase, sigma=blur_kernel_size)

        self.visualization_data = RasterVisualizationData(
            camera_images=camera_images,
            fitted_images=fitted_images,
            measured_phase=phase,
            superpixel_coordinates=superpixel_coordinates,
            lattice_images=lattice_images,
            fitted_lattice_images=fitted_lattice_images,
            lattice_shift_x=lattice_shift_x,
            lattice_shift_y=lattice_shift_y,
            lattice_shift_x_err=lattice_shift_x_err,
            lattice_shift_y_err=lattice_shift_y_err,
            displayed_slm_phases=(
                np.array(displayed_slm_phases)
                if record_displayed_phases
                else None
            ),
            full_frame_image=full_frame_image,
            full_frame_marker_positions=(
                {
                    "interference pattern": (
                        float(main_center[0]),
                        float(main_center[1]),
                    ),
                    "optical lattice": (
                        (float(lattice_center[0]), float(lattice_center[1]))
                        if compensate_pointing
                        else None
                    ),
                    "zeroth order": (
                        (
                            float(self.camera_mapping.zeroth_order_position[1]),
                            float(self.camera_mapping.zeroth_order_position[0]),
                        )
                        if self.camera_mapping is not None
                        else None
                    ),
                }
                if full_frame_image is not None
                else None
            ),
        )

        timer.stop()
        return phase, camera_images, fitted_images

    def calibrate(
        self,
        number_of_superpixels: tuple[int, int] | None = None,
        superpixel_size: tuple[int, int] | None = None,
        linear_phase_tilt: tuple[float, float] | None = None,
        camera_roi_size: tuple[int, int] | None = None,
        compensate_pointing: bool = False,
        lattice_phase_tilt: tuple[float, float] | None = None,
        lattice_superpixel_size: int | None = None,
        lattice_roi_size: tuple[int, int] | None = None,
        camera_mapping: CameraMapping | None = None,
        slm_camera_model: SLMFourierLensModel | None = None,
        normalize_power: bool = False,
        save_metadata: bool = False,
        verbose: bool = True,
    ) -> WavefrontCalibrationData:
        if number_of_superpixels is None and superpixel_size is not None:
            number_of_superpixels_x, number_of_superpixels_y = (
                self.get_number_of_superpixels(*superpixel_size)
            )
            superpixel_width = self.slm.shape[1] // number_of_superpixels_x
            superpixel_height = self.slm.shape[0] // number_of_superpixels_y
        elif superpixel_size is None and number_of_superpixels is not None:
            superpixel_width, superpixel_height = self.get_superpixel_size(
                *number_of_superpixels
            )
            number_of_superpixels_x = self.slm.shape[1] // superpixel_width
            number_of_superpixels_y = self.slm.shape[0] // superpixel_height
        elif superpixel_size is None and number_of_superpixels is None:
            superpixel_width, superpixel_height = self.get_superpixel_size()
            number_of_superpixels_x = self.slm.shape[1] // superpixel_width
            number_of_superpixels_y = self.slm.shape[0] // superpixel_height
        else:
            number_of_superpixels_x, number_of_superpixels_y = number_of_superpixels
            superpixel_width, superpixel_height = superpixel_size

        if camera_roi_size is None:
            camera_roi_size = self.get_roi_size(superpixel_width, superpixel_height)

        # Build the coarse mapping once up front (when a tilt must be auto-placed,
        # the fits need orienting, or the power reference needs placing) so both
        # scans reuse it.
        need_auto = (
            linear_phase_tilt is None
            or normalize_power
            or (compensate_pointing and lattice_phase_tilt is None)
        )
        if camera_mapping is not None or slm_camera_model is not None:
            self._ensure_camera_mapping(camera_mapping, slm_camera_model)
        elif need_auto and self.camera_mapping is None:
            self._ensure_camera_mapping(None, None)

        intensity, camera_images_intensity = self.measure_intensity(
            number_of_superpixels_x,
            number_of_superpixels_y,
            superpixel_width,
            superpixel_height,
            linear_phase_tilt,
            camera_roi_size,
            normalize_power=normalize_power,
            verbose=verbose,
        )

        phase, _, _ = self.measure_phase(
            number_of_superpixels_x,
            number_of_superpixels_y,
            superpixel_width // 2,
            superpixel_height // 2,
            linear_phase_tilt,
            camera_roi_size,
            measured_intensity=intensity,
            compensate_pointing=compensate_pointing,
            lattice_phase_tilt=lattice_phase_tilt,
            lattice_superpixel_size=lattice_superpixel_size,
            lattice_roi_size=lattice_roi_size,
            verbose=verbose,
        )

        complex_amplitude = np.sqrt(intensity) * np.exp(1j * phase)

        calibration_name = (
            f"Raster Calibration - {number_of_superpixels_x}x{number_of_superpixels_y}"
        )

        if save_metadata:
            # Scan parameters, the measured intensity map, and the intensity-scan
            # images. Everything from the phase scan (camera/fitted/lattice images,
            # lattice shifts, ...) already lives in visualization_data, so it is not
            # duplicated here.
            metadata = {
                "camera_images_intensity": camera_images_intensity,
                "intensity": intensity,
                "number_of_superpixels_x": number_of_superpixels_x,
                "number_of_superpixels_y": number_of_superpixels_y,
                "superpixel_width": superpixel_width,
                "superpixel_height": superpixel_height,
                "focal_length": self.focal_length,
                "linear_phase_tilt": linear_phase_tilt,
                "camera_roi_size": camera_roi_size,
                "compensate_pointing": compensate_pointing,
                "lattice_superpixel_size": lattice_superpixel_size,
                "lattice_phase_tilt": lattice_phase_tilt,
                "normalize_power": normalize_power,
                "power_reference": self.power_reference,
            }
        else:
            metadata = {}

        return WavefrontCalibrationData(
            timestamp=datetime.now(),
            name=calibration_name,
            complex_amplitude=complex_amplitude,
            metadata=metadata,
            visualization_data=self.visualization_data,
        )
