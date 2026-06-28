from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from slmsuite.hardware.slms.slm import SLM
from slmsuite.hardware.cameras.camera import Camera

from scipy.ndimage import gaussian_filter

from . import SuperpixelSlicer

from ..abstract import WavefrontCalibratorBase, WavefrontCalibrationData
from ..utils import inpaint
from ....analysis.unwrapping import unwrap_nonuniform

from ...camera_mapping.utils import get_diffraction_spot_position
from ....hardware.utils import set_camera_woi
from ....propagation.fourier import get_spatial_grid
from ....propagation.phase_profiles import linear_phase

from ....utils import gpu_to_numpy, roi_bounds

from ....analysis.fitting import (
    interferometric_fringes,
    fit_interferometric_fringes,
    optical_lattice_fringes,
    fit_optical_lattice_fringes,
    gaussian_beam_intensity,
    fit_gaussian_beam_intensity,
)


# TODO: Test this class
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
    ) -> None:
        super().__init__(slm, camera, device)
        self.focal_length = focal_length

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

    def calibrate_lattice_corner_tilts(
        self,
        corner_slices: list[tuple[slice, slice]],
        lattice_phase_tilt: tuple[float, float],
        roi_size: tuple[int, int],
        exposure_time: float | None = None,
    ) -> list[tuple[float, float]]:
        """Per-corner linear phase tilts that steer all four corner superpixels
        onto the same (nominal) camera spot.

        Local SLM wavefront aberration gives each corner a slightly different
        effective tilt, so with a single shared grating the four corner beams
        land at slightly different camera positions and do not interfere
        cleanly. Each corner is displayed on its own, its diffraction spot is
        located within a window around the nominal lattice position (so the
        zeroth-order spot at the SLM centre is excluded), and the tilt that
        brings it back onto the nominal position is returned.
        """
        base_grating = self.get_blazed_grating(lattice_phase_tilt)

        # Window around the nominal lattice spot, excluding the bright
        # zeroth-order spot at the camera centre.
        nominal_pixels = tuple(
            int(
                lattice_phase_tilt[i] / (self.camera.pitch_um[i] * 1e-6)
                + self.camera.shape[::-1][i] // 2
            )
            for i in range(2)
        )
        self.camera.set_woi(
            [
                nominal_pixels[0] - roi_size[1] // 2,
                roi_size[1],
                nominal_pixels[1] - roi_size[0] // 2,
                roi_size[0],
            ]
        )

        if exposure_time is None:
            full_lattice = np.zeros(self.slm.shape)
            for corner_slice in corner_slices:
                full_lattice[corner_slice] = base_grating[corner_slice]
            self.slm.set_phase(full_lattice)
            exposure_time = self.camera.autoexposure(
                set_fraction=0.7, exposure_bounds_s=(0, 1), timeout_s=0.1
            )

        # Grid centred on the window, i.e. relative to the nominal spot.
        grid = [
            gpu_to_numpy(g)
            for g in get_spatial_grid(roi_size, self.camera.pitch_um * 1e-6)
        ]
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

            camera_image = self.camera.get_image(exposure_time)
            popt, _ = fit_gaussian_beam_intensity(
                *grid, camera_image, beam_radius_guess=spot_radius_guess
            )
            # Spot offset from the nominal position; steer it back by that much.
            offset_x, offset_y = popt[1], popt[2]
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
        """Return (superpixel_width, superpixel_height) using the closest
        divisors of the SLM dimensions to slm_size / target_superpixels."""
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
        pattern in the Fourier plane for a rectangular aperture, in camera
        pixels.
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

    def measure_intensity(
        self,
        number_of_superpixels_x: int,
        number_of_superpixels_y: int,
        superpixel_width: int,
        superpixel_height: int,
        linear_phase_tilt: tuple[float, float],
        camera_roi_size: tuple[int, int] | None = None,
        verbose: bool = True,
    ) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
        """
        This function measures the intensity profile of the laser beam incident
        onto the SLM by displaying a sequence of rectangular phase masks on the
        SLM. The phase mask contains a linear phase which creates a diffraction
        spot on the camera. The position of the phase mask is varied across the
        entire area of the SLM and the intensity of each diffraction spot is
        measured using the camera. Read the SI of
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
                x and y gradient of the linear phase in units of the resulting
                spot displacement in the Fourier plane in metres.
            camera_roi_size : tuple[int, int] | None
                Height and width of the region of interest on the camera around
                each diffraction spot. If None, it is sized automatically from
                the superpixel's sinc^2 spot (see below).
        Returns:
            superpixel_intensity : NDArray
                Intensity of the superpixels.
        """
        # Size the camera ROI from the superpixel's sinc^2 diffraction spot when
        # not given. A square aperture of side a produces a sinc^2 whose first
        # zero lies at lambda * f / a from the peak, so get_roi_size returns the
        # null-to-null central-lobe width (2 * lambda * f / a). Use a window
        # twice that wide so it comfortably contains the main lobe and the first
        # sidelobes of every superpixel's diffraction spot.
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

        spot_center, _, _, _ = get_diffraction_spot_position(
            self.slm,
            self.camera,
            linear_phase_tilt,
            focal_length=self.focal_length,
            exposure_time=0.1,
            units="pixels",
            verbose=verbose,
        )
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

        # Background: a vertical binary 0/pi grating (every other pixel column)
        # instead of a flat zero phase, so the unmodulated SLM area diffracts
        # into the edge Nyquist orders instead of a bright central zeroth-order
        # spot. Each displayed superpixel overwrites this background.
        base_phase = np.zeros(self.slm.shape)
        base_phase[:, 1::2] = np.pi

        # Weights array to handle overlapping superpixels
        weights = np.zeros(self.slm.shape)

        # Display central sub-aperture on SLM and check if camera is
        # over-exposed.
        slm_phase_central_superpixel = np.copy(base_phase)
        slm_phase_central_superpixel[slicer.central_slice] = linear_slm_phase[
            slicer.central_slice
        ]

        self.slm.set_phase(slm_phase_central_superpixel)

        # Find camera exposure time
        exposure_time = self.camera.autoexposure(
            set_fraction=0.7,
            exposure_bounds_s=(0, 1),
            timeout_s=0.1,  # TODO: set to a more reasonable time
        )

        camera_images = np.zeros(
            (
                slicer.number_of_superpixels,
                *camera_roi_size,
            )
        )
        superpixel_power = np.zeros(self.slm.shape)

        # Take camera images
        for i, superpixel_slice in enumerate(slicer.slices):
            masked_phase = np.copy(base_phase)

            superpixel_slice = slicer.get_slice(i)

            masked_phase[superpixel_slice] = linear_slm_phase[superpixel_slice]

            self.slm.set_phase(masked_phase)

            camera_images[i, ...] = self.camera.get_image(exposure_time)

            weights[superpixel_slice] += 1

            superpixel_power[superpixel_slice] += np.sum(camera_images[i, ...]) / (
                np.size(camera_images[i, ...]) * exposure_time
            )
            print(
                f"Superpixel {i + 1}/{slicer.number_of_superpixels} "
                f"({100 * (i + 1) / slicer.number_of_superpixels:.2f}%)"
            )

        # Find SLM intensity profile
        weights[weights == 0] = 1
        superpixel_intensity = superpixel_power / weights

        blur_kernel_size = max(slicer.superpixel_separation) / 2
        superpixel_intensity = gaussian_filter(
            superpixel_intensity, sigma=blur_kernel_size
        )
        return superpixel_intensity, camera_images

    def measure_phase(
        self,
        number_of_superpixels_x: int,
        number_of_superpixels_y: int,
        superpixel_width: int,
        superpixel_height: int,
        linear_phase_tilt: tuple[float, float],
        camera_roi_size: tuple[int, int] | None = None,
        measured_intensity: NDArray[np.float_] | None = None,
        compensate_pointing: bool = False,
        lattice_phase_tilt: tuple[float, float] | None = None,
        lattice_superpixel_size: int | None = None,
        lattice_roi_size: tuple[int, int] | None = None,
        verbose: bool = True,
    ) -> tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.float_]]:
        """This function measures the constant phase at the SLM by displaying
        a sequence of rectangular phase masks on the SLM. This scheme was
        adapted from Phillip Zupancic's work
        (https://doi.org/10.1364/OE.24.013881). For details of our
        implementation, see the supplementary material of
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
            Height and width of the region of interest on the camera around the
            main interference spot. If None, sized automatically to the
            superpixel's sinc^2 central lobe (out to the first zero).
        compensate_pointing : bool, optional
            If True, also display four corner superpixels forming a 2D optical
            lattice and use its phase to correct for beam pointing drift.
            Default is False.
        lattice_phase_tilt : tuple[float, float] | None, optional
            x and y gradient (metres in the Fourier plane) steering the optical
            lattice to a separate camera region. Required if
            compensate_pointing is True.
        lattice_superpixel_size : int | None, optional
            Side length [px] of the square corner superpixels. If None, sized
            automatically from measured_intensity to match the fringe
            brightness.
        lattice_roi_size : tuple[int, int] | None, optional
            Height and width of the camera region of interest around the
            optical lattice. If None, sized automatically to the corner
            superpixel's sinc^2 central lobe (out to the first zero).
        verbose : bool, optional
            If True, prints the progress of the measurement. Default is True.

        Returns
        -------
        superpixel_phase : NDArray
            Phase of the superpixels.
        camera_images : NDArray
            Camera images.
        """
        if compensate_pointing and lattice_phase_tilt is None:
            raise ValueError(
                "lattice_phase_tilt must be provided when compensate_pointing "
                "is True."
            )
        if (
            compensate_pointing
            and measured_intensity is None
            and lattice_superpixel_size is None
        ):
            raise ValueError(
                "compensate_pointing requires measured_intensity (to size the "
                "corner superpixels) or an explicit lattice_superpixel_size."
            )

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

        # TODO: Make this a grating function in propagation.phase_profiles
        base_phase = np.zeros(self.slm.shape)
        base_phase[:, 1::2] = np.pi

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

        # Set up the optical lattice: four constant corner superpixels sharing a
        # steeper grating, sized so the lattice is about as bright as the
        # interference fringes.
        if compensate_pointing:
            corner_size = lattice_superpixel_size or slicer.get_lattice_corner_size()

            if lattice_roi_size is None:
                roi_width, roi_height = self.get_roi_size(corner_size, corner_size)
                lattice_roi_size = (roi_height, roi_width)
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
            # Local SLM aberration gives each corner a slightly different
            # effective tilt, so steer each corner individually onto the common
            # nominal lattice spot, so the four beams interfere cleanly.
            corner_tilts = self.calibrate_lattice_corner_tilts(
                corner_slices, lattice_phase_tilt, lattice_roi_size
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
            lattice_center, _, _, _ = get_diffraction_spot_position(
                self.slm,
                self.camera,
                lattice_phase_tilt,
                focal_length=self.focal_length,
                exposure_time=0.1,
                units="pixels",
                verbose=verbose,
            )

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

        # Pick a second superpixel (neighbour of the reference within the kept
        # slices) so the exposure test sees two-beam interference. Indexing the
        # reduced slice list directly avoids running off the end when slices are
        # removed (intensity compensation or corner exclusion).
        reference_position = slicer.slices.index(slicer.reference_slice)
        test_position = (reference_position + 1) % len(slicer.slices)
        test_slice = slicer.slices[test_position]
        exposure_test_phase = np.copy(reference_superpixel_phase)
        exposure_test_phase[test_slice] = linear_slm_phase[test_slice]

        self.slm.set_phase(exposure_test_phase)

        # Find camera exposure time
        # TODO: Set timeout_s to a more reasonable value.
        exposure_time = self.camera.autoexposure(
            set_fraction=0.7, exposure_bounds_s=(0, 1), timeout_s=0.1
        )

        camera_images = np.zeros((len(slicer.slices), *camera_roi_size))
        fitted_images = np.zeros((len(slicer.slices), *camera_roi_size))
        # Grid centred on the main-spot crop, preserving the original behaviour.
        main_grid = [
            gpu_to_numpy(grid)
            for grid in get_spatial_grid(
                camera_roi_size, self.camera.pitch_um * 1e-6
            )
        ]

        # TODO: This is a bit messy
        lattice_phase_x = np.zeros(slicer.number_of_superpixels)
        lattice_phase_y = np.zeros(slicer.number_of_superpixels)
        lattice_shift_x = np.zeros(slicer.number_of_superpixels)
        lattice_shift_y = np.zeros(slicer.number_of_superpixels)
        lattice_shift_x_err = np.zeros(slicer.number_of_superpixels)
        lattice_shift_y_err = np.zeros(slicer.number_of_superpixels)
        lattice_images = None
        fitted_lattice_images = None

        if compensate_pointing:
            lattice_grid = [
                gpu_to_numpy(grid)
                for grid in get_spatial_grid(
                    lattice_roi_size, self.camera.pitch_um * 1e-6
                )
            ]
            # Captured and fitted lattice ROI images, kept for troubleshooting
            # and plotting (mirrors camera_images / fitted_images).
            lattice_images = np.zeros(
                (slicer.number_of_superpixels, *lattice_roi_size)
            )
            fitted_lattice_images = np.zeros(
                (slicer.number_of_superpixels, *lattice_roi_size)
            )
            # Baseline lattice phase from the displayed reference/exposure
            # pattern (which already shows the constant lattice). This anchors
            # the measured drift to zero at the un-drifted reference state.
            baseline_image = self.camera.get_image(exposure_time)[lattice_box]
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

        # Take camera images
        fitted_phase = 0
        for i, superpixel_slice in enumerate(slicer.slices):
            masked_phase = np.copy(reference_superpixel_phase)

            masked_phase[superpixel_slice] = linear_slm_phase[superpixel_slice]

            self.slm.set_phase(masked_phase)

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

            # Measure the camera-plane displacement from beam pointing drift via
            # the optical lattice, then fit the main fringes on shifted
            # coordinates to remove it (shift stays 0 when not compensating).
            shift_x = 0.0
            shift_y = 0.0
            if compensate_pointing:
                lattice_image = full_image[lattice_box]
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
                # A pattern shift by d lowers the fitted phase by k*d, so the
                # camera-plane displacement is (phase0 - phase) / k. The main
                # fringes are then fitted on (grid - displacement).
                shift_x = (phase_x0 - phase_x_prev) / k_lattice_x
                shift_y = (phase_y0 - phase_y_prev) / k_lattice_y
                # 1-sigma fit uncertainty on the phases, propagated to the shift.
                phase_err = np.sqrt(np.diag(pcov_lattice))
                lattice_shift_x_err[i] = phase_err[0] / k_lattice_x
                lattice_shift_y_err[i] = phase_err[1] / k_lattice_y
                lattice_phase_x[i] = phase_x_prev
                lattice_phase_y[i] = phase_y_prev
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

        # Store lattice diagnostics for inspection/troubleshooting/plotting.
        self.lattice_phase_x = lattice_phase_x
        self.lattice_phase_y = lattice_phase_y
        self.lattice_shift_x = lattice_shift_x
        self.lattice_shift_y = lattice_shift_y
        self.lattice_shift_x_err = lattice_shift_x_err
        self.lattice_shift_y_err = lattice_shift_y_err
        self.lattice_images = lattice_images
        self.fitted_lattice_images = fitted_lattice_images

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

        intensity, camera_images_intensity = self.measure_intensity(
            number_of_superpixels_x,
            number_of_superpixels_y,
            superpixel_width,
            superpixel_height,
            linear_phase_tilt,
            camera_roi_size,
            verbose=verbose,
        )

        phase, camera_images_phase, fitted_images_phase = self.measure_phase(
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
        
        # TODO: Make sure these are sensible.
        if save_metadata:
            metadata = {
                "camera_images_intensity": camera_images_intensity,
                "camera_images_phase": camera_images_phase,
                "fitted_images_phase": fitted_images_phase,
                "number_of_superpixels_x": number_of_superpixels_x,
                "number_of_superpixels_y": number_of_superpixels_y,
                "superpixel_width": superpixel_width,
                "superpixel_height": superpixel_height,
                "linear_phase_tilt": linear_phase_tilt,
                "camera_roi_size": camera_roi_size,
                "compensate_pointing": compensate_pointing,
                "lattice_phase_x": self.lattice_phase_x,
                "lattice_phase_y": self.lattice_phase_y,
                "lattice_shift_x": self.lattice_shift_x,
                "lattice_shift_y": self.lattice_shift_y,
                "lattice_images": self.lattice_images,
                "fitted_lattice_images": self.fitted_lattice_images,
            }
        else:
            metadata = {}

        return WavefrontCalibrationData(
            timestamp=datetime.now(),
            name=calibration_name,
            complex_amplitude=complex_amplitude,
            metadata=metadata,
        )
