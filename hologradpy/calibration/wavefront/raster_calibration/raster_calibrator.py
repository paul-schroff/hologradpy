import numpy as np
from numpy.typing import NDArray

from slmsuite.hardware.slms.slm import SLM
from slmsuite.hardware.cameras.camera import Camera

from . import SuperpixelSlicer

from ..abstract import WavefrontCalibratorBase
from ...camera_mapping.utils import get_diffraction_spot_position
from ....propagation.utils.fourier_utils import get_spatial_grid
from ....propagation.utils.optics_utils import linear_phase

from ....propagation.utils.tensor_utils import gpu_to_numpy

from ....analysis.fitting import (
    interferometric_fringes,
    fit_interferometric_fringes,
    gaussian_beam_intensity
)


# TODO: Test this class
# TODO: Add wavefront calibration
class RasterCalibrator(WavefrontCalibratorBase):
    """
    Class to calibrate the SLM wavefront using a raster scan.
    """

    def __init__(
        self,
        slm: SLM,
        camera: Camera,
        focal_length: float,
        device: str = 'cpu'
        ) -> None:
        super().__init__(slm, camera, device)
        self.focal_length = focal_length
        self.spatial_grid_slm = self.virtual_slm.get_spatial_grid_input()
    
    def auto_camera_roi(
        self,
        linear_phase_tilt: tuple[float, float],
        roi_size: tuple[int, int],
        exposure_time: float | None = 0.1,
    ) -> tuple[tuple[float, float], NDArray[np.float_]]:
        # Find Camera position with respect to SLM
        self.camera.set_woi(None)

        spot_position, calibration_image = (
            get_diffraction_spot_position(
                self.slm,
                self.camera,
                linear_phase_tilt,
                focal_length=self.focal_length,
                exposure_time=exposure_time,
            )
        )

        spot_position_pixels = tuple(
            int(
                spot_position[i] / (self.camera.pitch_um[i] * 1e-6)
                + self.camera.shape[::-1][i] // 2
            )
            for i in range(2)
        )

        print(
            f"Auto camera ROI - Spot position (x, y): "
            f"({spot_position[0]}, {spot_position[1]}) m."
        )
        print(
            f"Auto camera ROI - Spot position (x, y): "
            f"({spot_position_pixels[0]}, {spot_position_pixels[1]}) px."
        )

        woi = [
            spot_position_pixels[0] - roi_size[1] // 2,
            roi_size[1],
            spot_position_pixels[1] - roi_size[0] // 2,
            roi_size[0],
        ]
        self.camera.set_woi(woi)
        return spot_position_pixels, calibration_image

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

    def measure_intensity(
        self,
        number_of_superpixels_x: int,
        number_of_superpixels_y: int,
        superpixel_width: int,
        superpixel_height: int,
        linear_phase_tilt: tuple[float, float],
        camera_roi_size: tuple[int, int],
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
            camera_roi_size : tuple[int, int]
                Width and height of the region of interest on the camera to 
                remove the zeroth-order diffraction spot.
        Returns:
            superpixel_intensity : NDArray
                Intensity of the superpixels.
        """
        spot_position_x, spot_position_y = (
            self.auto_camera_roi(linear_phase_tilt, camera_roi_size)[0]
        )

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

        # Weights array to handle overlapping superpixels
        weights = np.zeros(self.slm.shape)

        # Display central sub-aperture on SLM and check if camera is 
        # over-exposed.
        slm_phase_central_superpixel = np.zeros(self.slm.shape)
        slm_phase_central_superpixel[slicer.central_slice] = (
            linear_slm_phase[slicer.central_slice]
        )

        self.slm.set_phase(slm_phase_central_superpixel)

        # Find camera exposure time
        exposure_time = self.camera.autoexposure(
            set_fraction=0.7,
            exposure_bounds_s=(0, 1),
            timeout_s=1, #TODO: set to a more reasonable time
        )

        camera_images = np.zeros(
            (slicer.number_of_superpixels, *camera_roi_size)
        )
        superpixel_power = np.zeros(self.slm.shape)

        # Take camera images
        for i, superpixel_slice in enumerate(slicer.slices):
            masked_phase = np.zeros(self.slm.shape)

            superpixel_slice = slicer.get_slice(i)

            masked_phase[superpixel_slice] = linear_slm_phase[superpixel_slice]
            
            self.slm.set_phase(masked_phase)

            camera_images[i, ...] = self.camera.get_image(exposure_time)

            weights[superpixel_slice] += 1

            superpixel_power[superpixel_slice] += (
                np.sum(camera_images[i, ...]) / 
                (np.size(camera_images[i, ...]) * exposure_time)
            )
            print(
                f"Superpixel {i + 1}/{slicer.number_of_superpixels} "
                f"({100 * (i + 1) / slicer.number_of_superpixels:.2f}%)"
            )

        # Find SLM intensity profile
        weights[weights == 0] = 1
        superpixel_intensity = superpixel_power / weights
        return superpixel_intensity, camera_images
    
    # TODO: Add optical lattice to compensate for beam pointing instability.
    def measure_phase(
        self,
        number_of_superpixels_x: int,
        number_of_superpixels_y: int,
        superpixel_width: int,
        superpixel_height: int,
        linear_phase_tilt: tuple[float, float],
        camera_roi_size: tuple[int, int],
        measured_intensity: NDArray[np.float_] | None = None,
        verbose: bool = True,
        ) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
        """ This function measures the constant phase at the SLM by displaying 
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
        camera_roi_size : tuple[int, int]
            Width and height of the region of interest on the camera to 
            remove the zeroth-order diffraction spot.
        verbose : bool, optional
            If True, prints the progress of the measurement. Default is True.

        Returns
        -------
        superpixel_phase : NDArray
            Phase of the superpixels.
        camera_images : NDArray
            Camera images.
        """
        if measured_intensity is None:
            intensity = np.ones(self.slm.shape)
        else:
            beam_radius, shift_x, shift_y = (
                self.fit_gaussian_beam(measured_intensity)
            )
            intensity = gpu_to_numpy(gaussian_beam_intensity(
                *self.spatial_grid_slm,
                beam_radius,
                shift_x,
                shift_y
            ))
        
        spot_position_x, spot_position_y = (
            self.auto_camera_roi(linear_phase_tilt, camera_roi_size)[0]
        )
        
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
        
        linear_slm_phase = self.get_blazed_grating(linear_phase_tilt)

        # Weights array to handle overlapping superpixels
        weights = np.zeros(self.slm.shape)
        superpixel_phase = np.zeros(self.slm.shape)

        # Display central sub-aperture on SLM and check if camera is
        # over-exposed.
        reference_superpixel_phase = np.zeros(self.slm.shape)
        reference_superpixel_phase[slicer.reference_slice] = (
            linear_slm_phase[slicer.reference_slice]
        )

        reference_superpixel_center_x = (
            slicer.reference_slice[1].start + slicer.reference_slice[1].stop
        ) / 2
        reference_superpixel_center_y = (
            slicer.reference_slice[0].start + slicer.reference_slice[0].stop
        ) / 2

        test_index = slicer.reference_index - 2
        test_slice = slicer.slices[test_index]
        exposure_test_phase = np.copy(reference_superpixel_phase)
        exposure_test_phase[test_slice] = linear_slm_phase[test_slice]

        self.slm.set_phase(exposure_test_phase)

        # Find camera exposure time
        exposure_time = self.camera.autoexposure(
            set_fraction=0.7, exposure_bounds_s=(0, 1)
        )
        
        camera_images = np.zeros(
            (len(slicer.slices), *camera_roi_size)
        )
        fitted_images = np.zeros(
            (len(slicer.slices), *camera_roi_size)
        )
        fitting_grid = [
            gpu_to_numpy(i) for i in get_spatial_grid(
            (self.camera.woi[3], self.camera.woi[1]),
            self.camera.pitch_um * 1e-6
            )
        ]
        
        # Take camera images
        fitted_phase = 0
        for i, superpixel_slice in enumerate(slicer.slices):
            masked_phase = np.copy(reference_superpixel_phase)

            masked_phase[superpixel_slice] = linear_slm_phase[superpixel_slice]
            
            self.slm.set_phase(masked_phase)

            camera_images[i, ...] = self.camera.get_image(exposure_time)

            superpixel_center_x = (
                superpixel_slice[1].start + superpixel_slice[1].stop
                ) / 2
            superpixel_center_y = (
                superpixel_slice[0].start + superpixel_slice[0].stop
                ) / 2
            
            superpixel_separation_x = (
                superpixel_center_x - reference_superpixel_center_x
            ) * self.slm.pitch_um[1] * 1e-6
            superpixel_separation_y = (
                superpixel_center_y - reference_superpixel_center_y
            ) * self.slm.pitch_um[0] * 1e-6

            amplitude_guess = np.max(camera_images[i, ...]) / np.sqrt(2)

            popt, _ = fit_interferometric_fringes(
                *fitting_grid,
                camera_images[i, ...],
                superpixel_separation_x,
                superpixel_separation_y,
                2 * np.pi / (self.slm.wav_um * 1e-6),
                self.focal_length,
                phase_guess=0,
                amplitude_guess=amplitude_guess,
            )

            fitted_phase = popt[0]
            if superpixel_slice == slicer.reference_slice:
                fitted_phase = 0

            fitted_images[i, ...] = interferometric_fringes(
                *fitting_grid,
                superpixel_separation_x,
                superpixel_separation_y,
                2 * np.pi / (self.slm.wav_um * 1e-6),
                self.focal_length,
                *popt
            )

            weights[superpixel_slice] += 1
            superpixel_phase[superpixel_slice] += fitted_phase

            print(
                f"Superpixel {i + 1}/{slicer.number_of_superpixels} "
                f"({100 * (i + 1) / slicer.number_of_superpixels:.2f}%)"
            )
        weights[weights == 0] = 1
        phase = superpixel_phase / weights
        return phase, camera_images, fitted_images





            