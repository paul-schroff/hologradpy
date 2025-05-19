import numpy as np
from numpy.typing import NDArray

from slmsuite.hardware.slms.slm import SLM
from slmsuite.hardware.cameras.camera import Camera

from .abstract import WavefrontCalibratorBase
from ..utils import get_diffraction_spot_position
from ...torch_modules.utils.optics_utils import (
    linear_phase,
    quadratic_phase,
)


class SuperpixelSlicer:
    def __init__(
            self,
            number_of_superpixels_x: int,
            number_of_superpixels_y: int,
            superpixel_width: int,
            superpixel_height: int,
            start_index_x: int = 0,
            start_index_y: int = 0,
            end_index_x: int = None,
            end_index_y: int = None
        ) -> list[NDArray[np.int_],
                  NDArray[np.int_],
                  NDArray[np.int_],
                  NDArray[np.int_]]:
        # TODO: Make this function less convoluted: The same can be achieved 
        # with fewer lines of code.
        start_indices_x = (
            np.floor(np.linspace(
                start_index_x, end_index_x - superpixel_width,
                number_of_superpixels_x
                )
            ).astype('int')
        )
        
        end_indices_x = start_indices_x + superpixel_width

        self.start_indices_x = np.tile(
            start_indices_x,
            number_of_superpixels_y
        )
        self.end_indices_x = np.tile(
            end_indices_x,
            number_of_superpixels_y
        )
        
        start_indices_y = (
            np.floor(np.linspace(
                start_index_y, end_index_y - superpixel_height,
                number_of_superpixels_y
                )
            ).astype('int')
        )

        end_indices_y = start_indices_y + superpixel_height

        self.start_indices_y = np.repeat(
            start_indices_y,
            number_of_superpixels_x
        )
        self.end_indices_y = np.repeat(
            end_indices_y,
            number_of_superpixels_x
        )
    
    def get_slice(self, superpixel_index: int) -> tuple[slice, slice]:
        superpixel_slice =  (
            slice(
                self.start_indices_y[superpixel_index], 
                self.end_indices_y[superpixel_index]
            ),
            slice(
                self.start_indices_x[superpixel_index],
                self.end_indices_x[superpixel_index]
            )
        )
        return superpixel_slice


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
            device: str = 'cpu'):
        super().__init__(slm, camera, device)
        self.focal_length = focal_length
        self.spatial_grid_slm = self.virtual_slm.get_spatial_grid_input()

    def measure_slm_intensity(
            self,
            number_of_superpixels_x: int,
            number_of_superpixels_y: int,
            superpixel_width: int,
            superpixel_height: int,
            linear_phase_tilt: tuple[float, float],
            camera_roi_size: tuple[int, int],
        ):
        """
        This function measures the intensity profile of the laser beam incident 
        onto the SLM by displaying a sequence of rectangular phase masks on the 
        SLM. The phase mask contains a linear phase which creates a diffraction 
        spot on the camera. The position of the phase mask is varied across the 
        entire area of the SLM and the intensity of each diffraction spot is 
        measured using the camera. Read the SI of 
        https://doi.org/10.1038/s41598-023-30296-6 for details.

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
        Returns
        -------
        superpixel_intensity : NDArray
            Intensity of the superpixels.
        """
        number_of_superpixels = (
            number_of_superpixels_x * number_of_superpixels_y
        )

        number_of_pixels = np.min(self.slm.shape)

        linear_slm_phase = linear_phase(
            *self.spatial_grid_slm,
            *linear_phase_tilt,
            tilt_units='metres',
            focal_length=self.focal_length
        )

        slicer = SuperpixelSlicer(
            number_of_superpixels_x,
            number_of_superpixels_y,
            superpixel_width,
            superpixel_height,
            start_index_x=0,
            start_index_y=0,
            end_index_x=number_of_pixels,
            end_index_y=number_of_pixels,
        )

        # Display central sub-aperture on SLM and check if camera is 
        # over-exposed.
        central_index = (
            number_of_superpixels_y // 2 * number_of_superpixels_x + 
            number_of_superpixels_x // 2
        )
        central_slice = slicer.get_slice(central_index)

        slm_phase_central_superpixel = (
            np.zeros((number_of_pixels, number_of_pixels))
        )
        slm_phase_central_superpixel[central_slice] = (
            linear_slm_phase[central_slice]
        )

        self.slm.display(slm_phase_central_superpixel)

        # Find camera exposure time
        exposure_time = self.camera.autoexposure(
            set_fraction=0.7,
            exposure_bounds_s=(0, 1)
        )
        
        # Find Camera position with respect to SLM
        (spot_position_x, spot_position_y), calibration_image = (
            get_diffraction_spot_position(
                self.slm,
                self.camera,
                linear_phase_tilt,
                exposurre_time=None,
                camera_roi_size=camera_roi_size
            )
        )

        # Take camera images
        woi = [
            int(spot_position_x),
            camera_roi_size[0],
            int(spot_position_y),
            camera_roi_size[1],
        ]

        self.camera.set_woi(woi)

        camera_images = np.zeros((woi[1], woi[0], number_of_superpixels))
        superpixel_power = np.zeros(number_of_superpixels)

        for i in range(number_of_superpixels):
            masked_phase = np.zeros(self.slm.shape)

            superpixel_slice = slicer.get_slice(i)

            masked_phase[superpixel_slice] = linear_slm_phase[superpixel_slice]
            
            self.slm.display(masked_phase)

            camera_images[..., i] = self.camera.get_image(exposure_time)
            
            superpixel_power[i] = (
                np.sum(camera_images[..., i]) / 
                (np.size(camera_images[..., i]) * exposure_time)
            )

        # Find SLM intensity profile
        superpixel_intensity = np.reshape(
            superpixel_power,
            (number_of_superpixels_y, number_of_superpixels_x)
        )
        return superpixel_intensity
            