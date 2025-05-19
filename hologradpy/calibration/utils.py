from typing import Literal
import numpy as np
from numpy.typing import NDArray
import torch

from slmsuite.hardware.slms.slm import SLM
from slmsuite.hardware.cameras.camera import Camera

from ..torch_modules.utils.optics_utils import (
    rectangular_mask,
    circular_mask,
    linear_phase
)

from ..analysis.fitting import fit_gaussian
from ..torch_modules.utils.fourier_utils import get_spatial_grid
from .. torch_modules.utils.tensor_utils import gpu_to_numpy

def get_diffraction_spot_position(
        slm: SLM,
        camera: Camera,
        linear_phase_tilt: tuple[float, float],
        exposure_time: float | None = None,
        slm_mask_diameter: float | None = None,
        camera_roi_size: tuple[int, int] | None = None,
    ) -> tuple[NDArray]:
    """
    This function generates a spot on the camera by displaying a circular 
    aperture on the SLM containing a linear phase gradient. The position of the 
    spot is found by fitting a Gaussian to the camera image.

    Parameters
    ----------
    slm : SLM
        Instance of your SLM subclass.
    camera : Camera
        Instance of your camera subclass.
    linear_phase_tilt : tuple[float, float]
        x and y gradient of the linear phase.
    exposure_time : float | None
        Exposure time in seconds. If None, the camera will perform autoexposure.
    slm_mask_diameter : float | None
        Diameter of the circular aperture in meters. If None, the diameter is 
        set to the size of the SLM.
    camera_roi_size : tuple[int, int] | None
        Width and height of the region of interest on the camera to remove 
        the zeroth-order diffraction spot. If None, the size is set to the 
        camera size.
    Returns
    -------
    tuple[float, float]
        x and y coordinates of the spot on the camera.   
    """
    if slm_mask_diameter is None:
        slm_mask_diameter = min(slm.shape) * slm.pitch_um * 1e-6
    
    if camera_roi_size is None:
        camera_roi_size = camera.shape
    
    slm_grid = get_spatial_grid(slm.shape, slm.pitch_um * 1e-6)
    camera_grid = get_spatial_grid(camera.shape, camera.pitch_um * 1e-6)

    slm_phase = linear_phase(*slm_grid, *linear_phase_tilt)
    aperture = circular_mask(*slm_grid, slm_mask_diameter / 2)

    # Mask to crop camera image (removes the zeroth-order diffraction spot)
    crop_mask = gpu_to_numpy(rectangular_mask(*camera_grid, *camera_roi_size))

    # Display phase pattern on SLM
    slm.display(gpu_to_numpy(slm_phase * aperture))

    # Perform autoexposure() on camera if exposure_time is not provided
    if exposure_time is None:
        exposure_time = camera.autoexposure(
            set_fraction=0.8,
            exposure_bounds_s=(0, 1),
            timeout_s=5,
            window=crop_mask,
            verbose=True
        )
    
    camera.set_exposure_time(exposure_time)
    camera_image = camera.get_image()

    # Fit Gaussian to camera image
    popt, _ = fit_gaussian(camera_image * crop_mask)
    spot_position_x = popt[0] + camera.shape[1] // 2
    spot_position_y = popt[1] + camera.shape[0] // 2
    return (spot_position_x, spot_position_y), camera_image