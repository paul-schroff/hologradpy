from typing import Literal
import numpy as np
from numpy.typing import NDArray
import torch

import matplotlib.pyplot as plt

from slmsuite.hardware.slms.slm import SLM
from slmsuite.hardware.cameras.camera import Camera

from ..torch_modules.utils.optics_utils import (
    rectangular_mask,
    circular_mask,
    linear_phase,
    focal_spot_radius
)

from ..analysis.fitting import fit_gaussian_beam_intensity
from ..torch_modules.utils.fourier_utils import get_spatial_grid
from .. torch_modules.utils.tensor_utils import gpu_to_numpy

def get_diffraction_spot_position(
    slm: SLM,
    camera: Camera,
    linear_phase_tilt: tuple[float, float],
    focal_length: float,
    device: str = 'cpu',
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
        slm_mask_diameter = min(slm.shape) * slm.pitch_um[0] * 1e-6
    
    slm_grid = get_spatial_grid(
        slm.shape, slm.pitch_um * 1e-6, device=device
    )

    slm_phase = linear_phase(
        *slm_grid, *linear_phase_tilt, focal_length=focal_length,
        wavenumber=2 * np.pi / (slm.wav_um * 1e-6),
    )

    aperture = circular_mask(*slm_grid, slm_mask_diameter / 2)

    # Display phase pattern on SLM
    slm.set_phase(gpu_to_numpy(slm_phase * aperture))

    # Perform autoexposure() on camera if exposure_time is not provided
    if exposure_time is None:
        exposure_time = camera.autoexposure(
            set_fraction=0.8,
            exposure_bounds_s=(0, 1),
            timeout_s=10,
            window=None,
            verbose=True
        )
    
    camera.set_exposure(exposure_time)
    camera_image = camera.get_image()

    pixel_grid = np.meshgrid(
        np.arange(camera.shape[1]), np.arange(camera.shape[0])
    )

    # Fit Gaussian intensity profile to camera image
    beam_radius_guess = focal_spot_radius(
        beam_radius=slm_mask_diameter / 2,
        wavelength=slm.wav_um * 1e-6,
        focal_length=focal_length
    )

    print("Fitting Gaussian to camera image...")
    popt, _ = fit_gaussian_beam_intensity(
        *pixel_grid, camera_image, beam_radius_guess=beam_radius_guess
    )
    shift_x, shift_y = popt[1:3]
    print("Gaussian fit complete.")

    return (shift_x, shift_y), camera_image