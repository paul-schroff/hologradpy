import numpy as np
from numpy.typing import NDArray

from slmsuite.hardware.slms.slm import SLM
from slmsuite.hardware.cameras.camera import Camera

from ...propagation.utils.optics_utils import (
    circular_mask,
    linear_phase,
    get_focal_spot_radius,
)

from ...analysis.fitting import fit_gaussian_beam_intensity
from ...propagation.utils.fourier_utils import get_spatial_grid
from ...propagation.utils.tensor_utils import gpu_to_numpy


# TODO: Reformat docstrings.
def get_diffraction_spot_position(
    slm: SLM,
    camera: Camera,
    linear_phase_tilt: tuple[float, float],
    focal_length: float,
    exposure_time: float | None = None,
    slm_mask_diameter: float | None = None,
    verbose: bool = True,
) -> tuple[tuple[float, float], float, NDArray]:
    """
    This function generates a spot on the camera by displaying a circular
    aperture on the SLM containing a linear phase gradient. The position of the
    spot is found by fitting a Gaussian to the camera image.

    Args:
        slm : SLM
            Instance of your SLM subclass.
        camera : Camera
            Instance of your camera subclass.
        linear_phase_tilt : tuple[float, float]
            x and y gradient of the linear phase.
        focal_length : float
            Focal length of the Fourier lens in metres.
        exposure_time : float | None
            Exposure time in seconds. If None, the camera will perform
            autoexposure.
        slm_mask_diameter : float | None
            Diameter of the circular aperture in meters. If None, the diameter
            is set to the size of the SLM.
        verbose : bool
            If True, prints progress messages to the console.

    Returns:
        tuple[tuple[float, float], float, NDArray]
            Tuple of x and y coordinates of the spot on the camera in metres,
            the focal spot radius in metres, and captured camera image.
    """
    if slm_mask_diameter is None:
        slm_mask_diameter = min(
            [slm.shape[i] * slm.pitch_um[i] * 1e-6 for i in range(2)]
        )

    slm_grid = get_spatial_grid(slm.shape, slm.pitch_um * 1e-6)

    slm_phase = linear_phase(
        *slm_grid,
        *linear_phase_tilt,
        focal_length=focal_length,
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
            verbose=verbose,
        )

    camera.set_exposure(exposure_time)
    camera_image = camera.get_image()

    camera_grid = get_spatial_grid(camera.shape, camera.pitch_um * 1e-6)

    # Fit Gaussian intensity profile to camera image
    focal_spot_radius_guess = get_focal_spot_radius(
        beam_radius=slm_mask_diameter / 2,
        wavelength=slm.wav_um * 1e-6,
        focal_length=focal_length,
    )

    if verbose:
        print("Fitting Gaussian to camera image...")

    popt, _ = fit_gaussian_beam_intensity(
        *camera_grid, camera_image, beam_radius_guess=focal_spot_radius_guess
    )

    if verbose:
        print("Gaussian fit complete.")

    focal_spot_radius = popt[0]
    shift_x, shift_y = popt[1:3]

    return (shift_x, shift_y), focal_spot_radius, camera_image
