import numpy as np
from numpy.typing import NDArray

from slmsuite.hardware.slms.slm import SLM
from slmsuite.hardware.cameras.camera import Camera

from ...propagation.phase_profiles import linear_phase
from ...propagation.amplitude_profiles import (
    circular_mask,
    get_focal_spot_radius,
)

from ...analysis.fitting import fit_gaussian_beam_intensity
from ...propagation.fourier import get_spatial_grid
from ...utils import gpu_to_numpy


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


def checkerboard(
    resolution: tuple[int, int],
    number_of_squares: tuple[int, int] = None,
    square_size: int = None,
    shift_x: int = 0,
    shift_y: int = 0,
    dark_square_brightness: float = 0.0,
    light_square_brightness: float = 1.0,
) -> NDArray[np.float_]:
    """
    Creates a checkerboard pattern with specified resolution, square size,
    number of squares, and brightness values for dark and light squares.

    Args:
        resolution (tuple[int, int]): Resolution of the output image in
            pixels in the format (height, width).
        number_of_squares (tuple[int, int]): Number of squares along
            (y, x) axes. Overrides square_size.
        square_size (int, optional): Size of each square in pixels. If None,
            calculated from number_of_squares and resolution.
        shift_x (int, optional): Horizontal shift of the pattern. Defaults to
            0.
        shift_y (int, optional): Vertical shift of the pattern. Defaults to 0.
        dark_square_brightness (float, optional): Value for dark square
            brightness. Defaults to 0.0.
        light_square_brightness (float, optional): Value for light square
            brightness. Defaults to 1.0.

    Returns:
        NDArray[np.float_]: A 2D array with the checkerboard pattern.
    """
    height, width = resolution
    number_of_squares_y, number_of_squares_x = number_of_squares

    if square_size is None:
        square_size = min(width // number_of_squares_y, height // number_of_squares_x)

    checkerboard_resolution = (
        number_of_squares_y * square_size,
        number_of_squares_x * square_size,
    )

    y, x = np.indices(checkerboard_resolution)

    cb = np.where(
        ((x // square_size) + (y // square_size)) % 2 == 0,
        dark_square_brightness,
        light_square_brightness,
    )

    # Distribute the padding asymmetrically (before gets the floor, after the
    # remainder) so the output is always exactly `resolution`. Symmetric
    # padding loses a pixel whenever resolution - checkerboard_resolution is
    # odd, which happens e.g. for odd square_size.
    pad_y_total = resolution[0] - checkerboard_resolution[0]
    pad_x_total = resolution[1] - checkerboard_resolution[1]
    pad_y_before = pad_y_total // 2
    pad_x_before = pad_x_total // 2
    pad_y_after = pad_y_total - pad_y_before
    pad_x_after = pad_x_total - pad_x_before

    return np.pad(
        cb,
        (
            (pad_y_before + shift_y, pad_y_after - shift_y),
            (pad_x_before + shift_x, pad_x_after - shift_x),
        ),
    )
