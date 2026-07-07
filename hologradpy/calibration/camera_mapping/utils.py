from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from slmsuite.hardware.slms.slm import SLM
from slmsuite.hardware.cameras.camera import Camera

from ...propagation.phase_profiles import linear_phase
from ...propagation.amplitude_profiles import (
    circular_mask,
    get_focal_spot_radius,
)

from ...analysis.fitting import fit_gaussian_beam_intensity
from ...propagation.fourier import get_spatial_grid
from ...utils import gpu_to_numpy, find_roi, crop_to_roi

# Analysis window half-width in focal-spot radii
_WINDOW_SPOT_RADII = 12.0

# Enclosure threshold (fraction of the peak height)
_ENCLOSURE_EDGE_FRACTION = 0.6

# 1/e^2-intensity HWHM
_HALF_MAX_RADIUS_FACTOR = np.sqrt(np.log(2) / 2)

# Median absolute deviation to Gaussian sigma
_MAD_TO_SIGMA = 1.0 / norm.ppf(0.75)

# TODO: The background noise on cameras is Possonian
def background_noise(image: NDArray) -> float:
    """Robust estimate of the background noise sigma of ``image``.

    Uses the median absolute deviation (MAD) scaled to a Gaussian standard deviation, so
    a few bright spot pixels do not inflate the estimate the way the plain standard
    deviation would.
    """
    median = float(np.median(image))
    return _MAD_TO_SIGMA * float(np.median(np.abs(image - median)))


def has_prominent_peak(
    image: NDArray,
    camera: Camera,
    signal_to_noise_ratio: float = 8.0,
    lower_relative_intensity_threshold: float = 0.1,
) -> bool:
    """Whether ``image`` holds a peak prominent enough to be real signal.

    The two spot-shape-agnostic gates shared with :func:`detect_spot`: the peak must
    rise above the background by both ``signal_to_noise_ratio`` noise sigma and
    ``lower_relative_intensity_threshold`` of the camera's full-scale value. Unlike
    ``detect_spot`` this makes no single-spot assumption, so it also suits a multi-spot
    array (e.g. confirming an autoexposed calibration array did not simply rail on read
    noise).

    Args:
        image: Captured camera frame.
        camera: Supplies the full-scale pixel value (``camera.bitresolution``). Only 
            read, never captured from or mutated.
        signal_to_noise_ratio: Peak must exceed the background by this many noise sigma.
        lower_relative_intensity_threshold: Peak must also reach this fraction of the 
            camera's full-scale value.
    """
    prominence = float(image.max()) - float(np.median(image))
    sigma = max(background_noise(image), np.finfo(float).eps)
    return (
        prominence >= signal_to_noise_ratio * sigma
        and prominence
        >= lower_relative_intensity_threshold * float(camera.bitresolution)
    )


def detect_spot(
    image: NDArray,
    spot_radius: float,
    camera: Camera,
    signal_to_noise_ratio: float = 8.0,
    lower_relative_intensity_threshold: float = 0.1,
) -> tuple[int, int] | None:
    """Locate one localized bright spot in ``image``, or return ``None``.

    A sequence of rejection tests, all sized from the physical spot radius, tell a
    genuine focal spot apart from noise, stray light or the clipped tail of an order
    sitting just off the sensor. Intended for the coarse search, where most frames
    contain no on-sensor spot.

    Args:
        image: Captured camera frame. spot_radius: Diffraction-limited focal-spot radius
        (1/e^2 intensity) in metres.
        camera: Supplies the pixel pitch (``camera.pitch_um``) and the full-scale pixel 
            value (``camera.bitresolution``); only read, never captured from or mutated.
        signal_to_noise_ratio: Peak must exceed the background by this many noise sigma.
        lower_relative_intensity_threshold: Peak must also reach this fraction of the 
            camera's full-scale value.

    Returns:
        The ``(row, column)`` of the spot peak, or ``None`` if no spot is found.
    """
    pixel_pitch = min(camera.pitch_um) * 1e-6
    spot_radius_px = spot_radius / pixel_pitch

    # Checking for prominence (peak-vs-noise SNR and peak-vs-full-scale gates).
    if not has_prominent_peak(
        image,
        camera,
        signal_to_noise_ratio=signal_to_noise_ratio,
        lower_relative_intensity_threshold=lower_relative_intensity_threshold,
    ):
        return None

    background = float(np.median(image))
    prominence = float(image.max()) - background
    row, column = np.unravel_index(int(np.argmax(image)), image.shape)

    # Making sure detected maximum is not sitting at the border of the frame.
    border_margin = max(int(round(spot_radius_px)), 1)
    if (
        min(row, image.shape[0] - 1 - row) < border_margin
        or min(column, image.shape[1] - 1 - column) < border_margin
    ):
        return None

    half_window = int(round(_WINDOW_SPOT_RADII * spot_radius_px))
    top = max(row - half_window, 0)
    left = max(column - half_window, 0)
    window = image[top:row + half_window + 1, left:column + half_window + 1]

    # Checking the spot has a reasonable size
    half_max_radius_px = _HALF_MAX_RADIUS_FACTOR * spot_radius_px
    min_core_pixels = max(int(round(0.25 * np.pi * half_max_radius_px**2)), 1)
    core = (window - background) > 0.5 * prominence
    if int(core.sum()) < min_core_pixels:
        return None

    edge_maximum = float(
        max(
            window[0, :].max(),
            window[-1, :].max(),
            window[:, 0].max(),
            window[:, -1].max(),
        )
    )
    if edge_maximum - background > _ENCLOSURE_EDGE_FRACTION * prominence:
        return None

    return int(row), int(column)


def addressable_half_extent(slm: SLM, focal_length: float) -> tuple[float, float]:
    """Half-extent ``(x, y)`` of the focal-plane region the SLM can address, in metres: 
    the first-order deflection of a grating at the SLM's Nyquist frequency, 
    ``wavelength * focal_length / (2 * pitch)`` per axis. Focal spots cannot be placed 
    beyond it (the grating would alias).
    """
    wavelength = slm.wav_um * 1e-6
    return (
        wavelength * focal_length / (2.0 * slm.pitch_um[0] * 1e-6),
        wavelength * focal_length / (2.0 * slm.pitch_um[1] * 1e-6),
    )


def disc_mask(
    shape: tuple[int, int], center: tuple[float, float], radius: float
) -> NDArray:
    """Boolean pixel mask, True inside the disc of ``radius`` around ``center``.

    Args:
        shape: Image shape ``(height, width)``.
        center: Disc centre ``(x, y)`` in pixels.
        radius: Disc radius in pixels.
    """
    rows, columns = np.indices(shape)
    return (columns - center[0]) ** 2 + (rows - center[1]) ** 2 <= radius**2


def metres_to_pixel(
    position: tuple[float, float],
    pitch: NDArray,
    camera_shape: tuple[int, int],
) -> tuple[float, float]:
    """Camera-plane ``(x, y)`` metres -> ``(x, y)`` camera pixels (full sensor).

    Args:
        position: (x, y) position in metres, relative to the sensor centre.
        pitch: Camera pixel pitch in metres, (x, y).
        camera_shape: Sensor resolution (height, width).
    """
    return (
        position[0] / pitch[0] + camera_shape[1] // 2,
        position[1] / pitch[1] + camera_shape[0] // 2,
    )


def get_diffraction_spot_position(
    slm: SLM,
    camera: Camera,
    linear_phase_tilt: tuple[float, float],
    focal_length: float,
    exposure_time: float | None = None,
    slm_mask_diameter: float | None = None,
    units: Literal["metres", "pixels"] = "metres",
    roi_pad: int = 50,
    roi_threshold: float = 0.5,
    verbose: bool = True,
) -> tuple[tuple[float, float], float, NDArray, tuple[int, int, int, int]]:
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
        units : str
            Units of the returned spot position: "metres" (default) for the
            (x, y) coordinates in the camera plane, or "pixels" for integer
            camera pixel coordinates. The focal spot radius is always in metres.
        roi_pad : int
            Padding in pixels added around the detected spot when cropping the
            camera image before the fit (passed to find_roi).
        roi_threshold : float
            Fraction of the peak intensity used to detect the spot region of
            interest (passed to find_roi).
        verbose : bool
            If True, prints progress messages to the console.

    Returns:
        tuple[tuple[float, float], float, NDArray, tuple[int, int, int, int]]
            Tuple of x and y coordinates of the spot on the full sensor (in
            metres or pixels, see ``units``), the focal spot radius in metres,
            the cropped camera image used for the fit, and the (top, bottom,
            left, right) region of interest used to crop it.
    """
    if units not in ("metres", "pixels"):
        raise ValueError(f"units must be 'metres' or 'pixels', got {units!r}.")

    # Capture the whole frame so the spot can be located anywhere on the sensor.
    camera.set_woi(None)

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

    # Crop to a region of interest around the spot before fitting, so the
    # Gaussian fit runs on a small image instead of the whole sensor. The grid
    # is cropped with the same ROI, so it keeps full-sensor coordinates and the
    # fitted position is already referenced to the full sensor.
    roi = find_roi(camera_image, threshold=roi_threshold, pad=roi_pad)
    cropped_camera_image = crop_to_roi(camera_image, roi)

    camera_grid = get_spatial_grid(camera.shape, camera.pitch_um * 1e-6)
    cropped_grid = [crop_to_roi(grid, roi) for grid in camera_grid]

    focal_spot_radius_guess = get_focal_spot_radius(
        beam_radius=slm_mask_diameter / 2,
        wavelength=slm.wav_um * 1e-6,
        focal_length=focal_length,
    )

    if verbose:
        print("Fitting Gaussian to camera image...")

    popt, _ = fit_gaussian_beam_intensity(
        *cropped_grid,
        cropped_camera_image,
        beam_radius_guess=focal_spot_radius_guess,
    )

    if verbose:
        print("Gaussian fit complete.")

    focal_spot_radius = popt[0]
    position = (popt[1], popt[2])

    if units == "pixels":
        position = tuple(
            int(
                position[i] / (camera.pitch_um[i] * 1e-6)
                + camera.shape[::-1][i] // 2
            )
            for i in range(2)
        )

    if verbose:
        if units == "pixels":
            print(f"Diffraction spot position (x, y): {position} px.")
        else:
            print(
                "Diffraction spot position (x, y): "
                f"({position[0] * 1e6:.2f}, {position[1] * 1e6:.2f}) um."
            )

    return position, focal_spot_radius, cropped_camera_image, roi


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
