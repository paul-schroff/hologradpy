"""Shared spot-detection and spot-localization helpers for the calibrators.

These are used by both the camera-mapping and wavefront calibrators, so they live at the
``calibration`` level rather than inside ``camera_mapping``.
"""

from __future__ import annotations

import warnings
from typing import Literal, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter
from scipy.stats import norm

if TYPE_CHECKING:
    from .camera_mapping.mapping import CameraMapping

from ..hardware import Camera, SLM, as_camera, as_slm

from ..profiles.phase import linear_phase
from ..profiles.amplitude import (
    get_focal_spot_radius,
)
from ..profiles.masks import circular_mask

from ..analysis.fitting import fit_gaussian_beam_intensity
from ..grids import get_spatial_grid
from ..utils import gpu_to_numpy
from ..roi import ROI

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
    """Whether ``image`` holds a peak prominent enough to be a real signal.

    The peak must
    rise above the background by both ``signal_to_noise_ratio`` noise sigma and
    ``lower_relative_intensity_threshold`` of the camera's full-scale value. Unlike
    ``detect_spot``, this makes no single-spot assumption, so it also suits a multi-spot
    array (e.g. confirming an autoexposed calibration array did not simply rail on read
    noise).

    Args:
        image: Captured camera frame.
        camera: Supplies the full-scale pixel value (``camera.adu_levels``). Only
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
        >= lower_relative_intensity_threshold * float(camera.adu_levels)
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
        camera: Supplies the pixel pitch (``camera.pixel_size``) and the full-scale
            pixel value (``camera.adu_levels``); only read, never captured from or
            mutated.
        signal_to_noise_ratio: Peak must exceed the background by this many noise sigma.
        lower_relative_intensity_threshold: Peak must also reach this fraction of the
            camera's full-scale value.

    Returns:
        The ``(row, column)`` of the spot peak, or ``None`` if no spot is found.
    """
    pixel_pitch = min(camera.pixel_size)
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

# TODO: Should go to profiles/maks.py
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
            camera image before the fit (passed to ROI.detect).
        roi_threshold : float
            Fraction of the peak intensity used to detect the spot region of
            interest (passed to ROI.detect).
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

    slm = as_slm(slm)
    camera = as_camera(camera)

    # Capture the whole frame so the spot can be located anywhere on the sensor.
    camera.set_roi(None)

    if slm_mask_diameter is None:
        slm_mask_diameter = min(
            slm.resolution[i] * slm.pixel_size[i] for i in range(2)
        )

    slm_grid = get_spatial_grid(slm.resolution, slm.pixel_size)

    slm_phase = linear_phase(
        *slm_grid,
        *linear_phase_tilt,
        focal_length=focal_length,
        wavenumber=2 * np.pi / slm.wavelength,
    )

    aperture = circular_mask(*slm_grid, slm_mask_diameter / 2)

    # Display phase pattern on SLM
    slm.set_phase(gpu_to_numpy(slm_phase * aperture))

    # Perform autoexposure() on camera if exposure_time is not provided
    if exposure_time is None:
        exposure_time = camera.autoexpose(
            set_fraction=0.8,
            exposure_bounds=(0, 1),
            max_iterations=10,
            roi=None,
            verbose=verbose,
        )

    camera.set_exposure(exposure_time)
    camera_image = camera.get_image()

    # Crop to a region of interest around the spot before fitting, so the
    # Gaussian fit runs on a small image instead of the whole sensor. The grid
    # is cropped with the same ROI, so it keeps full-sensor coordinates and the
    # fitted position is already referenced to the full sensor.
    roi = ROI.detect(camera_image, threshold=roi_threshold, pad=roi_pad)
    cropped_camera_image = roi.crop(camera_image)

    camera_grid = get_spatial_grid(camera.resolution, camera.pixel_size)
    cropped_grid = [roi.crop(grid) for grid in camera_grid]

    focal_spot_radius_guess = get_focal_spot_radius(
        beam_radius=slm_mask_diameter / 2,
        wavelength=slm.wavelength,
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
        pixel_size = camera.pixel_size  # (y, x) metres
        height, width = camera.resolution
        position = (
            int(position[0] / pixel_size[1] + width // 2),  # x uses width pitch
            int(position[1] / pixel_size[0] + height // 2),  # y uses height pitch
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


def _clamped_roi(
    centre: tuple[float, float], size: tuple[int, int], sensor: tuple[int, int]
) -> ROI:
    """A box of ``size`` around ``centre`` in ``(row, column)``, kept on the sensor.

    Rounded before :meth:`ROI.centered`, which truncates, then clamped so a spot near
    an edge still yields a full-size box rather than one that runs off the sensor.
    """
    height, width = size
    sensor_height, sensor_width = sensor
    box = ROI.centered((round(centre[0]), round(centre[1])), (height, width))
    return ROI(
        max(0, min(box.top_row, sensor_height - height)),
        max(0, min(box.left_column, sensor_width - width)),
        height,
        width,
    )


def _meter_and_capture(
    camera: Camera, roi: ROI, set_fraction: float
) -> NDArray[np.float_]:
    """One frame, metered on ``roi``, with the previous exposure put back afterwards."""
    previous_exposure: float = camera.get_exposure()
    try:
        camera.autoexpose(set_fraction=set_fraction, roi=roi)
        return np.asarray(camera.get_image(), dtype=float)
    finally:
        camera.set_exposure(previous_exposure)


def _brightest_pixel(
    image: NDArray[np.float_],
    exclude: tuple[float, float] | None = None,
    exclude_radius: float = 0.0,
) -> tuple[int, int]:
    """The brightest ``(row, column)``, optionally ignoring a disc.

    Smoothed with a 3x3 mean first, so a single hot pixel cannot claim the peak.
    """
    smoothed = uniform_filter(image, size=3)
    if exclude is not None and exclude_radius > 0.0:
        rows, columns = np.indices(image.shape)
        blocked = (rows - exclude[0]) ** 2 + (columns - exclude[1]) ** 2 <= (
            exclude_radius**2
        )
        smoothed = np.where(blocked, -np.inf, smoothed)
    peak = np.unravel_index(int(np.argmax(smoothed)), image.shape)
    return (int(peak[0]), int(peak[1]))


def tilt_to_sensor_centre(
    camera: Camera, camera_mapping: CameraMapping
) -> tuple[float, float]:
    """The focal-plane tilt ``(x, y)`` in metres that steers a spot to the sensor
    centre. The tilt is a displacement in the model plane, which is rotated and scaled
    with respect to the sensor.

    Args:
        camera: The camera, for its resolution and pitch.
        camera_mapping: The fitted camera mapping, for its affine and its
            ``zeroth_order_position``.

    Returns:
        tuple[float, float]: ``(tilt_x, tilt_y)`` in metres, ready for
        :func:`~hologradpy.profiles.phase.linear_phase` with ``tilt_units="metres"``.
    """
    sensor_height, sensor_width = tuple(camera.resolution)
    zeroth = camera_mapping.zeroth_order_position  # (row, column) in camera pixels

    # The mapping works in (x, y), so both points are flipped out of (row, column).
    mapped = camera_mapping.affine.transform_points(
        [
            (float(zeroth[1]), float(zeroth[0])),
            (sensor_width / 2.0, sensor_height / 2.0),
        ]
    )
    displacement = mapped[1] - mapped[0]  # model pixels, (x, y)

    pitch_y, pitch_x = (float(pitch) for pitch in camera.pixel_size)
    return (float(displacement[0]) * pitch_x, float(displacement[1]) * pitch_y)


def capture_focal_spot(
    slm: SLM,
    camera: Camera,
    camera_mapping: CameraMapping,
    focal_length: float,
    kernel_size: int | tuple[int, int],
    set_fraction: float = 0.8,
    search_factor: float = 3.0,
) -> NDArray[np.float_]:
    """Capture the focal spot, steered to the middle of the sensor, in amplitude.

    This is the point spread function itself, so it is the natural seed for a
    :class:`~hologradpy.optics.modules.slm_fields.PSFSLMField`: it carries whatever
    aberration is actually present, where a Gaussian of the fitted waist carries only
    the width.

    A linear phase steers the spot to the sensor centre rather than leaving it wherever
    the zeroth order happens to land.

    Args:
        slm: The SLM, which is given the steering tilt.
        camera: The camera to capture from.
        camera_mapping: The fitted mapping, which sets the steering tilt.
        focal_length: Fourier lens focal length in metres, which converts the tilt into
            a phase ramp.
        kernel_size: Crop side in camera pixels, as an int or ``(height, width)``.
        set_fraction: Fraction of full scale to meter the spot to.
        search_factor: How many kernel widths wide to meter and search over before
            falling back to the whole sensor.

    Returns:
        NDArray: The cropped amplitude, ``sqrt`` of the background-subtracted counts,
        shaped ``kernel_size``.

    Raises:
        RuntimeError: If no spot is found anywhere on the sensor.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    height, width = int(kernel_size[0]), int(kernel_size[1])

    sensor = tuple(camera.resolution)
    centre = (sensor[0] / 2.0, sensor[1] / 2.0)

    tilt = tilt_to_sensor_centre(camera, camera_mapping)
    slm_grid = get_spatial_grid(slm.resolution, slm.pixel_size)
    # Over the full aperture, unlike get_diffraction_spot_position: the seed has to be
    # the point spread function of the whole SLM, and an aperture would broaden it.
    slm.set_phase(
        gpu_to_numpy(
            linear_phase(
                *slm_grid,
                *tilt,
                focal_length=focal_length,
                wavenumber=2 * np.pi / slm.wavelength,
            )
        )
    )

    search = _clamped_roi(
        centre,
        (
            min(int(search_factor * height), sensor[0]),
            min(int(search_factor * width), sensor[1]),
        ),
        sensor,
    )

    image = _meter_and_capture(camera, search, set_fraction)

    if has_prominent_peak(search.crop(image), camera):
        found = _brightest_pixel(search.crop(image))
        found = (search.top_row + found[0], search.left_column + found[1])
    else:
        zeroth = (
            float(camera_mapping.zeroth_order_position[0]),
            float(camera_mapping.zeroth_order_position[1]),
        )
        separation = float(np.hypot(zeroth[0] - centre[0], zeroth[1] - centre[1]))
        found = _brightest_pixel(image, exclude=zeroth, exclude_radius=separation / 2)

        offset = np.hypot(found[0] - centre[0], found[1] - centre[1])
        # Re-metered around where it really is: the first exposure was set on a window
        # the spot was not in, so the frame is metered on background or saturated.
        search = _clamped_roi(found, (search.height, search.width), sensor)
        image = _meter_and_capture(camera, search, set_fraction)
        if not has_prominent_peak(search.crop(image), camera):
            raise RuntimeError(
                "No focal spot found anywhere on the sensor after steering by "
                f"{tilt[0] * 1e3:.2f} x {tilt[1] * 1e3:.2f} mm. Check the camera "
                "mapping, and that the requested tilt is within the SLM's "
                "diffraction angle."
            )
        warnings.warn(
            f"The steered focal spot landed {offset:.0f} px from the sensor centre, "
            f"outside the {search.height} x {search.width} px search window, so the "
            "whole sensor was searched. That offset is the camera mapping's error: "
            "the seed is still good, but the mapping is worth refitting.",
            stacklevel=2,
        )

    spot_roi = _clamped_roi(found, (height, width), sensor)

    crop: NDArray[np.float_] = spot_roi.crop(image)
    # A robust floor, so read-out background does not become part of the seed.
    background: float = float(np.percentile(crop, 10.0))
    return np.sqrt(np.clip(crop - background, 0.0, None))
