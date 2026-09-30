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
from ..fourier_optics import get_focal_spot_radius
from ..profiles.masks import circular_mask

from ..analysis.fitting import fit_gaussian_beam_intensity
from ..grids import get_spatial_grid, metres_to_pixel
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
def background_noise(image: NDArray, mask: NDArray[np.bool_] | None = None) -> float:
    """Robust estimate of the background noise sigma of ``image``.

    Uses the median absolute deviation (MAD), scaled to a Gaussian standard deviation,
    so a few bright spot pixels do not inflate the estimate.

    Args:
        image: The frame.
        mask: True at the pixels to estimate from, in the shape of ``image``. Every
            pixel is used when None.
    """
    pixels = _measured_pixels(image, mask)
    median = float(np.median(pixels))
    return _MAD_TO_SIGMA * float(np.median(np.abs(pixels - median)))


def peak_prominence(image: NDArray, mask: NDArray[np.bool_] | None = None) -> float:
    """How far the brightest pixel of ``image`` rises above the background level, in
    counts. The background level is the median pixel value.

    Args:
        image: The frame.
        mask: True at the pixels to measure, in the shape of ``image``. Every pixel is
            measured when None.
    """
    pixels = _measured_pixels(image, mask)
    return float(np.max(pixels)) - float(np.median(pixels))


def _measured_pixels(image: NDArray, mask: NDArray[np.bool_] | None) -> NDArray:
    """The pixels of ``image`` where ``mask`` is True, or every pixel for None.

    Raises:
        ValueError: ``mask`` is not the shape of ``image``, or keeps no pixel.
    """
    image = np.asarray(image)
    if mask is None:
        return image
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != image.shape:
        raise ValueError(
            f"The mask has shape {mask.shape}, and the image is {image.shape}."
        )
    if not mask.any():
        raise ValueError("The mask keeps no pixel of the image.")
    return image[mask]


def has_prominent_peak(
    image: NDArray,
    camera: Camera,
    signal_to_noise_ratio: float = 8.0,
    lower_relative_intensity_threshold: float = 0.1,
    *,
    mask: NDArray[np.bool_] | None = None,
) -> bool:
    """Whether ``image`` holds a peak prominent enough to be a real signal.

    The peak must rise above the background by both ``signal_to_noise_ratio`` noise
    sigma and ``lower_relative_intensity_threshold`` of the camera's full-scale value.
    ``detect_spot`` assumes a single spot. This test makes no such assumption, so it
    also suits a multi-spot array. For example, it checks that an autoexposed
    calibration array did not simply rail on read noise.

    Args:
        image: Captured camera frame.
        camera: Supplies the full-scale pixel value (``camera.max_pixel_value``). Only
            read, never captured from or mutated. A driver that
            :func:`~hologradpy.hardware.as_native.as_camera` wraps is accepted.
        signal_to_noise_ratio: Peak must exceed the background by this many noise sigma.
        lower_relative_intensity_threshold: Peak must also reach this fraction of the
            camera's full-scale value.
        mask: True at the pixels to measure, in the shape of ``image``. The peak, the
            background and the noise are taken from these pixels only. Every pixel is
            measured when None.
    """
    camera = as_camera(camera)
    prominence = peak_prominence(image, mask)
    sigma = max(background_noise(image, mask), np.finfo(float).eps)
    return (
        prominence >= signal_to_noise_ratio * sigma
        and prominence
        >= lower_relative_intensity_threshold * float(camera.max_pixel_value)
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
        image: Captured camera frame.
        spot_radius: Diffraction-limited focal-spot radius (1/e^2 intensity) in metres.
        camera: Supplies the pixel pitch (``camera.pixel_size``) and the full-scale
            pixel value (``camera.max_pixel_value``), only read, never captured from or
            mutated. A driver that :func:`~hologradpy.hardware.as_native.as_camera`
            wraps is accepted.
        signal_to_noise_ratio: Peak must exceed the background by this many noise sigma.
        lower_relative_intensity_threshold: Peak must also reach this fraction of the
            camera's full-scale value.

    Returns:
        tuple[int, int] | None: The ``(row, column)`` of the spot peak, or ``None`` if
        no spot is found.
    """
    camera = as_camera(camera)
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
    prominence = peak_prominence(image)
    row, column = np.unravel_index(int(np.argmax(image)), image.shape)

    # Making sure detected maximum is not sitting at the border of the frame.
    border_margin = max(int(round(spot_radius_px)), 1)
    if (
        min(row, image.shape[0] - 1 - row) < border_margin
        or min(column, image.shape[1] - 1 - column) < border_margin
    ):
        return None

    half_window = round(_WINDOW_SPOT_RADII * spot_radius_px)
    top = max(row - half_window, 0)
    left = max(column - half_window, 0)
    window = image[top:row + half_window + 1, left:column + half_window + 1]

    # Checking the spot has a reasonable size
    half_max_radius_px = _HALF_MAX_RADIUS_FACTOR * spot_radius_px
    min_core_pixels = max(round(0.25 * np.pi * half_max_radius_px**2), 1)
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


def zeroth_order_mask_radius(
    spot_radius: float, pixel_size: NDArray | tuple[float, float]
) -> float:
    """The radius in pixels of the disc that keeps the zeroth order out of a spot search
    or fit, twice the fitted focal-spot radius.

    Args:
        spot_radius: The fitted focal-spot radius (1/e^2 intensity) in metres.
        pixel_size: The pixel pitch ``(y, x)`` of the sensor in metres.

    Returns:
        float: The radius in pixels.
    """
    return 2.0 * spot_radius / float(np.min(pixel_size))


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
    *,
    mask: NDArray[np.bool_] | None = None,
    search_roi: ROI | None = None,
) -> tuple[tuple[float, float], float, NDArray, ROI]:
    """This function generates a spot on the camera by displaying a linear phase
    gradient inside a circular aperture on the SLM. The position of the spot is found by
    fitting a Gaussian to the camera image.

    The whole sensor is read out while the spot is measured, so every position, mask
    and region is in whole-sensor pixels. The camera's exposure and region of interest
    are put back afterwards
    (:meth:`~hologradpy.hardware.camera.Camera.preserve_exposure_and_roi`).

    Args:
        slm: Instance of your SLM subclass.
        camera: Instance of your camera subclass.
        linear_phase_tilt: x and y gradient of the linear phase.
        focal_length: Focal length of the Fourier lens in metres.
        exposure_time: Exposure time in seconds. If None, the camera autoexposes on the
            searched pixels. The spot is fitted with a warning when it is overexposed
            even at the shortest exposure.
        slm_mask_diameter: Diameter of the circular aperture in metres. If None, the
            diameter is set to the size of the SLM.
        units: Units of the returned spot position: "metres" (default) for the (x, y)
            coordinates in the camera plane, or "pixels" for integer camera pixel
            coordinates. The focal spot radius is always in metres.
        roi_pad: Padding in pixels added around the detected spot when cropping the
            camera image before the fit (passed to ROI.detect).
        roi_threshold: Fraction of the peak intensity used to detect the spot region of
            interest (passed to ROI.detect).
        verbose: If True, prints progress messages to the console.
        mask: True at the pixels of the whole sensor to measure, and False where light
            must not steer the exposure or the fit, such as the zeroth order. The
            pixels outside the mask are left out of the exposure, the spot search and
            the fit. Everything is measured when None.
        search_roi: The region of the whole sensor to search for the spot. The
            exposure is metered on this region, and the fitted region stays within it.
            The region is trimmed to the sensor. The whole sensor is searched when None.

    Returns:
        tuple[tuple[float, float], float, NDArray, ROI]: The x and y coordinates of the
        spot on the full sensor (in metres or pixels, see ``units``), the focal spot
        radius in metres, the cropped camera image for the fit, and the region of
        interest of the crop in whole-sensor pixels. The cropped image is as captured,
        including any pixels outside the mask.

    Raises:
        ValueError: When ``units`` is neither "metres" nor "pixels", when ``mask`` is
            not the shape of the sensor or keeps no pixel of the searched region, or
            when no part of ``search_roi`` lies on the sensor.
        RuntimeError: When the camera autoexposed and the searched pixels hold no
            prominent peak.
    """
    if units not in ("metres", "pixels"):
        raise ValueError(f"units must be 'metres' or 'pixels', got {units!r}.")

    slm = as_slm(slm)
    camera = as_camera(camera)

    if slm_mask_diameter is None:
        slm_mask_diameter = min(slm.aperture_extent)

    slm_grid = get_spatial_grid(slm.resolution, slm.pixel_size)

    slm_phase = linear_phase(
        *slm_grid,
        *linear_phase_tilt,
        focal_length=focal_length,
        wavenumber=2 * np.pi / slm.wavelength,
    )

    aperture = circular_mask(*slm_grid, slm_mask_diameter / 2)

    focal_spot_radius_guess = get_focal_spot_radius(
        beam_radius=slm_mask_diameter / 2,
        wavelength=slm.wavelength,
        focal_length=focal_length,
    )

    # The whole sensor is read out, so the spot can be located anywhere on it.
    with camera.preserve_exposure_and_roi(full_sensor=True):
        sensor_resolution = tuple(camera.resolution)
        searched_region = (
            ROI(0, 0, *sensor_resolution)
            if search_roi is None
            else search_roi.trimmed_to(sensor_resolution)
        )
        searched_mask = None
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != sensor_resolution:
                raise ValueError(
                    f"The mask has shape {mask.shape}, and the sensor is "
                    f"{sensor_resolution}."
                )
            searched_mask = searched_region.crop(mask)
            if not searched_mask.any():
                raise ValueError(
                    f"The mask keeps no pixel of the searched region {searched_region}."
                )

        # Display phase pattern on SLM
        slm.set_phase(gpu_to_numpy(slm_phase * aperture))

        # Autoexpose on the searched pixels when no exposure time is given. The spot is
        # still fitted when it is overexposed at the shortest exposure.
        autoexposed = exposure_time is None
        if autoexposed:
            exposure_time = camera.autoexpose(
                set_fraction=0.8,
                max_iterations=10,
                roi=searched_region,
                mask=searched_mask,
                raise_on_rail=False,
                verbose=verbose,
            )

        camera.set_exposure(exposure_time)
        camera_image = np.asarray(camera.get_image(), dtype=np.float64)
        searched_image = searched_region.crop(camera_image)

        if autoexposed and not has_prominent_peak(
            searched_image, camera, mask=searched_mask
        ):
            raise RuntimeError(
                f"No spot on the sensor after autoexposure at tilt "
                f"{linear_phase_tilt}. Check that the tilt lands on the sensor."
            )

        # Crop to a region of interest around the spot before fitting, so the Gaussian
        # fit runs on a small image.
        detected = ROI.detect(
            searched_image, threshold=roi_threshold, pad=roi_pad, mask=searched_mask
        )
        roi = ROI(
            searched_region.top_row + detected.top_row,
            searched_region.left_column + detected.left_column,
            detected.height,
            detected.width,
        )
        cropped_camera_image = roi.crop(camera_image)

        camera_grid = get_spatial_grid(sensor_resolution, camera.pixel_size)
        cropped_grid = [roi.crop(grid) for grid in camera_grid]

        if verbose:
            print("Fitting Gaussian to camera image...")

        # The fit starts at the peak of the frame blurred over one spot radius, so a
        # spot beside the mask starts at its own peak.
        popt, _ = fit_gaussian_beam_intensity(
            *cropped_grid,
            cropped_camera_image,
            beam_radius_guess=focal_spot_radius_guess,
            blur_sigma=max(
                focal_spot_radius_guess / float(np.min(camera.pixel_size)), 1.0
            ),
            mask=None if mask is None else roi.crop(mask),
        )

        if verbose:
            print("Gaussian fit complete.")

    focal_spot_radius = popt[0]
    position = (popt[1], popt[2])

    if units == "pixels":
        position = tuple(
            int(value)
            for value in metres_to_pixel(position, camera.pixel_size, sensor_resolution)
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

# TODO: Move to exposure.py?
def _meter_and_capture(
    camera: Camera, roi: ROI, set_fraction: float
) -> NDArray[np.float64]:
    """One frame, metered on ``roi``, with the camera's exposure and region of interest
    put back afterwards.
    """
    with camera.preserve_exposure_and_roi():
        camera.autoexpose(set_fraction=set_fraction, roi=roi)
        return np.asarray(camera.get_image(), dtype=float)


def _brightest_pixel(
    image: NDArray[np.float64],
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


def tilt_to_sensor_center(
    camera: Camera, camera_mapping: CameraMapping
) -> tuple[float, float]:
    """The tilt that steers a spot to the sensor centre, as ``(x, y)`` in metres. It is
    the image-plane position of the sensor centre, measured from the zeroth order,
    through
    :meth:`~hologradpy.calibration.camera_mapping.CameraMapping.sensor_to_image_plane`.

    Args:
        camera: The camera, for its sensor resolution, or a driver that
            :func:`~hologradpy.hardware.as_native.as_camera` wraps. The tilt aims at the
            middle of the whole sensor, whatever region of interest is set.
        camera_mapping: The fitted camera mapping, for its partial affine and its
            output plane.

    Returns:
        tuple[float, float]: ``(tilt_x, tilt_y)`` in metres, ready for
        :func:`~hologradpy.profiles.phase.linear_phase` with ``tilt_units="metres"``.

    Raises:
        ValueError: The mapping records no output plane, or was measured on a camera
            with another sensor, pitch or orientation.
    """
    camera = as_camera(camera)
    camera_mapping.check_camera(camera)
    sensor_height, sensor_width = tuple(camera.sensor_resolution)
    tilt_x, tilt_y = camera_mapping.sensor_to_image_plane(
        [(sensor_height / 2.0, sensor_width / 2.0)]
    )[0]
    return (float(tilt_x), float(tilt_y))


def capture_focal_spot(
    slm: SLM,
    camera: Camera,
    camera_mapping: CameraMapping,
    focal_length: float,
    kernel_size: int | tuple[int, int],
    set_fraction: float = 0.8,
    search_factor: float = 3.0,
) -> NDArray[np.float64]:
    """Capture the amplitude of the focal spot, steered to the middle of the sensor.

    The captured spot is the point spread function itself, so it is the natural seed
    for a :class:`~hologradpy.optics.modules.slm_fields.PSFSLMField`. It carries
    whatever aberration is actually present. A Gaussian of the fitted waist carries
    only the width.

    A linear phase steers the spot to the sensor centre.

    The whole sensor is read out while the spot is captured, and the camera's exposure
    and region of interest are put back afterwards
    (:meth:`~hologradpy.hardware.camera.Camera.preserve_exposure_and_roi`).

    Args:
        slm: The SLM, or a driver that :func:`~hologradpy.hardware.as_native.as_slm`
            wraps. It displays the steering tilt.
        camera: The camera to capture from, or a driver that
            :func:`~hologradpy.hardware.as_native.as_camera` wraps.
        camera_mapping: The fitted mapping, which sets the steering tilt.
        focal_length: Fourier lens focal length in metres, which converts the tilt into
            a phase ramp.
        kernel_size: Crop side in camera pixels, as an int or ``(height, width)``.
        set_fraction: Fraction of full scale to meter the spot to.
        search_factor: The width of the metered and searched region, in kernel widths.
            The whole sensor is searched when the spot is not found in this region.

    Returns:
        NDArray: The cropped amplitude, ``sqrt`` of the background-subtracted counts,
        shaped ``kernel_size``.

    Raises:
        RuntimeError: If no spot is found anywhere on the sensor.
    """
    slm = as_slm(slm)
    camera = as_camera(camera)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    height, width = int(kernel_size[0]), int(kernel_size[1])

    with camera.preserve_exposure_and_roi(full_sensor=True):
        sensor = tuple(camera.sensor_resolution)
        center = (sensor[0] / 2.0, sensor[1] / 2.0)

        tilt = tilt_to_sensor_center(camera, camera_mapping)
        slm_grid = get_spatial_grid(slm.resolution, slm.pixel_size)
        # The tilt is displayed over the full aperture, since the seed is the point
        # spread function of the whole SLM. An aperture broadens this function.
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

        search = ROI.centered(
            center,
            (
                min(int(search_factor * height), sensor[0]),
                min(int(search_factor * width), sensor[1]),
            ),
        ).moved_inside(sensor)

        image = _meter_and_capture(camera, search, set_fraction)

        if has_prominent_peak(search.crop(image), camera):
            found = _brightest_pixel(search.crop(image))
            found = (search.top_row + found[0], search.left_column + found[1])
        else:
            zeroth = (
                float(camera_mapping.zeroth_order_position[0]),
                float(camera_mapping.zeroth_order_position[1]),
            )
            separation = float(
                np.hypot(zeroth[0] - center[0], zeroth[1] - center[1])
            )
            found = _brightest_pixel(
                image, exclude=zeroth, exclude_radius=separation / 2
            )

            offset = np.hypot(found[0] - center[0], found[1] - center[1])
            # The first exposure was metered on background in a window without the
            # spot, so the spot can be overexposed. The frame is metered again around
            # the spot.
            search = ROI.centered(
                found, (search.height, search.width)
            ).moved_inside(sensor)
            image = _meter_and_capture(camera, search, set_fraction)
            if not has_prominent_peak(search.crop(image), camera):
                raise RuntimeError(
                    "No focal spot found anywhere on the sensor after steering by "
                    f"{tilt[0] * 1e3:.2f} x {tilt[1] * 1e3:.2f} mm. Check the camera "
                    "mapping, and that the requested tilt is within the SLM's "
                    "diffraction angle."
                )
            warnings.warn(
                f"The steered focal spot landed {offset:.0f} px from the sensor "
                f"center, outside the {search.height} x {search.width} px search "
                "window, so the whole sensor was searched. That offset is the camera "
                "mapping's error: the seed is still good, but the mapping is worth "
                "refitting.",
                stacklevel=2,
            )

    spot_roi = ROI.centered(found, (height, width)).moved_inside(sensor)

    crop: NDArray[np.float64] = spot_roi.crop(image)
    # A robust floor, so read-out background does not become part of the seed.
    background: float = float(np.percentile(crop, 10.0))
    return np.sqrt(np.clip(crop - background, 0.0, None))
