"""Analytic amplitude/intensity profiles, apertures and blurring on a 2D grid."""

from __future__ import annotations

import math
from typing import Literal, TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy.special import erf as erf_scipy

import torch
from torch import erf as erf_torch

from array_api_compat import array_namespace

from ..grids import get_spatial_grid
from ..utils import unsqueeze_to

ArrayLike = TypeVar("ArrayLike", torch.Tensor, NDArray)

# Intensity profiles
def gaussian_beam_intensity_1D(
    x: ArrayLike,
    beam_radius: float,
    shift: float = 0.0,
    intensity: float = 1.0,
    offset: float = 0.0,
) -> ArrayLike:
    """Gaussian beam intensity along one axis.

    Args:
        x: Coordinates along the axis.
        beam_radius: 1/e^2 intensity radius of the beam.
        shift: Shift of the beam center. Defaults to 0.0.
        intensity: Peak intensity of the beam. Defaults to 1.0.
        offset: Offset added to the intensity. Defaults to 0.0.

    Returns:
        ArrayLike: Gaussian beam intensity.
    """
    xp = array_namespace(x)
    return intensity * xp.exp(-2 * (x - shift) ** 2 / beam_radius**2) + offset


def gaussian_beam_intensity(
    x: ArrayLike,
    y: ArrayLike,
    beam_radius: float,
    shift_x: float = 0.0,
    shift_y: float = 0.0,
    intensity: float = 1.0,
    offset: float = 0.0,
) -> ArrayLike:
    """Gaussian beam intensity on a 2D grid with given radius and center.

    Args:
        x: X coordinates.
        y: Y coordinates.
        beam_radius: Radius of the beam.
        shift_x: X shift of the beam center. Defaults to 0.0.
        shift_y: Y shift of the beam center. Defaults to 0.0.
        intensity: Peak intensity of the beam. Defaults to 1.0.
        offset: Offset added to the intensity. Defaults to 0.0.

    Returns:
        ArrayLike: Gaussian beam intensity.
    """
    return (
        intensity
        * gaussian_beam_intensity_1D(x, beam_radius, shift_x)
        * gaussian_beam_intensity_1D(y, beam_radius, shift_y)
        + offset
    )

def super_gaussian(
    x: ArrayLike,
    y: ArrayLike,
    shift_x: int,
    shift_y: int,
    number_of_pixels_x: int,
    number_of_pixels_y: int,
    sigma_x: float,
    sigma_y: float,
    amplitude: float = 1.0,
    offset: float = 0.0,
) -> ArrayLike:
    """2D super-Gaussian intensity distribution.

    Args:
        x: X meshgrid.
        y: Y meshgrid.
        shift_x: X-offset of the Gaussian.
        shift_y: Y-offset of the Gaussian.
        number_of_pixels_x: X-order.
        number_of_pixels_y: Y-order.
        sigma_x: X-width.
        sigma_y: Y-width.
        amplitude: Amplitude. Defaults to 1.0.
        offset: Offset. Defaults to 0.0.

    Returns:
        ArrayLike: 2D super-Gaussian.
    """
    xp = array_namespace(x, y)
    return (
        amplitude
        * xp.exp(-2 * (xp.abs(x - shift_x) / sigma_x) ** number_of_pixels_x)
        * xp.exp(-2 * (xp.abs(y - shift_y) / sigma_y) ** number_of_pixels_y)
        + offset
    )

def gaussian_spot_array(
    x: ArrayLike,
    y: ArrayLike,
    number_of_rows: int,
    number_of_columns: int,
    shift_x: int,
    shift_y: int,
    spot_separation: float,
    beam_radius: float,
) -> ArrayLike:
    """Array of Gaussian spots with equal spacing.

    Args:
        x: X coordinates.
        y: Y coordinates.
        number_of_rows: Number of array rows.
        number_of_columns: Number of array columns.
        shift_x: X-offset of the array.
        shift_y: Y-offset of the array.
        spot_separation: Separation between neighboring spots.
        beam_radius: Beam radius of the Gaussian spots.

    Returns:
        ArrayLike: Spot array.
    """
    xp = array_namespace(x, y)

    spots = xp.zeros_like(x)
    vertical_extent = (number_of_rows - 1) * spot_separation
    horizontal_extent = (number_of_columns - 1) * spot_separation

    for i in range(number_of_rows):
        for j in range(number_of_columns):
            spots = spots + gaussian_beam_intensity(
                x,
                y,
                beam_radius,
                shift_x=j * spot_separation - horizontal_extent // 2 - shift_x,
                shift_y=i * spot_separation - vertical_extent // 2 - shift_y,
            )
    return spots

def gaussian_ring(
    x: ArrayLike,
    y: ArrayLike,
    shift_x: int,
    shift_y: int,
    radius: float,
    ring_sigma: float,
    amplitude: float = 1.0,
) -> ArrayLike:
    """Ring with a Gaussian radial profile.

    Args:
        x: X meshgrid.
        y: Y meshgrid.
        shift_x: X-offset of the ring.
        shift_y: Y-offset of the ring.
        radius: Radius of the ring.
        ring_sigma: Width of the Gaussian profile.
        amplitude: Amplitude. Defaults to 1.0.

    Returns:
        ArrayLike: Ring with a Gaussian profile.
    """
    xp = array_namespace(x, y)
    return amplitude * xp.exp(
        -2
        * (xp.sqrt((x - shift_x) ** 2 + (y - shift_y) ** 2) - radius) ** 2
        / ring_sigma ** 2
    )

def top_hat_gaussian_shoulders(
    x: ArrayLike, 
    shift: float,
    plateau_width: float, 
    shoulder_radius: float,
    amplitude: float,
) -> ArrayLike:
    """This function describes the convolution of a boxcar function with the
    intensity profile of a Gaussian beam, resulting in a top hat with soft shoulders.
    The Gaussian beam radius determines `shoulder_radius`. The width of the flat
    plateau is specified by `plateau_width` (1 - 1/e^4 intensity threshold). The width
    of the original boxcar function is `plateau_width + 2 * shoulder_radius`.

    Args:
        x: X coordinates.
        shift: Center of the top hat.
        plateau_width: Width of the flat plateau (1 - 1/e^4 intensity threshold).
        shoulder_radius: Radius of the Gaussian shoulders.
        amplitude: Amplitude of the top hat.

    Returns:
        ArrayLike: Top hat with Gaussian shoulders.

    Raises:
        ValueError: when the array namespace of `x` is neither numpy nor torch.
    """
    xp = array_namespace(x)
    backend_name = xp.__name__.split(".")[1]
    if backend_name == "torch":
        erf = erf_torch
    elif backend_name == "numpy":
        erf = erf_scipy
    else:
        raise ValueError(f"Unsupported array namespace: {xp}")

    width = plateau_width + 2 * shoulder_radius
    a = amplitude * math.sqrt(2) * shoulder_radius
    x = x - shift
    return 0.5 * (erf((width - 2 * x) / a) + erf((width + 2 * x) / a))

def top_hat_2D(
    x: ArrayLike,
    y: ArrayLike,
    shift_x: float,
    shift_y: float,
    plateau_width_x: float,
    plateau_width_y: float,
    shoulder_radius_x: float,
    shoulder_radius_y: float,
    amplitude: float,
) -> ArrayLike:
    """2D top hat with Gaussian shoulders.

    Args:
        x: X meshgrid.
        y: Y meshgrid.
        shift_x: X-offset of the top hat.
        shift_y: Y-offset of the top hat.
        plateau_width_x: Width of the flat plateau in the x-direction
            (1 - 1/e^4 intensity threshold).
        plateau_width_y: Width of the flat plateau in the y-direction
            (1 - 1/e^4 intensity threshold).
        shoulder_radius_x: Radius of the Gaussian shoulders in the x-direction.
        shoulder_radius_y: Radius of the Gaussian shoulders in the y-direction.
        amplitude: Amplitude of the top hat.

    Returns:
        ArrayLike: 2D top hat with Gaussian shoulders.
    """
    x_term = top_hat_gaussian_shoulders(
        x, shift_x, plateau_width_x, shoulder_radius_x, amplitude
    )
    y_term = top_hat_gaussian_shoulders(
        y, shift_y, plateau_width_y, shoulder_radius_y, 1.0
    )
    return x_term * y_term


def top_hat_1D(
    x: ArrayLike,
    y: ArrayLike,
    plateau_width: float,
    shoulder_radius: float,
    beam_radius: float,
    axis: Literal["x", "y"] = "y",
    shift_x: float = 0.0,
    shift_y: float = 0.0,
    intensity: float = 1.0,
) -> ArrayLike:
    """A top hat with Gaussian shoulders along one axis and a Gaussian beam profile
    across the other.

    Args:
        x: X meshgrid.
        y: Y meshgrid.
        plateau_width: Length of the flat plateau along the line
            (1 - 1/e^4 intensity threshold).
        shoulder_radius: Radius of the Gaussian shoulders capping the ends.
        beam_radius: 1/e^2 intensity radius across the line.
        axis: The axis the flat direction runs along, `"x"` or `"y"`. Defaults to
            `"y"`, a line standing upright with the Gaussian across it in x.
        shift_x: X-offset of the line. Defaults to 0.0.
        shift_y: Y-offset of the line. Defaults to 0.0.
        intensity: Peak intensity. Defaults to 1.0.

    Returns:
        ArrayLike: The line profile, peaking at `intensity`.

    Raises:
        ValueError: when `axis` is neither `"x"` nor `"y"`.
    """
    match axis:
        case "y":
            along, along_shift = y, shift_y
            across, across_shift = x, shift_x
        case "x":
            along, along_shift = x, shift_x
            across, across_shift = y, shift_y
        case _:
            raise ValueError(f"axis must be 'x' or 'y', not {axis!r}.")

    plateau = top_hat_gaussian_shoulders(
        along, along_shift, plateau_width, shoulder_radius, 1.0
    )
    profile = gaussian_beam_intensity_1D(across, beam_radius, across_shift)
    return intensity * plateau * profile


# Binary masks
def gaussian_blur(input: torch.Tensor, beam_radius: float) -> torch.Tensor:
    """Blurs the input tensor with the intensity distribution of a Gaussian
    focal spot with a given `beam_radius`.

    Args:
        input: Input tensor.
        beam_radius: Radius of the Gaussian beam.

    Returns:
        torch.Tensor: Blurred output tensor.
    """
    kernel_size = int(3 * beam_radius // 2 * 2 + 1)
    kernel_grid = get_spatial_grid(
        (kernel_size, kernel_size), pixel_size=(1, 1), device=input.device
    )
    kernel = gaussian_beam_intensity(*kernel_grid, beam_radius)
    kernel /= kernel.sum()
    kernel = unsqueeze_to(kernel, 4)

    input = unsqueeze_to(input, 4)

    return torch.nn.functional.conv2d(input, kernel, padding="same").squeeze()


def laser_speckle_intensity(
    resolution: tuple[int, int],
    pixel_size: float,
    grain_radius: float,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """A static, laser-speckle intensity pattern.

    Models a fixed scattering surface: complex Gaussian white noise is low-pass filtered
    (Gaussian blur of radius ``grain_radius``) to set the speckle grain size, and the
    squared magnitude gives the intensity. The first-order statistics are the expected
    negative-exponential (``std/mean ~= 1``). The result is normalized to unit mean,
    and the physical scale is applied separately (e.g. by
    :class:`~hologradpy.optics.modules.hardware_models.background_scatter.BackgroundScatter`,
    which scales it to a total background power).

    Args:
        resolution: Output resolution ``(height, width)`` in pixels.
        pixel_size: Pixel size in metres, used to convert ``grain_radius`` to pixels
            (square pixels assumed).
        grain_radius: Speckle grain radius in metres (the Gaussian-blur radius).
        device: Device for the generated tensor.
        dtype: Real dtype of the returned intensity map.
        generator: Optional RNG for reproducible speckle.

    Returns:
        torch.Tensor: A ``(height, width)`` unit-mean speckle-intensity tensor.
    """
    height, width = resolution
    grain_radius_pixels = max(grain_radius / pixel_size, 1.0)
    real = torch.randn((height, width), device=device, generator=generator)
    imaginary = torch.randn((height, width), device=device, generator=generator)
    real = gaussian_blur(real, grain_radius_pixels)
    imaginary = gaussian_blur(imaginary, grain_radius_pixels)
    intensity = real**2 + imaginary**2
    intensity = intensity / intensity.mean()
    return intensity.to(dtype)


def checkerboard(
    resolution: tuple[int, int],
    number_of_squares: tuple[int, int],
    square_size: int | None = None,
    shift_x: int = 0,
    shift_y: int = 0,
    dark_square_brightness: float = 0.0,
    light_square_brightness: float = 1.0,
    device: torch.device | None = None,
) -> NDArray | torch.Tensor:
    """A checkerboard pattern, centered (and shifted) in a zero border.

    The cheap integer pattern is built with numpy and returned as a numpy array, or as
    a torch tensor on ``device`` when one is given.

    Args:
        resolution: Output resolution ``(height, width)`` in pixels.
        number_of_squares: Number of squares along the ``(y, x)`` axes.
        square_size: Side length of each square in pixels. If ``None``, the largest
            that fits ``number_of_squares`` into ``resolution``.
        shift_x: Horizontal (x) shift of the board in pixels.
        shift_y: Vertical (y) shift of the board in pixels.
        dark_square_brightness: Value of the dark squares.
        light_square_brightness: Value of the light squares.
        device: If given, return a torch tensor on that device, otherwise numpy.

    Returns:
        NDArray | torch.Tensor: A ``(height, width)`` array with the checkerboard
        pattern.
    """
    height, width = resolution
    number_of_squares_y, number_of_squares_x = number_of_squares
    if square_size is None:
        square_size = min(
            width // number_of_squares_x, height // number_of_squares_y
        )
    board_height = number_of_squares_y * square_size
    board_width = number_of_squares_x * square_size

    rows = np.arange(board_height)[:, None]
    columns = np.arange(board_width)[None, :]
    parity = ((columns // square_size) + (rows // square_size)) % 2
    board = dark_square_brightness + parity * (
        light_square_brightness - dark_square_brightness
    )

    # Center the board in a zero border, then apply the (x, y) shift. The caller sizes
    # the board so it stays within the frame.
    output = np.zeros((height, width))
    top = (height - board_height) // 2 + shift_y
    left = (width - board_width) // 2 + shift_x
    output[top : top + board_height, left : left + board_width] = board

    if device is not None:
        return torch.as_tensor(output, dtype=torch.float32, device=device)
    return output
