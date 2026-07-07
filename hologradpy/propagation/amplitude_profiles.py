"""Analytic amplitude/intensity profiles, apertures and blurring on a 2D grid."""

from __future__ import annotations
from typing import TypeVar

from numpy.typing import NDArray
import torch

from array_api_compat import array_namespace

from .fourier import get_spatial_grid
from ..utils import unsqueeze_to

ArrayLike = TypeVar("ArrayLike", torch.Tensor, NDArray)


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
        x (ArrayLike): X coordinates.
        y (ArrayLike): Y coordinates.
        beam_radius (float): Radius of the beam.
        shift_x (float, optional): X shift of the beam center. Defaults to 0.0.
        shift_y (float, optional): Y shift of the beam center. Defaults to 0.0.
        intensity (float, optional): Peak intensity of the beam. Defaults to
            1.0.
        offset (float, optional): Offset added to the intensity. Defaults to
            0.0.

    Returns:
        ArrayLike: Gaussian beam intensity.
    """
    xp = array_namespace(x, y)
    return (
        intensity
        * xp.exp(-2 * ((x - shift_x) ** 2 + (y - shift_y) ** 2) / (beam_radius**2))
        + offset
    )


def get_focal_spot_radius(
    beam_radius: float,
    wavelength: float,
    focal_length: float,
) -> float:
    """Calculates the radius of the focal spot for a Gaussian beam with a given
    `beam_radius`, focussed by a lens with a given `focal_length`.

    Args:
        beam_radius (float): The radius of the Gaussian beam at the lens in
            meters.
        wavelength (float): The wavelength of the light in meters.
        focal_length (float): The focal length of the lens in meters.

    Returns:
        float: The radius of the focal spot in meters.
    """
    return (wavelength * focal_length) / (torch.pi * beam_radius)


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
        x (ArrayLike): X meshgrid.
        y (ArrayLike): Y meshgrid.
        shift_x (int): X-offset of the Gaussian.
        shift_y (int): Y-offset of the Gaussian.
        number_of_pixels_x (int): X-order.
        number_of_pixels_y (int): Y-order.
        sigma_x (float): X-width.
        sigma_y (float): Y-width.
        amplitude (float, optional): Amplitude. Defaults to 1.0.
        offset (float, optional): Offset. Defaults to 0.0.

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
        x (ArrayLike): X coordinates.
        y (ArrayLike): Y coordinates.
        number_of_rows (int): Number of array rows.
        number_of_columns (int): Number of array columns.
        shift_x (int): X-offset of the array.
        shift_y (int): Y-offset of the array.
        spot_separation (float): Separation between neighbouring spots.
        beam_radius (float): Beam radius of the Gaussian spots.

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
                shift_x=j * spot_separation - horizontal_extent // 2 - shift_y,
                shift_y=i * spot_separation - vertical_extent // 2 - shift_x,
            )
    return spots


def ring_gauss(
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
        x (ArrayLike): X meshgrid.
        y (ArrayLike): Y meshgrid.
        shift_x (int): X-offset of the ring.
        shift_y (int): Y-offset of the ring.
        radius (float): Radius of the ring.
        ring_sigma (float): Width of the Gaussian profile.
        amplitude (float, optional): Amplitude. Defaults to 1.0.

    Returns:
        ArrayLike: Ring with a Gaussian profile.
    """
    xp = array_namespace(x, y)
    return amplitude * xp.exp(
        -2
        * (xp.sqrt((x - shift_x) ** 2 + (y - shift_y) ** 2) - radius) ** 2
        / ring_sigma ** 2
    )


def rectangular_mask(
    x: ArrayLike,
    y: ArrayLike,
    width: float,
    height: float,
    shift_x: float = 0.0,
    shift_y: float = 0.0,
) -> ArrayLike:
    """Rectangular mask with given width, height, and center.

    Args:
        x (ArrayLike): X coordinates.
        y (ArrayLike): Y coordinates.
        width (float): Width of the rectangle.
        height (float): Height of the rectangle.
        shift_x (float, optional): X shift of the rectangle center. Defaults
            to 0.0.
        shift_y (float, optional): Y shift of the rectangle center. Defaults
            to 0.0.

    Returns:
        ArrayLike: Binary mask.
    """
    xp = array_namespace(x, y)
    return (xp.abs(x - shift_x) < width / 2) & (xp.abs(y - shift_y) < height / 2)


def circular_mask(
    x: ArrayLike,
    y: ArrayLike,
    radius: float,
    shift_x: float = 0.0,
    shift_y: float = 0.0,
) -> ArrayLike:
    """Create a circular mask with a given radius and center.

    Args:
        x (ArrayLike): X coordinates.
        y (ArrayLike): Y coordinates.
        radius (float): Radius of the circle.
        shift_x (float, optional): X shift of the circle center. Defaults to
            0.0.
        shift_y (float, optional): Y shift of the circle center. Defaults to
            0.0.

    Returns:
        ArrayLike: Binary mask.
    """
    return ((x - shift_x) ** 2 + (y - shift_y) ** 2) ** 0.5 < radius


def gaussian_blur(input: torch.Tensor, beam_radius: float):
    """Blurs the input tensor with the intensity distribution of a Gaussian
    focal spot with a given `beam_radius`.

    Args:
        input (torch.Tensor): Input tensor.
        beam_radius (float): Radius of the Gaussian beam.

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
    negative-exponential (``std/mean ~= 1``). The result is normalised to unit mean --
    the physical scale is applied separately (e.g. by
    :class:`~hologradpy.propagation.background_scatter.BackgroundScatter`, which scales
    it to a total background power).

    Args:
        resolution: Output resolution ``(height, width)`` in pixels.
        pixel_size: Pixel size in metres, used to convert ``grain_radius`` to pixels
            (square pixels assumed).
        grain_radius: Speckle grain radius in metres (the Gaussian-blur radius).
        device: Device for the generated tensor.
        dtype: Real dtype of the returned intensity map.
        generator: Optional RNG for reproducible speckle.

    Returns:
        A ``(height, width)`` unit-mean speckle-intensity tensor.
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
