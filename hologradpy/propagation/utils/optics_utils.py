from __future__ import annotations
from typing import Literal, TypeVar

from numpy.typing import NDArray
import torch

from .fourier_utils import get_spatial_grid
from .tensor_utils import unsqueeze_to

from array_api_compat import array_namespace

ArrayLike = TypeVar("ArrayLike", torch.Tensor, NDArray)
TiltUnits = Literal["degrees", "radians", "metres", "lines_per_mm"]
CurvatureUnits = Literal[
        "radians_per_pixel_squared", "radians_per_metre_squared"
    ]

# %% Phase functions
def lens_phase(
    x: ArrayLike, y: ArrayLike, focal_length: float, wavenumber: float
) -> ArrayLike:
    """Calculates the phase of an ideal lens on a 2D grid.

    Args:
        x (ArrayLike): X coordinates.
        y (ArrayLike): Y coordinates.
        focal_length (float): Focal length of the lens.
        wavenumber (float): Wavenumber of the light.

    Returns:
        ArrayLike: Lens phase.
    """
    return -0.5 * wavenumber / focal_length * (x ** 2 + y ** 2)

def spherical_surface(
    x: ArrayLike, 
    y: ArrayLike, 
    radius: float, 
    shift_x: float = 0.0, 
    shift_y: float = 0.0,
) -> ArrayLike:
    """Calculates a spherical surface.

    Args:
        x (ArrayLike): X-meshgrid [m].
        y (ArrayLike): Y-meshgrid [m].
        wavenumber (float): Wavenumber [rad/m].
        radius (float): Radius of curvature [m].
        shift_x (float): X-offset of lens [m].
        shift_y (float): Y-offset of lens [m].
    """
    xp = array_namespace(x, y)
    surface = (
        radius * (
        1 - xp.sqrt(
            1 - ((x - shift_x) ** 2 + (y - shift_y) ** 2) / radius ** 2
            )
        )
    )
    return surface
    
def doublet_lens(
    x: ArrayLike, 
    y: ArrayLike, 
    wavenumber: float, 
    refractive_index_flint: float, 
    refractive_index_crown: float, 
    radius_crown: float, 
    radius_crown_flint: float, 
    radius_flint: float, 
    shift_x: float = 0.0, 
    shift_y: float = 0.0,
) -> ArrayLike:
    """Calculates the phase of a doublet lens.

    Args:
        x (ArrayLike): X-meshgrid [m].
        y (ArrayLike): Y-meshgrid [m].
        wavenumber (float): Wavenumber [rad/m].
        refractive_index_flint (float): Refractive index of flint.
        refractive_index_crown (float): Refractive index of crown.
        radius_crown (float): Radius of curvature of the first crown surface 
            [m].
        radius_crown_flint (float): Radius of curvature of the second crown 
            surface/ first flint surface [m].
        radius_flint (float): Radius of curvature of the second flint surface 
            [m].
        shift_x (float): X-offset of lens [m].
        shift_y (float): Y-offset of lens [m].
        wavenumber (float): Phase of the doublet lens [rad].

    Returns:
        ArrayLike: Phase of the doublet lens.
    """
    crown_surface = spherical_surface(x, y, radius_crown, shift_x, shift_y)

    crown_flint_surface = spherical_surface(
        x, y, radius_crown_flint, shift_x, shift_y
    )

    flint_surface = spherical_surface(x, y, radius_flint, shift_x, shift_y)

    delta1 = - crown_surface + crown_flint_surface
    delta2 = flint_surface - crown_flint_surface

    return (
        wavenumber * (
            (refractive_index_flint - 1) * delta1 + 
            (refractive_index_crown - 1) * delta2
        )
    )

def linear_phase(
    x: ArrayLike,
    y: ArrayLike,
    tilt_x: float,
    tilt_y: float,
    tilt_units: TiltUnits = "metres",
    wavenumber: float | None = None,
    focal_length: float | None = None,
) -> ArrayLike:
    """Calculates a linear phase on a 2D grid.

    Args:
        x (ArrayLike): X coordinates.
        y (ArrayLike): Y coordinates.
        tilt_x (float): Tilt in the x direction.
        tilt_y (float): Tilt in the y direction.
        tilt_units (TiltUnits, optional): Units for tilt. Defaults to "metres".
        wavenumber (float | None, optional): Wavenumber required if 
            `tilt_units` is "degrees", "radians", or "metres". Defaults to 
            None.
        focal_length (float | None, optional): Focal length required if 
            `tilt_units` is "metres". Defaults to None.
    Returns:
        ArrayLike: Linear phase.
    """
    xp = array_namespace(x, y)
    match tilt_units:
        case "degrees":
            slope_x = xp.tan(tilt_x) * wavenumber
            slope_y = xp.tan(tilt_y) * wavenumber
        case "radians":
            slope_x = tilt_x * wavenumber
            slope_y = tilt_y * wavenumber
        case "metres":
            if focal_length is None:
                raise ValueError(
                    'Focal length must be provided when tilt_units is "metres".'
                )
            slope_x = tilt_x / focal_length * wavenumber
            slope_y = tilt_y / focal_length * wavenumber
        case "lines_per_mm":
            raise NotImplementedError("lines_per_mm not implemented yet")
    return slope_x * x + slope_y * y

def quadratic_phase(
    x: ArrayLike,
    y: ArrayLike,
    curvature: float,
    aspect_ratio: float = 0.5,
    curvature_units: CurvatureUnits = "radians_per_metre_squared",
) -> ArrayLike:
    """Calculates a quadratic phase with some `curvature` and an 
    `aspect_ratio` on a 2D grid.

    Args:
        x (ArrayLike): X coordinates.
        y (ArrayLike): Y coordinates.
        curvature (float): Curvature for quadratic phase.
        aspect_ratio (float, optional): Aspect ratio for quadratic phase. 
            Defaults to 0.5.
        curvature_units (CurvatureUnits, optional): Units for curvature.
            Defaults to "radians_per_metre_squared".
    Returns:
        ArrayLike: Quadratic phase.    
    """
    if curvature_units == "radians_per_pixel_squared":
        x = x / (x.max() - x.min()) * x.shape[1]
        y = y / (y.max() - y.min()) * y.shape[0]
    return (
        4 * curvature * (aspect_ratio * y ** 2 + (1 - aspect_ratio) * x ** 2)
    )

def analytic_phase_guess(
    x: ArrayLike,
    y: ArrayLike,
    tilt_x: float,
    tilt_y: float,
    curvature: float,
    aspect_ratio: float = 0.5,
    tilt_units: TiltUnits = "metres",
    curvature_units: CurvatureUnits = "radians_per_metre_squared",
    wavenumber: float | None = None,
    focal_length: float | None = None
) -> ArrayLike:
    """Calculates a combined linear and quadratic phase term.
    
    Args:
        x (ArrayLike): X coordinates.
        y (ArrayLike): Y coordinates.
        tilt_x (float): Tilt in the x direction.
        tilt_y (float): Tilt in the y direction.
        curvature (float): Curvature for quadratic phase.
        aspect_ratio (float, optional): Aspect ratio for quadratic phase. 
            Defaults to 0.5.
        tilt_units (TiltUnits, optional): Units for tilt. Defaults to "metres".
        curvature_units (CurvatureUnits, optional): Units for curvature.
            Defaults to "radians_per_metre_squared".
        wavenumber (float | None, optional): Wavenumber for linear phase.
            Required if tilt_units is "degrees", "radians", or "metres".
            Defaults to None.
        focal_length (float | None, optional): Focal length for linear phase.
            Required if tilt_units is "metres". Defaults to None.
    
    Returns:
        ArrayLike: Combined linear and quadratic phase.
    """
    linear_phase_term = linear_phase(
        x, y, tilt_x, tilt_y, tilt_units, wavenumber, focal_length
        )
    quadratic_phase_term = quadratic_phase(
        x, y, curvature, aspect_ratio, curvature_units
        )
    return linear_phase_term + quadratic_phase_term


# %% Intensity distributions
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
        intensity * xp.exp(
            -2 * ((x - shift_x) ** 2 + (y - shift_y) ** 2) / (beam_radius**2)
        ) 
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


# %% Binary aperture functions
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
    return (
        xp.abs(x - shift_x) < width / 2) & (xp.abs(y - shift_y) < height / 2
    )

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


# %%
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
        (kernel_size, kernel_size),
        pixel_size=(1, 1),
        device=input.device
    )
    kernel = gaussian_beam_intensity(*kernel_grid, beam_radius)
    kernel /= kernel.sum()
    kernel = unsqueeze_to(kernel, 4)

    input = unsqueeze_to(input, 4)

    return torch.nn.functional.conv2d(input, kernel, padding="same").squeeze()
