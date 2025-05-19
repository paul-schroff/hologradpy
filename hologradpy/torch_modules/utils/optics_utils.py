from __future__ import annotations
from typing import Literal
import torch


def lens_phase(
        x: torch.Tensor,
        y: torch.Tensor,
        focal_length: float,
        wavenumber:float
    ) -> torch.Tensor:
    return -0.5 * wavenumber / focal_length * (x ** 2 + y ** 2)

# TODO: Sanity check this function
def linear_phase(
        x: torch.Tensor,
        y: torch.Tensor,
        tilt_x: float,
        tilt_y: float,
        tilt_units: Literal[
            'degrees',
            'radians',
            'metres',
            'lines_per_mm',
            ] = 'metres',
        wavenumber: float | None = None,
        focal_length: float | None = None,
    ) -> torch.Tensor:
    match tilt_units:
        case 'degrees':
            slope_x = torch.tan(tilt_x) * wavenumber
            slope_y = torch.tan(tilt_y) * wavenumber
        case 'radians':
            slope_x = tilt_x * wavenumber
            slope_y = tilt_y * wavenumber
        case 'metres':
            if focal_length is None:
                raise ValueError(
                    'Focal length must be provided when tilt_units is '
                    '"metres".'
                )
            slope_x = tilt_x / focal_length
            slope_y = tilt_y / focal_length
        case 'lines_per_mm':
            raise NotImplementedError('lines_per_mm not implemented yet')
    return slope_x * x + slope_y * y

# TODO: Sanity check this function
def quadratic_phase(
        x: torch.Tensor,
        y: torch.Tensor,
        curvature: float,
        aspect_ratio: float = 1.0,
        curvature_units: Literal[
            'radians_per_pixel_squared',
            'radians_per_metre_squared'] = 'radians_per_metre_squared',
    ) -> torch.Tensor:
    if curvature_units == 'radians_per_pixel_squared':
        x = x / (x.max() - x.min()) * x.shape[1]
        y = y / (y.max() - y.min()) * y.shape[0]
    return (
        4 * curvature * (aspect_ratio * y ** 2 + (1 - aspect_ratio) * x ** 2)
    )

def gaussian_beam_intensity(
        x: torch.Tensor,
        y: torch.Tensor,
        beam_radius: float,
        shift_x: float = 0.0,
        shift_y: float = 0.0,
        intensity: float = 1.0,
        offset: float = 0.0
    ) -> torch.Tensor:
    """Gaussian beam with given radius and center."""
    return intensity * torch.exp(
        - 2 * ((x - shift_x) ** 2 + (y - shift_y) ** 2) / (beam_radius ** 2)
        ) + offset

# %% Binary aperture functions
def rectangular_mask(
        x: torch.Tensor, 
        y: torch.Tensor,
        width: float, 
        height: float,
        shift_x: float = 0.0, 
        shift_y: float = 0.0, 
    ) -> torch.Tensor[torch.bool]:
    """Rectangular mask with given width, height, and center."""
    return (
        ((x - shift_x).abs() < width / 2) & ((y - shift_y).abs() < height / 2)
    )

def circular_mask(
        x: torch.Tensor,
        y: torch.Tensor,
        radius: float,
        shift_x: float = 0.0,
        shift_y: float = 0.0
    ) -> torch.Tensor[torch.bool]:
    """Create a circular mask with a given radius and center."""
    return ((x - shift_x) ** 2 + (y - shift_y) ** 2) ** 0.5 < radius

