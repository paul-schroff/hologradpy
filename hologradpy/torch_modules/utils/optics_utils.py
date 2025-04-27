from __future__ import annotations
import torch

# %% Phase function(s)
def lens_phase(
        x: torch.Tensor,
        y: torch.Tensor,
        focal_length: float,
        wavenumber:float
    ) -> torch.Tensor:
    return -0.5 * wavenumber / focal_length * (x ** 2 + y ** 2)

# %% Binary aperture functions
def rect_mask(
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