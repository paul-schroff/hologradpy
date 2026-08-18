from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def get_pixel_grid(
    resolution: tuple[int, int], device: torch.device = "cpu"
) -> Tuple[Tensor, Tensor]:
    height, width = resolution

    pixel_indices_x = torch.arange(-width // 2, width // 2, device=device)
    pixel_indices_y = torch.arange(-height // 2, height // 2, device=device)

    return torch.meshgrid(pixel_indices_x, pixel_indices_y, indexing="xy")


def get_spatial_grid(
    resolution: tuple[int, int],
    pixel_size: tuple[float, float],
    device: torch.device = "cpu",
) -> Tuple[Tensor, Tensor]:
    resolution = torch.tensor(resolution, device=device)
    pixel_size = torch.tensor(pixel_size, device=device)

    spatial_extent = resolution * pixel_size

    pixel_grid_x, pixel_grid_y = get_pixel_grid(resolution, device)

    spatial_grid_x = pixel_grid_x / resolution[1] * spatial_extent[1]
    spatial_grid_y = pixel_grid_y / resolution[0] * spatial_extent[0]

    return spatial_grid_x, spatial_grid_y


def plane_center(resolution: tuple[int, int]) -> tuple[int, int]:
    """The ``(x, y)`` pixel where :func:`get_pixel_grid` crosses zero."""
    height, width = resolution
    first_x = -width // 2
    first_y = -height // 2
    return (-first_x, -first_y)


def metres_to_pixel(
    position: tuple[float, float],
    pixel_size: tuple[float, float],
    resolution: tuple[int, int],
) -> tuple[float, float]:
    """Plane ``(x, y)`` metres (relative to the center) to ``(x, y)`` pixels, the
    inverse of :func:`get_spatial_grid`.

    ``pixel_size`` is ``(y, x)`` metres and ``resolution`` is ``(height, width)``, so
    x uses the width pitch and y the height pitch.
    """
    center_x, center_y = plane_center(resolution)
    return (
        position[0] / pixel_size[1] + center_x,
        position[1] / pixel_size[0] + center_y,
    )


def pixel_to_metres(
    pixel: tuple[float, float],
    pixel_size: tuple[float, float],
    resolution: tuple[int, int],
) -> tuple[float, float]:
    """``(x, y)`` pixels to plane ``(x, y)`` metres from the center. The inverse of 
    :func:`metres_to_pixel`.
    """
    center_x, center_y = plane_center(resolution)
    return (
        (pixel[0] - center_x) * pixel_size[1],
        (pixel[1] - center_y) * pixel_size[0],
    )


def get_frequency_grid(
    resolution: tuple[int, int],
    pixel_size: tuple[float, float],
    device: torch.device = "cpu",
) -> Tuple[Tensor, Tensor]:
    resolution = torch.tensor(resolution, device=device)
    pixel_size = torch.tensor(pixel_size, device=device)

    frequency_extent = 2 * torch.pi / pixel_size

    pixel_grid_x, pixel_grid_y = get_pixel_grid(resolution, device)

    frequency_grid_x = pixel_grid_x / resolution[1] * frequency_extent[1]
    frequency_grid_y = pixel_grid_y / resolution[0] * frequency_extent[0]

    return frequency_grid_x, frequency_grid_y
