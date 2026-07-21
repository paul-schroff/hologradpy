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


def metres_to_pixel(
    position: tuple[float, float],
    pixel_size: tuple[float, float],
    resolution: tuple[int, int],
) -> tuple[float, float]:
    """Plane ``(x, y)`` metres (relative to the centre) to ``(x, y)`` pixels, the
    inverse of :func:`get_spatial_grid`.

    ``pixel_size`` is ``(y, x)`` metres and ``resolution`` is ``(height, width)``, so
    x uses the width pitch and y the height pitch. Works for any plane (the camera
    sensor or a model output grid) given that plane's pitch and shape.
    """
    return (
        position[0] / pixel_size[1] + resolution[1] // 2,
        position[1] / pixel_size[0] + resolution[0] // 2,
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


def get_zoom_frequency_grid(
    resolution: tuple[int, int],
    resolution_out: tuple[int, int],
    magnification: tuple[float, float],
    shift: tuple[float, float] = (0.0, 0.0),
    device: torch.device = "cpu",
) -> Tuple[Tensor, Tensor]:
    """Per-axis k-space sample points (rad/sample) of a scaled + shifted output
    window, shared by the zoom transforms (NUFFT, chirp-z) so they sample
    identical points.

    The native DFT bin spacing is ``2*pi / resolution`` (rad/sample); the window
    samples ``resolution_out`` points at spacing ``(2*pi / resolution) /
    magnification`` (``magnification > 1`` zooms in), centred and offset by
    ``shift`` (rad/sample). ``shift`` and ``magnification`` are ``(x, y)``;
    ``resolution`` / ``resolution_out`` are ``(height, width)``.

    Returns the 1D omega arrays ``(omega_x, omega_y)``.
    """

    def axis(length_in: int, length_out: int, mag: float, offset: float) -> Tensor:
        step = (2 * torch.pi / length_in) / mag
        indices = torch.arange(
            -(length_out // 2), length_out - length_out // 2, device=device
        )
        return indices * step + offset

    omega_x = axis(resolution[1], resolution_out[1], magnification[0], shift[0])
    omega_y = axis(resolution[0], resolution_out[0], magnification[1], shift[1])
    return omega_x, omega_y
