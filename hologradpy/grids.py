"""Centred coordinate grids sharing one convention.

The origin of a plane of length ``n`` is the sample at index ``n // 2``, so sample ``i``
sits at ``(i - n // 2) * pitch``.

The grid of an even-length axis is asymmetric, running ``-n/2`` to ``n/2 - 1``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .utils import resolve_device


def get_pixel_grid(
    resolution: tuple[int, int],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor]:
    """Centred pixel indices, as an ``(x, y)`` pair of grids.

    Args:
        resolution: ``(height, width)`` in pixels.
        device: Where to build the grid, the CPU for None.
        dtype: Dtype of the indices. Defaults to the integer one ``arange`` picks.

    Returns:
        tuple[Tensor, Tensor]: The ``x`` and ``y`` index grids.
    """
    device = resolve_device(device)
    height, width = resolution

    pixel_indices_x = torch.arange(
        -(width // 2), width - width // 2, device=device, dtype=dtype
    )
    pixel_indices_y = torch.arange(
        -(height // 2), height - height // 2, device=device, dtype=dtype
    )

    return torch.meshgrid(pixel_indices_x, pixel_indices_y, indexing="xy")


def get_spatial_grid(
    resolution: tuple[int, int],
    pixel_size: tuple[float, float] | Tensor,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """Centred ``(x, y)`` coordinates in metres, one point per pixel.

    The grid is returned in ``pixel_size``'s dtype if a tensor is passed.

    Args:
        resolution: ``(height, width)`` in pixels.
        pixel_size: ``(height, width)`` pitch in metres, as a pair or a tensor.
        device: Where to build the grid, the CPU for None.

    Returns:
        tuple[Tensor, Tensor]: The ``x`` and ``y`` coordinate grids.
    """
    device = resolve_device(device)
    pixel_size = torch.as_tensor(pixel_size, device=device)
    extent = torch.as_tensor(resolution, device=device) * pixel_size

    pixel_grid_x, pixel_grid_y = get_pixel_grid(
        resolution, device, dtype=pixel_size.dtype
    )

    spatial_grid_x = pixel_grid_x / resolution[1] * extent[1]
    spatial_grid_y = pixel_grid_y / resolution[0] * extent[0]

    return spatial_grid_x, spatial_grid_y


def plane_center(resolution: tuple[int, int]) -> tuple[int, int]:
    """The ``(x, y)`` pixel holding the origin, ``(width // 2, height // 2)``.

    Args:
        resolution: ``(height, width)`` in pixels.

    Returns:
        tuple[int, int]: The origin as ``(x, y)`` pixel indices.
    """
    height, width = resolution
    return (width // 2, height // 2)


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
    pixel_size: tuple[float, float] | Tensor,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """Centred angular frequencies in rad/m, one per sample.

    Spaced by ``2 * pi / (n * pitch)`` and spanning ``2 * pi / pitch``.

    Args:
        resolution: ``(height, width)`` in pixels.
        pixel_size: ``(height, width)`` pitch in metres, as a pair or a tensor. The
            grid comes back in its dtype.
        device: Where to build the grid, the CPU for None.

    Returns:
        tuple[Tensor, Tensor]: The ``x`` and ``y`` frequency grids.
    """
    device = resolve_device(device)
    pixel_size = torch.as_tensor(pixel_size, device=device)
    frequency_extent = 2 * torch.pi / pixel_size

    pixel_grid_x, pixel_grid_y = get_pixel_grid(
        resolution, device, dtype=pixel_size.dtype
    )

    frequency_grid_x = pixel_grid_x / resolution[1] * frequency_extent[1]
    frequency_grid_y = pixel_grid_y / resolution[0] * frequency_extent[0]

    return frequency_grid_x, frequency_grid_y


def coordinates_to_indices(
    x: torch.Tensor,
    y: torch.Tensor,
    coordinates: torch.Tensor,
) -> list[tuple[int, int]]:
    """Convert coordinates to pixel indices.

    Args:
        x: The x-coordinates of the spatial grid.
        y: The y-coordinates of the spatial grid.
        coordinates: The coordinates to convert. x-coordinates are in
            ``coordinates[:, 0]`` and y-coordinates are in
            ``coordinates[:, 1]``.

    Returns:
        list[tuple[int, int]]: The indices of the coordinates.
    """
    indices: list[tuple[int, int]] = []
    for i in range(coordinates.shape[0]):
        indices.append(
            torch.unravel_index(
                torch.argmin(
                    (x - coordinates[i, 0]).abs() + (y - coordinates[i, 1]).abs()
                ),
                x.shape,
            )
        )
    return indices
