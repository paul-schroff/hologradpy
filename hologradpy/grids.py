"""Centred coordinate grids sharing one convention.

The origin of a plane of length ``n`` is the sample at index ``n // 2``. Column ``i``
sits at ``x = (i - n // 2) * pitch``, and row ``i`` at ``y = (n // 2 - i) * pitch``.
So y points up when row 0 is drawn at the top, and a positive angle, from +x towards
+y, turns counterclockwise.

The grid of an even-length axis is asymmetric. x runs from ``-n/2`` to ``n/2 - 1``
pitches, and y from ``n/2`` down to ``-(n/2 - 1)``.

Arrays are indexed ``image[row, column]``, with shapes ``(height, width)`` and pitches
``(y, x)``. Pixel positions and pixel-frame displacements are ``(column, row)``, in the
order of ``(x, y)``. Coordinates ``(x, y)`` are physical. :func:`pixel_frame_vector` and
:func:`pixel_frame_angle` convert vectors and angles between the two frames, and
:func:`metres_to_pixel` and :func:`pixel_to_metres` convert positions.
"""

from __future__ import annotations

import math
from typing import TypeVar

import torch
from torch import Tensor

from .utils import resolve_device

T = TypeVar("T")


def pixel_frame_vector(vector: tuple[T, T]) -> tuple[T, T]:
    """The ``(column, row)`` components of an ``(x, y)`` vector, or the ``(x, y)``
    components of a ``(column, row)`` vector.

    The row component is minus the y component, since y increases with decreasing row
    index. The function is its own inverse. It takes floats, arrays or tensors, and
    gradients pass through it.

    Args:
        vector: The two components.

    Returns:
        tuple[T, T]: The components in the other frame.
    """
    x, y = vector
    return x, -y


def pixel_frame_angle(angle_degrees: T) -> T:
    """The angle in the pixel frame, from +column towards +row, of an angle in the
    physical frame, from +x towards +y, or the other way round.

    The pixel frame is the mirror image of the physical one, so the angle is negated.
    The function is its own inverse. It takes floats, arrays or tensors.

    Args:
        angle_degrees: The angle in degrees.

    Returns:
        T: The angle in degrees in the other frame.
    """
    return -angle_degrees


def centered_indices(
    length: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """The centred indices ``i - length // 2`` of an axis of ``length`` samples.

    The indices are pixel-frame offsets, so along the rows they grow down the screen,
    towards -y. :func:`get_pixel_grid` converts them to ``(x, y)`` pixels.

    Args:
        length: The number of samples.
        device: Where to build the indices, the CPU for None.
        dtype: Dtype of the indices. Defaults to the integer one ``arange`` picks.
    """
    return torch.arange(
        -(length // 2), length - length // 2, device=resolve_device(device), dtype=dtype
    )


def get_pixel_grid(
    resolution: tuple[int, int],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor]:
    """Centred ``(x, y)`` coordinates in pixels, as a pair of grids. y increases with
    decreasing row index, so row 0 holds the largest y.

    Args:
        resolution: ``(height, width)`` in pixels.
        device: Where to build the grid, the CPU for None.
        dtype: Dtype of the indices. Defaults to the integer one ``arange`` picks.

    Returns:
        tuple[Tensor, Tensor]: The ``x`` and ``y`` grids.
    """
    height, width = resolution
    return pixel_frame_vector(
        torch.meshgrid(
            centered_indices(width, device, dtype),
            centered_indices(height, device, dtype),
            indexing="xy",
        )
    )


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
    """The pixel holding the origin, ``(width // 2, height // 2)``.

    Args:
        resolution: ``(height, width)`` in pixels.

    Returns:
        tuple[int, int]: The origin as ``(column, row)`` pixel indices.
    """
    height, width = resolution
    return (width // 2, height // 2)


def odd_sample_count(extent: float, pitch: float) -> int:
    """The number of samples at ``pitch`` that cover ``extent``, rounded up to an odd
    number of at least 3, so the grid has a centre sample.

    Args:
        extent: The length to cover, in the units of ``pitch``.
        pitch: The sample spacing.

    Returns:
        int: The odd sample count.
    """
    count = math.ceil(extent / pitch)
    return max(3, count + 1 - count % 2)


def metres_to_pixel(
    position: tuple[float, float],
    pixel_size: tuple[float, float],
    resolution: tuple[int, int],
) -> tuple[float, float]:
    """Plane ``(x, y)`` metres (relative to the center) to the pixel ``(column, row)``,
    the inverse of :func:`get_spatial_grid`.

    ``pixel_size`` is ``(y, x)`` metres and ``resolution`` is ``(height, width)``, so
    x uses the width pitch and y the height pitch.
    """
    center_column, center_row = plane_center(resolution)
    column, row = pixel_frame_vector(
        (position[0] / pixel_size[1], position[1] / pixel_size[0])
    )
    return column + center_column, row + center_row


def metres_to_pixels(
    shift: tuple[float, float],
    pixel_size: tuple[float, float],
) -> tuple[float, float]:
    """A shift ``(x, y)`` in metres as a shift ``(x, y)`` in pixels.

    ``pixel_size`` is ``(y, x)`` metres, so x is divided by the width pitch and y by
    the height pitch. The result stays a physical ``(x, y)`` shift with y pointing up,
    as a
    :class:`~hologradpy.optics.modules.learnable_partial_affine.LearnablePartialAffine`
    takes its ``shift``. :func:`metres_to_pixel` converts a position to the pixel
    ``(column, row)``.
    """
    return (
        float(shift[0]) / float(pixel_size[1]),
        float(shift[1]) / float(pixel_size[0]),
    )


def pixel_to_metres(
    pixel: tuple[float, float],
    pixel_size: tuple[float, float],
    resolution: tuple[int, int],
) -> tuple[float, float]:
    """The pixel ``(column, row)`` to plane ``(x, y)`` metres from the center. The
    inverse of :func:`metres_to_pixel`.
    """
    center_column, center_row = plane_center(resolution)
    x, y = pixel_frame_vector((pixel[0] - center_column, pixel[1] - center_row))
    return x * pixel_size[1], y * pixel_size[0]


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
        list[tuple[int, int]]: The ``(row, column)`` index of the pixel nearest each
        coordinate, as ``image[row, column]`` takes it.
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
