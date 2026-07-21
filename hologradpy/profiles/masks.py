"""Binary geometric masks on a 2D coordinate grid."""

from __future__ import annotations

from typing import TypeVar

import torch
from numpy.typing import NDArray

from array_api_compat import array_namespace

ArrayLike = TypeVar("ArrayLike", torch.Tensor, NDArray)


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
