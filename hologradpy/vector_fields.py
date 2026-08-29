"""Differential and integral operators on sampled 2D fields."""

from __future__ import annotations

import torch
from torch import Tensor


def integrate_along_path(
    field_x: Tensor,
    field_y: Tensor,
    pixel_size: tuple[float, float],
) -> Tensor:
    """Integrate a 2D vector field along an L-shaped path out from the origin.

    Args:
        field_x: X component on an ``(..., height, width)`` grid.
        field_y: Y component, same shape.
        pixel_size: Sample spacing ``(y, x)`` in metres.

    Returns:
        Tensor: The integral at every sample, same shape as the input, zero at the
            origin sample.
    """
    height, width = field_x.shape[-2:]

    origin_y, origin_x = height // 2, width // 2
    spacing_y, spacing_x = pixel_size

    # Trapezoid rule using cumulative sum
    row = field_x[..., origin_y, :]
    along_row = torch.cumsum(row, dim=-1) - row / 2
    along_row = (along_row - along_row[..., origin_x : origin_x + 1]) * spacing_x

    down_columns = torch.cumsum(field_y, dim=-2) - field_y / 2
    down_columns = (
        down_columns - down_columns[..., origin_y : origin_y + 1, :]
    ) * spacing_y

    return along_row[..., None, :] + down_columns


def gradient(
    potential: Tensor, pixel_size: tuple[float, float]
) -> tuple[Tensor, Tensor]:
    """Gradient of a scalar field by central differences, one-sided at the edges.

    The companion of :func:`integrate_along_path`, mainly so that a round trip can be
    tested.

    Args:
        potential: An ``(..., height, width)`` scalar field.
        pixel_size: Sample spacing ``(y, x)`` in metres.

    Returns:
        tuple[Tensor, Tensor]: The ``(x, y)`` components of the gradient.
    """
    spacing_y, spacing_x = pixel_size
    return (
        torch.gradient(potential, spacing=spacing_x, dim=-1)[0],
        torch.gradient(potential, spacing=spacing_y, dim=-2)[0],
    )


def forward_difference(field: Tensor) -> tuple[Tensor, Tensor]:
    """Differences between neighbouring samples, in pixels rather than metres.

    The cheap counterpart of :func:`gradient`.

    Args:
        field: An ``(..., height, width)`` scalar field.

    Returns:
        tuple[Tensor, Tensor]: The ``(x, y)`` differences, where x runs along the last
            axis. They are the width and the height respectively that are one shorter.
    """
    return torch.diff(field, dim=-1), torch.diff(field, dim=-2)


def mean_curvature(
    surface: Tensor, pixel_size: tuple[float, float] = (1.0, 1.0)
) -> Tensor:
    """Mean curvature of a scalar field read as a surface, by finite differences.

    Args:
        surface: An ``(..., height, width)`` scalar field, its value read as a height.
        pixel_size: Sample spacing ``(y, x)``, in the same units as the height so the
            result is a reciprocal length.

    Returns:
        Tensor: The mean curvature at each sample, same shape as the input.
    """
    spacing_y, spacing_x = pixel_size
    slope_y, slope_x = torch.gradient(
        surface, spacing=(spacing_y, spacing_x), dim=(-2, -1), edge_order=2
    )

    curvature_xy, curvature_xx = torch.gradient(
        slope_x, spacing=(spacing_y, spacing_x), dim=(-2, -1), edge_order=2
    )
    curvature_yy, _ = torch.gradient(
        slope_y, spacing=(spacing_y, spacing_x), dim=(-2, -1), edge_order=2
    )

    return (
        0.5
        * (
            (1 + slope_x**2) * curvature_yy
            + (1 + slope_y**2) * curvature_xx
            - 2 * slope_x * slope_y * curvature_xy
        )
        / (1 + slope_x**2 + slope_y**2) ** 1.5
    )
