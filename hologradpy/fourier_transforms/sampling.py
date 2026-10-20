"""Where the zoom transforms sample k-space.

The chirp-z and the NUFFT zooms sample the window of :func:`get_zoom_frequency_grid`,
turned by :func:`rotated_zoom_trajectory`. The two sample identical points, so one can
be checked against the other.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ..grids import centered_indices
from ..utils import resolve_device

if TYPE_CHECKING:
    from ..roi import ROI


def get_zoom_frequency_axes(
    resolution: tuple[int, int],
    resolution_out: tuple[int, int],
    magnification: tuple[float | Tensor, float | Tensor],
    shift: tuple[float | Tensor, float | Tensor] = (0.0, 0.0),
    device: torch.device | None = None,
    output_roi: ROI | None = None,
) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
    """The first k-space sample point and the step along each axis of a zoomed and
    shifted output window, in rad/sample and in double precision.

    The native DFT bin spacing of an axis of ``N`` samples is ``2 * pi / N``. The window
    samples ``resolution_out`` points at that spacing divided by the magnification, so a
    magnification above one zooms in. The points are centred on the offset ``shift``.
    Point ``k`` of an axis is ``first + k * step``, and
    :func:`get_zoom_frequency_grid` gives the points themselves.

    Args:
        resolution: The ``(height, width)`` of the grid being transformed.
        resolution_out: The ``(height, width)`` of the window.
        magnification: The zoom ``(column, row)``.
        shift: The offset ``(column, row)`` of the window in rad/sample.
        device: Where to build the values, the CPU for None.
        output_roi: The rectangle of the window to sample, or None for all of it.

    Returns:
        tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]: ``(first, step)`` along
        the columns and along the rows, in float64. For an ``(n_wl,)`` magnification or
        shift, each has a trailing axis of length 1 after the wavelength axis.

    Raises:
        ValueError: ``output_roi`` reaches outside the window or spans fewer than 2
            pixels along an axis.
    """
    device = resolve_device(device)
    if output_roi is not None and (
        not output_roi.lies_inside(resolution_out)
        or min(output_roi.height, output_roi.width) < 2
    ):
        raise ValueError(
            f"The output region of interest {output_roi} has to lie inside the "
            f"{tuple(resolution_out)} output grid and span at least 2 pixels "
            "along each axis."
        )

    def axis(
        length_in: int,
        length_out: int,
        mag: float | Tensor,
        offset: float | Tensor,
        first_index: int,
    ) -> tuple[Tensor, Tensor]:
        mag = torch.as_tensor(mag, dtype=torch.float64, device=device)
        offset = torch.as_tensor(offset, dtype=torch.float64, device=device)
        step = (2 * torch.pi / length_in) / mag
        if step.ndim > 0:
            step = step.unsqueeze(-1)
        if offset.ndim > 0:
            offset = offset.unsqueeze(-1)
        index = centered_indices(length_out, device)[first_index]
        return index * step + offset, step

    first_column = 0 if output_roi is None else output_roi.left_column
    first_row = 0 if output_roi is None else output_roi.top_row
    return (
        axis(
            resolution[1], resolution_out[1], magnification[0], shift[0], first_column
        ),
        axis(resolution[0], resolution_out[0], magnification[1], shift[1], first_row),
    )


def get_zoom_frequency_grid(
    resolution: tuple[int, int],
    resolution_out: tuple[int, int],
    magnification: tuple[float | Tensor, float | Tensor],
    shift: tuple[float | Tensor, float | Tensor] = (0.0, 0.0),
    device: torch.device | None = None,
    output_roi: ROI | None = None,
) -> tuple[Tensor, Tensor]:
    """The k-space sample points along each axis of a zoomed and shifted output window,
    in rad/sample.

    The points are those of :func:`get_zoom_frequency_axes`, in the default dtype, or in
    the dtype of a floating-point magnification or shift tensor that is wider.

    Args:
        resolution: The ``(height, width)`` of the grid being transformed.
        resolution_out: The ``(height, width)`` of the window.
        magnification: The zoom ``(column, row)``.
        shift: The offset ``(column, row)`` of the window in rad/sample.
        device: Where to build the points, the CPU for None.
        output_roi: The rectangle of the window to sample, or None for all of it.

    Returns:
        tuple[Tensor, Tensor]: The points ``(omega_x, omega_y)`` along the columns and
        the rows, with a leading wavelength axis for an ``(n_wl,)`` magnification or
        shift.

    Raises:
        ValueError: ``output_roi`` reaches outside the window or spans fewer than 2
            pixels along an axis.
    """
    (first_x, step_x), (first_y, step_y) = get_zoom_frequency_axes(
        resolution, resolution_out, magnification, shift, device, output_roi
    )
    height, width = (
        resolution_out
        if output_roi is None
        else (output_roi.height, output_roi.width)
    )
    dtype = torch.get_default_dtype()
    for value in (*magnification, *shift):
        if torch.is_tensor(value) and value.is_floating_point():
            dtype = torch.promote_types(dtype, value.dtype)

    def points(first: Tensor, step: Tensor, count: int) -> Tensor:
        index = torch.arange(count, dtype=torch.float64, device=first.device)
        return (first + index * step).to(dtype)

    return points(first_x, step_x, width), points(first_y, step_y, height)


def rotated_zoom_trajectory(
    omega_x: Tensor, omega_y: Tensor, angle: float | Tensor
) -> Tensor:
    """The k-space sample points that turn the image of a zoom transform by ``angle``.

    The points are the grid of ``omega_x`` by ``omega_y`` turned by ``-angle``, so the
    grid point ``(gx, gy)`` becomes ``(gx * cos + gy * sin, gy * cos - gx * sin)``. A
    positive angle turns the image from +column towards +row, clockwise on a screen
    with row 0 at the top.

    Args:
        omega_x: The ``(..., width)`` sample points along the columns, in rad/sample.
        omega_y: The ``(..., height)`` sample points along the rows, in rad/sample.
        angle: The angle in radians.

    Returns:
        Tensor: The points as ``(2, ..., height * width)``, the column component first,
        with the output pixels in row-major order.
    """
    grid_x, grid_y = torch.broadcast_tensors(
        omega_x[..., None, :], omega_y[..., :, None]
    )
    angle = torch.as_tensor(angle, dtype=grid_x.dtype, device=grid_x.device)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    return torch.stack(
        (
            (grid_x * cosine + grid_y * sine).flatten(-2),
            (grid_y * cosine - grid_x * sine).flatten(-2),
        ),
        dim=0,
    )


def window_offset_from_pixels(
    shift: tuple[float, float] | Tensor,
    resolution: tuple[int, int],
    magnification: tuple[float, float] | Tensor,
) -> tuple[Tensor, Tensor]:
    """The rad/sample window offset that moves the image by ``shift`` output pixels.
    This produces the ``shift`` that :func:`get_zoom_frequency_grid` takes.

    Args:
        shift: Image translation ``(column, row)`` in output pixels.
        resolution: ``(height, width)`` of the grid being transformed.
        magnification: Zoom ``(column, row)``, as :func:`get_zoom_frequency_grid`
            takes it.

    Returns:
        The ``(column, row)`` offset in rad/sample. Broadcasts, so a per-wavelength
        magnification gives a per-wavelength offset.
    """
    height, width = resolution
    two_pi = 2 * torch.pi
    return (
        -two_pi * shift[0] / (width * magnification[0]),
        -two_pi * shift[1] / (height * magnification[1]),
    )
