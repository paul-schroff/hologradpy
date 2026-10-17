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

    The native DFT bin spacing of an axis of ``N`` samples is ``2 * pi / N``. The window
    samples ``resolution_out`` points at that spacing divided by the magnification, so a
    magnification above one zooms in. The points are centred on the offset ``shift``.

    Args:
        resolution: The ``(height, width)`` of the grid being transformed.
        resolution_out: The ``(height, width)`` of the window.
        magnification: The zoom ``(x, y)``.
        shift: The offset ``(x, y)`` of the window in rad/sample.
        device: Where to build the points, the CPU for None.
        output_roi: The rectangle of the window to sample, or None for all of it.

    Returns:
        tuple[Tensor, Tensor]: The points ``(omega_x, omega_y)`` along x and y, with a
        leading wavelength axis for an ``(n_wl,)`` magnification or shift.

    Raises:
        ValueError: ``output_roi`` reaches outside the window or spans fewer than 2
            pixels along an axis.
    """
    device = resolve_device(device)

    def axis(
        length_in: int, length_out: int, mag: float | Tensor, offset: float | Tensor
    ) -> Tensor:
        step = (2 * torch.pi / length_in) / mag
        if torch.is_tensor(step) and step.ndim > 0:
            step = step.unsqueeze(-1)
        if torch.is_tensor(offset) and offset.ndim > 0:
            offset = offset.unsqueeze(-1)
        return centered_indices(length_out, device) * step + offset

    omega_x = axis(resolution[1], resolution_out[1], magnification[0], shift[0])
    omega_y = axis(resolution[0], resolution_out[0], magnification[1], shift[1])
    if output_roi is None:
        return omega_x, omega_y
    if not output_roi.lies_inside(resolution_out) or min(
        output_roi.height, output_roi.width
    ) < 2:
        raise ValueError(
            f"The output region of interest {output_roi} has to lie inside the "
            f"{tuple(resolution_out)} output grid and span at least 2 pixels "
            "along each axis."
        )
    return omega_x[..., output_roi.columns], omega_y[..., output_roi.rows]


def rotated_zoom_trajectory(
    omega_x: Tensor, omega_y: Tensor, angle: float | Tensor
) -> Tensor:
    """The k-space sample points that turn the image of a zoom transform by ``angle``.

    The points are the grid of ``omega_x`` by ``omega_y`` turned by ``-angle``, so the
    grid point ``(gx, gy)`` becomes ``(gx * cos + gy * sin, gy * cos - gx * sin)``. A
    positive angle turns the image from +x towards +y.

    Args:
        omega_x: The ``(..., width)`` sample points along x, in rad/sample.
        omega_y: The ``(..., height)`` sample points along y, in rad/sample.
        angle: The angle in radians.

    Returns:
        Tensor: The points as ``(2, ..., height * width)``, x first, with the output
        pixels in row-major order.
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
        shift: Image translation ``(x, y)`` in output pixels.
        resolution: ``(height, width)`` of the grid being transformed.
        magnification: Zoom ``(x, y)``, as :func:`get_zoom_frequency_grid` takes it.

    Returns:
        The ``(x, y)`` offset in rad/sample. Broadcasts, so a per-wavelength
        magnification gives a per-wavelength offset.
    """
    height, width = resolution
    two_pi = 2 * torch.pi
    return (
        -two_pi * shift[0] / (width * magnification[0]),
        -two_pi * shift[1] / (height * magnification[1]),
    )
