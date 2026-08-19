"""Where the zoom transforms sample k-space.

Shared by the chirp-z and NUFFT zooms so they sample identical points, so one can be
checked against the other.
"""

from __future__ import annotations


import torch
from torch import Tensor


def get_zoom_frequency_grid(
    resolution: tuple[int, int],
    resolution_out: tuple[int, int],
    magnification: tuple[float, float],
    shift: tuple[float, float] = (0.0, 0.0),
    device: torch.device = "cpu",
) -> tuple[Tensor, Tensor]:
    """Per-axis k-space sample points (rad/sample) of a scaled + shifted output
    window, shared by the zoom transforms (NUFFT, chirp-z) so they sample
    identical points.

    The native DFT bin spacing is ``2*pi / resolution`` (rad/sample); the window
    samples ``resolution_out`` points at spacing ``(2*pi / resolution) /
    magnification`` (``magnification > 1`` zooms in), centered and offset by
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
