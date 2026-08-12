"""Where the zoom transforms sample k-space.

Shared by the chirp-z and NUFFT zooms so they sample identical points, which is what
lets one be checked against the other.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


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

