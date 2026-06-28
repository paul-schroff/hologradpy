from __future__ import annotations

import math

import torch
from torch import Tensor


def _pad_axis(field: Tensor, axis: int, pad: int) -> Tensor:
    """Zero-pad ``pad`` samples on both sides of ``axis`` (complex-safe)."""
    shape = list(field.shape)
    shape[axis] = pad
    zeros = torch.zeros(shape, dtype=field.dtype, device=field.device)
    return torch.cat([zeros, field, zeros], dim=axis)


def _fft_shear(field: Tensor, axis: int, shifts: Tensor) -> Tensor:
    """Shear along ``axis`` by an exact, band-limited per-line shift.

    Each line along ``axis`` is translated by the corresponding entry of
    ``shifts`` (one shift per line of the *other* axis) using the Fourier shift
    theorem. The axis is zero-padded first so the cyclic FFT shift does not wrap
    real signal around the edge.
    """
    length = field.shape[axis]
    pad = int(math.ceil(float(shifts.abs().max()))) + 1
    padded = _pad_axis(field, axis, pad)
    padded_length = length + 2 * pad

    k = torch.fft.fftfreq(padded_length, device=field.device)  # cycles/sample
    # Phase ramp exp(-2j*pi*k*shift); broadcast (n_lines, padded_length) against
    # the two transformed dims (..., axis=-2, axis=-1).
    if axis == -1:
        phase = torch.exp(-2j * math.pi * k[None, :] * shifts[:, None])
    else:
        phase = torch.exp(-2j * math.pi * k[:, None] * shifts[None, :])

    spectrum = torch.fft.fft(padded, dim=axis) * phase
    sheared = torch.fft.ifft(spectrum, dim=axis)

    index = torch.arange(pad, pad + length, device=field.device)
    return sheared.index_select(axis, index)


def shear_rotate(field: Tensor, angle: float | Tensor) -> Tensor:
    """Rotate ``field`` ``(*batch, H, W)`` about its centre by ``angle`` (radians)
    using three FFT shears.

    Decomposes ``R = Sx(-tan(theta/2)) . Sy(sin theta) . Sx(-tan(theta/2))``, each
    shear an exact band-limited per-line shift. The rotation is area-preserving
    (so power-conserving) and differentiable w.r.t. the field. Valid for
    ``|angle| < pi/2``; for larger rotations pre-rotate by a multiple of 90
    degrees (exact transpose/flip) and apply the residual here.

    ``angle`` may be a Python float (fast path) or a 0-d :class:`torch.Tensor`,
    in which case the rotation is also differentiable w.r.t. the angle (so it can
    be a learnable parameter) -- the shears are then computed with torch ops and
    the ``angle == 0`` short circuit is skipped so a gradient still flows at 0.
    """
    if torch.is_tensor(angle):
        return _shear_rotate_tensor(field, angle)

    if angle == 0.0:
        return field

    # Centre matches get_pixel_grid / fftshift convention (index N // 2).
    height, width = field.shape[-2:]
    rows = torch.arange(height, device=field.device) - height // 2
    cols = torch.arange(width, device=field.device) - width // 2

    shear_x = -math.tan(angle / 2)
    shear_y = math.sin(angle)

    field = _fft_shear(field, -1, shear_x * rows)   # shift x per row y
    field = _fft_shear(field, -2, shear_y * cols)   # shift y per column x
    field = _fft_shear(field, -1, shear_x * rows)
    return field


def _shear_rotate_tensor(field: Tensor, angle: Tensor) -> Tensor:
    """Differentiable (w.r.t. ``angle``) variant of :func:`shear_rotate` for a
    0-d tensor angle. No ``angle == 0`` short circuit, so a gradient flows even
    at zero rotation."""
    height, width = field.shape[-2:]
    rows = (torch.arange(height, device=field.device) - height // 2).to(angle.dtype)
    cols = (torch.arange(width, device=field.device) - width // 2).to(angle.dtype)

    shear_x = -torch.tan(angle / 2)
    shear_y = torch.sin(angle)

    field = _fft_shear(field, -1, shear_x * rows)
    field = _fft_shear(field, -2, shear_y * cols)
    field = _fft_shear(field, -1, shear_x * rows)
    return field
