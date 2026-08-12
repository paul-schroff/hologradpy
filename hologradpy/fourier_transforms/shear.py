"""Framing a field, and shearing it by an exact band-limited shift.

The shear is what a rotation costs once ``ChirpZPartialAffine`` has absorbed everything
it can into its own sampling. The triangular part of a rotation is free there, and only
this is left.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from scipy.fft import next_fast_len


def _pad_axis_asymmetric(
    field: Tensor, axis: int, before: int, after: int
) -> Tensor:
    """Zero-pad ``before`` and ``after`` samples either side of ``axis`` (complex-safe).
    """

    def zeros(count: int) -> Tensor:
        shape = list(field.shape)
        shape[axis] = count
        return torch.zeros(shape, dtype=field.dtype, device=field.device)

    return torch.cat([zeros(before), field, zeros(after)], dim=axis)


def padded_resolution_for_rotation(
    resolution: tuple[int, int], angle_degrees: float
) -> tuple[int, int]:
    """The frame a rotation of ``angle_degrees`` needs to keep every input sample.

    A rotation carries the corners of a frame outside it, so a field that fills its
    frame loses them unless the frame grows. Deliberately generous, since the shear
    reaches further than the final rotation does.

    Args:
        resolution: The unpadded ``(height, width)``.
        angle_degrees: Rotation to be applied.

    Returns:
        tuple[int, int]: The ``(height, width)`` to rotate into.
    """
    reach = max(resolution) * abs(math.sin(math.radians(angle_degrees)))
    margin = int(math.floor(reach)) + 1
    return (resolution[0] + 2 * margin, resolution[1] + 2 * margin)


def place(field: Tensor, resolution: tuple[int, int]) -> Tensor:
    """Centre ``field``'s last two axes in a frame of ``resolution``, padding or
    cropping each axis as needed.
    """
    for axis, target in ((-2, resolution[0]), (-1, resolution[1])):
        length = field.shape[axis]
        if target == length:
            continue

        offset = target // 2 - length // 2
        if target > length:
            index = [slice(None)] * field.ndim
            shape = list(field.shape)
            shape[axis] = target
            placed = torch.zeros(shape, dtype=field.dtype, device=field.device)
            index[axis] = slice(offset, offset + length)
            placed[tuple(index)] = field
            field = placed
        else:
            index = [slice(None)] * field.ndim
            index[axis] = slice(-offset, -offset + target)
            field = field[tuple(index)]
    return field


def fft_shear(field: Tensor, axis: int, shifts: Tensor) -> Tensor:
    """Shear along ``axis`` by an exact, band-limited per-line shift.

    Each line along ``axis`` is translated by the corresponding entry of ``shifts`` (one
    shift per line of the *other* axis) using the Fourier shift theorem. The axis is
    zero-padded first so the cyclic FFT shift does not wrap real signal around the edge.
    """
    length = field.shape[axis]
    pad = int(math.floor(float(shifts.abs().max()))) + 1
    padded_length = next_fast_len(length + 2 * pad)
    padded = _pad_axis_asymmetric(field, axis, pad, padded_length - length - pad)

    k = torch.fft.fftfreq(padded_length, device=field.device)  # cycles/sample
    # Phase ramp exp(-2j*pi*k*shift). Broadcast (n_lines, padded_length) against
    # the two transformed dims (..., axis=-2, axis=-1).
    if axis == -1:
        phase = torch.exp(-2j * math.pi * k[None, :] * shifts[:, None])
    else:
        phase = torch.exp(-2j * math.pi * k[:, None] * shifts[None, :])

    spectrum = torch.fft.fft(padded, dim=axis) * phase
    sheared = torch.fft.ifft(spectrum, dim=axis)

    index = torch.arange(pad, pad + length, device=field.device)
    return sheared.index_select(axis, index)
