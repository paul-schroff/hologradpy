from __future__ import annotations

import torch
from torch import Tensor

from .shear import fft_shear


def fft_translate(field: Tensor, shift: tuple[float, float]) -> Tensor:
    """``field`` translated by ``shift``, given as ``(rows, columns)`` samples.

    Positive values move towards increasing index, matching how a pattern is pasted.
    Either component may be fractional, and a whole-sample component costs no more.

    Args:
        field: The last two axes are translated.
        shift: ``(rows, columns)`` in samples.

    Returns:
        Tensor: The translated field, complex as the transform leaves it. A shift of
        nothing returns the field itself, real or complex as it came in. Take
        :func:`translate_intensity` for a real, non-negative image.
    """
    rows, columns = float(shift[0]), float(shift[1])
    shifted = field

    if columns:
        along_rows = torch.full(
            (shifted.shape[-2],), columns, device=field.device
        )
        shifted = fft_shear(shifted, -1, along_rows)
    if rows:
        along_columns = torch.full(
            (shifted.shape[-1],), rows, device=field.device
        )
        shifted = fft_shear(shifted, -2, along_columns)
    return shifted


def translate_intensity(image: Tensor, shift: tuple[float, float]) -> Tensor:
    """An intensity translated by ``shift``. Sharp edges might cause ringing."""
    amplitude = fft_translate(image.clamp(min=0.0).sqrt(), shift)
    return amplitude.abs().square().to(image.dtype)

