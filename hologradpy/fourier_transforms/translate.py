from __future__ import annotations

import torch
from torch import Tensor

from .fft import fft_2d, ifft_2d
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


def translate_real(image: Tensor, shift: tuple[float, float]) -> Tensor:
    """A real ``image`` translated by ``shift``, the real part of :func:`fft_translate`,
    so the translation stays linear in the image.

    Args:
        image: The last two axes are translated.
        shift: ``(rows, columns)`` in samples.

    Returns:
        Tensor: The translated image, real.
    """
    moved = fft_translate(image, shift)
    return moved.real if moved.is_complex() else moved


def translate_intensity(image: Tensor, shift: tuple[float, float]) -> Tensor:
    """An intensity translated by ``shift``. Sharp edges might cause ringing."""
    amplitude = fft_translate(image.clamp(min=0.0).sqrt(), shift)
    return amplitude.abs().square().to(image.dtype)


def cross_correlation_peak(
    image: Tensor, pattern: Tensor, region: Tensor
) -> tuple[float, float]:
    """The whole-sample shift that moves ``pattern`` onto ``image``, at the peak of
    their cross-correlation inside ``region``.

    Both are taken about their mean inside the region and set to zero outside it, so
    light elsewhere on the frame, such as the zeroth order, plays no part. The
    correlation is cyclic, so the shift found lies within half the frame along each
    axis.

    Args:
        image: The image the pattern is moved onto, ``(H, W)``.
        pattern: The pattern on the same grid.
        region: True where the two are compared.

    Returns:
        tuple[float, float]: The shift ``(rows, columns)`` in whole samples, positive
        towards increasing index, as :func:`fft_translate` takes it.
    """
    dtype = torch.promote_types(image.dtype, torch.float32)
    image = image.to(dtype)
    pattern = pattern.to(dtype)
    weights = region.to(dtype)
    inside = region.to(torch.bool)
    image_centred = (image - image[inside].mean()) * weights
    pattern_centred = (pattern - pattern[inside].mean()) * weights
    # In the centred layout of fft_2d, the zero shift sits at the centre sample.
    correlation = ifft_2d(fft_2d(image_centred) * fft_2d(pattern_centred).conj()).real

    height, width = correlation.shape[-2:]
    row, column = divmod(int(torch.argmax(correlation)), width)
    return (float(row - height // 2), float(column - width // 2))

