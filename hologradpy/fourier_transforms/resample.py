from __future__ import annotations

import torch
from torch import Tensor

from ..utils import to_canvas


def fft_resample(field: Tensor, resolution: tuple[int, int]) -> Tensor:
    """``field`` resampled onto ``resolution`` by padding or cropping its spectrum.
    Patterns with sharp edges will cause ringing.

    Args:
        field: The last two axes are resampled. Leading batch and wavelength axes are
            left alone.
        resolution: The ``(height, width)`` to resample onto.

    Returns:
        Tensor: The field on ``resolution``, scaled so amplitudes are preserved.
        Complex as the transform leaves it, except that a real input comes back real.
    """
    source = tuple(field.shape[-2:])
    resolution = tuple(int(length) for length in resolution)
    if resolution == source:
        return field

    was_real = not field.is_complex()

    spectrum = torch.fft.fftshift(torch.fft.fft2(field), dim=(-2, -1))
    spectrum = to_canvas(spectrum, resolution)
    scale = (resolution[0] * resolution[1]) / (source[0] * source[1])
    resampled = torch.fft.ifft2(torch.fft.ifftshift(spectrum, dim=(-2, -1))) * scale

    return resampled.real if was_real else resampled
