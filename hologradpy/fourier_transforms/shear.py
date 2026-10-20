"""Shearing a field by a shift that changes from line to line.

The shear is what a rotation costs once ``ChirpZPartialAffine`` has absorbed everything
it can into its own sampling. The triangular part of a rotation is free there, and only
this is left.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional

from scipy.fft import next_fast_len
from scipy.special import erfcinv


def fft_shear(field: Tensor, axis: int, shifts: Tensor) -> Tensor:
    """Shear along ``axis`` by a per-line shift, exact at the DFT bins of the padded
    line.

    Each line along ``axis`` is translated by the corresponding entry of ``shifts`` (one
    shift per line of the *other* axis) using the Fourier shift theorem. The axis is
    zero-padded first so the cyclic FFT shift does not wrap real signal around the edge.
    """
    length = field.shape[axis]
    pad = math.floor(float(shifts.detach().abs().max())) + 1
    padded_length = next_fast_len(length + 2 * pad)
    margins = (pad, padded_length - length - pad)
    # functional.pad takes the margins of the last axis first.
    padded = functional.pad(field, margins if axis == -1 else (0, 0, *margins))

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


def _window_transition(
    band: tuple[float, float], tolerance: float
) -> tuple[float, float]:
    """The width of each fall of the shear window and the scale of its error functions,
    both in rad/sample.

    Each fall takes half the gap that ``band`` leaves on the circle of frequencies. The
    error functions are scaled so that the window is within ``tolerance`` of one at the
    edges of the band.

    Raises:
        ValueError: The band covers the whole circle.
    """
    low, high = band
    gap = 2 * math.pi - (high - low)
    if gap <= 0:
        raise ValueError(
            f"The band {band} spans {high - low:.3f} rad/sample and covers the whole "
            "circle of frequencies, so it leaves no room for a window."
        )
    transition = gap / 2
    return transition, transition / (2 * float(erfcinv(tolerance)))


def _shear_window(
    offsets: Tensor, band: tuple[float, float], tolerance: float
) -> Tensor:
    """The window of :func:`windowed_fft_shear` at ``offsets`` from the centre of the
    band, in rad/sample within ``[-pi, pi)``.

    The window is the band widened by half its gap at each edge, convolved with a
    Gaussian. It is within ``tolerance`` of one on the band and of zero at the frequency
    opposite the centre of the band.
    """
    low, high = band
    transition, erf_scale = _window_transition(band, tolerance)
    edge = (high - low) / 2 + transition / 2
    return 0.5 * (
        torch.special.erf((offsets + edge) / erf_scale)
        - torch.special.erf((offsets - edge) / erf_scale)
    )


def windowed_shift_half_width(band: tuple[float, float], tolerance: float) -> int:
    """The samples that a shift of :func:`windowed_fft_shear` spreads a sample over,
    on each side of the shifted position.

    The kernel of the window is a sinc times ``exp(-(scale * m) ** 2 / 4)``, with
    ``scale`` the scale of the error functions of the window. The returned half-width
    is where this envelope falls to ``tolerance``.

    Args:
        band: The band ``(low, high)`` in rad/sample.
        tolerance: The largest error the window leaves on the band.

    Returns:
        int: The half-width in samples.
    """
    _, erf_scale = _window_transition(band, tolerance)
    return math.ceil(2 * math.sqrt(math.log(1 / tolerance)) / erf_scale)


def windowed_fft_shear(
    field: Tensor,
    axis: int,
    shifts: Tensor,
    band: tuple[float, float],
    tolerance: float,
) -> Tensor:
    """Shear along ``axis`` by a per-line shift, exact at every frequency of ``band``.

    The phase ramp of each line is multiplied by a window, equal to one on ``band``
    within ``tolerance`` and falling to zero opposite it, so the shift spreads each
    sample over :func:`windowed_shift_half_width` samples. The ramp is evaluated on the
    interval of ``2 * pi`` centred on the band, so its jump falls opposite the band,
    where the window is zero, and a band beyond ``+-pi`` is shifted by its own
    frequencies. The FFT runs over the whole axis with no padding, and the caller
    leaves the largest shift plus that half-width free at both ends. With ``-shifts``
    the function is its own adjoint.

    Args:
        field: The field to shear.
        axis: The axis along which the lines are shifted, -1 or -2.
        shifts: One shift in samples per line of the other axis.
        band: The band ``(low, high)`` in rad/sample, narrower than ``2 * pi``.
        tolerance: The largest error the window leaves on the band.

    Returns:
        Tensor: The sheared field, with the shape of ``field``.
    """
    length = field.shape[axis]
    bins = 2 * math.pi * torch.fft.fftfreq(
        length, device=field.device, dtype=shifts.dtype
    )
    centre = (band[0] + band[1]) / 2
    offsets = torch.remainder(bins - centre + math.pi, 2 * math.pi) - math.pi
    window = _shear_window(offsets, band, tolerance)
    frequencies = centre + offsets
    if axis == -1:
        ramp = torch.exp(-1j * frequencies[None, :] * shifts[:, None]) * window[None, :]
    else:
        ramp = torch.exp(-1j * frequencies[:, None] * shifts[None, :]) * window[:, None]
    return torch.fft.ifft(torch.fft.fft(field, dim=axis) * ramp, dim=axis)
