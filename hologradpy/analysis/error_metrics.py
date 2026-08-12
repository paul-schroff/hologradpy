"""Metrics characterising light potentials and wavefront errors."""

from __future__ import annotations

import math
from typing import TypeVar

import torch
from numpy.typing import NDArray

from array_api_compat import array_namespace, device

ArrayLike = TypeVar("ArrayLike", torch.Tensor, NDArray)


def normalize(img: ArrayLike, roi: ArrayLike, thres: float = 0.5) -> ArrayLike:
    """Normalises an image by the mean of its bright pixels in a region of interest.

    Only pixels brighter than ``thres * max(roi * img)`` are averaged, so the result
    does not depend on how much dark area the region happens to include.

    Args:
        img (ArrayLike): Input image.
        roi (ArrayLike): Binary mask containing the region of interest.
        thres (float, optional): Pixel value threshold as a fraction of the brightest
            pixel in the region. Defaults to 0.5.

    Returns:
        ArrayLike: Normalised image, the same shape as ``img``.
    """
    xp = array_namespace(img, roi)

    img_roi = img * roi
    threshold = thres * xp.max(img_roi)
    mask = img_roi > threshold
    return img / xp.mean(img[mask])


def fidelity(
    signal: ArrayLike,
    a_tar: ArrayLike,
    phi_tar: ArrayLike,
    a_out: ArrayLike,
    phi_out: ArrayLike,
) -> ArrayLike:
    """Fidelity between two electric fields in a region of interest.

    The normalised overlap integral, so it is 1 when the two fields match up to a global
    scale and phase, and 0 when they are orthogonal.

    Args:
        signal (ArrayLike): Binary mask containing the region of interest.
        a_tar (ArrayLike): Target amplitude pattern.
        phi_tar (ArrayLike): Target phase pattern.
        a_out (ArrayLike): Amplitude of the light potential.
        phi_out (ArrayLike): Phase of the light potential.

    Returns:
        ArrayLike: Fidelity in ``[0, 1]``, as a scalar of the input backend.
    """
    xp = array_namespace(signal, a_tar, phi_tar, a_out, phi_out)

    e_tar_s = (a_tar * xp.exp(1j * phi_tar)) * signal
    e_out_s = (a_out * xp.exp(1j * phi_out)) * signal

    fid = (
        xp.sum(e_tar_s * xp.conj(e_out_s))
        / (xp.sum(xp.abs(e_tar_s) ** 2) * xp.sum(xp.abs(e_out_s) ** 2)) ** 0.5
    )
    return xp.abs(fid) ** 2


def rms(
    signal: ArrayLike, i_target: ArrayLike, i_out: ArrayLike, frac: float = 0.5
) -> ArrayLike:
    """Normalised root-mean-squared error between two images in a region of interest.

    Only pixels brighter than ``(1 - frac) * max(i_target)`` inside the region are
    scored, so dark background does not dilute the result. Both images are normalised to
    unit sum over those pixels first, which makes the answer independent of exposure.

    Args:
        signal (ArrayLike): Binary mask containing the region of interest.
        i_target (ArrayLike): Target intensity pattern.
        i_out (ArrayLike): Intensity pattern of the light potential.
        frac (float, optional): Sets the brightness threshold as described above.
            Defaults to 0.5, where the threshold is half the peak.

    Returns:
        ArrayLike: Normalised RMS error, as a scalar of the input backend.
    """
    xp = array_namespace(signal, i_target, i_out)

    i_target = i_target * signal
    i_out = i_out * signal

    # Boolean throughout, where this used to build a 0/1 array of the mask's dtype. The
    # arithmetic below promotes it the same way, so the result is unchanged.
    mr_mask = i_target > (1 - frac) * xp.max(i_target)
    mr = int(xp.count_nonzero(mr_mask))

    i_target_w_norm = i_target * mr_mask / xp.sum(i_target * mr_mask)
    i_out_w_norm = i_out * mr_mask / xp.sum(i_out * mr_mask)

    # Outside the mask this divides zero by zero. Those entries are dropped by the
    # selection below, which is why the division is left as it is.
    n = (mr_mask * (i_out_w_norm - i_target_w_norm) / i_target_w_norm) ** 2

    return xp.sqrt(xp.sum(n[mr_mask]) / mr)


def rms_phase(phi: ArrayLike) -> ArrayLike:
    """Root-mean-squared deviation of a phase pattern about its own mean.

    Requires *unwrapped* phase. Passing the output of ``angle``, or any phase that has
    been wrapped into ``(-pi, pi]``, gives a meaningless answer as soon as the true
    phase leaves that interval, because the 2 pi jumps dominate the deviation. Use
    :func:`wavefront_rms` for a wrapped phase, or unwrap first with
    :func:`hologradpy.analysis.unwrapping.unwrap_2d_poisson`.

    Args:
        phi (ArrayLike): Unwrapped phase pattern.

    Returns:
        ArrayLike: RMS phase deviation, as a scalar of the input backend.
    """
    xp = array_namespace(phi)

    return xp.sqrt(xp.mean((phi - xp.mean(phi)) ** 2))


def psnr(signal: ArrayLike, i_target: ArrayLike, i_out: ArrayLike) -> ArrayLike:
    """Peak signal-to-noise ratio between two images in a region of interest.

    Follows https://doi.org/10.1364/OE.24.006249. Both images are normalised to unit sum
    over the region first, so the result does not depend on exposure.

    Args:
        signal (ArrayLike): Binary mask containing the region of interest.
        i_target (ArrayLike): Target intensity pattern.
        i_out (ArrayLike): Intensity pattern of the light potential.

    Returns:
        ArrayLike: Peak signal-to-noise ratio in dB, as a scalar of the input backend.
    """
    xp = array_namespace(signal, i_target, i_out)

    i_target_w = i_target * signal
    i_out_w = i_out * signal

    i_target_w_norm = i_target_w / xp.sum(i_target_w)
    i_out_w_norm = i_out_w / xp.sum(i_out_w)

    mr = int(xp.count_nonzero(signal))

    mse = xp.sum(signal * (i_out_w_norm - i_target_w_norm) ** 2) / mr

    return 20 * xp.log10(xp.max(i_target_w_norm * signal) / xp.sqrt(mse))


def eff(signal: ArrayLike, i_out: ArrayLike) -> ArrayLike:
    """Predicted efficiency of a light potential.

    The fraction of the total power landing inside the signal region.

    Args:
        signal (ArrayLike): Binary mask containing the signal region.
        i_out (ArrayLike): Intensity pattern of the light potential.

    Returns:
        ArrayLike: Efficiency in ``[0, 1]``, as a scalar of the input backend.
    """
    xp = array_namespace(signal, i_out)

    return xp.sum(i_out * signal) / xp.sum(i_out)


def remove_linear_phase(
    phasor: ArrayLike, mask: ArrayLike, iterations: int = 3
) -> ArrayLike:
    """Divides the mean linear phase ramp out of a complex phasor.

    A ramp across a pupil shifts the focal spot rather than degrading it. The slope is
    estimated as the mean phase step between neighbouring pixels, taken as
    ``angle(z[n + 1] * conj(z[n]))``. That step stays far below pi even when the
    accumulated phase spans many multiples of 2 pi, so the estimate is immune to
    wrapping, unlike fitting a plane to the phase itself.

    Iterated a few times, since removing most of the ramp makes the estimate of what is
    left more accurate.

    Args:
        phasor (ArrayLike): Complex array, e.g. ``exp(1j * phase)``.
        mask (ArrayLike): Binary mask selecting the region to fit over.
        iterations (int, optional): Number of refinement passes. Defaults to 3.

    Returns:
        ArrayLike: The phasor with the ramp divided out.
    """
    xp = array_namespace(phasor, mask)

    field = xp.asarray(phasor, copy=True)

    height, width = field.shape[-2], field.shape[-1]
    rows = xp.reshape(xp.arange(height, device=device(field)), (height, 1))
    columns = xp.reshape(xp.arange(width, device=device(field)), (1, width))

    for _ in range(iterations):
        interior_x = mask[:, 1:] & mask[:, :-1]
        interior_y = mask[1:, :] & mask[:-1, :]
        step_x = xp.mean(xp.angle(field[:, 1:] * xp.conj(field[:, :-1]))[interior_x])
        step_y = xp.mean(xp.angle(field[1:, :] * xp.conj(field[:-1, :]))[interior_y])
        field = field * xp.exp(-1j * (step_x * columns + step_y * rows))

    return field


def strehl_amplitude(
    phase_difference: ArrayLike, mask: ArrayLike, remove_ramp: bool = True
) -> float:
    """Fraction of the field amplitude surviving a phase error.

    Computes ``abs(mean(exp(1j * phase_difference)))``, the amplitude ratio. This is not
    the Strehl ratio as usually defined: that compares peak *intensities* and is the
    square of what this returns, so square it before comparing against a Strehl quoted
    elsewhere. The name says amplitude for exactly that reason.

    Works on a phasor throughout, so the phase may be wrapped: a 2 pi offset leaves
    ``exp(1j * phase)`` unchanged. Piston needs no special handling either, since a
    global phase does not change a magnitude.

    Args:
        phase_difference (ArrayLike): Phase error, wrapped or unwrapped.
        mask (ArrayLike): Binary mask selecting the region to average over.
        remove_ramp (bool, optional): Discount any mean tilt, which displaces the focus
            rather than aberrating it. Defaults to True.

    Returns:
        float: Amplitude ratio in ``(0, 1]``, whose square is the intensity Strehl
        ratio. Floored at 1e-12 so the logarithm in :func:`wavefront_rms` stays finite.
    """
    xp = array_namespace(phase_difference, mask)

    phasor = xp.exp(1j * phase_difference)
    if remove_ramp:
        phasor = remove_linear_phase(phasor, mask)
    return float(xp.clip(xp.abs(xp.mean(phasor[mask])), 1e-12, 1.0))


def wavefront_rms(
    phase_difference: ArrayLike, mask: ArrayLike, remove_ramp: bool = True
) -> float:
    """Equivalent RMS of a phase error in radians, safe on wrapped phase.

    Inverts the Marechal relation ``S = exp(-sigma ** 2)``, where ``S`` is the intensity
    Strehl ratio and so the *square* of what :func:`strehl_amplitude` returns. That
    squaring is why the expression below reads ``-2 * log`` rather than ``-log``.

    Exact whenever the phase error is Gaussian distributed, at any size, because
    ``mean(exp(1j * phi))`` is then exactly ``exp(-sigma ** 2 / 2)``. For other
    distributions it is exact only to leading order in sigma, and it reads high once the
    error is both large and far from Gaussian: phase spread uniformly over ``(-pi, pi]``
    has a true RMS of 1.81 but is reported as 3.23. It stays monotonic throughout, so it
    ranks reliably even where the value is no longer an RMS.

    Saturates at 7.43 radians, where :func:`strehl_amplitude` reaches its own floor of
    1e-12, so a result near that ceiling means only that the error is large.

    Args:
        phase_difference (ArrayLike): Phase error, wrapped or unwrapped.
        mask (ArrayLike): Binary mask selecting the region to average over.
        remove_ramp (bool, optional): Discount any mean tilt. Defaults to True.

    Returns:
        float: Equivalent RMS phase error in radians.
    """
    return math.sqrt(
        -2.0 * math.log(strehl_amplitude(phase_difference, mask, remove_ramp))
    )


def wavefront_residual(
    recovered_phase: ArrayLike,
    target_phase: ArrayLike,
    mask: ArrayLike,
    allow_sign_flip: bool = True,
) -> float:
    """Aberration left after applying a recovered wavefront, over what was there.

    Returns 1.0 when the correction buys nothing and 0.0 when it cancels the aberration
    completely, so it answers the question a calibration is actually for. Both phases
    may be wrapped, which matters because a recovered field's phase comes from ``angle``
    and is therefore always wrapped, while an aberration of any size is not.

    Args:
        recovered_phase (ArrayLike): Recovered wavefront phase, wrapped or unwrapped.
        target_phase (ArrayLike): Phase that was present, wrapped or unwrapped.
        mask (ArrayLike): Binary mask selecting the illuminated region.
        allow_sign_flip (bool, optional): Take whichever global sign leaves less behind.
            Intensity-only sensing recovers a wavefront up to a conjugate, since
            ``abs(FT(f)) == abs(FT(conj(f)))``, so the sign is not observable. Defaults
            to True.

    Returns:
        float: Residual aberration over the original, 1.0 meaning no improvement.
    """
    signs = (1.0, -1.0) if allow_sign_flip else (1.0,)
    corrected = min(
        wavefront_rms(target_phase - sign * recovered_phase, mask) for sign in signs
    )
    return corrected / wavefront_rms(target_phase, mask)
