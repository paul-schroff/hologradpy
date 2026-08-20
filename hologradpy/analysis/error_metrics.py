"""Metrics characterizing light potentials and wavefront errors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

import torch
from numpy.typing import NDArray

from array_api_compat import array_namespace, device

ArrayLike = TypeVar("ArrayLike", torch.Tensor, NDArray)


def normalize(
    image: ArrayLike, roi: ArrayLike, threshold: float = 0.5
) -> ArrayLike:
    """Normalizes an image by the mean of its bright pixels in a region of interest.

    Only pixels brighter than ``threshold * max(roi * image)`` are averaged, so the
    result does not depend on how much dark area the region happens to include.

    Args:
        image: Input image.
        roi: Binary mask containing the region of interest.
        threshold: Pixel value threshold as a fraction of the brightest pixel in the
            region. Defaults to 0.5.

    Returns:
        ArrayLike: Normalized image, the same shape as ``image``.
    """
    xp = array_namespace(image, roi)

    image_roi = image * roi

    threshold_intensity = threshold * xp.max(image_roi)
    return image / xp.mean(image[image_roi > threshold_intensity])


def fidelity(
    signal_region: ArrayLike,
    target_amplitude: ArrayLike,
    target_phase: ArrayLike,
    measured_amplitude: ArrayLike,
    measured_phase: ArrayLike,
) -> ArrayLike:
    """Fidelity between two complex amplitudes in a region of interest.

    Args:
        signal_region: Binary mask containing the region of interest.
        target_amplitude: Target amplitude pattern.
        target_phase: Target phase pattern.
        measured_amplitude: Measured amplitude pattern.
        measured_phase: Measured phase pattern.

    Returns:
        ArrayLike: Fidelity in ``[0, 1]``, as a scalar of the input backend.
    """
    xp = array_namespace(
        signal_region, 
        target_amplitude, 
        target_phase, 
        measured_amplitude, 
        measured_phase
    )

    target_complex_amplitude = (
        (target_amplitude * xp.exp(1j * target_phase)) * signal_region
    )
    measured_complex_amplitude = (
        (measured_amplitude * xp.exp(1j * measured_phase)) * signal_region
    )

    fidelity = (
        xp.sum(target_complex_amplitude * xp.conj(measured_complex_amplitude))
        / (
            xp.sum(xp.abs(target_complex_amplitude) ** 2) 
            * xp.sum(xp.abs(measured_complex_amplitude) ** 2)
        ) ** 0.5
    )
    return xp.abs(fidelity) ** 2


def rmse(
    signal: ArrayLike,
    target_intensity: ArrayLike,
    measured_intensity: ArrayLike,
    threshold: float = 0.5,
) -> ArrayLike:
    """Normalized root-mean-squared error between two images in a region of interest.

    Only pixels brighter than ``threshold * max(target_intensity * signal)`` are
    considered, to avoid small values in the denominator blowing up the error. Both
    images are normalized to unit sum over those pixels first, which makes the answer
    independent of exposure.

    Args:
        signal: Binary mask containing the region of interest.
        target_intensity: Target intensity pattern.
        measured_intensity: Measured intensity pattern.
        threshold: Brightness threshold as a fraction of the target's peak inside the
            region. Defaults to 0.5, which is half the peak.

    Returns:
        ArrayLike: Normalized RMS error, as a scalar of the input backend.
    """
    xp = array_namespace(signal, target_intensity, measured_intensity)

    target_intensity_masked = target_intensity * signal
    measured_intensity_masked = measured_intensity * signal

    metric_mask = target_intensity_masked > threshold * xp.max(target_intensity_masked)

    target_bright = target_intensity_masked[metric_mask]
    measured_bright = measured_intensity_masked[metric_mask]

    target_normalized = target_bright / xp.sum(target_bright)
    measured_normalized = measured_bright / xp.sum(measured_bright)

    relative_error = (measured_normalized - target_normalized) / target_normalized
    return xp.sqrt(xp.mean(relative_error**2))


def rmse_phase(phase: ArrayLike) -> ArrayLike:
    """Root-mean-squared error of a phase pattern about its own mean.

    Requires *unwrapped* phase. Use :func:`wavefront_rms` for a wrapped phase, or
    unwrap first with :func:`hologradpy.analysis.unwrapping.unwrap_2d_poisson`.

    Args:
        phase: Unwrapped phase pattern.

    Returns:
        ArrayLike: RMS phase deviation, as a scalar of the input backend.
    """
    xp = array_namespace(phase)

    return xp.sqrt(xp.mean((phase - xp.mean(phase)) ** 2))


def psnr(
    signal_region: ArrayLike, 
    target_intensity: ArrayLike, 
    measured_intensity: ArrayLike
) -> ArrayLike:
    """Peak signal-to-noise ratio between two images in a region of interest.

    Follows https://doi.org/10.1364/OE.24.006249. Both images are normalized to unit sum
    over the region first, so the result does not depend on exposure.

    Args:
        signal_region: Binary mask containing the region of interest.
        target_intensity: Target intensity pattern.
        measured_intensity: Intensity pattern of the light potential.

    Returns:
        ArrayLike: Peak signal-to-noise ratio in dB, as a scalar of the input backend.
    """
    xp = array_namespace(signal_region, target_intensity, measured_intensity)

    signal_region = signal_region > 0
    target_intensity_signal = target_intensity[signal_region]
    measured_intensity_signal = measured_intensity[signal_region]

    target_intensity_normalized = (
        target_intensity_signal / xp.sum(target_intensity_signal)
    )
    measured_intensity_normalized = (
        measured_intensity_signal / xp.sum(measured_intensity_signal)
    )

    mean_squared_error = (
        xp.mean((measured_intensity_normalized - target_intensity_normalized) ** 2)
    )
    return (
        20 * xp.log10(xp.max(target_intensity_normalized) / xp.sqrt(mean_squared_error))
    )


def efficiency(signal_region: ArrayLike, measured_intensity: ArrayLike) -> ArrayLike:
    """Predicted efficiency of a light potential.

    The fraction of the total power landing inside the signal region.

    Args:
        signal_region: Binary mask containing the signal region.
        measured_intensity: Intensity pattern of the light potential.

    Returns:
        ArrayLike: Efficiency in ``[0, 1]``, as a scalar of the input backend.
    """
    xp = array_namespace(signal_region, measured_intensity)

    return xp.sum(measured_intensity * signal_region) / xp.sum(measured_intensity)


@dataclass(frozen=True)
class IntensityMetric:
    """A metric comparing a light potential against a target within a ``signal_region``.

    Args:
        name: Label for the metric.
        function: Takes ``(signal_region, target_intensity, measured_intensity)`` and
            returns a scalar.
        lower_is_better: Flag indicating which direction is better.
    """

    name: str
    function: Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike]
    lower_is_better: bool = True

    def __call__(
        self,
        signal_region: ArrayLike,
        target_intensity: ArrayLike,
        measured_intensity: ArrayLike,
    ) -> float:
        return float(
            self.function(signal_region, target_intensity, measured_intensity)
        )


DEFAULT_INTENSITY_METRICS: tuple[IntensityMetric, ...] = (
    IntensityMetric("rmse", rmse),
    IntensityMetric("psnr [dB]", psnr, lower_is_better=False),
)


@dataclass(frozen=True)
class WavefrontMetric:
    """A metric comparing two phase patterns within a ``mask``.

    Args:
        name: Label for the metric.
        function: Takes ``(recovered_phase, reference_phase, mask)`` and returns a
            scalar.
    """

    name: str
    function: Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike]

    def __call__(
        self,
        recovered_phase: ArrayLike,
        reference_phase: ArrayLike,
        mask: ArrayLike,
    ) -> float:
        return float(self.function(recovered_phase, reference_phase, mask))


DEFAULT_WAVEFRONT_METRICS: tuple[WavefrontMetric, ...] = (
    WavefrontMetric(
        "residual_phase_rms",
        lambda recovered, reference, mask: wavefront_rms(reference - recovered, mask),
    ),
    WavefrontMetric(
        "residual_fraction",
        lambda recovered, reference, mask: wavefront_residual(
            recovered, reference, mask, allow_sign_flip=False
        ),
    ),
)


def evaluate_wavefront_metrics(
    metrics: Sequence[WavefrontMetric],
    recovered_phase: ArrayLike,
    reference_phase: ArrayLike,
    mask: ArrayLike,
) -> dict[str, float]:
    return {
        metric.name: metric(recovered_phase, reference_phase, mask)
        for metric in metrics
    }


def evaluate_metrics(
    metrics: Sequence[IntensityMetric],
    signal_region: ArrayLike,
    target_intensity: ArrayLike,
    measured_intensity: ArrayLike,
    history: dict[str, list[float]] | None = None,
) -> dict[str, list[float]]:
    """Evaluate the metrics and append the results to ``history``.

    Args:
        metrics: The metrics to evaluate.
        signal_region: Binary mask containing the region of interest.
        target_intensity: The intensity that was asked for.
        measured_intensity: The intensity that came out.
        history: Where to append. A new dictionary is started when None.

    Returns:
        dict[str, list[float]]: ``history``, with one more entry per metric.
    """
    if history is None:
        history = {}
    for metric in metrics:
        history.setdefault(metric.name, []).append(
            metric(signal_region, target_intensity, measured_intensity)
        )
    return history


def remove_linear_phase(
    phasor: ArrayLike, mask: ArrayLike, iterations: int = 3
) -> ArrayLike:
    """Divides the mean linear phase ramp out of a complex phasor.

    A ramp across a pupil shifts the focal spot without degrading it. The slope is
    estimated as the mean phase step between neighbouring pixels, taken as
    ``angle(z[n + 1] * conj(z[n]))``. That step stays far below pi even when the
    accumulated phase spans many multiples of 2 pi, so the estimate is immune to
    wrapping.

    Iterated a few times, since removing most of the ramp makes the estimate of what is
    left more accurate.

    Args:
        phasor: Complex array, e.g. ``exp(1j * phase)``.
        mask: Binary mask selecting the region to fit over.
        iterations: Number of refinement passes. Defaults to 3.

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

    Computes ``abs(mean(exp(1j * phase_difference)))``, the amplitude ratio. The Strehl
    ratio as usually defined compares peak *intensities*, so it is the square of this
    value. Square this value before comparing against a Strehl quoted elsewhere. The
    name says amplitude for exactly that reason.

    Works on a phasor throughout, so the phase may be wrapped: a 2 pi offset leaves
    ``exp(1j * phase)`` unchanged. Piston needs no special handling either, since a
    global phase does not change a magnitude.

    Args:
        phase_difference: Phase error, wrapped or unwrapped.
        mask: Binary mask selecting the region to average over.
        remove_ramp: Discount any mean tilt, which displaces the focus without
            aberrating it. Defaults to True.

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

    Args:
        phase_difference: Phase error, wrapped or unwrapped.
        mask: Binary mask selecting the region to average over.
        remove_ramp: Discount any mean tilt. Defaults to True.

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

    Args:
        recovered_phase: Recovered wavefront phase, wrapped or unwrapped.
        target_phase: Phase that was present, wrapped or unwrapped.
        mask: Binary mask selecting the illuminated region.
        allow_sign_flip: Take whichever global sign leaves less behind. Intensity-only
            sensing recovers a wavefront up to a conjugate, since
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


def captured_power(
    measured_intensity: ArrayLike,
    pixel_area: float,
    region: ArrayLike | None = None,
) -> ArrayLike:
    """Optical power in a frame, or in a region of it: ``sum(I) * pixel_area``.

    Args:
        measured_intensity: Intensity on the output grid.
        pixel_area: Area of one output pixel in square metres.
        region: Restrict to this mask. The whole frame when None.

    Returns:
        The power, in the units ``measured_intensity * pixel_area`` carries.
    """
    xp = array_namespace(measured_intensity)
    if region is not None:
        measured_intensity = measured_intensity * region
    return xp.sum(measured_intensity) * pixel_area


def efficiency_metric(
    incident_power: float,
    pixel_area: float,
    in_signal_region: bool = True,
    name: str | None = None,
) -> IntensityMetric:
    """A metric reporting diffraction efficiency against a fixed reference.

    Args:
        incident_power: Power entering the lens, from
            :meth:`~hologradpy.optics.systems.SLMFourierLensModel.incident_power`.
        pixel_area: Area of one output pixel, from ``output_pixel_area()``.
        in_signal_region: Measure over the signal region only, rather than the whole
            frame.
        name: Label.

    Returns:
        IntensityMetric: The metric, higher being better.
    """
    reference = float(incident_power)
    area = float(pixel_area)
    label = name or ("signal efficiency" if in_signal_region else "window efficiency")

    def measure(signal_region, target_intensity, measured_intensity):
        region = signal_region if in_signal_region else None
        return captured_power(measured_intensity, area, region) / reference

    return IntensityMetric(label, measure, lower_is_better=False)
