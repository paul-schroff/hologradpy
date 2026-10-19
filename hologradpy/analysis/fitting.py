"""This module contains functions for curve fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
from scipy.optimize import curve_fit


from ..fourier_transforms import cross_correlation_peak, translate_real
from ..profiles.zernike import Zernike


def interferometric_fringes(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    separation_x: float,
    separation_y: float,
    wavenumber: float,
    focal_length: float,
    phase: float,
    amplitude: float,
) -> NDArray[np.float64]:
    """Interference pattern on the camera caused by two superpixels
    on the SLM separated by separation_x and separation_y. Equation adapted
    from https://doi.org/10.1364/OE.24.013881.

    Args:
        x: x coordinates.
        y: y coordinates.
        separation_x: Separation between two superpixels along x.
        separation_y: Separation between two superpixels along y.
        wavenumber: Wavenumber of the light.
        focal_length: Focal length of the Fourier lens.
        phase: Phase difference between the two superpixels.
        amplitude: Amplitude due to each superpixel.

    Returns:
        NDArray[np.float64]: Interference pattern on the camera.
    """
    angle_x = np.arctan(separation_x / focal_length)
    angle_y = np.arctan(separation_y / focal_length)

    intesity = 2 * amplitude**2 + 2 * amplitude**2 * np.cos(
        wavenumber * (np.sin(angle_x) * x + np.sin(angle_y) * y) + phase
    )
    return intesity


def curve_fit_2d(
    x: NDArray,
    y: NDArray,
    data: NDArray,
    func: Callable[[NDArray, NDArray, Sequence[float]], NDArray],
    *args: Any,
    **kwargs: Any,
) -> tuple[NDArray, NDArray]:
    x_data = np.vstack((x.ravel(), y.ravel()))

    def func_flat(x_data_flat: NDArray, *params: float) -> NDArray:
        x_flat, y_flat = x_data_flat
        return func(x_flat, y_flat, *params)

    popt, pcov = curve_fit(func_flat, x_data, data.ravel(), *args, **kwargs)
    return popt, pcov


def remove_tilt(phase: NDArray, mask: NDArray | None = None) -> NDArray:
    """Fit and subtract piston and tilt from an image.

    Fits the first three Zernike modes (piston and tip/tilt) to ``phase`` over the
    region defined by ``mask`` as a linear least-squares problem and subtracts the
    fitted surface. Higher-order modes (defocus, astigmatism, ...) are left untouched.

    Args:
        phase: Input phase.
        mask: Boolean mask of the region to fit over. Defaults to the whole
            image.

    Returns:
        NDArray: Phase with piston and tilt removed.
    """
    # Radial orders 0 and 1 are piston and tip/tilt (the first three Zernike
    # modes). The default unit disk mode, ``"fill"``, covers the whole image.
    zernike = Zernike(
        resolution=phase.shape,
        number_of_radial_orders=2,
    )

    phase_tensor = torch.as_tensor(phase, dtype=torch.float64)
    mask_tensor = None if mask is None else torch.as_tensor(mask, dtype=bool)

    coefficients = zernike.fit(phase_tensor, mask=mask_tensor)
    fitted = zernike.get_phase(coefficients).numpy()
    return phase - fitted


def fit_interferometric_fringes(
    x: NDArray,
    y: NDArray,
    data: NDArray,
    separation_x: float,
    separation_y: float,
    wavenumber: float,
    focal_length: float,
    phase_guess: float = 0,
    amplitude_guess: float = 1,
    max_iterations: int = 10000,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fit the interference pattern on the camera caused by two superpixels
    on the SLM separated by separation_x and separation_y. Equation adapted
    from https://doi.org/10.1364/OE.24.013881.

    Args:
        x: x coordinates.
        y: y coordinates.
        data: Interference pattern on the camera.
        separation_x: Separation between two superpixels along x.
        separation_y: Separation between two superpixels along y.
        wavenumber: Wavenumber of the light.
        focal_length: Focal length of the Fourier lens.
        phase_guess: Phase difference between the two superpixels. Defaults
            to 0.
        amplitude_guess: Amplitude due to each superpixel. Defaults to 1.
        max_iterations: Maximum number of calls to the fit function. Defaults
            to 10000.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: Fitting parameters
            (phase, amplitude) and covariance matrix.
    """

    def fit_function(
        x_: NDArray,
        y_: NDArray,
        phase: float,
        amplitude: float,
    ) -> NDArray[np.float64]:
        return interferometric_fringes(
            x_,
            y_,
            separation_x,
            separation_y,
            wavenumber,
            focal_length,
            phase,
            amplitude,
        )

    p0 = (phase_guess, amplitude_guess)
    bounds = ((-np.pi, 0), (np.pi, np.inf))
    popt, pcov = curve_fit_2d(
        x, y, data, fit_function, p0=p0, bounds=bounds, maxfev=max_iterations
    )
    return popt, pcov


def optical_lattice_fringes(
    x: NDArray,
    y: NDArray,
    separation_x: float,
    separation_y: float,
    wavenumber: float,
    focal_length: float,
    phase_x: float,
    phase_y: float,
    amplitude: float,
) -> NDArray[np.float64]:
    """2D optical lattice on the camera caused by four superpixels in the SLM
    corners.

    It is the separable product of two orthogonal ``interferometric_fringes``
    patterns and is used to track beam pointing drift during the phase
    measurement.

    Args:
        x: x coordinates.
        y: y coordinates.
        separation_x: Horizontal separation between the corner superpixels.
        separation_y: Vertical separation between the corner superpixels.
        wavenumber: Wavenumber of the light.
        focal_length: Focal length of the Fourier lens.
        phase_x: Phase of the lattice along x.
        phase_y: Phase of the lattice along y.
        amplitude: Amplitude of the lattice.

    Returns:
        NDArray[np.float64]: 2D optical lattice pattern on the camera.
    """
    angle_x = np.arctan(separation_x / focal_length)
    angle_y = np.arctan(separation_y / focal_length)

    fringes_x = 1 + np.cos(wavenumber * np.sin(angle_x) * x + phase_x)
    fringes_y = 1 + np.cos(wavenumber * np.sin(angle_y) * y + phase_y)
    return 2 * amplitude ** 2 * fringes_x * fringes_y


def fit_optical_lattice_fringes(
    x: NDArray,
    y: NDArray,
    data: NDArray,
    separation_x: float,
    separation_y: float,
    wavenumber: float,
    focal_length: float,
    phase_x_guess: float = 0.0,
    phase_y_guess: float = 0.0,
    amplitude_guess: float = 1.0,
    max_iterations: int = 10000,
    bound_phase: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fit the 2D optical lattice on the camera caused by four corner
    superpixels.

    Args:
        x: x coordinates.
        y: y coordinates.
        data: Optical lattice pattern on the camera.
        separation_x: Horizontal separation between the corner superpixels.
        separation_y: Vertical separation between the corner superpixels.
        wavenumber: Wavenumber of the light.
        focal_length: Focal length of the Fourier lens.
        phase_x_guess: Initial guess for the lattice phase along x. Defaults
            to 0.0.
        phase_y_guess: Initial guess for the lattice phase along y. Defaults
            to 0.0.
        amplitude_guess: Initial guess for the lattice amplitude. Defaults to
            1.0.
        max_iterations: Maximum number of calls to the fit function. Defaults
            to 10000.
        bound_phase: If True, constrain the phases to (-pi, pi). Set False to
            track a continuously drifting phase across frames (warm-started
            from the previous value), avoiding the wrap discontinuity at the
            bound. Defaults to True.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: Fitting parameters
            (phase_x, phase_y, amplitude) and covariance matrix.
    """

    def fit_function(
        x_: NDArray,
        y_: NDArray,
        phase_x: float,
        phase_y: float,
        amplitude: float,
    ) -> NDArray[np.float64]:
        return optical_lattice_fringes(
            x_,
            y_,
            separation_x,
            separation_y,
            wavenumber,
            focal_length,
            phase_x,
            phase_y,
            amplitude,
        )

    p0 = (phase_x_guess, phase_y_guess, amplitude_guess)
    if bound_phase:
        bounds = ((-np.pi, -np.pi, 0), (np.pi, np.pi, np.inf))
    else:
        bounds = ((-np.inf, -np.inf, 0), (np.inf, np.inf, np.inf))
    popt, pcov = curve_fit_2d(
        x, y, data, fit_function, p0=p0, bounds=bounds, maxfev=max_iterations
    )
    return popt, pcov


@dataclass(frozen=True)
class TranslationFit:
    """The translation that moves a pattern onto an image, and the brightness that
    matches them.

    The image is modelled as ``scale * pattern(r - shift) + offset``. ``shift`` is the
    translation ``(columns, rows)`` in samples, positive towards increasing index, as
    :func:`~hologradpy.fourier_transforms.translate.fft_translate` takes it. ``scale``
    is the brightness gain and ``offset`` the background level between the two.
    Fitting them keeps a difference in brightness out of the shift. ``offset`` is zero
    unless it was fitted.
    """

    shift: tuple[float, float]
    scale: float
    offset: float


def fit_scale(
    image: Tensor, pattern: Tensor, region: Tensor, fit_offset: bool = False
) -> tuple[float, float]:
    """The brightness gain, and with ``fit_offset`` the background level, that best
    match ``pattern`` to ``image`` inside ``region``.

    The least-squares fit of ``image ~ scale * pattern``, or of
    ``image ~ scale * pattern + offset`` with ``fit_offset``. A fitted offset absorbs
    the mean of any light the pattern lacks in the region, such as a leak into a dark
    margin. The offset is therefore fitted only on request.

    Args:
        image: The measured image.
        pattern: The pattern on the same grid.
        region: True where the fit is evaluated.
        fit_offset: Fit the background level as well.

    Returns:
        tuple[float, float]: The gain ``scale``, and the background level ``offset``,
        which is zero without ``fit_offset``.

    Raises:
        ValueError: The pattern has no structure inside the region to fit.
    """
    inside = region.to(torch.bool)
    image_inside = image[inside].to(torch.float64)
    pattern_inside = pattern[inside].to(torch.float64)
    if fit_offset:
        pattern_inside = pattern_inside - pattern_inside.mean()

    power = float((pattern_inside**2).sum())
    if power <= 0.0:
        raise ValueError(
            "The pattern is constant inside the region, so neither its scale nor its "
            "translation can be fitted."
        )
    if not fit_offset:
        return float((pattern_inside * image_inside).sum()) / power, 0.0

    scale = float((pattern_inside * (image_inside - image_inside.mean())).sum()) / power
    offset = float(image_inside.mean()) - scale * float(pattern[inside].mean())
    return scale, offset


def fit_translation(
    image: Tensor,
    pattern: Tensor,
    region: Tensor,
    fit_offset: bool = False,
    max_iterations: int = 20,
    tolerance: float = 1e-4,
) -> TranslationFit:
    """The sub-sample translation that moves ``pattern`` onto ``image`` inside
    ``region``.

    The image is modelled as ``scale * pattern(r - shift) + offset``. The brightness
    gain ``scale``, and with ``fit_offset`` the background level ``offset``, are fitted
    with the translation, so the translation follows the shape alone. The translation
    starts at the peak of the cross-correlation inside the region
    (:func:`~hologradpy.fourier_transforms.translate.cross_correlation_peak`).
    Gauss-Newton steps then refine it, translating the pattern by
    :func:`~hologradpy.fourier_transforms.translate.translate_real`. The gain and the
    background level follow in closed form at every step (:func:`fit_scale`).

    Args:
        image: The measured image ``(H, W)``.
        pattern: The pattern on the same grid.
        region: True where the fit is evaluated. The pattern has to vary inside it.
        fit_offset: Fit the background level as well.
        max_iterations: The most Gauss-Newton steps taken.
        tolerance: The fit stops once a step moves the translation by less than this,
            in samples.

    Returns:
        TranslationFit: The translation, the gain and the background level.
    """
    dtype = torch.promote_types(image.dtype, torch.float32)
    image = image.to(dtype)
    pattern = pattern.to(dtype)
    weights = region.to(dtype)

    shift = cross_correlation_peak(image, pattern, region)
    for _ in range(max_iterations):
        moved = translate_real(pattern, shift)
        scale, offset = fit_scale(image, moved, region, fit_offset)
        residual = (image - scale * moved - offset) * weights

        # The derivative of scale * pattern(r - shift) by the shift is minus the scale
        # times the gradient of the translated pattern.
        gradient_columns, gradient_rows = torch.gradient(moved, dim=(-1, -2))
        jacobian_columns = -scale * gradient_columns * weights
        jacobian_rows = -scale * gradient_rows * weights
        normal = torch.tensor(
            [
                [
                    float((jacobian_columns**2).sum()),
                    float((jacobian_columns * jacobian_rows).sum()),
                ],
                [
                    float((jacobian_columns * jacobian_rows).sum()),
                    float((jacobian_rows**2).sum()),
                ],
            ],
            dtype=torch.float64,
        )
        projected = torch.tensor(
            [
                float((jacobian_columns * residual).sum()),
                float((jacobian_rows * residual).sum()),
            ],
            dtype=torch.float64,
        )
        step = torch.linalg.lstsq(normal, projected[:, None]).solution[:, 0]
        shift = (shift[0] + float(step[0]), shift[1] + float(step[1]))
        if float(step.abs().max()) < tolerance:
            break

    scale, offset = fit_scale(image, translate_real(pattern, shift), region, fit_offset)
    return TranslationFit(shift=shift, scale=scale, offset=offset)
