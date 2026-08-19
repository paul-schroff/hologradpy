"""This module contains functions for curve fitting."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray
import torch
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit


from ..profiles.amplitude import gaussian_beam_intensity
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


def fit_gaussian_beam_intensity(
    x: NDArray,
    y: NDArray,
    data: NDArray,
    beam_radius_guess: float,
    blur_sigma: float = 10,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

    data_blurred = gaussian_filter(data, blur_sigma)
    index = np.unravel_index(np.argmax(data_blurred), data.shape)
    p0 = (beam_radius_guess, x[index], y[index], np.max(data), 0)

    popt, pcov = curve_fit_2d(
        x,
        y,
        data,
        gaussian_beam_intensity,
        p0=p0,
    )
    return popt, pcov


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
