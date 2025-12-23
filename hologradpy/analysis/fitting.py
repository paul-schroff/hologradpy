"""
This module contains functions for curve fitting.
"""

from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit

from .. import patterns as pt

from .functions import tilt, interferometric_fringes

from ..propagation.utils.optics_utils import gaussian_beam_intensity


def curve_fit_2d(
    x: NDArray,
    y: NDArray,
    data: NDArray,
    func: Callable[[NDArray, NDArray, Sequence[float]], NDArray],
    *args,
    **kwargs,
) -> tuple[NDArray, NDArray]:
    x_data = np.vstack((x.ravel(), y.ravel()))

    def func_flat(x_data_flat, *params):
        x_flat, y_flat = x_data_flat
        return func(x_flat, y_flat, *params)

    popt, pcov = curve_fit(func_flat, x_data, data.ravel(), *args, **kwargs)
    return popt, pcov


def remove_tilt(img: NDArray, mask: NDArray | None = None) -> NDArray:
    """
    This function removes fits the first three Zernike polynomials (Piston and
    tilt) to an image and subtracts the
    fitted function from the original image.
    :param ndarray img: Input image.
    :param ndarray mask: Binary mask in which to remove tilt.
    :return: Image without tilt.
    """
    if mask is None:
        mask = np.ones_like(img)

    def tilt_mask(xy: tuple[NDArray, NDArray], *args: float) -> NDArray:
        return tilt(xy, *args, mask=mask.ravel())

    x_, y_ = pt.make_grid(img)
    xdata = np.vstack((x_.ravel(), y_.ravel()))
    p0 = np.zeros(4)

    p_opt, p_cov = curve_fit(tilt_mask, xdata, img.ravel(), p0)
    img_tilt = np.reshape(tilt(xdata, *p_opt), img.shape)
    return img - img_tilt


def fit_gaussian_beam_intensity(
    x: NDArray,
    y: NDArray,
    data: NDArray,
    beam_radius_guess: float,
    blur_sigma: float = 10,
) -> tuple[NDArray[np.float_], NDArray[np.float_]]:

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
) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
    """Fit the interference pattern on the camera caused by two superpixels
    on the SLM separated by separation_x and separation_y. Equation adapted
    from https://doi.org/10.1364/OE.24.013881.
    Args:
        x (NDArray): x coordinates.
        y (NDArray): y coordinates.
        data (NDArray): Interference pattern on the camera.
        separation_x (float): Separation between two superpixels along x.
        separation_y (float): Separation between two superpixels along y.
        wavenumber (float): Wavenumber of the light.
        focal_length (float): Focal length of the Fourier lens.
        phase_guess (float, optional): Phase difference between the two
            superpixels. Defaults to 0.
        amplitude_guess (float, optional): Amplitude due to superpixel a.
            Defaults to 1.
    Returns:
        tuple[NDArray[np.float_], NDArray[np.float_]]: Fitting parameters
            (phase, amplitude) and covariance matrix.
    """

    def fit_function(x_, y_, phase, amplitude):
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
