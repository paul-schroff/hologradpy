"""
This module contains functions for curve fitting.
"""

import numpy as np
from . import patterns as pt
import scipy
import scipy.optimize as opt


def tilt(xy, *args, mask=None):
    """
    Fit function containing the first three Zernike polynomials of the form z = c0 + c1 * x + c2 * y + c3 * 2xy.

    :param xy: x, y coordinate vectors.
    :param args: Vector of length 4, containing Zernike coefficients.
    :return: First 3 Zernike polynomials.
    """
    if mask is None:
        mask = np.ones_like(xy[0])

    x, y = xy
    c = np.array(args)
    arr = c[0] + c[1] * x + c[2] * y + c[3] * 2 * x * y
    return arr * mask


def remove_tilt(img, mask=None):
    """
    This function removes fits the first three Zernike polynomials (Piston and tilt) to an image and subtracts the
    fitted function from the original image.
    :param ndarray img: Input image.
    :param ndarray mask: Binary mask in which to remove tilt.
    :return: Image without tilt.
    """
    if mask is None:
        mask = np.ones_like(img)

    def tilt_mask(xy, *args):
        return tilt(xy, *args, mask=mask.ravel())

    x_, y_ = pt.make_grid(img)
    xdata = np.vstack((x_.ravel(), y_.ravel()))
    p0 = np.zeros(4)

    p_opt, p_cov = opt.curve_fit(tilt_mask, xdata, img.ravel(), p0)
    img_tilt = np.reshape(tilt(xdata, *p_opt), img.shape)
    return img - img_tilt


def gaussian(xy, *args):
    """
    Gaussian fit function.

    :param xy: x, y coordinate vectors.
    :param args: Fitting parameters passed to patterns.gaussian.
    :return: Gaussian.
    """
    x, y = xy
    arr = pt.gaussian(x, y, *args)
    return arr


def fit_gaussian(img, dx=None, dy=None, sig_x=15, sig_y=15, a=None, c=0, blur_width=10, xy=None):
    """
    Fits a 2D Gaussian to an image. The image s blurred using a Gaussian filer before fitting.

    :param img: Input image.
    :param dx: X-offset of Gaussian [px].
    :param dy: Y-offset of Gaussian [px].
    :param sig_x: X-width of Gaussian [px].
    :param sig_y: -width of Gaussian [px].
    :param a: Amplitude.
    :param c: Offset.
    :param blur_width: Width of Gaussian blurring kernel [px].
    :param xy: X, Y meshgrid. If not specified, pixel coordinates are used.
    :return: Fitting parameters, parameter errors.
    """
    if xy is None:
        x, y = pt.make_grid(img)
    else:
        x, y = xy
    x_data = np.vstack((x.ravel(), y.ravel()))
    img_blur = scipy.ndimage.gaussian_filter(img, blur_width)
    if dx is None or dy is None:
        dy, dx = np.unravel_index(np.argmax(img_blur), img.shape)
        dx -= img.shape[1] / 2
        dy -= img.shape[0] / 2
    if a is None:
        a = np.max(img_blur)
    # Define initial parameter guess.
    p0 = [(dx, dy, sig_x, sig_y, a, c)]
    popt, pcov = opt.curve_fit(gaussian, x_data, img.ravel(), p0, maxfev=10000)

    # Calculate errors
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


class FitSine:
    """
    This class is used in the wavefront measurement to fit a 2D sine to the interference pattern on the camera. The
    distance between the sample and reference patch can be set by calling the method set_dx_dy.
    """
    def __init__(self, fl, k, dx=None, dy=None):
        """
        Initialises the class by defining fixed parameters.

        :param fl: Focal length of the Fourier lens [m].
        :param k: Wavenumber [rad/m].
        :param dx: Separation between the reference and the sampling patch along x.
        :param dy: Separation between the reference and the sampling patch along y.
        """
        self.fl = fl
        self.dx = dx
        self.dy = dy
        self.k = k

    def set_dx_dy(self, dx, dy):
        """
        Method to set parameters dx and dy.
        :param dx: New dx.
        :param dy: New dy.
        """
        self.dx = dx
        self.dy = dy

    def fit_sine(self, xy, *args):
        """
        Method to perform 2D sine fit.
        :param xy: x, y coordinate vectors.
        :param args: Args passed to patterns.fringes_wavefront
        :return: 2D sine.
        """
        x, y = xy
        arr = pt.fringes_wavefront(x, y, self.dx, self.dy, self.k, self.fl, *args)
        return arr
