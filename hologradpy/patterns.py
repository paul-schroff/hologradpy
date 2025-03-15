"""
This module contains utility functions for array manipulation and functions to 
create binary masks, intensity and phase patterns of various shapes.

The :py:class:`Hologram` class provides arrays needed for the CG minimisation: 
The target light potential and the signal region, the measured constant SLM 
phase and intensity at the required resolution, and the initial SLM phase guess
to start the CG minimisation.
"""

import numpy as np
from numpy.typing import NDArray
import scipy
from scipy import ndimage
import cv2 as cv


# Utility functions for array manipulation
def make_grid(im: NDArray[np.float_],
              scale: float | None = None
              ) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
    """
    Return a xy meshgrid based in an input array, im, ranging from -scal * 
    im.shape[0] // 2 to scal * im.shape[0] // 2.

    :param im: Input array.
    :param scale: Optional scaling factor.
    :return: x and y meshgrid arrays.
    """
    if scale is None:
        scale = 1
    h, w = im.shape
    y_lim, x_lim = h // 2, w // 2
    
    x = np.linspace(-x_lim * scale, x_lim * scale, w)
    y = np.linspace(-y_lim * scale, y_lim * scale, h)
    return np.meshgrid(x, y)


def pixel_corr(img: NDArray[np.float_],
               x: list[int],
               y: list[int]
               ) -> NDArray[np.float_]:
    """
    Replace a pixel value with coordinates x and y by the mean value of its 3x3
    neighbourhood.

    :param img: Input image.
    :param x: x-coordinate of pixel.
    :param y: y-coordinate of pixel.
    :return: Corrected image.
    """
    for i, j in y, x:
        img[i, j] = (np.sum(img[i-1:i+2, j-1:j+2]) - img[i, j]) / 8
    return img


def unwrap_2d(img: NDArray[np.float_], **kwargs) -> NDArray[np.float_]:
    """
    Unwraps an image along the x- and y-axis.

    :param img: Input image.
    :param kwargs: kwargs for ``np.unwrap()`` function.
    :return: Unwrapped image.
    """
    if kwargs is None:
        kwargs = {'period': 2 * np.pi}
    return np.unwrap(np.unwrap(img, **kwargs), axis=0, **kwargs)


def unwrap_2d_mask(img: NDArray[np.float_],
                   mask: NDArray[np.float_],
                   **kwargs
                   ) -> NDArray[np.float_]:
    """
    Unwraps an image within a region of interest defined by a binary mask.

    :param img: Input image.
    :param mask: Binary mask with region of interest.
    :param kwargs: kwargs for np.unwrap() function.
    :return: Unwrapped image.
    """
    if kwargs is None:
        kwargs = {'period': 2 * np.pi}
    img_uw_1d = np.zeros_like(img)
    for i in range(mask.shape[0]):
        img_uw_1d[i, mask[i, :]] = np.unwrap(img[i, mask[i, :]], **kwargs)

    img_uw = np.zeros_like(img)
    for i in range(mask.shape[1]):
        img_uw[mask[:, i], i] = np.unwrap(img_uw_1d[mask[:, i], i], **kwargs)
    img_uw[~mask] = 0
    return img_uw


def crop(img: NDArray[np.float_],
         n_crop: int
         ) -> NDArray[np.float_]:
    """
    Crops an image around all four edges by n_crop pixels.

    :param img: Input image.
    :param n_crop: Number of pixels to crop at both end of each dimension.
    :return: Cropped image.
    """
    img = img[n_crop:-n_crop, n_crop:-n_crop]
    return img


def crop_to_mask(img: NDArray[np.float_],
                 mask: NDArray[np.float_]
                 ) -> NDArray[np.float_]:
    """
    Crops an image to the smallest size taken up by a binary mask.

    :param img: Input image.
    :param mask: Binary mask.
    :return: Cropped image.
    """
    idx_min_0 = np.min(np.argwhere(mask > 0)[:, 0])
    idx_min_1 = np.min(np.argwhere(mask > 0)[:, 1])
    idx_max_0 = np.max(np.argwhere(mask > 0)[:, 0])
    idx_max_1 = np.max(np.argwhere(mask > 0)[:, 1])
    idx = [idx_min_0, idx_max_0, idx_min_1, idx_max_1]
    pad = ((idx_min_0, img.shape[0] - idx_max_0),
           (idx_min_1, img.shape[1] - idx_max_1))
    return img[idx[0]:idx[1], idx[2]:idx[3]], [pad, idx]


def load_filter_upscale(path: str,
                        npx: int,
                        pix_res: int,
                        crop: int | None = None,
                        filter_size: int | None = None
                        ) -> NDArray[np.float]:
    """
    Loads a 2D numpy array and crops its edges. A uniform filter is applied to 
    the cropped image before it is upscaled using Lanczos interpolation.

    :param path: Numpy array or path to numpy array.
    :param npx: Number of SLM pixels.
    :param pix_res: Number of pixels per SLM pixel.
    :param crop: Number of unused pixels [SLM pixels].
    :param filter_size: Size of the uniform filter.
    :return: Upscaled image.
    """

    if crop is None:
        crop = 0
    if filter_size is None:
        filter_size = 1

    if isinstance(path, np.ndarray):
        out = path
    else:
        out = np.load(path)
    res = out.shape[0]
    pix = int(np.round(npx / (res + 2)))
    if crop == 0:
        out = scipy.ndimage.uniform_filter(out,
                                           size=filter_size, mode='nearest')
        out = cv.resize(out,
                        (npx * pix_res, npx * pix_res),
                        interpolation=cv.INTER_LANCZOS4)
    else:
        out = out[crop // pix:-crop // pix, crop // pix:-crop // pix]
        out = scipy.ndimage.uniform_filter(out,
                                           size=filter_size,
                                           mode='nearest')
        out = cv.resize(out,
                        ((npx - 2 * crop) * pix_res,
                         (npx - 2 * crop) * pix_res),
                        interpolation=cv.INTER_LANCZOS4)
        out = np.pad(out,
                     ((crop * pix_res, crop * pix_res),
                      (crop * pix_res, crop * pix_res)))
    return out


# Binary masks
def rect_mask(im: NDArray[np.float_],
              dx: int,
              dy: int,
              w: int,
              h: int
              ) -> NDArray[np.float_]:
    """
    Rectangular mask using pixel coordinates of an input image.

    :param im: Input image
    :param dx: X-offset of rectangle from the centre of the image.
    :param dy: Y-offset of rectangle from the centre of the image.
    :param w: Width of rectangle.
    :param h: Height of rectangle.
    :return: Binary mask.
    """
    height, width = im.shape
    y_grid, x_grid = np.ogrid[-height // 2:height // 2, -width // 2:width // 2]
    
    idx = ((x_grid - dx > -w // 2) & (x_grid - dx < w // 2) & 
           (y_grid - dy > -h // 2) & (y_grid - dy < h // 2))
    mask = np.zeros_like(im)
    mask[idx] = 1
    return mask


def rect_mask_xy(x: NDArray[np.float_], 
                 y: NDArray[np.float_], 
                 dx: int, 
                 dy: int, 
                 w: int, 
                 h: int
                 ) -> NDArray[np.float_]:
    """
    Rectangular mask using XY meshgrid coordinates.

    :param x: X meshgrid
    :param y: Y meshgrid
    :param dx: X-offset of rectangle from the centre of the image.
    :param dy: Y-offset of rectangle from the centre of the image.
    :param w: Width of rectangle.
    :param h: Height of rectangle.
    :return: Binary mask.
    """
    idx = (np.abs(x - dx) < w / 2) & (np.abs(y - dy) < h / 2)
    mask = np.zeros_like(x)
    mask[idx] = 1
    return mask


def circ_mask(im: NDArray[np.float_],
              dx: int,
              dy: int,
              r: int
              ) -> NDArray[np.float_]:
    """
    Circular mask using pixel coordinates of an input image.

    :param im: Input image
    :param dx: X-offset of circle.
    :param dy: Y-offset of circle.
    :param r: Radius of circle.
    :return: Binary mask.
    """
    height, width = im.shape
    y, x = np.ogrid[-height / 2:height / 2, -width / 2:width / 2]
    
    idx = (x - dx) ** 2 + (y - dy) ** 2 < r ** 2
    mask = np.zeros_like(im)
    mask[idx] = 1
    return mask


def circ_mask_xy(x: NDArray[np.float_], 
                 y: NDArray[np.float_], 
                 dx: int, 
                 dy: int, 
                 r: int, 
                 sparse: bool | None = None
                 ) -> NDArray[np.float_]:
    """
    Circular mask using XY meshgrid coordinates.

    :param x: X meshgrid.
    :param y: Y meshgrid.
    :param dx: X-offset of circle.
    :param dy: Y-offset of circle.
    :param r: Radius of circle.
    :param sparse: Whether to use a sparse mask.
    :return: Binary mask.
    """
    idx = (x - dx) ** 2 + (y - dy) ** 2 < r ** 2
    if sparse:
        mask = np.zeros((max(x.shape), max(x.shape)))
    else:
        mask = np.zeros_like(x)
    mask[idx] = 1
    return mask


# Intensity patterns
def gaussian(x: NDArray[np.float_], 
             y: NDArray[np.float_], 
             dx: int, 
             dy: int, 
             sig_x: float, 
             sig_y: float | None = None, 
             a: float = 1, 
             c: float = 0
             ) -> NDArray[np.float_]:
    """
    2D Gaussian.

    :param x: X meshgrid.
    :param y: Y meshgrid.
    :param dx: X-offset of Gaussian.
    :param dy: Y-offset of Gaussian.
    :param sig_x: X width of Gaussian.
    :param sig_y: Y width of Gaussian
    :param a: Amplitude.
    :param c: Offset.
    :return: 2D Gaussian.
    """
    if sig_y is None:
        sig_y = sig_x

    return (a * np.exp(-0.5 * ((x - dx) ** 2 / sig_x ** 2 + 
                               (y - dy) ** 2 / sig_y ** 2)) + c)


def super_gaussian(x: NDArray[np.float_], 
                   y: NDArray[np.float_], 
                   dx: int, 
                   dy: int, 
                   nx: int, 
                   ny: int, 
                   sig_x: float, 
                   sig_y: float, 
                   a: float = 1, 
                   c: float = 0
                   ) -> NDArray[np.float_]:
    """
    2D super-Gaussian.

    :param x: X meshgrid.
    :param y: Y meshgrid.
    :param dx: X-offset of Gaussian.
    :param dy: Y-offset of Gaussian.
    :param nx: X-order.
    :param ny: Y-order.
    :param sig_x: X-width.
    :param sig_y: Y-width.
    :param a: Amplitude.
    :param c: Offset.
    :return: 2D super-Gaussian.
    """
    return (a * np.exp(-2 * (np.abs(x - dx) / sig_x) ** nx) *
            np.exp(-2 * (np.abs(y - dy) / sig_y) ** ny) + c)


def gauss_array(im: NDArray[np.float_], 
                nx: int, 
                ny: int, 
                dx: int, 
                dy: int, 
                d: float, 
                sigma: float
                ) -> NDArray[np.float_]:
    """
    Gaussian spot array using coordinates of input image.

    :param im: Input image.
    :param nx: Number of array columns.
    :param ny: Number of array rows.
    :param dx: X-offset of array.
    :param dy: Y-offset of array.
    :param d: Separation between neighbouring spots.
    :param sigma: Width of Gaussian spots.
    :return: Spot array.
    """
    height, width = im.shape
    x, y = np.ogrid[-height / 2:height / 2, -width / 2:width / 2]
    a = np.zeros_like(im)
    wx = (nx - 1) * d
    wy = (ny - 1) * d
    
    for i in range(nx):
        for j in range(ny):
            a = a + gaussian(x,
                             y,
                             j * d - wy // 2 - dy,
                             i * d - wx // 2 - dx,
                             sigma)
    return a


def ring_gauss(x: NDArray[np.float_], 
               y: NDArray[np.float_], 
               dx: int, 
               dy: int, 
               r: float, 
               w: float, 
               a: float = 1
               ) -> NDArray[np.float_]:
    """
    Ring with Gaussian profile.

    :param x: X meshgrid.
    :param y: Y meshgrid.
    :param dx: X-offset of ring.
    :param dy: Y-offset of ring.
    :param r: Radius of ring.
    :param w: Width of Gaussian profile.
    :param a: Amplitude.
    :return: Ring with Gaussian profile.
    """
    return (a * np.exp(-2 * (np.sqrt((x - dx) ** 2 +
                                     (y - dy) ** 2) - r) ** 2 / w ** 2))


def checkerboard(npx: int, 
                 dx: int, 
                 dy: int, 
                 rows: int, 
                 columns: int, 
                 square_size: int
                 ) -> NDArray[np.float_]:
    """
    Creates a checkerboard on a canvas of `(npx, npx)` pixels.

    :param npx: Size of canvas.
    :param dx: X-offset of checkerboard.
    :param dy: Y-Offset of checkerboard.
    :param rows: Checkerboard rows.
    :param columns: Checkerboard columns.
    :param square_size: Size of a square in pixels
    :return: Checkerboard.
    """
    cb = np.indices((columns, rows)).sum(axis=0) % 2
    cb = np.repeat(np.repeat(cb, square_size, axis=0), square_size, axis=1)
    cb_w, cb_h = cb.shape
    pad_w = (npx - cb_w) // 2
    pad_h = (npx - cb_h) // 2
    return np.pad(cb, ((pad_w+dx, pad_w-dx), (pad_h+dy, pad_h-dy)))


def fringes_wavefront(x: NDArray[np.float_], 
                      y: NDArray[np.float_], 
                      dx: float, 
                      dy: float, 
                      k: float, 
                      f: float, 
                      phi: float, 
                      a: float, 
                      b: float
                      ) -> NDArray[np.float_]:
    """
    Standing wave interference pattern on the camera caused by two patches on 
    the SLM seperated by dx and dy. Equation adapted from 
    https://doi.org/10.1364/OE.24.013881.

    :param x: X meshgrid.
    :param y: Y meshgrid.
    :param dx: Separation between reference and sample patch along x [m].
    :param dy: Separation between reference and sample patch along y [m].
    :param k: Wavenumber [rad/m].
    :param f: Focal length of Fourier lens [m].
    :param phi: Phase difference between reference and sample patches (see paper above) [rad].
    :param a: Amplitude on reference patch.
    :param b: Amplitude on sample patch.
    :return: Interference pattern.
    """
    gamma_x = np.arctan(dx / (np.abs(f) + 1e-12))  # x component of angle between reference and sample beam
    gamma_y = np.arctan(dy / (np.abs(f) + 1e-12))  # y component of angle between reference and sample beam
    i_out = (a ** 2 + b ** 2 + 
             2 * a * b * np.cos(k * (np.sin(gamma_x) * x + 
                                     np.sin(gamma_y) * y) + phi))
    return i_out
