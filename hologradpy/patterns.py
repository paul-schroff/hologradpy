"""
This module contains utility functions for array manipulation and functions to create binary masks, intensity and
phase patterns of various shapes.

The :py:class:`Hologram` class provides arrays needed for the CG minimisation: The target light
potential and the signal region, the measured constant SLM phase and intensity at the required resolution, and
the initial SLM phase guess to start the CG minimisation.
"""

import numpy as np
import scipy
from scipy import ndimage
import cv2 as cv


# Utility functions for array manipulation
def make_grid(im, scale=None):
    """
    Return a xy meshgrid based in an input array, im, ranging from -scal * im.shape[0] // 2 to scal * im.shape[0] // 2.

    :param im: Input array.
    :param scale: Optional scaling factor.
    :return: x and y meshgrid arrays.
    """
    if scale is None:
        scale = 1
    h, w = im.shape
    y_lim, x_lim = h // 2, w // 2
    
    x, y = np.linspace(-x_lim * scale, x_lim * scale, w), np.linspace(-y_lim * scale, y_lim * scale, h)
    x, y = np.meshgrid(x, y)
    return x, y


def pixel_corr(img, x, y):
    """
    Replace a pixel value with coordinates x and y by the mean value of its 3x3 neighbourhood.

    :param img: Input image.
    :param x: x-coordinate of pixel.
    :param y: y-coordinate of pixel.
    :return: Corrected image.
    """
    for i, j in y, x:
        img[i, j] = (np.sum(img[i-1:i+2, j-1:j+2]) - img[i, j]) / 8
    return img


def unwrap_2d(img, **kwargs):
    """
    Unwraps an image along the x- and y-axis.

    :param img: Input image.
    :param kwargs: kwargs for ``np.unwrap()`` function.
    :return: Unwrapped image.
    """
    if kwargs is None:
        kwargs = {'period': 2 * np.pi}
    return np.unwrap(np.unwrap(img, **kwargs), axis=0, **kwargs)


def unwrap_2d_mask(img, mask, **kwargs):
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


def crop(img, n_crop):
    """
    Crops an image around all four edges by n_crop pixels.

    :param img: Input image.
    :param n_crop: Number of pixels to crop at both end of each dimension.
    :return: Cropped image.
    """
    img = img[n_crop:-n_crop, n_crop:-n_crop]
    return img


def crop_to_mask(img, mask):
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
    pad = ((idx_min_0, img.shape[0] - idx_max_0), (idx_min_1, img.shape[1] - idx_max_1))
    return img[idx[0]:idx[1], idx[2]:idx[3]], [pad, idx]


def load_filter_upscale(path, npx, pix_res, crop=None, filter_size=None):
    """
    Loads a 2D numpy array and crops its edges. A uniform filter is applied to the cropped image before it is upscaled
    using Lanczos interpolation.

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
        out = scipy.ndimage.uniform_filter(out, size=filter_size, mode='nearest')
        out = cv.resize(out, (npx * pix_res, npx * pix_res), interpolation=cv.INTER_LANCZOS4)
    else:
        out = out[crop // pix:-crop // pix, crop // pix:-crop // pix]
        out = scipy.ndimage.uniform_filter(out, size=filter_size, mode='nearest')
        out = cv.resize(out, ((npx - 2 * crop) * pix_res, (npx - 2 * crop) * pix_res),
                        interpolation=cv.INTER_LANCZOS4)
        out = np.pad(out, ((crop * pix_res, crop * pix_res), (crop * pix_res, crop * pix_res)))
    return out


# Binary masks
def rect_mask(im, dx, dy, w, h):
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
    
    idx = (x_grid - dx > -w // 2) & (x_grid - dx < w // 2) & (y_grid - dy > -h // 2) & (y_grid - dy < h // 2)
    mask = np.zeros_like(im)
    mask[idx] = 1
    return mask


def rect_mask_xy(x, y, dx, dy, w, h):
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


def circ_mask(im, dx, dy, r):
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


def circ_mask_xy(x, y, dx, dy, r, sparse=None):
    """
    Circular mask using XY meshgrid coordinates.

    :param x: X meshgrid.
    :param y: Y meshgrid.
    :param dx: X-offset of circle.
    :param dy: Y-offset of circle.
    :param r: Radius of circle.
    :return: Binary mask.
    """
    idx = (x - dx) ** 2 + (y - dy) ** 2 < r ** 2
    if sparse is True:
        mask = np.zeros((max(x.shape), max(x.shape)))
    else:
        mask = np.zeros_like(x)
    mask[idx] = 1
    return mask


# Intensity patterns
def gaussian(x, y, dx, dy, sig_x, sig_y=None, a=1, c=0):
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

    return a * np.exp(-0.5 * ((x - dx) ** 2 / sig_x ** 2 + (y - dy) ** 2 / sig_y ** 2)) + c


def super_gaussian(x, y, dx, dy, nx, ny, sig_x, sig_y, a=1, c=0):
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
    return a * np.exp(-2 * (np.abs(x - dx) / sig_x) ** nx) * np.exp(-2 * (np.abs(y - dy) / sig_y) ** ny) + c


def gauss_array(im, nx, ny, dx, dy, d, sigma):
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
            a = a + gaussian(x, y, j * d - wy // 2 - dy, i * d - wx // 2 - dx, sigma)
    return a


def ring_gauss(x, y, dx, dy, r, w, a=1):
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
    return a * np.exp(-2 * (np.sqrt((x - dx) ** 2 + (y - dy) ** 2) - r) ** 2 / w ** 2)


def checkerboard(npx, dx, dy, rows, columns, square_size):
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


def fringes_wavefront(x, y, dx, dy, k, f, phi, a, b):
    """
    Standing wave interference pattern on the camera caused by two patches on the SLM seperated by dx and dy.
    Equation adapted from https://doi.org/10.1364/OE.24.013881.

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
    i_out = a ** 2 + b ** 2 + 2 * a * b * np.cos(k * (np.sin(gamma_x) * x + np.sin(gamma_y) * y) + phi)
    return i_out


# Phase patterns
def init_phase(img, slm_disp_obj, pms_obj, lin_phase=None, quad_phase=None, lin_method=None):
    """
    SLM phase guess to initialise phase-retrieval algorithm (see https://doi.org/10.1364/OE.16.002176).

    :param ndarray img: 2D array with size of desired output.
    :param slm_disp_obj: Instance of Params class
    :param ndarray lin_phase: Vector of length 2, containing parameters for the linear phase term
    :param ndarray quad_phase: Vector of length 2, containing parameters for the quadratic phase term
    :param str lin_method: Determines how the linear phase term is parameterised. The options are:

        -'pixel'
            Defines the linear phase in terms of Fourier pixels [px].
        -'angles'
            Defines the linear phase in terms of angles [rad].

    :return: Phase pattern of shape ``img.shape``
    """
    if lin_phase is None:
        lin_phase = np.zeros(2)
    if lin_method is None:
        lin_method = 'pixel'
    if quad_phase is None:
        quad_phase = np.zeros(2)

    x, y = make_grid(img)
    
    if lin_method == 'pixel':
        pix_x, pix_y = lin_phase
        mx = np.pi * pix_x / slm_disp_obj.res[0]
        my = np.pi * pix_y / slm_disp_obj.res[0]
    if lin_method == 'angles':
        alpha_x, alpha_y = lin_phase
        mx = np.tan(alpha_x) * pms_obj.k * slm_disp_obj.pitch
        my = np.tan(alpha_y) * pms_obj.k * slm_disp_obj.pitch
    
    r, gamma = quad_phase
    kl = mx * x + my * y
    kq = 4 * r * (gamma * y ** 2 + (1 - gamma) * x ** 2)
    return kl + kq


def lens(x, y, k, f):
    """
    Phase of a parabolic lens.

    :param x: X-meshgrid [m].
    :param y: Y-meshgrid [m].
    :param k: Wavenumber [rad/m].
    :param f: Focal length [m].
    :return: Phase of the lens [rad].
    """
    return -0.5 * k / f * (x ** 2 + y ** 2)


def doublet(x, y, k, n1, n2, r1, r2, r3, dx=None, dy=None):
    """
    Phase of a doublet lens.

    :param x: X-meshgrid [m].
    :param y: Y-meshgrid [m].
    :param k: Wavenumber [rad/m].
    :param n1: Refractive index of flint.
    :param n2: Refractive index of crown.
    :param r1: Radius of curvature of the first crown surface [m].
    :param r2: Radius of curvature of the second crown surface/ first flint surface [m].
    :param r3: Radius of curvature of the second flint surface [m].
    :param dx: X-offset of lens [m].
    :param dy: Y-offset of lens [m].
    :return: Phase of the doublet lens [rad].
    """
    if dx is None:
        dx = 0
    if dy is None:
        dy = 0
    delta1 = -r1 * (1 - np.sqrt(1 - ((x - dx) ** 2 + (y - dy) ** 2) / r1 ** 2)) + r2 * (1 - np.sqrt(1 - ((x - dx) ** 2 + (y - dy) ** 2) / r2 ** 2))
    delta2 = r3 * (1 - np.sqrt(1 - ((x - dx) ** 2 + (y - dy) ** 2) / r3 ** 2)) - r2 * (1 - np.sqrt(1 - ((x - dx) ** 2 + (y - dy) ** 2) / r2 ** 2))
    return k * ((n1 - 1) * delta1 + (n2 - 1) * delta2)


def slm_phase_doublet(dx, dy, k, xf, yf, z1, z2, fl, n1, n2, r1, r2, r3):
    """
    Models the phase difference in the wavefront measurement caused by the doublet lens and an out-of-focus camera
    placement (see equation S8 in the supplementary information of https://doi.org/10.1038/s41598-023-30296-6).

    :param dx: X-position of sample patch [m].
    :param dy: Y-position of sample patch [m].
    :param k: Wavenumber [rad/m].
    :param xf: X-position of phase measurement in the image plane [m].
    :param yf: Y-position of phase measurement in the image plane [m].
    :param z1: Distance between SLM and lens [m].
    :param z2: Distance between lens and camera [m].
    :param fl: Focal length of doublet lens [m].
    :param n1: Refractive index of flint.
    :param n2: Refractive index of crown.
    :param r1: Radius of curvature of the first crown surface [m].
    :param r2: Radius of curvature of the second crown surface/ first flint surface [m].
    :param r3: Radius of curvature of the second flint surface [m].
    :return: Corrective phase pattern [rad].
    """

    alpha_x = np.arctan(xf / fl)
    alpha_y = np.arctan(yf / fl)

    d_samp_x = dx + z1 * np.tan(alpha_x)
    d_samp_y = dy + z1 * np.tan(alpha_y)

    d_ref_x = z1 * xf / fl
    d_ref_y = z1 * yf / fl

    phi_ls = doublet(d_samp_x, d_samp_y, k, n1, n2, r1, r2, r3)
    phi_lr = doublet(d_ref_x, d_ref_y, k, n1, n2, r1, r2, r3)

    ps = np.sqrt((d_samp_x - xf) ** 2 + (d_samp_y - yf) ** 2 + z2 ** 2)
    pr = np.sqrt((d_ref_x - xf) ** 2 + (d_ref_y - yf) ** 2 + z2 ** 2)

    phi = k * (ps - pr) + phi_ls - phi_lr

    return phi


def pixel_ct_kernel(slm_pitch, pix_res, extent, m, sigma):
    """
    2D blurring kernel to model pixel crosstalk on the SLM (see https://doi.org/10.1186/s41476-021-00174-7).

    :param slm_pitch: Pixel pitch of SLM [m].
    :param pix_res: Up-scaling factor (computational pixels per SLM pixel).
    :param extent: Spatial extent of kernel in SLM pixels.
    :param m: Order of the kernel.
    :param sigma: Width of the kernel.
    :return: 2D blurring kernel.
    """
    n_kernel = pix_res * extent                             # Number of pixels in crosstalk kernel (n_kernel * n_kernel)
    f_max = 1 / (slm_pitch * extent)                        # Largest spatial frequency of crosstalk kernel
    k = np.arange(-n_kernel / 2, n_kernel / 2, 1) * f_max   # Spatial frequncies of crosstalk kernel
    kx, ky = np.meshgrid(k, k)
    h = np.exp(-((np.abs(kx)) / sigma) ** m) * np.exp(-((np.abs(ky)) / sigma) ** m)
    kernel = np.abs(np.fft.fftshift(np.fft.ifft2(h)))
    return kernel / np.sum(kernel)


# Vortex detection
def vortex_field(img, m, x, y):
    """
    Creates the phase of a vortex field of charge ``m`` at positions ``x`` and ``y`` . The origin of the coordinate
    system is in the top-left corner of ``img`` .

    :param img: 2D array with size of desired output.
    :param m: Vector of vortex charge (1 or -1).
    :param x: Vector of vortex x-coordinate [px].
    :param y: Vector of vortex y-coordinate [px].
    :return: Phase of vortex field with size ``img.shape``.
    """
    n = len(x)
    npix_y, npix_x = img.shape
    x_lin = np.arange(0, npix_x, 1)
    y_lin = np.arange(0, npix_y, 1)
    xx, yy = np.meshgrid(x_lin, y_lin)
    out = np.ones_like(img)
    for i in range(n):
        out = out * np.exp(1j * m[i] * np.angle(xx - x[i] + 1j * (yy - y[i])))
    return out


def detect_vortices(n_pix, e_holo, i_tar, threshold=None):
    """
    This function detects the positions and charges of optical vortices in an electric field.
    Todo: Tidy up this function and improve documentation.

    :param n_pix: Number of pixels.
    :param e_holo: Electric field.
    :param i_tar: Target intensity pattern.
    :param threshold: Only look for vortices in areas which are brighter than ``theshold * max(abs(i_tar) ** 2)``.
                      Vortices in low-intensity regions are hard to detect.
    :return: Charge of vortices and their xy coordinates.
    """
    if threshold is None:
        threshold = 0.2

    tar_res = i_tar.shape[0]
    n_pix = n_pix // 2

    e_crop = e_holo[tar_res // 2 - n_pix:tar_res // 2 + n_pix, tar_res // 2 - n_pix:tar_res // 2 + n_pix]
    i_tar_crop = i_tar[tar_res//2 - n_pix:tar_res//2 + n_pix, tar_res//2 - n_pix:tar_res//2 + n_pix]
    i_tar_crop = i_tar_crop / np.max(np.sqrt(i_tar_crop))
    e_phi = e_crop / np.max(np.abs(e_crop)) / np.sqrt(i_tar_crop)
    roi = i_tar > (threshold * np.max(i_tar))
    roi = roi[tar_res//2 - n_pix:tar_res//2 + n_pix, tar_res//2 - n_pix:tar_res//2 + n_pix]

    phi = np.angle(e_crop)
    phi_crop = phi * roi

    img_r = np.real(e_phi)
    mask_r = (np.abs(img_r) < 0.5)
    img_i = np.imag(e_phi)
    mask_i = (np.abs(img_i) < 0.5)
    mask = mask_r & mask_i
    mask[:, 1] = 0
    mask[:, -1] = 0
    mask[1, :] = 0
    mask[-1, :] = 0

    idx = np.nonzero(mask * roi)
    print(idx[0].size)
    charge = np.zeros_like(idx[0])
    thres = 1.5 * np.pi
    for i in range(idx[0].size):
        nbrs = phi[idx[0][i] - 1, idx[1][i] - 1:idx[1][i] + 2]
        nbrs = np.append(nbrs, phi[idx[0][i], idx[1][i] + 1])
        nbrs = np.append(nbrs, np.flip(phi[idx[0][i] + 1, idx[1][i] - 1:idx[1][i] + 2]))
        nbrs = np.append(nbrs, phi[idx[0][i], idx[1][i] - 1])
        nbrs = np.append(nbrs, nbrs[0])
        nbrs_diff = np.diff(nbrs)
        if np.any(nbrs_diff > thres) and np.sum(np.abs(nbrs_diff) > thres / 2) == 1:
            charge[i] = 1
        elif np.any(nbrs_diff < -thres) and np.sum(np.abs(nbrs_diff) > thres / 2) == 1:
            charge[i] = -1
        else:
            charge[i] = 0

    idx_pos = charge > 0
    idx_neg = charge < 0
    idx_filt = (idx[0][charge != 0], idx[1][charge != 0])
    charge_filt = charge[charge != 0]
    mask_filt = np.zeros_like(mask)
    mask_filt[idx_filt[0], idx_filt[1]] = 1
    labelled_mask, n_vtx = ndimage.measurements.label(mask_filt)

    vtx = np.zeros((n_vtx, 3))
    for i in range(n_vtx):
        idx_label = np.argwhere(labelled_mask == i + 1)
        vtx[i, :2] = np.mean(idx_label, axis=0)
        idx_charge = np.argwhere((idx_filt[0] == idx_label[0, 0]) & (idx_filt[1] == idx_label[0, 1]))
        vtx[i, -1] = charge_filt[idx_charge]

    print('Number of detected vortices:', n_vtx)
    return vtx


class Hologram:
    """
    This class provides arrays needed for the CG minimisation:

        - The target light potential
        - the signal region
        - the measured constant SLM phase and intensity at the required resolution
        - and the initial SLM phase guess to start the phase retrieval.

    Some patterns, including the patterns from our publication (https://doi.org/10.1038/s41598-023-30296-6), are
    pre-defined here. Feel free to define your own patterns, you don't have to use this class to do this.

    Todo: Tidy up this class and improve documentation.
    """
    def __init__(self, slm_disp_obj, pms_obj, name, npix, npix_pad=None, pix_res=1, phase_guess_type='random',
                 linear_phase=None, quadratic_phase=None, slm_field_type='guess', propagation_type='fft',
                 target_position=None, target_width=None, target_blur=None,
                 checkerboard_rows=8, checkerboard_columns=10, checkerboard_square_size=32):
        """
        Initialisation.

        :param slm_disp_obj: Object created by a subclass of :py:class:`hardware.SlmBase`
        :param pms_obj: Object created by a subclass of :py:class:`hardware.ParamsBase`
        :param str name: Name of the pattern

            -'disc'
                Disc-shaped potential.
            -'super_gauss'
                Super Gaussian potential.
            -'ring_gauss'
                Ring with Gaussian profile.
            -'spot_array'
                Gaussian spot array.
            -'harm_conf'
                Hamonic confinement.
            -'square'
                Square.
            -'rectangle'
                Rectangle.
            -'checkerboard'
                Checkerboard.
            -'duke'
                Duke of Wellington.
            -'or_gate'
                OR gate.

        :param npix: Size of the used SLM area ``(npix, npix)`` [px].
        :param npix_pad: Size of padded SLM plane ``(npix_pad, npix_pad)`` [px].
        :param pix_res: Computational pixels per SLM pixel (``(pix_res, pix_res)`` computational pixels per SLM pixel).
        :param str phase_guess_type: Type of the SLM phase guess to initialise the phase retrieval algorithm.

            -'random'
                Entirely random phase guess.
            -'guess'
                Linear and quadratic phase guess.

        :param linear_phase: Linear phase term parameters.
        :param quadratic_phase: Quadratic phase term parameters.
        :param slm_field_type: Determines the constant electric field at the SLM. Options are:

            - 'guess'
                Assumes flat constant phase at the SLM and a Gaussian beam.
            - 'measured'
                Loads the measured constant intensity and phase at the SLM and upscales them to the required resolution.

        :param propagation_type: Type of light propagation. Can be 'fft' or 'asm'.
        :param target_position: Position of target light potential in x- and y- direction.
        :param target_width: Size of target light potential.
        :param target_blur: Width of blurring kernel for target light potential.
        :param checkerboard_rows: Number of rows for checkerboard pattern.
        :param checkerboard_columns: Number of columns for checkerboard pattern.
        :param checkerboard_square_size: Size of a single checkerboard square.
        """

        self.name = name
        self.npix = npix
        if npix_pad is None:
            npix_pad = 2 * self.npix
        self.npix_pad = npix_pad
        self.slm_ones = np.ones((self.npix, self.npix))
        self.pix_res = pix_res
        self.tar_res = self.npix_pad * self.pix_res
        self.tar_ones = np.ones((self.tar_res, self.tar_res))
        self.guess_type = phase_guess_type
        if linear_phase is None:
            linear_phase = np.asarray([0, 0])
        self.lin_phase = linear_phase
        if quadratic_phase is None:
            quadratic_phase = np.asarray([0, 0])
        self.quad_phase = quadratic_phase
        self.slm_field_type = slm_field_type
        self.prop = propagation_type
        self.img_pitch_fft = pms_obj.wavelength * pms_obj.fl / (slm_disp_obj.pitch * 2 * self.npix)
        if self.prop == 'asm':
            self.asm_corr = 0.5 * self.img_pitch_fft / (slm_disp_obj.pitch / self.pix_res)
        elif self.prop == 'fft':
            self.asm_corr = 1

        if target_position is None:
            target_position = -self.npix // 4
        self.mask_pos = int(target_position * self.asm_corr)

        if target_width is None:
            target_width = self.npix // 2
        self.tar_width = int(target_width * self.asm_corr)

        if target_blur is None:
            target_blur = self.npix_pad / self.npix
        self.tar_blur = target_blur * self.asm_corr

        if self.slm_field_type == 'guess':
            i_slm = gaussian(slm_disp_obj.meshgrid_slm[0], slm_disp_obj.meshgrid_slm[1], 0, 0,
                             pms_obj.beam_diameter / 4)
            phi_slm_native = np.zeros_like(i_slm)
            i_slm = load_filter_upscale(i_slm, self.npix, self.pix_res)
            phi_slm = np.zeros_like(i_slm)
        if self.slm_field_type == 'measured':
            i_slm = load_filter_upscale(pms_obj.i_path, self.npix, self.pix_res, crop=pms_obj.crop,
                                        filter_size=pms_obj.i_filter_size)
            if np.min(i_slm) < 0:
                i_slm -= np.min(i_slm)
            phi_slm = load_filter_upscale(pms_obj.phi_path, self.npix, self.pix_res, crop=pms_obj.crop,
                                          filter_size=pms_obj.phi_filter_size)
            phi_slm_native = load_filter_upscale(pms_obj.phi_path, self.npix, 1, crop=pms_obj.crop,
                                                 filter_size=pms_obj.phi_filter_size)

        self.phi_slm = phi_slm
        self.phi_slm_native = phi_slm_native
        self.laser_pow = np.sum(i_slm)
        self.i_slm = i_slm / self.laser_pow
        self.a_slm = np.sqrt(self.i_slm)
        self.e_slm = self.a_slm * np.exp(1j * self.phi_slm)

        # Initial SLM phase-guess
        if self.guess_type == 'guess':
            self.phi_init = np.remainder(init_phase(self.slm_ones, slm_disp_obj, pms_obj, self.lin_phase,
                                                    self.quad_phase) - self.phi_slm_native, slm_disp_obj.max_phase)
        elif self.guess_type == 'constant':
            self.phi_init = np.zeros((self.npix, self.npix)) - self.phi_slm_native
        elif self.guess_type == 'load':
            self.phi_init = scipy.io.loadmat('phi.mat')
            self.phi_init = self.phi_init['phi']
        elif self.guess_type == 'random':
            np.random.seed(0)
            self.phi_init = np.random.uniform(low=0, high=2 * np.pi, size=(self.npix, self.npix))

        # Target intensity and signal region
        if self.name == 'disc':
            self.sig_mask = circ_mask(self.tar_ones, self.mask_pos, self.mask_pos, self.tar_width / 2)
            a_tar_mask = circ_mask(self.tar_ones, self.mask_pos, self.mask_pos, self.tar_width / 2 - self.tar_width / 4)
            a_tar = scipy.ndimage.gaussian_filter(a_tar_mask, self.tar_blur)
        if self.name == 'super_gauss':
            self.sig_mask = circ_mask(self.tar_ones, self.mask_pos, self.mask_pos, self.tar_width / 2)
            x, y = make_grid(self.tar_ones)
            a_tar = np.sqrt(
                super_gaussian(x, y, self.mask_pos, self.mask_pos, 10, 2, self.tar_width / 4, self.tar_width / 16))
        if self.name == 'ring_gauss':
            self.sig_mask = circ_mask(self.tar_ones, self.mask_pos, self.mask_pos, self.tar_width / 2)
            x, y = make_grid(self.tar_ones)
            a_tar = ring_gauss(x, y, self.mask_pos, self.mask_pos, self.tar_width / 4, 15)
            a_tar = np.sqrt(a_tar ** 2 / np.max(a_tar ** 2) * 0.9 + 0.1)
        elif self.name == 'spot_array':
            self.sig_mask = rect_mask(self.tar_ones, self.mask_pos, self.mask_pos, 270 * self.asm_corr,
                                      270 * self.asm_corr)
            cropped_array, [pad, idx] = crop_to_mask(self.tar_ones, self.sig_mask)
            a_tar = gauss_array(cropped_array, 14, 14, 0, 0, 15 * self.asm_corr, 4 * self.asm_corr)
            a_tar = np.pad(a_tar, pad)
            a_tar = a_tar / np.max(a_tar)
            a_tar = (a_tar + 0.5) * self.sig_mask
        elif self.name == 'harm_conf':
            self.sig_mask = circ_mask(self.tar_ones, self.mask_pos, self.mask_pos, 200 * self.asm_corr)
            x, y = make_grid(self.tar_ones)
            a_tar_mask = circ_mask(self.tar_ones, self.mask_pos, self.mask_pos, 140 * self.asm_corr)
            a_tar = gaussian(x, y, self.mask_pos, self.mask_pos, 600, 600)
            a_tar = a_tar * a_tar_mask
            mask = a_tar > 0
            a_tar[mask] -= np.min(a_tar[a_tar > 0])
            a_tar[mask] += 0.5 * np.max(a_tar[a_tar > 0])
            a_tar = a_tar / np.max(a_tar)
            a_tar[mask] += np.max(a_tar[mask])
            a_tar = scipy.ndimage.gaussian_filter(a_tar, 2)
        elif self.name == 'square':
            self.sig_mask = rect_mask(self.tar_ones, self.mask_pos, self.mask_pos, self.tar_width, self.tar_width)
            a_tar_mask = rect_mask(self.tar_ones, self.mask_pos, self.mask_pos, self.tar_width / 2, self.tar_width / 2)
            a_tar = scipy.ndimage.gaussian_filter(a_tar_mask, self.tar_blur * self.asm_corr)
        elif self.name == 'rectangle':
            a_tar_mask = rect_mask(self.tar_ones, self.mask_pos, self.mask_pos + 1, 5 * self.tar_width, self.tar_width)
            a_tar = scipy.ndimage.gaussian_filter(a_tar_mask, self.tar_blur * self.asm_corr)
            self.sig_mask = ((a_tar ** 2) > (0.1 * np.max(a_tar ** 2))) + 0.0
        elif self.name == 'checkerboard':
            self.cb_square_size = int(checkerboard_square_size * self.asm_corr)
            self.cb_m = checkerboard_columns
            self.cb_n = checkerboard_rows
            self.sig_mask = rect_mask(self.tar_ones, self.mask_pos, self.mask_pos,
                                      (self.cb_m + 4) * self.cb_square_size, (self.cb_n + 4) * self.cb_square_size)
            i_tar = checkerboard(self.tar_res, self.mask_pos, self.mask_pos, self.cb_m, self.cb_n, self.cb_square_size)
            i_tar = scipy.ndimage.gaussian_filter(i_tar * 255, self.tar_blur * self.asm_corr)
            a_tar = np.sqrt(i_tar)
        elif self.name == 'duke':
            # Photograph taken from Ed O’Keeffe Photography.
            img_size = 400
            self.sig_mask = rect_mask(self.tar_ones, self.mask_pos, self.mask_pos, img_size, img_size)
            a_tar = cv.cvtColor(cv.imread('./data/duke.jpg'), cv.COLOR_BGR2GRAY)
            a_tar = a_tar[a_tar.shape[0] - a_tar.shape[1]:-1, 1:-1]
            a_tar = cv.resize(a_tar, (img_size, img_size), interpolation=cv.INTER_LANCZOS4)
            a_tar = a_tar.astype('float64')
            a_tar = scipy.ndimage.gaussian_filter(a_tar, 1)
            a_tar = np.pad(a_tar, ((npix//2 + (npix//2 - img_size) // 2, npix + (npix//2 - img_size) // 2),
                                   (npix//2 + (npix//2 - img_size) // 2, npix + (npix//2 - img_size) // 2))) / np.max(a_tar)
        elif self.name == 'or_gate':
            self.sig_mask = rect_mask(self.tar_ones, self.mask_pos, self.mask_pos, 400, 400)
            img = np.sqrt(cv.cvtColor(cv.imread('../../targets/transistor_target.bmp'), cv.COLOR_BGR2GRAY))
            img_shape = img.shape
            a_tar = img.astype('float64')
            a_tar = np.pad(a_tar,
                           (((self.tar_res - img_shape[0]) // 2 + self.mask_pos,
                             (self.tar_res - img_shape[0]) // 2 - self.mask_pos),
                            ((self.tar_res - img_shape[1]) // 2 + self.mask_pos,
                             (self.tar_res - img_shape[1]) // 2 - self.mask_pos))) / np.max(a_tar)


        # Normalise target intensity
        i_tar = a_tar ** 2 * self.sig_mask
        i_tar_w = np.sum(i_tar)
        self.a_tar = a_tar * np.sqrt(self.laser_pow / i_tar_w)
        i_tar = np.abs(a_tar) ** 2
        tar_pow_w = np.sum(i_tar * self.sig_mask)
        self.i_tar = i_tar / tar_pow_w
