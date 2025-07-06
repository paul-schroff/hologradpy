import numpy as np
from numpy.typing import NDArray

from .. import patterns as pt


def tilt(
    xy: tuple[NDArray, NDArray], *args: float, mask: NDArray | None = None
) -> NDArray:
    """
    Fit function containing the first three Zernike polynomials of the form
    z = c0 + c1 * x + c2 * y + c3 * 2xy.

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


def gaussian(xy: tuple[NDArray, NDArray], *args: float) -> NDArray:
    """
    Gaussian fit function.

    :param xy: x, y coordinate vectors.
    :param args: Fitting parameters passed to patterns.gaussian.
    :return: Gaussian.
    """
    x, y = xy
    arr = pt.gaussian(x, y, *args)
    return arr


def interferometric_fringes(
    x: NDArray[np.float_],
    y: NDArray[np.float_],
    separation_x: float,
    separation_y: float,
    wavenumber: float,
    focal_length: float,
    phase: float,
    amplitude: float,
) -> NDArray[np.float_]:
    """Interference pattern on the camera caused by two superpixels
    on the SLM seperated by separation_x and separation_y. Equation adapted
    from https://doi.org/10.1364/OE.24.013881.

    Args:
        x (NDArray[np.float_]): x coordinates.
        y (NDArray[np.float_]): y coordinates.
        separation_x (float): Separation between two superpixels along x.
        separation_y (float): Separation between two superpixels along y.
        wavenumber (float): Wavenumber of the light.
        focal_length (float): Focal length of the Fourier lens.
        phase (float): Phase difference between the two superpixels.
        amplitude_a (float): Amplitude due to superpixel a.
        amplitude_b (float): Amplitude due to superpixel b.
    Returns:
        NDArray[np.float_]: Interference pattern on the camera.
    """
    angle_x = np.arctan(separation_x / focal_length)
    angle_y = np.arctan(separation_y / focal_length)

    intesity = 2 * amplitude**2 + 2 * amplitude**2 * np.cos(
        wavenumber * (np.sin(angle_x) * x + np.sin(angle_y) * y) + phase
    )
    return intesity