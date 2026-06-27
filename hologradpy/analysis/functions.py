import numpy as np
from numpy.typing import NDArray



def tilt(
    xy: tuple[NDArray, NDArray], *args: float, mask: NDArray | None = None
) -> NDArray:
    """Fit function for the first Zernike terms (piston and tilt) of the form
    ``c0 + c1*x + c2*y + c3*2*x*y``.

    Args:
        xy (tuple[NDArray, NDArray]): ``(x, y)`` coordinate vectors.
        *args (float): The four coefficients ``(c0, c1, c2, c3)``.
        mask (NDArray | None, optional): Mask multiplied into the output.
            Defaults to all ones.

    Returns:
        NDArray: The evaluated tilt surface.
    """
    if mask is None:
        mask = np.ones_like(xy[0])

    x, y = xy
    c = np.array(args)
    arr = c[0] + c[1] * x + c[2] * y + c[3] * 2 * x * y
    return arr * mask


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


# TODO: Make this work for odd values of square_size
def checkerboard(
    resolution: tuple[int, int],
    number_of_squares: tuple[int, int] = None,
    square_size: int = None,
    shift_x: int = 0,
    shift_y: int = 0,
    dark_square_brightness: float = 0.0,
    light_square_brightness: float = 1.0,
) -> NDArray[np.float_]:
    """
    Creates a checkerboard pattern with specified resolution, square size,
    number of squares, and brightness values for dark and light squares.

    Args:
        resolution (tuple[int, int]): Resolution of the output image in
            pixels in the format (height, width).
        number_of_squares (tuple[int, int]): Number of squares along
            (y, x) axes. Overrides square_size.
        square_size (int, optional): Size of each square in pixels. If None,
            calculated from number_of_squares and resolution.
        shift_x (int, optional): Horizontal shift of the pattern. Defaults to
            0.
        shift_y (int, optional): Vertical shift of the pattern. Defaults to 0.
        dark_square_brightness (float, optional): Value for dark square
            brightness. Defaults to 0.0.
        light_square_brightness (float, optional): Value for light square
            brightness. Defaults to 1.0.

    Returns:
        NDArray[np.float_]: A 2D array with the checkerboard pattern.
    """
    height, width = resolution
    number_of_squares_y, number_of_squares_x = number_of_squares

    if square_size is None:
        square_size = min(width // number_of_squares_y, height // number_of_squares_x)

    checkerboard_resolution = (
        number_of_squares_y * square_size,
        number_of_squares_x * square_size,
    )

    y, x = np.indices(checkerboard_resolution)

    cb = np.where(
        ((x // square_size) + (y // square_size)) % 2 == 0,
        dark_square_brightness,
        light_square_brightness,
    )

    pad_x = (resolution[1] - checkerboard_resolution[1]) // 2
    pad_y = (resolution[0] - checkerboard_resolution[0]) // 2

    return np.pad(
        cb, ((pad_y + shift_y, pad_y - shift_y), (pad_x + shift_x, pad_x - shift_x))
    )
