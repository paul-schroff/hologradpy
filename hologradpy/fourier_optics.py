from __future__ import annotations

from numpy.typing import NDArray
from torch import Tensor


def fourier_lens_pixel_size(
    wavelength: Tensor | float,
    focal_length: float,
    pixel_size_in: Tensor | float,
    resolution_in: Tensor | int,
) -> Tensor | float:
    """Natural sample spacing of the plane a Fourier lens transforms into.

    A lens of focal length ``f`` maps an input sampled at ``pixel_size_in`` over
    ``resolution_in`` points to a focal plane with spacing
    ``wavelength * focal_length / (pixel_size_in * resolution_in)``.

    Args:
        wavelength: Wavelength in metres.
        focal_length: Focal length in metres.
        pixel_size_in: Input plane sample spacing in metres.
        resolution_in: Number of input samples along the same axis.

    Returns:
        The output plane's sample spacing in metres.
    """
    return wavelength * focal_length / (pixel_size_in * resolution_in)


def fourier_lens_resolution(
    wavelength: Tensor | float,
    focal_length: float,
    pixel_size_in: Tensor | float,
    pixel_size_out: Tensor | float,
) -> Tensor | float:
    """How many input samples a Fourier lens needs to achieve the taget output spacing.

    Args:
        wavelength: Wavelength in metres.
        focal_length: Focal length in metres.
        pixel_size_in: Input plane sample spacing in metres.
        pixel_size_out: The output plane sample spacing wanted, in metres.

    Returns:
        The number of input samples along that axis.
    """
    return wavelength * focal_length / (pixel_size_in * pixel_size_out)


def fourier_lens_power_prefactor(
    pixel_area_in: Tensor | float,
    wavelength: Tensor | float,
    focal_length: float,
) -> Tensor | float:
    """Amplitude prefactor making a Fourier lens conserve optical power.

    Args:
        pixel_area_in: Area of one input pixel, from
            :func:`~hologradpy.optics.complex_amplitude.pixel_area`.
        wavelength: Wavelength in metres.
        focal_length: Focal length in metres.

    Returns:
        The prefactor, one per wavelength.
    """
    return pixel_area_in / (wavelength * focal_length)


def fourier_lens_half_extent(
    wavelength: Tensor | float,
    focal_length: float,
    pixel_size_in: Tensor | float,
) -> Tensor | float:
    """Half the focal-plane extent a Fourier lens can address, in metres.

    ``wavelength * focal_length / (2 * pixel_size_in)``: the first-order deflection of a
    grating at the input's Nyquist frequency. A spot cannot be placed beyond it, because
    the grating that would put it there aliases.

    The companion of :func:`fourier_lens_pixel_size`, and the same relation seen
    from the other end: the output plane holds ``resolution_in`` samples of that
    spacing, so it spans ``resolution_in`` times it, and this is half of that.
    Unlike the spacing, it does not depend on the sampling, only on the pitch.

    Args:
        wavelength: Wavelength in metres.
        focal_length: Focal length in metres.
        pixel_size_in: Input plane sample spacing in metres.

    Returns:
        The half-extent in metres. Broadcasts, so it accepts per-axis or per-wavelength
        tensors.
    """
    return wavelength * focal_length / (2.0 * pixel_size_in)

# TODO: Make this return (y, x)?
def addressable_half_extent(
    wavelength: float,
    focal_length: float,
    pixel_size: tuple[float, float] | NDArray | Tensor,
) -> tuple[float, float]:
    """Half-extent ``(x, y)`` of the focal-plane region an SLM can address, in metres.
    The first-order deflection of a grating at the SLM's Nyquist frequency, per axis.
    ``pixel_size`` is ``(y, x)``, so x takes the second entry.
    """
    pitch_y, pitch_x = (float(pitch) for pitch in pixel_size)
    return (
        fourier_lens_half_extent(wavelength, focal_length, pitch_x),
        fourier_lens_half_extent(wavelength, focal_length, pitch_y),
    )


def fourier_lens_magnification(
    wavelength: Tensor | float,
    focal_length: float,
    pixel_size_in: Tensor | float,
    resolution_in: Tensor | int,
    pixel_size_out: Tensor | float,
) -> Tensor | float:
    """Zoom a Fourier-lens transform needs to land on a chosen output spacing.

    The ratio of the spacing from :func:`fourier_lens_pixel_size` to the spacing wanted,
    and the magnification the zoom transforms take. Greater than one samples the focal
    plane more finely than the plain transform would.

    Args:
        wavelength: Wavelength in metres.
        focal_length: Focal length in metres.
        pixel_size_in: Input plane sample spacing in metres.
        resolution_in: Number of input samples along the same axis.
        pixel_size_out: Wanted output sample spacing in metres.

    Returns:
        The magnification, dimensionless.
    """
    return (
        fourier_lens_pixel_size(
            wavelength, focal_length, pixel_size_in, resolution_in
        )
        / pixel_size_out
    )
