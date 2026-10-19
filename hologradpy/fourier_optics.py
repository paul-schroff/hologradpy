from __future__ import annotations

import torch
from numpy.typing import ArrayLike, NDArray
from torch import Tensor

from .grids import pixel_frame_vector


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


def intensity_nyquist_pixel_size(
    wavelength: Tensor | float,
    focal_length: float,
    pixel_size_in: Tensor | float,
    resolution_in: Tensor | int,
) -> Tensor | float:
    """Coarsest focal-plane sampling at which the focal intensity is still resolved.

    ``resolution_in`` is the number of *illuminated* samples, not a padded transform
    length.

    Args:
        wavelength: Wavelength in metres.
        focal_length: Focal length in metres.
        pixel_size_in: Input plane sample spacing in metres.
        resolution_in: Number of input samples along the same axis.

    Returns:
        The coarsest focal-plane spacing that still resolves the intensity, in metres.
    """
    return fourier_lens_pixel_size(
        wavelength, focal_length, pixel_size_in, 2 * resolution_in
    )


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


def central_lobe_width(
    wavelength: Tensor | float,
    focal_length: float,
    aperture_width: Tensor | float,
) -> Tensor | float:
    """The zero-to-zero width of the central lobe of the sinc^2 that a uniformly lit
    rectangular aperture forms in the focal plane of a Fourier lens.

    Args:
        wavelength: Wavelength in metres.
        focal_length: Focal length in metres.
        aperture_width: The width of the aperture along the same axis, in metres.

    Returns:
        The width ``2 * wavelength * focal_length / aperture_width``, in metres.
    """
    return 2 * wavelength * focal_length / aperture_width


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

def addressable_half_extent(
    wavelength: float,
    focal_length: float,
    pixel_size: tuple[float, float] | NDArray | Tensor,
) -> tuple[float, float]:
    """Half-extent ``(x, y)`` of the focal-plane region an SLM can address, in metres.
    The first-order deflection of a grating at the SLM's Nyquist frequency, per axis.
    ``pixel_size`` is ``(y, x)``, so x takes the second entry. A per-wavelength stack
    of pitches is read from its first row, all wavelengths sharing the one SLM.
    """
    pitch_y, pitch_x = (
        float(pitch) for pitch in torch.as_tensor(pixel_size).reshape(-1, 2)[0]
    )
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


def beam_shaping_focal_length(
    beam_radius: Tensor | float | tuple[float, float],
    focal_length: float,
    target_beam_radius: Tensor | float | tuple[float, float],
    wavelength: Tensor | float,
) -> Tensor | float | tuple[float, float]:
    """Focal length of the lens that sets a Gaussian beam's radius in the focal plane.

    Args:
        beam_radius: Radius of the incident beam in metres, or a pair of radii along
            two axes.
        focal_length: Focal length of the Fourier lens in metres.
        target_beam_radius: Wanted radius in the focal plane, in metres, or a pair
            along the same axes.
        wavelength: Wavelength in metres.

    Returns:
        The shaping lens's focal length in metres, or a pair along the same axes.
        Broadcasts, so it accepts per-axis tensors.
    """
    if isinstance(beam_radius, tuple):
        return tuple(
            beam_shaping_focal_length(radius, focal_length, target, wavelength)
            for radius, target in zip(beam_radius, target_beam_radius, strict=True)
        )
    focal_spot = get_focal_spot_radius(beam_radius, wavelength, focal_length)
    return (
        beam_radius
        * focal_length
        / (target_beam_radius**2 - focal_spot**2) ** 0.5
    )


def get_focal_spot_radius(
    beam_radius: Tensor | float | tuple[float, float],
    wavelength: Tensor | float,
    focal_length: float,
) -> Tensor | float | tuple[float, float]:
    """Calculates the radius of the focal spot for a Gaussian beam with a given
    `beam_radius`, focused by a lens with a given `focal_length`.

    The 1/e^2 intensity radius, and the smallest radius any shaping lens can reach.

    Args:
        beam_radius: The radius of the Gaussian beam at the lens in metres, or a pair
            of radii along two axes.
        wavelength: The wavelength of the light in metres.
        focal_length: The focal length of the lens in metres.

    Returns:
        The radius of the focal spot in metres, or a pair along the same axes.
        Broadcasts, so it accepts per-axis tensors.
    """
    if isinstance(beam_radius, tuple):
        return tuple(
            get_focal_spot_radius(radius, wavelength, focal_length)
            for radius in beam_radius
        )
    return (wavelength * focal_length) / (torch.pi * beam_radius)


def focal_field(
    field: Tensor,
    positions: Tensor | ArrayLike,
    pixel_size: tuple[float, float],
    wavelength: float,
    focal_length: float,
) -> Tensor:
    """Return the field that a Fourier lens makes at focal-plane positions, from the
    field on a grid before the lens.

    The field at each position is the lens's Fourier transform there, so a position
    can lie between output pixels. It includes the lens's prefactor,
    :func:`fourier_lens_power_prefactor`.

    Args:
        field: The ``(height, width)`` field on the grid. Its centre pixel is
            ``(height // 2, width // 2)``.
        positions: ``(N, 2)`` focal-plane positions, ``(x, y)`` in metres from the
            zeroth order.
        pixel_size: The grid's ``(y, x)`` pitch in metres.
        wavelength: The wavelength in metres.
        focal_length: The focal length of the lens in metres.

    Returns:
        Tensor: The ``(N,)`` field at the positions, in the dtype of ``field``.
    """
    rows, columns = _fourier_phasors(
        positions, tuple(field.shape), pixel_size, wavelength, focal_length, field
    )
    prefactor = fourier_lens_power_prefactor(
        pixel_size[0] * pixel_size[1], wavelength, focal_length
    )
    return prefactor * (rows * (field @ columns.T).T).sum(-1)


def focal_field_adjoint(
    spot_field: Tensor,
    positions: Tensor | ArrayLike,
    resolution: tuple[int, int],
    pixel_size: tuple[float, float],
    wavelength: float,
    focal_length: float,
) -> Tensor:
    """Return the field on a grid that the adjoint of :func:`focal_field` makes from
    the field at focal-plane positions.

    Args:
        spot_field: The ``(N,)`` field at the positions.
        positions: ``(N, 2)`` focal-plane positions, ``(x, y)`` in metres from the
            zeroth order.
        resolution: The grid's ``(height, width)``.
        pixel_size: The grid's ``(y, x)`` pitch in metres.
        wavelength: The wavelength in metres.
        focal_length: The focal length of the lens in metres.

    Returns:
        Tensor: The ``(height, width)`` field on the grid, in the dtype of
        ``spot_field``.
    """
    rows, columns = _fourier_phasors(
        positions, resolution, pixel_size, wavelength, focal_length, spot_field
    )
    prefactor = fourier_lens_power_prefactor(
        pixel_size[0] * pixel_size[1], wavelength, focal_length
    )
    return (rows.conj().T * (prefactor * spot_field)) @ columns.conj()


def _fourier_phasors(
    positions: Tensor | ArrayLike,
    resolution: tuple[int, int],
    pixel_size: tuple[float, float],
    wavelength: float,
    focal_length: float,
    like: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return the factors of a Fourier lens's transform at focal-plane positions,
    ``(N, height)`` for the rows and ``(N, width)`` for the columns.

    The field at position ``n`` is the sum over the grid of the field times
    ``rows[n, :, None] * columns[n, None, :]``. The factors are in the dtype and on the
    device of ``like``.
    """
    height, width = resolution
    pitch_y, pitch_x = pixel_size
    positions = torch.as_tensor(positions, dtype=torch.float64, device=like.device)
    # Radians per metre of focal-plane position and per metre across the grid.
    scale = 2 * torch.pi / (wavelength * focal_length)
    rows = torch.arange(height, dtype=torch.float64, device=like.device)
    columns = torch.arange(width, dtype=torch.float64, device=like.device)
    # The x of each column and the y of each row, in metres.
    x, y = pixel_frame_vector(
        ((columns - width // 2) * pitch_x, (rows - height // 2) * pitch_y)
    )
    row_phasors = torch.exp(-1j * scale * positions[:, 1:] * y)
    column_phasors = torch.exp(-1j * scale * positions[:, :1] * x)
    return row_phasors.to(like.dtype), column_phasors.to(like.dtype)
