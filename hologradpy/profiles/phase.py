"""Analytic phase profiles on a 2D grid (lenses, tilts, curvatures)."""

from __future__ import annotations
from typing import Literal, TypeVar

import numpy as np
from numpy.typing import NDArray
import torch

from array_api_compat import array_namespace

from ..fourier_transforms import fft_2d, ifft_2d

ArrayLike = TypeVar("ArrayLike", torch.Tensor, NDArray)
TiltUnits = Literal["degrees", "radians", "metres", "lines_per_mm"]
CurvatureUnits = Literal["radians_per_pixel_squared", "radians_per_metre_squared"]


def tilt_to_angle(
    tilt: ArrayLike,
    tilt_units: TiltUnits = "metres",
    wavenumber: float | None = None,
    focal_length: float | None = None,
) -> ArrayLike:
    """Convert a tilt in ``tilt_units`` to the beam deflection angle in radians.

    A linear phase ramp ``k * angle * x`` steers the focal spot. This is the single
    place that maps every supported tilt representation onto that angle:

    - ``"radians"``: the beam deflection angle directly.
    - ``"degrees"``: the angle in degrees (small-angle, consistent with``"radians"``).
    - ``"metres"``: a focal-plane displacement ``d``. Paraxially 
        ``angle = d / focal_length``.
    - ``"lines_per_mm"``: a grating spatial frequency ``nu``; the first order deflects 
        by ``angle = nu * 1e3 * wavelength`` with ``wavelength = 2 * pi / wavenumber``.

    Args:
        tilt: Tilt value(s) in ``tilt_units``. tilt_units: One of ``"radians"``,
            ``"degrees"``, ``"metres"``, ``"lines_per_mm"``.
        wavenumber: Wavenumber ``2 * pi / wavelength`` [rad/m]. Required for
            ``"lines_per_mm"``.
        focal_length: Focal length of the downstream Fourier lens [m]. Required for 
            ``"metres"``.

    Returns:
        The beam deflection angle in radians (same type/shape as ``tilt``).
    """
    match tilt_units:
        case "radians":
            return tilt
        case "degrees":
            return tilt * (np.pi / 180.0)
        case "metres":
            if focal_length is None:
                raise ValueError(
                    'focal_length must be provided when tilt_units is "metres".'
                )
            return tilt / focal_length
        case "lines_per_mm":
            if wavenumber is None:
                raise ValueError(
                    "wavenumber must be provided when tilt_units is "
                    '"lines_per_mm".'
                )
            wavelength = 2 * np.pi / wavenumber
            return tilt * 1e3 * wavelength
        case _:
            raise ValueError(
                f"Unknown tilt_units {tilt_units!r}; expected one of "
                '"radians", "degrees", "metres", "lines_per_mm".'
            )


def lens_phase(
    x: ArrayLike, y: ArrayLike, focal_length: float, wavenumber: float
) -> ArrayLike:
    """Calculates the phase of an ideal lens on a 2D grid.

    Args:
        x (ArrayLike): X coordinates.
        y (ArrayLike): Y coordinates.
        focal_length (float): Focal length of the lens.
        wavenumber (float): Wavenumber of the light.

    Returns:
        ArrayLike: Lens phase.
    """
    return -0.5 * wavenumber / focal_length * (x**2 + y**2)


def spherical_surface(
    x: ArrayLike,
    y: ArrayLike,
    radius: float,
    shift_x: float = 0.0,
    shift_y: float = 0.0,
) -> ArrayLike:
    """Calculates a spherical surface.

    Args:
        x (ArrayLike): X-meshgrid [m].
        y (ArrayLike): Y-meshgrid [m].
        wavenumber (float): Wavenumber [rad/m].
        radius (float): Radius of curvature [m].
        shift_x (float): X-offset of lens [m].
        shift_y (float): Y-offset of lens [m].
    """
    xp = array_namespace(x, y)
    surface = radius * (
        1 - xp.sqrt(1 - ((x - shift_x) ** 2 + (y - shift_y) ** 2) / radius**2)
    )
    return surface


def doublet_lens(
    x: ArrayLike,
    y: ArrayLike,
    wavenumber: float,
    refractive_index_flint: float,
    refractive_index_crown: float,
    radius_crown: float,
    radius_crown_flint: float,
    radius_flint: float,
    shift_x: float = 0.0,
    shift_y: float = 0.0,
) -> ArrayLike:
    """Calculates the phase of a doublet lens.

    Args:
        x (ArrayLike): X-meshgrid [m].
        y (ArrayLike): Y-meshgrid [m].
        wavenumber (float): Wavenumber [rad/m].
        refractive_index_flint (float): Refractive index of flint.
        refractive_index_crown (float): Refractive index of crown.
        radius_crown (float): Radius of curvature of the first crown surface
            [m].
        radius_crown_flint (float): Radius of curvature of the second crown
            surface/ first flint surface [m].
        radius_flint (float): Radius of curvature of the second flint surface
            [m].
        shift_x (float): X-offset of lens [m].
        shift_y (float): Y-offset of lens [m].
        wavenumber (float): Phase of the doublet lens [rad].

    Returns:
        ArrayLike: Phase of the doublet lens.
    """
    crown_surface = spherical_surface(x, y, radius_crown, shift_x, shift_y)

    crown_flint_surface = spherical_surface(x, y, radius_crown_flint, shift_x, shift_y)

    flint_surface = spherical_surface(x, y, radius_flint, shift_x, shift_y)

    delta1 = -crown_surface + crown_flint_surface
    delta2 = flint_surface - crown_flint_surface

    return wavenumber * (
        (refractive_index_flint - 1) * delta1 + (refractive_index_crown - 1) * delta2
    )


def linear_phase(
    x: ArrayLike,
    y: ArrayLike,
    tilt_x: float,
    tilt_y: float,
    tilt_units: TiltUnits = "metres",
    wavenumber: float | None = None,
    focal_length: float | None = None,
) -> ArrayLike:
    """Calculates a linear phase ramp on a 2D grid.

    The phase is ``wavenumber * (angle_x * x + angle_y * y)``, where the beam
    deflection angles are obtained from the tilt via :func:`tilt_to_angle`.

    Args:
        x (ArrayLike): X coordinates.
        y (ArrayLike): Y coordinates.
        tilt_x (float): Tilt in the x direction.
        tilt_y (float): Tilt in the y direction.
        tilt_units (TiltUnits, optional): Units for tilt. Defaults to "metres".
        wavenumber (float | None, optional): Wavenumber ``2 * pi / wavelength``.
            Always required; also sets the wavelength for "lines_per_mm". Defaults to 
            None.
        focal_length (float | None, optional): Focal length required if `tilt_units` is 
            "metres". Defaults to None.
    Returns:
        ArrayLike: Linear phase.
    """
    if wavenumber is None:
        raise ValueError("wavenumber must be provided.")
    angle_x = tilt_to_angle(tilt_x, tilt_units, wavenumber, focal_length)
    angle_y = tilt_to_angle(tilt_y, tilt_units, wavenumber, focal_length)
    return wavenumber * (angle_x * x + angle_y * y)


def quadratic_phase(
    x: ArrayLike,
    y: ArrayLike,
    curvature: float,
    aspect_ratio: float = 0.5,
    curvature_units: CurvatureUnits = "radians_per_metre_squared",
) -> ArrayLike:
    """Calculates a quadratic phase with some `curvature` and an
    `aspect_ratio` on a 2D grid.

    Args:
        x (ArrayLike): X coordinates.
        y (ArrayLike): Y coordinates.
        curvature (float): Curvature for quadratic phase.
        aspect_ratio (float, optional): Aspect ratio for quadratic phase.
            Defaults to 0.5.
        curvature_units (CurvatureUnits, optional): Units for curvature.
            Defaults to "radians_per_metre_squared".
    Returns:
        ArrayLike: Quadratic phase.
    """
    if curvature_units == "radians_per_pixel_squared":
        x = x / (x.max() - x.min()) * x.shape[1]
        y = y / (y.max() - y.min()) * y.shape[0]
    return 4 * curvature * (aspect_ratio * y**2 + (1 - aspect_ratio) * x**2)


def analytic_phase_guess(
    x: ArrayLike,
    y: ArrayLike,
    tilt_x: float,
    tilt_y: float,
    curvature: float,
    aspect_ratio: float = 0.5,
    tilt_units: TiltUnits = "metres",
    curvature_units: CurvatureUnits = "radians_per_metre_squared",
    wavenumber: float | None = None,
    focal_length: float | None = None,
) -> ArrayLike:
    """Calculates a combined linear and quadratic phase term.

    Args:
        x (ArrayLike): X coordinates.
        y (ArrayLike): Y coordinates.
        tilt_x (float): Tilt in the x direction.
        tilt_y (float): Tilt in the y direction.
        curvature (float): Curvature for quadratic phase.
        aspect_ratio (float, optional): Aspect ratio for quadratic phase.
            Defaults to 0.5.
        tilt_units (TiltUnits, optional): Units for tilt. Defaults to "metres".
        curvature_units (CurvatureUnits, optional): Units for curvature.
            Defaults to "radians_per_metre_squared".
        wavenumber (float | None, optional): Wavenumber for linear phase.
            Required if tilt_units is "degrees", "radians", or "metres".
            Defaults to None.
        focal_length (float | None, optional): Focal length for linear phase.
            Required if tilt_units is "metres". Defaults to None.

    Returns:
        ArrayLike: Combined linear and quadratic phase.
    """
    linear_phase_term = linear_phase(
        x, y, tilt_x, tilt_y, tilt_units, wavenumber, focal_length
    )
    quadratic_phase_term = quadratic_phase(
        x, y, curvature, aspect_ratio, curvature_units
    )
    return linear_phase_term + quadratic_phase_term


def binary_phase_grating(
    shape: tuple[int, int],
    axis: int = 1,
    high: float = np.pi,
    low: float = 0.0,
) -> NDArray[np.float_]:
    """A period-2-pixel binary phase grating (the SLM Nyquist grating).

    Every other line along ``axis`` is set to ``high`` and the rest to ``low``.
    Applied to the unmodulated SLM area it deflects the light into the plus and minus
    first Nyquist orders, away from the bright zeroth order, instead of leaving a flat
    phase.

    Args:
        shape (tuple[int, int]): Output shape (height, width).
        axis (int, optional): Axis whose lines alternate. ``1`` (the default)
            alternates columns for a vertical grating, ``0`` alternates rows.
        high (float, optional): Phase of every other line. Defaults to ``np.pi``.
        low (float, optional): Phase of the remaining lines. Defaults to ``0.0``.

    Returns:
        NDArray[np.float_]: The grating phase, shape ``shape``.
    """
    grating = np.full(shape, low, dtype=float)
    if axis == 1:
        grating[:, 1::2] = high
    elif axis == 0:
        grating[1::2, :] = high
    else:
        raise ValueError("axis must be 0 or 1.")
    return grating


def band_limited_random_phase(
    band_mask: torch.Tensor,
    generator: torch.Generator | None = None,
    clip_sigma: float | None = None,
) -> torch.Tensor:
    """Smooth random phase with diffracted light landing inside ``band_mask`` in the 
    Fourier plane.

    Args:
        band_mask: The band-limiting mask.
        generator: Random number generator.
        clip_sigma: Clip the field at this many standard deviations before rescaling,
            or None (default).

    Returns:
        torch.Tensor: The phase in ``[0, 2 * pi]``, shaped like ``band_mask``.
    """
    white = torch.randn(band_mask.shape, generator=generator, device=band_mask.device)

    field = ifft_2d(fft_2d(white + 0j) * band_mask).real
    field = field / field.std()

    if clip_sigma is not None:
        field = field.clamp(-clip_sigma, clip_sigma)

    lowest = field.min()
    return (field - lowest) / (field.max() - lowest) * 2 * torch.pi
