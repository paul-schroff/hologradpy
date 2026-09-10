from __future__ import annotations

import torch
from jaxtyping import Complex
from torch import Tensor

from ..fourier_transforms.fft import fft_2d, ifft_2d
from ..grids import get_frequency_grid
from .complex_amplitude import ComplexAmplitude

# Plane waves travelling within this fraction of a wavenumber of the plane carry no
# power through it, and their longitudinal component is unbounded, so they are dropped.
IN_PLANE_FLOOR = 1e-6


def longitudinal_component(
    x: ComplexAmplitude, y: ComplexAmplitude
) -> ComplexAmplitude:
    """Calculates ``E_z`` from ``E_x`` and ``E_y`` of a transverse field using Gauss's
    law.

    Args:
        x: The ``E_x`` component, a scalar field on a transverse plane.
        y: The ``E_y`` component, on the same plane.

    Returns:
        ComplexAmplitude: The ``E_z`` component, a scalar field on the same plane.

    Raises:
        ValueError: Either component carries more than one, or the two are sampled on
            different planes.
    """
    for axis, part in (("x", x), ("y", y)):
        if not part.is_scalar:
            raise ValueError(
                f"The {axis} component carries {part.number_of_components} components, "
                "and each transverse component of a field vector is a scalar field."
            )
    if not x.geometry.is_transverse:
        raise ValueError(
            "The angular spectrum decomposes a field over a transverse plane, and this "
            "one is sampled on a plane tilted out of x-y."
        )

    spectrum_x = fft_2d(x.as_tensor())
    spectrum_y = fft_2d((y * torch.ones_like(x)).as_tensor())

    frequency_x, frequency_y = get_frequency_grid(
        x.resolution, x.pixel_size[0], x.device
    )
    wavenumber = x.wavenumber.reshape(-1, 1, 1).to(x.device)
    axial = torch.sqrt(wavenumber**2 - frequency_x**2 - frequency_y**2 + 0j)

    spectrum_z = -(frequency_x * spectrum_x + frequency_y * spectrum_y)

    propagating = axial.abs() > IN_PLANE_FLOOR * wavenumber
    spectrum_z = torch.where(
        propagating, spectrum_z / torch.where(propagating, axial, 1.0), 0.0
    )

    return ComplexAmplitude.from_tensor(
        ifft_2d(spectrum_z).to(x.dtype_c), x.wavelength, x.pixel_size
    )


def gauss_law_residual(
    field: ComplexAmplitude,
) -> Complex[Tensor, "*batch n_wavelengths H W"]:
    """``k . E`` for every plane wave in the field's spectrum, scaled by the wavenumber.

    Zero everywhere for a field vector free space allows, so this measures how far one
    departs from it.

    Args:
        field: A field vector on a transverse plane.

    Returns:
        Tensor: The residual, one value per plane wave, dimensionless.

    Raises:
        ValueError: The field is not a field vector.
    """
    if not field.is_vector:
        raise ValueError(
            f"Gauss's law relates all three components of a field, and this carries "
            f"{field.number_of_components}."
        )

    spectrum = fft_2d(field.as_tensor())
    frequency_x, frequency_y = get_frequency_grid(
        field.resolution, field.pixel_size[0], field.device
    )
    wavenumber = field.wavenumber.reshape(-1, 1, 1).to(field.device)
    axial = torch.sqrt(wavenumber**2 - frequency_x**2 - frequency_y**2 + 0j)

    return (
        frequency_x * spectrum.select(-4, 0)
        + frequency_y * spectrum.select(-4, 1)
        + axial * spectrum.select(-4, 2)
    ) / wavenumber
