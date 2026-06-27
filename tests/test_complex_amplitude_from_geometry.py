"""``ComplexAmplitude.from_geometry`` constructor.

Builds a field that matches a :class:`FieldGeometry`: either a uniform default
field (with a wavelength axis only when multi-wavelength) or a wrapper around
explicit data, always carrying the geometry's wavelength and pixel size.
"""

from __future__ import annotations

import pytest
import torch

from hologradpy.propagation.complex_amplitude import (
    ComplexAmplitude,
    FieldGeometry,
)


pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _single_geometry() -> FieldGeometry:
    return FieldGeometry(
        wavelength=torch.tensor(670e-9),
        pixel_size=torch.tensor([12.5e-6, 12.5e-6]),
        resolution=(8, 10),
    )


def _multi_geometry() -> FieldGeometry:
    return FieldGeometry(
        wavelength=torch.tensor([670e-9, 800e-9]),
        pixel_size=torch.tensor([12.5e-6, 12.5e-6]),
        resolution=(8, 10),
    )


def test_default_single_wavelength_is_uniform() -> None:
    geometry = _single_geometry()
    field = ComplexAmplitude.from_geometry(geometry)

    assert type(field) is ComplexAmplitude
    assert field.shape == geometry.resolution
    assert field.dtype == torch.complex64
    expected = torch.ones(geometry.resolution, dtype=torch.complex64)
    assert torch.equal(field._data, expected)


def test_default_multi_wavelength_adds_wavelength_axis() -> None:
    geometry = _multi_geometry()
    field = ComplexAmplitude.from_geometry(geometry)

    assert field.shape == (geometry.number_of_wavelengths, *geometry.resolution)


def test_explicit_data_is_wrapped_with_geometry() -> None:
    geometry = _single_geometry()
    data = torch.randn(8, 10, dtype=torch.complex64)

    field = ComplexAmplitude.from_geometry(geometry, data=data)

    assert torch.equal(field._data, data)
    torch.testing.assert_close(field.wavelength, geometry.wavelength.reshape(-1))
    assert field.resolution == geometry.resolution


def test_grid_matches_geometry_grid() -> None:
    geometry = _single_geometry()
    field = ComplexAmplitude.from_geometry(geometry)

    field_x, field_y = field.get_spatial_grid()
    geometry_x, geometry_y = geometry.get_spatial_grid()

    torch.testing.assert_close(field_x, geometry_x)
    torch.testing.assert_close(field_y, geometry_y)
