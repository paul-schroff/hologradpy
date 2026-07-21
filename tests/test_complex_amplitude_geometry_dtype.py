"""Geometry metadata (wavelength / pixel_size) dtype should match the field.

Float / tuple inputs used to be forced to float32 regardless of the field's
dtype; they now derive the real dtype from the field, so a complex128 field
carries float64 geometry (while complex64 stays float32). Tensor inputs keep
their own dtype (unchanged behaviour).
"""

import torch

from hologradpy.optics.complex_amplitude import ComplexAmplitude


def _field(dtype: torch.dtype) -> ComplexAmplitude:
    return ComplexAmplitude(torch.ones(4, 4, dtype=dtype), 0.5e-6, (1e-5, 1e-5))


def test_geometry_dtype_matches_complex64_field():
    field = _field(torch.complex64)
    assert field.wavelength.dtype == torch.float32
    assert field.pixel_size.dtype == torch.float32


def test_geometry_dtype_matches_complex128_field():
    field = _field(torch.complex128)
    assert field.wavelength.dtype == torch.float64
    assert field.pixel_size.dtype == torch.float64


def test_with_geometry_float_inputs_follow_field_dtype():
    field = _field(torch.complex128)
    updated = field.with_geometry(wavelength=0.6e-6, pixel_size=(2e-5, 2e-5))
    assert updated.wavelength.dtype == torch.float64
    assert updated.pixel_size.dtype == torch.float64


def test_tensor_wavelength_dtype_is_preserved():
    # A tensor input keeps its own dtype, independent of the field dtype.
    wavelength = torch.tensor([0.5e-6], dtype=torch.float64)
    field = ComplexAmplitude(
        torch.ones(4, 4, dtype=torch.complex64), wavelength, (1e-5, 1e-5)
    )
    assert field.wavelength.dtype == torch.float64
