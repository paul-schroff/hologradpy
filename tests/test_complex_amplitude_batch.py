"""Unit tests for the ND batch helpers on :class:`ComplexAmplitude`.

``flatten_batch`` / ``unflatten_batch`` are the primitives that fixed-rank
``OpticsModule`` implementations use to support arbitrary batch ranks, so they
are tested directly here in isolation from any module.
"""

from __future__ import annotations

import math

import pytest
import torch

from hologradpy.propagation.complex_amplitude import ComplexAmplitude

from .registry import make_field


# (shape, n_wavelengths, expected_batch_shape)
BATCH_SHAPE_CASES = [
    ((16, 16), 1, ()),
    ((2, 16, 16), 2, ()),
    ((3, 2, 16, 16), 2, (3,)),
    ((2, 3, 2, 16, 16), 2, (2, 3)),
]


@pytest.mark.parametrize("shape, n_wl, expected", BATCH_SHAPE_CASES)
def test_batch_shape(shape, n_wl, expected) -> None:
    field = make_field(shape, n_wl)
    assert field.batch_shape == expected


@pytest.mark.parametrize("shape, n_wl, batch_shape", BATCH_SHAPE_CASES)
def test_flatten_batch_canonical_shape(shape, n_wl, batch_shape) -> None:
    field = make_field(shape, n_wl)

    flat, spec = field.flatten_batch()

    expected_n = max(1, math.prod(batch_shape))
    assert flat.shape == (expected_n, n_wl, *field.resolution)
    assert spec.leading_shape == batch_shape
    assert spec.original_ndim == field.ndim


@pytest.mark.parametrize("shape, n_wl, _batch_shape", BATCH_SHAPE_CASES)
def test_flatten_unflatten_roundtrip(shape, n_wl, _batch_shape) -> None:
    field = make_field(shape, n_wl)

    flat, spec = field.flatten_batch()
    restored = ComplexAmplitude.unflatten_batch(
        flat, spec, field.wavelength, field.pixel_size
    )

    assert restored.shape == field.shape
    assert restored.batch_shape == field.batch_shape
    torch.testing.assert_close(restored._data, field._data)


@pytest.mark.parametrize("shape, n_wl, batch_shape", BATCH_SHAPE_CASES)
def test_unflatten_with_changed_resolution(shape, n_wl, batch_shape) -> None:
    """A resampling propagator changes spatial resolution while preserving
    batch and wavelength axes; unflatten must honour the new spatial size."""
    field = make_field(shape, n_wl)
    flat, spec = field.flatten_batch()

    new_resolution = (8, 8)
    resampled = torch.zeros(flat.shape[0], n_wl, *new_resolution, dtype=flat.dtype)

    restored = ComplexAmplitude.unflatten_batch(
        resampled, spec, field.wavelength, field.pixel_size
    )

    assert restored.batch_shape == batch_shape
    assert restored.resolution == new_resolution
    assert restored.ndim == field.ndim


def test_flatten_batch_is_view_of_underlying_data() -> None:
    """flatten_batch reshapes (no copy) so it stays cheap on the hot path."""
    field = make_field((3, 2, 16, 16), 2)
    flat, _ = field.flatten_batch()
    assert flat.data_ptr() == field._data.data_ptr()
