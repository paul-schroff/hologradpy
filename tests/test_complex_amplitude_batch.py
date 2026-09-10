"""Unit tests for the ND batch helpers on :class:`ComplexAmplitude`.

``flatten_batch`` / ``unflatten_batch`` are the primitives that fixed-rank
``OpticsModule`` implementations use to support arbitrary batch ranks, so they
are tested directly here in isolation from any module. The component axis folds into the
leading axis alongside the batch, so a module that treats the components independently
stays unaware of them.
"""

from __future__ import annotations

import math

import pytest
import torch

from hologradpy.optics.complex_amplitude import ComplexAmplitude

from .registry import make_field


# (shape as constructed, n_wavelengths, expected batch shape, expected component count)
BATCH_SHAPE_CASES = [
    ((16, 16), 1, (), 1),
    ((2, 16, 16), 2, (), 1),
    ((1, 2, 16, 16), 2, (), 1),
    ((3, 2, 16, 16), 2, (), 3),
    ((3, 1, 2, 16, 16), 2, (3,), 1),
    ((2, 3, 2, 16, 16), 2, (2,), 3),
    ((2, 3, 1, 2, 16, 16), 2, (2, 3), 1),
]


@pytest.mark.parametrize("shape, n_wl, expected, n_components", BATCH_SHAPE_CASES)
def test_batch_shape(shape, n_wl, expected, n_components) -> None:
    field = make_field(shape, n_wl)
    assert field.batch_shape == expected
    assert field.number_of_components == n_components


@pytest.mark.parametrize("shape, n_wl, batch_shape, n_components", BATCH_SHAPE_CASES)
def test_flatten_batch_canonical_shape(shape, n_wl, batch_shape, n_components) -> None:
    field = make_field(shape, n_wl)

    flat, spec = field.flatten_batch()

    expected_n = max(1, math.prod(batch_shape)) * n_components
    assert flat.shape == (expected_n, n_wl, *field.resolution)
    assert spec.batch_shape == batch_shape
    assert spec.number_of_components == n_components
    assert spec.original_ndim == field.ndim


@pytest.mark.parametrize("shape, n_wl, batch_shape, n_components", BATCH_SHAPE_CASES)
def test_flatten_batch_can_keep_the_components(
    shape, n_wl, batch_shape, n_components
) -> None:
    """A module that mixes components asks for them, and gets one leading axis."""
    field = make_field(shape, n_wl)

    flat, spec = field.flatten_batch(fold_components=False)

    assert flat.shape == (
        max(1, math.prod(batch_shape)),
        n_components,
        n_wl,
        *field.resolution,
    )
    assert spec.number_of_components == n_components


@pytest.mark.parametrize("shape, n_wl, _batch_shape, _n_components", BATCH_SHAPE_CASES)
def test_flatten_unflatten_roundtrip(shape, n_wl, _batch_shape, _n_components) -> None:
    field = make_field(shape, n_wl)

    flat, spec = field.flatten_batch()
    restored = ComplexAmplitude.unflatten_batch(
        flat, spec, field.wavelength, field.pixel_size
    )

    assert restored.shape == field.shape
    assert restored.batch_shape == field.batch_shape
    assert restored.number_of_components == field.number_of_components
    torch.testing.assert_close(restored._data, field._data)


@pytest.mark.parametrize("shape, n_wl, _batch_shape, _n_components", BATCH_SHAPE_CASES)
def test_unflatten_from_the_unfolded_layout(
    shape, n_wl, _batch_shape, _n_components
) -> None:
    """Both flattened layouts hold the same values in the same order, so either
    restores.
    """
    field = make_field(shape, n_wl)

    flat, spec = field.flatten_batch(fold_components=False)
    restored = ComplexAmplitude.unflatten_batch(
        flat, spec, field.wavelength, field.pixel_size
    )

    torch.testing.assert_close(restored._data, field._data)


@pytest.mark.parametrize("shape, n_wl, batch_shape, n_components", BATCH_SHAPE_CASES)
def test_unflatten_with_changed_resolution(
    shape, n_wl, batch_shape, n_components
) -> None:
    """A resampling propagator changes spatial resolution while preserving
    batch, component and wavelength axes. Unflatten must honour the new spatial size.
    """
    field = make_field(shape, n_wl)
    flat, spec = field.flatten_batch()

    new_resolution = (8, 8)
    resampled = torch.zeros(flat.shape[0], n_wl, *new_resolution, dtype=flat.dtype)

    restored = ComplexAmplitude.unflatten_batch(
        resampled, spec, field.wavelength, field.pixel_size
    )

    assert restored.batch_shape == batch_shape
    assert restored.number_of_components == n_components
    assert restored.resolution == new_resolution
    assert restored.ndim == field.ndim


def test_unflatten_can_change_the_component_count() -> None:
    """A module turning a scalar pupil into a field vector says so on the way back."""
    field = make_field((3, 1, 1, 16, 16), 1)
    flat, spec = field.flatten_batch()

    tripled = flat.repeat_interleave(3, dim=0)
    restored = ComplexAmplitude.unflatten_batch(
        tripled, spec, field.wavelength, field.pixel_size, number_of_components=3
    )

    assert restored.number_of_components == 3
    assert restored.batch_shape == (3,)


def test_flatten_batch_is_view_of_underlying_data() -> None:
    """flatten_batch reshapes (no copy) so it stays cheap on the hot path."""
    field = make_field((2, 3, 2, 16, 16), 2)
    flat, _ = field.flatten_batch()
    assert flat.data_ptr() == field._data.data_ptr()
