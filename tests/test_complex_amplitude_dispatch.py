"""Dispatch paths of :class:`ComplexAmplitude`.

The layout is ``(*batch, component, wavelength, H, W)``, and an operation that
removes one of the two reserved axes leaves it. What these pin is where the boundary
sits: which operations still return a field, which hand back plain values, and which
argument lists the special-cased branches have to accept.
"""

from __future__ import annotations

import pytest
import torch

from hologradpy.optics.complex_amplitude import ComplexAmplitude

PIXEL_SIZE = (10e-6, 10e-6)


def _field(shape: tuple[int, ...], number_of_wavelengths: int) -> ComplexAmplitude:
    generator = torch.Generator().manual_seed(1)
    data = torch.randn(*shape, generator=generator, dtype=torch.complex64)
    wavelength = (
        torch.tensor(800e-9)
        if number_of_wavelengths == 1
        else torch.linspace(800e-9, 900e-9, number_of_wavelengths)
    )
    return ComplexAmplitude(data, wavelength, PIXEL_SIZE)


def test_a_slice_with_the_schema_defaults_left_out_is_accepted() -> None:
    """``torch.gradient`` calls ``aten.slice.Tensor(self, dim, start)`` with three
    arguments, so ``end`` and ``step`` have to take their schema defaults.
    """
    field = _field((1, 2, 8, 8), 2)

    sliced = torch.ops.aten.slice.Tensor(field, 2, 3)

    assert isinstance(sliced, ComplexAmplitude)
    assert sliced.resolution == (5, 8)
    torch.testing.assert_close(sliced.as_tensor(), field.as_tensor()[:, :, 3:])


def test_slicing_the_wavelength_axis_carries_its_metadata() -> None:
    field = _field((1, 2, 8, 8), 2)

    one = field[:, 1:]

    assert one.number_of_wavelengths == 1
    torch.testing.assert_close(one.wavelength, field.wavelength[1:])
    torch.testing.assert_close(one.pixel_size, field.pixel_size[1:])


def test_slicing_the_component_axis_keeps_a_field_while_the_count_is_legal() -> None:
    field = _field((3, 2, 8, 8), 2)

    one = field[0:1]
    two = field[0:2]

    assert isinstance(one, ComplexAmplitude)
    assert one.number_of_components == 1
    # Two components describe nothing, so the values come back plain.
    assert not isinstance(two, ComplexAmplitude)


def test_selecting_a_reserved_axis_gives_plain_values() -> None:
    """Selecting drops the axis, which leaves the layout. ``narrow`` keeps the rank."""
    field = _field((3, 2, 8, 8), 2)

    assert not isinstance(field[0], ComplexAmplitude)
    assert not isinstance(field[:, 1], ComplexAmplitude)
    torch.testing.assert_close(field[:, 1], field.as_tensor()[:, 1])


def test_selecting_a_batch_element_keeps_the_field() -> None:
    field = _field((3, 1, 2, 8, 8), 2)

    element = field[1]

    assert isinstance(element, ComplexAmplitude)
    assert element.number_of_wavelengths == 2
    assert element.number_of_components == 1
    torch.testing.assert_close(element.as_tensor(), field.as_tensor()[1])


def test_reducing_a_reserved_axis_gives_plain_values() -> None:
    """A sum over the components or the wavelengths is not a field, whatever the
    resulting shape looks like. The batch length here is three, which the shape rule
    alone mistakes for a field vector.
    """
    field = _field((3, 1, 2, 8, 8), 2)

    over_components = field.sum(dim=-4)
    over_wavelengths = field.sum(dim=-3)

    assert not isinstance(over_components, ComplexAmplitude)
    assert not isinstance(over_wavelengths, ComplexAmplitude)
    torch.testing.assert_close(over_components, field.as_tensor().sum(dim=-4))


def test_reducing_a_batch_axis_keeps_the_field() -> None:
    field = _field((3, 1, 2, 8, 8), 2)

    summed = field.sum(dim=0)

    assert isinstance(summed, ComplexAmplitude)
    assert summed.number_of_components == 1
    assert summed.number_of_wavelengths == 2


def test_with_geometry_keeps_the_pose() -> None:
    field = _field((1, 1, 8, 8), 1).with_geometry(
        origin=torch.tensor([0.0, 0.0, 1e-3]), rotation=torch.eye(3)
    )

    moved = field.with_geometry(pixel_size=(20e-6, 20e-6))

    torch.testing.assert_close(moved.geometry.origin, torch.tensor([0.0, 0.0, 1e-3]))
    torch.testing.assert_close(moved.geometry.rotation, torch.eye(3))
    assert float(moved.pixel_size[0, 0]) == pytest.approx(20e-6)


def test_a_scalar_field_modulates_a_field_vector() -> None:
    """A lens, a mask and an SLM phase are scalar, so they must broadcast over the
    components of a field vector.
    """
    vector = _field((3, 1, 8, 8), 1)
    scalar = _field((1, 1, 8, 8), 1)

    product = vector * scalar

    assert isinstance(product, ComplexAmplitude)
    assert product.number_of_components == 3
    torch.testing.assert_close(
        product.as_tensor(), vector.as_tensor() * scalar.as_tensor()
    )
