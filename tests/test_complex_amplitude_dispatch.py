"""Dispatch paths of :class:`ComplexAmplitude` that torch reaches with argument lists
the special-cased branches have to accept.
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
    field = _field((2, 8, 8), 2)
    sliced = torch.ops.aten.slice.Tensor(field, 2, 3)
    assert isinstance(sliced, ComplexAmplitude)
    assert sliced.resolution == (8, 5)
    torch.testing.assert_close(sliced.as_tensor(), field.as_tensor()[:, :, 3:])

    wavelengths = torch.ops.aten.slice.Tensor(field, 0, 1)
    assert wavelengths.number_of_wavelengths == 1
    torch.testing.assert_close(wavelengths.wavelength, field.wavelength[1:])


def test_selecting_a_wavelength_of_a_batched_field_gives_a_plain_tensor() -> None:
    field = _field((3, 2, 8, 8), 2)
    selected = field[:, 1]
    assert not isinstance(selected, ComplexAmplitude)
    torch.testing.assert_close(selected, field.as_tensor()[:, 1])


def test_selecting_a_batch_element_keeps_the_field() -> None:
    field = _field((3, 2, 8, 8), 2)
    element = field[1]
    assert isinstance(element, ComplexAmplitude)
    assert element.number_of_wavelengths == 2
    torch.testing.assert_close(element.as_tensor(), field.as_tensor()[1])


def test_with_geometry_keeps_the_pose() -> None:
    field = _field((8, 8), 1).with_geometry(
        origin=torch.tensor([0.0, 0.0, 1e-3]), rotation=torch.eye(3)
    )
    moved = field.with_geometry(pixel_size=(20e-6, 20e-6))
    torch.testing.assert_close(moved.geometry.origin, torch.tensor([0.0, 0.0, 1e-3]))
    torch.testing.assert_close(moved.geometry.rotation, torch.eye(3))
    assert float(moved.pixel_size[0, 0]) == pytest.approx(20e-6)
