"""The component axis of :class:`ComplexAmplitude`.

A field is laid out as ``(*batch, component, wavelength, H, W)``, where the component
axis holds one value for a scalar field or three for the Cartesian components of a
field vector. Lower ranks are promoted at construction, so every field has both
reserved axes and no code downstream has to ask which form it was handed.

What these pin is the promotion, since it decides how an existing array is read, and the
errors it raises, since an array that cannot be read unambiguously must say so.
"""

from __future__ import annotations

import pytest
import torch

from hologradpy.optics.complex_amplitude import (
    SCALAR,
    VECTOR,
    ComplexAmplitude,
    FieldGeometry,
)

PIXEL_SIZE = (10e-6, 10e-6)
RESOLUTION = (8, 6)


def _field(shape: tuple[int, ...], number_of_wavelengths: int) -> ComplexAmplitude:
    wavelength = (
        torch.tensor(800e-9)
        if number_of_wavelengths == 1
        else torch.linspace(800e-9, 900e-9, number_of_wavelengths)
    )
    return ComplexAmplitude(
        torch.ones(shape, dtype=torch.complex64), wavelength, PIXEL_SIZE
    )


# --- promotion ----------------------------------------------------------------------


def test_a_plane_is_one_component_of_one_wavelength() -> None:
    field = _field(RESOLUTION, 1)

    assert field.shape == (1, 1, *RESOLUTION)
    assert field.number_of_components == SCALAR
    assert field.batch_shape == ()


def test_a_stack_of_planes_is_one_component_of_each_wavelength() -> None:
    field = _field((2, *RESOLUTION), 2)

    assert field.shape == (1, 2, *RESOLUTION)
    assert field.number_of_components == SCALAR
    assert field.number_of_wavelengths == 2


def test_a_canonical_field_is_taken_as_it_is() -> None:
    for count in (SCALAR, VECTOR):
        field = _field((count, 2, *RESOLUTION), 2)
        assert field.number_of_components == count
        assert field.batch_shape == ()


def test_a_batch_carries_its_own_component_axis() -> None:
    field = _field((4, VECTOR, 2, *RESOLUTION), 2)

    assert field.batch_shape == (4,)
    assert field.number_of_components == VECTOR


# --- what promotion refuses ---------------------------------------------------------


def test_a_plane_cannot_carry_several_wavelengths() -> None:
    with pytest.raises(ValueError, match="carries one wavelength"):
        _field(RESOLUTION, 2)


def test_a_stack_that_is_not_the_wavelengths_is_refused() -> None:
    """``(5, H, W)`` against two wavelengths is either a batch or a mistake, and reading
    it as either is a guess.
    """
    with pytest.raises(ValueError, match="does not start with"):
        _field((5, *RESOLUTION), 2)


def test_an_illegal_component_count_names_the_layout() -> None:
    """The message has to say what to do, since a batch written for the old layout lands
    here, and the fix is to add an axis.
    """
    with pytest.raises(ValueError, match="component axis"):
        _field((2, 2, *RESOLUTION), 2)


def test_a_wavelength_axis_that_disagrees_is_refused() -> None:
    with pytest.raises(ValueError, match="wavelength axis"):
        _field((1, 5, *RESOLUTION), 2)


def test_a_geometry_refuses_an_illegal_component_count() -> None:
    with pytest.raises(ValueError, match="number_of_components must be"):
        FieldGeometry(
            wavelength=torch.tensor(800e-9),
            pixel_size=torch.tensor([1e-5, 1e-5]),
            resolution=RESOLUTION,
            number_of_components=2,
        )


# --- the geometry travels with the data ---------------------------------------------


def test_the_geometry_reports_what_the_data_carries() -> None:
    """``number_of_components`` is derived from the data wherever there is data,
    exactly as
    ``resolution`` is, so the two cannot drift apart.
    """
    field = _field((VECTOR, 1, *RESOLUTION), 1)

    assert field.geometry.number_of_components == VECTOR
    assert field.geometry.resolution == RESOLUTION


def test_slicing_the_components_updates_the_geometry() -> None:
    field = _field((VECTOR, 1, *RESOLUTION), 1)

    one = field[0:1]

    assert one.number_of_components == SCALAR
    assert one.geometry.number_of_components == SCALAR


# --- the autograd crossing ----------------------------------------------------------


def test_a_plane_leaf_gets_a_plane_gradient() -> None:
    """``from_tensor`` promotes on the way in, so the gradient has to come back in the
    shape the leaf was written in.
    """
    phase = torch.zeros(RESOLUTION, requires_grad=True)

    field = ComplexAmplitude.from_tensor(
        torch.exp(1j * phase), torch.tensor(800e-9), PIXEL_SIZE
    )
    field.intensity.sum().backward()

    assert phase.grad is not None
    assert phase.grad.shape == RESOLUTION
