"""Tests for where a field is sampled, not just how finely.

``FieldGeometry`` grew an optional pose, an origin and a rotation, so the same
object that has always described a transverse plane can also describe one standing
on its edge, which an x-z cross section is. Two things are worth pinning. A
geometry without a pose still means exactly what it always did, and a section
reproduces the distances it was asked for.
"""

from __future__ import annotations

import pytest
import torch

from hologradpy.grids import get_spatial_grid
from hologradpy.optics.complex_amplitude import ComplexAmplitude, FieldGeometry

WAVELENGTH = 800e-9
PITCH = 10e-6
RESOLUTION = (5, 7)
DISTANCES = torch.linspace(-1e-3, 1e-3, 11, dtype=torch.float64)


def _plane(resolution: tuple[int, int] = RESOLUTION) -> FieldGeometry:
    return FieldGeometry(
        wavelength=torch.tensor([WAVELENGTH]),
        pixel_size=torch.tensor([[PITCH, PITCH]]),
        resolution=resolution,
    )


def test_a_geometry_without_a_pose_is_the_plane_through_zero() -> None:
    """The origin convention lives in grids.py, and positions() must not restate it."""
    geometry = _plane()

    positions = geometry.positions()
    grid_x, grid_y = get_spatial_grid(RESOLUTION, torch.tensor([PITCH, PITCH]))

    assert positions.shape == (*RESOLUTION, 3)
    assert torch.equal(positions[..., 0], grid_x)
    assert torch.equal(positions[..., 1], grid_y)
    assert torch.count_nonzero(positions[..., 2]) == 0
    assert geometry.is_transverse


@pytest.mark.parametrize("axis", ["x", "y"])
def test_a_section_reproduces_the_distances_it_was_asked_for(axis: str) -> None:
    """The whole point of the pose: row ``i`` of a section sits at ``distances[i]``."""
    offset = 3e-6
    section = FieldGeometry.cross_section(
        WAVELENGTH, DISTANCES, transverse_pitch=2e-6, width=9, axis=axis, offset=offset
    )

    positions = section.positions()

    assert section.resolution == (DISTANCES.numel(), 9)
    assert torch.allclose(positions[:, 0, 2], DISTANCES)
    # Constant down every column, so the section really is a plane of constant offset.
    other = 1 if axis == "x" else 0
    flat = torch.full_like(positions[..., 0], offset)
    assert torch.allclose(positions[..., other], flat)


def test_a_section_spans_the_transverse_axis_it_names() -> None:
    """An x section moves in x and an y section in y, which a shape check would miss."""
    for axis, moving in (("x", 0), ("y", 1)):
        section = FieldGeometry.cross_section(
            WAVELENGTH, DISTANCES, transverse_pitch=2e-6, width=9, axis=axis
        )
        row = section.positions()[0]
        wanted = torch.arange(-4, 5, dtype=row.dtype) * 2e-6
        assert torch.allclose(row[:, moving], wanted)


def test_a_section_is_not_a_transverse_plane() -> None:
    section = FieldGeometry.cross_section(WAVELENGTH, DISTANCES, 2e-6, 9)

    assert not section.is_transverse


def test_uneven_distances_are_refused() -> None:
    """Uneven z spacing is not an evenly spaced grid, and pretending otherwise
    would misplace every row but the middle one.
    """
    uneven = torch.tensor([0.0, 1e-3, 3e-3])

    with pytest.raises(ValueError, match="evenly spaced"):
        FieldGeometry.cross_section(WAVELENGTH, uneven, 2e-6, 4)


def test_an_unknown_axis_is_refused() -> None:
    with pytest.raises(ValueError, match="axis must be"):
        FieldGeometry.cross_section(WAVELENGTH, DISTANCES, 2e-6, 4, axis="z")


def _section_field() -> ComplexAmplitude:
    section = FieldGeometry.cross_section(WAVELENGTH, DISTANCES, 2e-6, 9)
    field = ComplexAmplitude.from_geometry(section)
    return field.with_geometry(origin=section.origin, rotation=section.rotation)


def test_the_pose_survives_arithmetic() -> None:
    """Multiplying a section by a mask must not quietly turn it back into a plane."""
    field = _section_field()

    scaled = field * 2.0

    assert not scaled.geometry.is_transverse
    assert torch.equal(scaled.geometry.rotation, field.geometry.rotation)
    assert torch.equal(scaled.geometry.origin, field.geometry.origin)


def test_fields_on_different_planes_cannot_be_combined() -> None:
    section = _section_field()
    plane = ComplexAmplitude(
        torch.ones(section.resolution, dtype=torch.complex64),
        section.wavelength,
        section.pixel_size,
    )

    with pytest.raises(ValueError, match="same plane"):
        section * plane


def test_power_refuses_on_a_section_and_still_answers_on_a_plane() -> None:
    """Power is the flux through a plane, and a section is not one it crosses."""
    plane = ComplexAmplitude.from_geometry(_plane(), power=1.0)
    assert float(plane.power()) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="flux through a transverse plane"):
        _section_field().power()
