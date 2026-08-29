"""Detecting optical vortices, and telling them apart from ordinary amplitude zeros.

The distinction is the whole point and it is easy to lose. ``VortexDetector`` labels
every place the real and imaginary parts of the field both cross zero, and reports how
many such places it found. Most of them are not vortices: an interference null is a
zero the phase does not wind around, and ``find_vortex_charge`` scores it zero. Reading
``number_of_vortices`` as a vortex count therefore overcounts, sometimes by an order of
magnitude, and anything that loops until it reaches zero never terminates.
"""

from __future__ import annotations

import pytest
import torch

from hologradpy.holography.vortices import VortexDetector
from hologradpy.optics.complex_amplitude import ComplexAmplitude, FieldGeometry

RESOLUTION = (64, 64)
PIXEL_SIZE = (1e-5, 1e-5)
WAVELENGTH = 1e-6


def _field(data: torch.Tensor) -> ComplexAmplitude:
    geometry = FieldGeometry(
        resolution=RESOLUTION,
        pixel_size=torch.tensor(list(PIXEL_SIZE)),
        wavelength=torch.tensor(WAVELENGTH),
    )
    return ComplexAmplitude.from_geometry(geometry, data=data.to(torch.complex64))


def _grid() -> tuple[torch.Tensor, torch.Tensor]:
    """Index coordinates centred where the detector puts its origin.

    Offset by half a pixel so no sample lands exactly on a zero. The detector tests for
    a sign change with a strict ``product < 0``, which an exact zero fails, so a field
    whose null sits precisely on a pixel is invisible to it.
    """
    rows = torch.arange(RESOLUTION[0], dtype=torch.float32) - RESOLUTION[0] // 2 + 0.5
    columns = (
        torch.arange(RESOLUTION[1], dtype=torch.float32) - RESOLUTION[1] // 2 + 0.5
    )
    return columns[None, :].expand(RESOLUTION), rows[:, None].expand(RESOLUTION)


def _charges(data: torch.Tensor) -> torch.Tensor:
    """The charges the detector finds, over a target that is lit everywhere."""
    detector = VortexDetector(RESOLUTION)
    detector.detect_vortices(
        _field(data), target_intensity=torch.ones(RESOLUTION), threshold=0.2
    )
    if detector.number_of_vortices == 0:
        return torch.zeros(0)
    return detector.charges.reshape(-1)


def test_a_single_vortex_is_found_and_carries_one_charge():
    """``x + iy`` winds once around the origin, which is the textbook charge +1."""
    x, y = _grid()

    charges = _charges(x + 1j * y)

    assert int((charges != 0).sum()) == 1
    assert abs(float(charges[charges != 0][0])) == 1


def test_the_opposite_winding_gets_the_opposite_sign():
    """Conjugating the field reverses the direction the phase turns."""
    x, y = _grid()

    forward = _charges(x + 1j * y)
    reversed_ = _charges(x - 1j * y)

    assert float(forward[forward != 0][0]) == -float(reversed_[reversed_ != 0][0])


def test_an_interference_null_is_a_zero_but_not_a_vortex():
    """The failure this module exists to prevent.

    Two beams crossing give a field that is a real fringe pattern times one overall
    phase, so its real and imaginary parts vanish together along whole lines and the
    detector labels plenty of components. None of them is a vortex: the phase jumps by
    pi across a null rather than winding around a point.
    """
    x, _ = _grid()
    fringes = torch.cos(x * torch.pi / 8).to(torch.complex64)
    interference = fringes * torch.exp(torch.tensor(1j * torch.pi / 4))

    detector = VortexDetector(RESOLUTION)
    detector.detect_vortices(
        _field(interference),
        target_intensity=torch.ones(RESOLUTION),
        threshold=0.2,
    )

    assert detector.number_of_vortices > 0, "expected zero crossings to be labelled"
    assert int((detector.charges.reshape(-1) != 0).sum()) == 0


def test_a_smooth_beam_has_neither():
    x, y = _grid()
    gaussian = torch.exp(-(x**2 + y**2) / (2 * 12.0**2)) + 0j

    charges = _charges(gaussian)

    assert int((charges != 0).sum()) == 0


@pytest.mark.parametrize("offset", [(0, 0), (7, -5)])
def test_a_vortex_is_found_where_it_was_put(offset: tuple[int, int]):
    """Pins the row/column convention of ``center_indices``, which a plot depends on."""
    x, y = _grid()
    shift_x, shift_y = offset

    detector = VortexDetector(RESOLUTION)
    detector.detect_vortices(
        _field((x - shift_x) + 1j * (y - shift_y)),
        target_intensity=torch.ones(RESOLUTION),
        threshold=0.2,
    )

    charged = detector.charges.reshape(-1) != 0
    rows, columns = zip(
        *[
            (int(row), int(column))
            for (row, column), keep in zip(detector.center_indices, charged)
            if keep
        ]
    )
    assert rows[0] == pytest.approx(RESOLUTION[0] // 2 + shift_y, abs=1)
    assert columns[0] == pytest.approx(RESOLUTION[1] // 2 + shift_x, abs=1)
