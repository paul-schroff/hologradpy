"""Tests for tilt-unit handling in phase_profiles.

`linear_phase` delegates every tilt representation to `tilt_to_angle`, the one
place that maps a tilt (metres / radians / degrees / lines_per_mm) onto a beam
deflection angle. These tests pin the conversions, the previously broken
`degrees` branch, the newly implemented `lines_per_mm` branch, and the argument
validation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from hologradpy.propagation.phase_profiles import linear_phase, tilt_to_angle

WAVELENGTH = 0.63e-6
WAVENUMBER = 2 * math.pi / WAVELENGTH
FOCAL_LENGTH = 0.25


def _grid():
    axis = np.linspace(-1e-3, 1e-3, 7)
    return np.meshgrid(axis, axis)


# --- tilt_to_angle ------------------------------------------------------------


def test_tilt_to_angle_values():
    assert tilt_to_angle(0.02, "radians") == pytest.approx(0.02)
    assert tilt_to_angle(5.0, "degrees") == pytest.approx(math.radians(5.0))
    assert tilt_to_angle(
        500e-6, "metres", focal_length=FOCAL_LENGTH
    ) == pytest.approx(500e-6 / FOCAL_LENGTH)
    # A grating of nu lines/mm deflects the first order by nu * 1e3 * lambda.
    assert tilt_to_angle(
        10.0, "lines_per_mm", wavenumber=WAVENUMBER
    ) == pytest.approx(10.0 * 1e3 * WAVELENGTH)


def test_tilt_to_angle_metres_requires_focal_length():
    with pytest.raises(ValueError):
        tilt_to_angle(500e-6, "metres")


def test_tilt_to_angle_lines_per_mm_requires_wavenumber():
    with pytest.raises(ValueError):
        tilt_to_angle(10.0, "lines_per_mm")


def test_tilt_to_angle_unknown_unit_raises():
    with pytest.raises(ValueError):
        tilt_to_angle(1.0, "furlongs")


# --- linear_phase -------------------------------------------------------------


def test_linear_phase_requires_wavenumber():
    x, y = _grid()
    with pytest.raises(ValueError):
        linear_phase(x, y, 1e-4, 0.0, tilt_units="metres", focal_length=FOCAL_LENGTH)


def test_linear_phase_degrees_matches_radians():
    """Regression: the degrees branch used to take tan() of a degree value with
    no deg->rad conversion. It must now equal the radians branch at the same
    angle."""
    x, y = _grid()
    degrees = linear_phase(x, y, 0.03, -0.02, tilt_units="degrees",
                           wavenumber=WAVENUMBER)
    radians = linear_phase(
        x, y, math.radians(0.03), math.radians(-0.02), tilt_units="radians",
        wavenumber=WAVENUMBER,
    )
    np.testing.assert_allclose(degrees, radians, rtol=1e-12)


def test_linear_phase_metres_matches_radians():
    x, y = _grid()
    metres = linear_phase(x, y, 500e-6, -200e-6, tilt_units="metres",
                          wavenumber=WAVENUMBER, focal_length=FOCAL_LENGTH)
    radians = linear_phase(
        x, y, 500e-6 / FOCAL_LENGTH, -200e-6 / FOCAL_LENGTH,
        tilt_units="radians", wavenumber=WAVENUMBER,
    )
    np.testing.assert_allclose(metres, radians, rtol=1e-12)


def test_linear_phase_ramp_value():
    x, y = _grid()
    angle_x, angle_y = 2e-3, -1e-3
    phase = linear_phase(x, y, angle_x, angle_y, tilt_units="radians",
                         wavenumber=WAVENUMBER)
    expected = WAVENUMBER * (angle_x * x + angle_y * y)
    np.testing.assert_allclose(phase, expected, rtol=1e-12)


def test_linear_phase_lines_per_mm_period():
    """A grating of nu lines/mm advances the phase by 2*pi over one line pitch,
    1 / (nu * 1e3) metres, along the grating axis."""
    lines_per_mm = 10.0
    pitch = 1.0 / (lines_per_mm * 1e3)  # metres
    x = np.array([[0.0, pitch]])
    y = np.zeros_like(x)
    phase = linear_phase(x, y, lines_per_mm, 0.0, tilt_units="lines_per_mm",
                         wavenumber=WAVENUMBER)
    assert (phase[0, 1] - phase[0, 0]) == pytest.approx(2 * math.pi)


def test_linear_phase_lines_per_mm_is_wavelength_independent():
    """A fixed grating imprints the same geometric phase ramp regardless of
    wavelength (the wavenumber cancels)."""
    x, y = _grid()
    phase_a = linear_phase(x, y, 8.0, 3.0, tilt_units="lines_per_mm",
                           wavenumber=2 * math.pi / 0.5e-6)
    phase_b = linear_phase(x, y, 8.0, 3.0, tilt_units="lines_per_mm",
                           wavenumber=2 * math.pi / 0.8e-6)
    np.testing.assert_allclose(phase_a, phase_b, rtol=1e-12)


def test_linear_phase_torch_backend_matches_numpy():
    x, y = _grid()
    phase_numpy = linear_phase(x, y, 500e-6, 0.0, tilt_units="metres",
                               wavenumber=WAVENUMBER, focal_length=FOCAL_LENGTH)
    phase_torch = linear_phase(
        torch.as_tensor(x), torch.as_tensor(y), 500e-6, 0.0, tilt_units="metres",
        wavenumber=WAVENUMBER, focal_length=FOCAL_LENGTH,
    )
    np.testing.assert_allclose(phase_torch.numpy(), phase_numpy, rtol=1e-12)
