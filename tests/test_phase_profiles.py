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

from hologradpy.fourier_optics import (
    beam_shaping_focal_length,
    get_focal_spot_radius,
)
from hologradpy.fourier_transforms import fft_2d, ifft_2d
from hologradpy.grids import get_frequency_grid, get_pixel_grid
from hologradpy.profiles.phase import (
    defocus_phase,
    hermite_gaussian_phase,
    lens_phase,
    linear_phase,
    tilt_to_angle,
    binary_phase_grating,
    band_limited_random_phase,
    gaussian_phase_guess,
    hyperbolic_lens_phase,
    random_diversity_phase,
)

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
    angle.
    """
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
    1 / (nu * 1e3) metres, along the grating axis.
    """
    lines_per_mm = 10.0
    pitch = 1.0 / (lines_per_mm * 1e3)  # metres
    x = np.array([[0.0, pitch]])
    y = np.zeros_like(x)
    phase = linear_phase(x, y, lines_per_mm, 0.0, tilt_units="lines_per_mm",
                         wavenumber=WAVENUMBER)
    assert (phase[0, 1] - phase[0, 0]) == pytest.approx(2 * math.pi)


def test_linear_phase_lines_per_mm_is_wavelength_independent():
    """A fixed grating imprints the same geometric phase ramp regardless of
    wavelength (the wavenumber cancels).
    """
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


# --- hyperbolic_lens_phase ----------------------------------------------------


def _lens_grid(radius: float, samples: int = 101):
    axis = np.linspace(-radius, radius, samples)
    return np.meshgrid(axis, axis)


def test_hyperbolic_lens_phase_brings_every_point_to_the_focus_in_phase():
    """The lens phase plus the path to the focal point is the same everywhere."""
    x, y = _lens_grid(12.7e-3)
    phase = hyperbolic_lens_phase(x, y, FOCAL_LENGTH, WAVENUMBER)
    arrival = phase + WAVENUMBER * np.sqrt(x**2 + y**2 + FOCAL_LENGTH**2)
    np.testing.assert_allclose(arrival, WAVENUMBER * FOCAL_LENGTH, rtol=1e-12)


def test_hyperbolic_lens_phase_departs_from_the_paraxial_lens_by_the_fourth_order():
    x, y = _lens_grid(2e-3)
    radius_squared = x**2 + y**2
    difference = hyperbolic_lens_phase(x, y, FOCAL_LENGTH, WAVENUMBER) - lens_phase(
        x, y, FOCAL_LENGTH, WAVENUMBER
    )
    fourth_order = WAVENUMBER * radius_squared**2 / (8 * FOCAL_LENGTH**3)
    np.testing.assert_allclose(difference, fourth_order, rtol=1e-3, atol=1e-12)


def test_hyperbolic_lens_phase_keeps_its_precision_in_single_precision():
    """At the edge of a one-inch lens the single-precision phase stays within a
    milliradian of the double-precision one, where the difference of the two lengths
    loses more than 10 mrad.
    """
    x, y = _lens_grid(12.7e-3)
    wavenumber = 2 * math.pi / 1039e-9
    exact = hyperbolic_lens_phase(x, y, FOCAL_LENGTH, wavenumber)
    x_single, y_single = (torch.as_tensor(axis, dtype=torch.float32) for axis in (x, y))
    single = hyperbolic_lens_phase(x_single, y_single, FOCAL_LENGTH, wavenumber)
    assert single.dtype == torch.float32
    assert np.abs(single.numpy() - exact).max() < 1e-3
    path_difference = (
        torch.sqrt(x_single**2 + y_single**2 + FOCAL_LENGTH**2) - FOCAL_LENGTH
    )
    assert np.abs(-wavenumber * path_difference.numpy() - exact).max() > 1e-2


# --- defocus_phase ------------------------------------------------------------


def test_defocus_phase_propagates_every_plane_wave():
    """The wave from ``r`` reaches the camera with transverse wavenumber ``k r / f``.
    The camera's plane lies the defocus before the focus, at ``z = -defocus``, and the
    phase is the axial phase over ``z``, less the piston ``k z``.
    """
    x, y = _lens_grid(12.7e-3)
    defocus = 5e-3
    phase = defocus_phase(x, y, defocus, FOCAL_LENGTH, WAVENUMBER)
    transverse = WAVENUMBER * np.sqrt(x**2 + y**2) / FOCAL_LENGTH
    axial = np.sqrt(WAVENUMBER**2 - transverse**2)
    np.testing.assert_allclose(phase, -defocus * (axial - WAVENUMBER), rtol=1e-9)


def test_defocus_phase_departs_from_the_paraxial_defocus_by_the_fourth_order():
    x, y = _lens_grid(2e-3)
    defocus = 5e-3
    radius_squared = x**2 + y**2
    paraxial = WAVENUMBER * defocus * radius_squared / (2 * FOCAL_LENGTH**2)
    difference = defocus_phase(x, y, defocus, FOCAL_LENGTH, WAVENUMBER) - paraxial
    fourth_order = WAVENUMBER * defocus * radius_squared**2 / (8 * FOCAL_LENGTH**4)
    np.testing.assert_allclose(difference, fourth_order, rtol=1e-3, atol=1e-15)


def test_defocus_phase_keeps_its_precision_in_single_precision():
    """Over a one-inch aperture and 10 mm, the single-precision phase stays within
    0.1 mrad of the double-precision one, where the difference of the axial phase and
    the piston loses more than a milliradian.
    """
    x, y = _lens_grid(12.7e-3)
    wavenumber, defocus = 2 * math.pi / 1039e-9, 10e-3
    exact = defocus_phase(x, y, defocus, FOCAL_LENGTH, wavenumber)
    x_single, y_single = (torch.as_tensor(axis, dtype=torch.float32) for axis in (x, y))
    single = defocus_phase(x_single, y_single, defocus, FOCAL_LENGTH, wavenumber)
    assert single.dtype == torch.float32
    assert np.abs(single.numpy() - exact).max() < 1e-4
    sine_squared = (x_single**2 + y_single**2) / FOCAL_LENGTH**2
    difference = wavenumber * defocus * (1 - torch.sqrt(1 - sine_squared))
    assert np.abs(difference.numpy() - exact).max() > 1e-3


def test_a_defocus_is_a_lens_of_minus_f_squared_over_the_defocus():
    """To second order in the radius, the defocus is a lens of focal length
    ``-f**2 / defocus``. On this 2 mm grid, the fourth-order term stays below 1e-5 of
    the phase.
    """
    x, y = _grid()

    phase = defocus_phase(x, y, 2e-3, FOCAL_LENGTH, WAVENUMBER)

    np.testing.assert_allclose(
        phase, lens_phase(x, y, -(FOCAL_LENGTH**2) / 2e-3, WAVENUMBER), rtol=1e-4
    )


def test_a_positive_defocus_is_a_diverging_lens():
    """The phase grows away from (0, 0), so the focus moves away from the SLM."""
    x, y = _grid()

    phase = defocus_phase(x, y, 2e-3, FOCAL_LENGTH, WAVENUMBER)

    assert phase[3, 3] == 0.0
    assert np.all(phase >= 0.0)


def test_no_defocus_is_no_phase():
    """A zero defocus gives an array of zeros in the shape of the grid."""
    x, y = _grid()

    phase = defocus_phase(x, y, 0.0, FOCAL_LENGTH, WAVENUMBER)

    assert np.shape(phase) == x.shape
    np.testing.assert_array_equal(phase, 0.0)


# --- binary_phase_grating -----------------------------------------------------


def test_binary_phase_grating_default_columns():
    grating = binary_phase_grating((4, 6))
    assert grating.shape == (4, 6)
    # Odd columns are pi, even columns are 0 (a vertical period-2 grating).
    np.testing.assert_allclose(grating[:, 0::2], 0.0)
    np.testing.assert_allclose(grating[:, 1::2], np.pi)


def test_binary_phase_grating_axis0_rows():
    grating = binary_phase_grating((5, 3), axis=0)
    np.testing.assert_allclose(grating[0::2, :], 0.0)
    np.testing.assert_allclose(grating[1::2, :], np.pi)


def test_binary_phase_grating_custom_levels():
    grating = binary_phase_grating((2, 4), high=1.5, low=-0.5)
    np.testing.assert_allclose(grating[:, 0::2], -0.5)
    np.testing.assert_allclose(grating[:, 1::2], 1.5)


def test_binary_phase_grating_invalid_axis():
    with pytest.raises(ValueError):
        binary_phase_grating((2, 2), axis=2)


# --- gaussian_phase_guess -----------------------------------------------------


@pytest.mark.parametrize("output_radius", [1e-3, 200e-6, 100e-6, 40e-6])
def test_beam_shaping_focal_length_hits_the_target_radius(output_radius):
    """Ray optics and diffraction add in quadrature, and the lens accounts for both."""
    input_radius = 2e-3
    lens = beam_shaping_focal_length(
        input_radius, FOCAL_LENGTH, output_radius, WAVELENGTH
    )
    focal_spot = get_focal_spot_radius(input_radius, WAVELENGTH, FOCAL_LENGTH)
    arrived = math.hypot(FOCAL_LENGTH * input_radius / lens, focal_spot)
    assert arrived == pytest.approx(output_radius)


def test_beam_shaping_focal_length_overshoots_without_the_diffraction_term():
    """The ray-optics-only lens lands the beam wide, which is what the fix removes."""
    input_radius, output_radius = 2e-3, 40e-6
    ray_optics_lens = input_radius * FOCAL_LENGTH / output_radius
    focal_spot = get_focal_spot_radius(input_radius, WAVELENGTH, FOCAL_LENGTH)
    arrived = math.hypot(FOCAL_LENGTH * input_radius / ray_optics_lens, focal_spot)
    assert arrived > 1.15 * output_radius


def test_gaussian_phase_guess_matches_an_explicit_cylindrical_lens():
    x, y = _grid()
    input_radius, output_radius = 2e-3, 100e-6
    focal_spot = get_focal_spot_radius(input_radius, WAVELENGTH, FOCAL_LENGTH)
    lens = input_radius * FOCAL_LENGTH / math.sqrt(output_radius**2 - focal_spot**2)
    expected = -WAVENUMBER * y**2 / (2 * lens)
    guess = gaussian_phase_guess(
        x,
        y,
        (input_radius, input_radius),
        (None, output_radius),
        FOCAL_LENGTH,
        WAVENUMBER,
    )
    np.testing.assert_allclose(guess, expected)


def test_gaussian_phase_guess_shapes_both_axes():
    """Shaping both axes gives the two cylindrical lenses summed."""
    x, y = _grid()
    kwargs = dict(
        input_beam_radius=(2e-3, 2e-3),
        focal_length=FOCAL_LENGTH,
        wavenumber=WAVENUMBER,
    )
    both = gaussian_phase_guess(x, y, output_beam_radius=(200e-6, 100e-6), **kwargs)
    along_x = gaussian_phase_guess(x, y, output_beam_radius=(200e-6, None), **kwargs)
    along_y = gaussian_phase_guess(x, y, output_beam_radius=(None, 100e-6), **kwargs)
    np.testing.assert_allclose(both, along_x + along_y)

    # Equal radii on both axes make the lens spherical.
    square = gaussian_phase_guess(x, y, output_beam_radius=(100e-6, 100e-6), **kwargs)
    np.testing.assert_allclose(square, square.T)


def test_gaussian_phase_guess_accepts_torch_and_per_axis_radii():
    axis = torch.linspace(-1e-3, 1e-3, 7, dtype=torch.float64)
    x, y = torch.meshgrid(axis, axis, indexing="xy")
    guess = gaussian_phase_guess(
        x, y, (2e-3, 1e-3), (200e-6, 100e-6), FOCAL_LENGTH, WAVENUMBER
    )
    assert isinstance(guess, torch.Tensor)
    assert guess.shape == x.shape


def test_gaussian_phase_guess_rejects_nothing_to_shape():
    x, y = _grid()
    radii = (2e-3, 2e-3)
    with pytest.raises(ValueError, match="[Aa]t least one axis"):
        gaussian_phase_guess(x, y, radii, (None, None), FOCAL_LENGTH, WAVENUMBER)
    focal_spot = get_focal_spot_radius(radii[1], WAVELENGTH, FOCAL_LENGTH)
    with pytest.raises(ValueError, match="diffraction limit"):
        gaussian_phase_guess(
            x, y, radii, (None, 0.5 * focal_spot), FOCAL_LENGTH, WAVENUMBER
        )
    with pytest.raises(ValueError, match="diffraction limit"):
        gaussian_phase_guess(x, y, radii, (None, -1e-6), FOCAL_LENGTH, WAVENUMBER)


def test_gaussian_phase_guess_requires_pairs():
    """Both radii are (x, y) pairs, so a bare number is refused, not broadcast."""
    x, y = _grid()
    with pytest.raises(TypeError):
        gaussian_phase_guess(x, y, 2e-3, (None, 100e-6), FOCAL_LENGTH, WAVENUMBER)


@pytest.mark.parametrize("angle_degrees", [0.0, 30.0])
def test_gaussian_phase_guess_shift_adds_a_linear_phase(angle_degrees):
    """The shift is a tilt on top, so it separates cleanly from the shaping lens. It
    stays along x and y when the lenses are turned.
    """
    x, y = _grid()
    radii, outputs = (2e-3, 2e-3), (200e-6, 100e-6)
    shift = (300e-6, -150e-6)
    shaped = gaussian_phase_guess(
        x, y, radii, outputs, FOCAL_LENGTH, WAVENUMBER, angle_degrees=angle_degrees
    )
    steered = gaussian_phase_guess(
        x,
        y,
        radii,
        outputs,
        FOCAL_LENGTH,
        WAVENUMBER,
        output_beam_shift=shift,
        angle_degrees=angle_degrees,
    )
    tilt = linear_phase(
        x,
        y,
        *shift,
        tilt_units="metres",
        wavenumber=WAVENUMBER,
        focal_length=FOCAL_LENGTH,
    )
    np.testing.assert_allclose(steered - shaped, tilt)


def test_gaussian_phase_guess_defaults_to_the_optical_axis():
    x, y = _grid()
    radii, outputs = (2e-3, 2e-3), (200e-6, 100e-6)
    np.testing.assert_array_equal(
        gaussian_phase_guess(x, y, radii, outputs, FOCAL_LENGTH, WAVENUMBER),
        gaussian_phase_guess(
            x, y, radii, outputs, FOCAL_LENGTH, WAVENUMBER, output_beam_shift=(0.0, 0.0)
        ),
    )


def test_gaussian_phase_guess_steers_an_unshaped_axis():
    """A tilt does not need a lens on the same axis."""
    x, y = _grid()
    steered = gaussian_phase_guess(
        x,
        y,
        (2e-3, 2e-3),
        (None, 100e-6),
        FOCAL_LENGTH,
        WAVENUMBER,
        output_beam_shift=(300e-6, 0.0),
    )
    # x carries only the tilt, so its phase is linear along that axis.
    along_x = steered[steered.shape[0] // 2, :]
    np.testing.assert_allclose(np.diff(along_x, 2), 0.0, atol=1e-6)


def test_gaussian_phase_guess_turns_its_lenses_from_x_towards_y():
    """At 45 degrees, the first lens runs along the diagonal x = y, so its phase
    depends on (x + y) / sqrt(2) only.
    """
    x, y = _grid()
    input_radius, output_radius = 2e-3, 100e-6
    lens = beam_shaping_focal_length(
        input_radius, FOCAL_LENGTH, output_radius, WAVELENGTH
    )
    along = (x + y) / math.sqrt(2)
    guess = gaussian_phase_guess(
        x,
        y,
        (input_radius, input_radius),
        (output_radius, None),
        FOCAL_LENGTH,
        WAVENUMBER,
        angle_degrees=45.0,
    )
    np.testing.assert_allclose(guess, -WAVENUMBER * along**2 / (2 * lens), atol=1e-12)


def test_turning_the_lenses_by_90_degrees_swaps_their_axes():
    x, y = _grid()
    turned = gaussian_phase_guess(
        x, y, (2e-3, 1e-3), (200e-6, 100e-6), FOCAL_LENGTH, WAVENUMBER, angle_degrees=90
    )
    swapped = gaussian_phase_guess(
        x, y, (1e-3, 2e-3), (100e-6, 200e-6), FOCAL_LENGTH, WAVENUMBER
    )
    np.testing.assert_allclose(turned, swapped)


def test_the_focal_spot_and_the_shaping_lens_of_a_pair_are_pairs():
    radii, targets = (2e-3, 1e-3), (200e-6, 100e-6)

    spots = get_focal_spot_radius(radii, WAVELENGTH, FOCAL_LENGTH)
    lenses = beam_shaping_focal_length(radii, FOCAL_LENGTH, targets, WAVELENGTH)

    assert spots == tuple(
        get_focal_spot_radius(radius, WAVELENGTH, FOCAL_LENGTH) for radius in radii
    )
    assert lenses == tuple(
        beam_shaping_focal_length(radius, FOCAL_LENGTH, target, WAVELENGTH)
        for radius, target in zip(radii, targets)
    )


# --- hermite_gaussian_phase ---------------------------------------------------


def test_hg_1_1_is_pi_where_x_times_y_is_negative():
    x, y = _grid()

    phase = hermite_gaussian_phase(x, y, 1, 1)

    np.testing.assert_array_equal(phase, np.where(x * y < 0, np.pi, 0.0))


def test_hg_1_0_is_pi_where_x_is_negative():
    x, y = _grid()

    phase = hermite_gaussian_phase(x, y, 1, 0)

    np.testing.assert_array_equal(phase, np.where(x < 0, np.pi, 0.0))


def test_hg_2_0_is_pi_within_half_the_beam_radius():
    """H_2(u) = 4 u^2 - 2 is negative for |u| < 1 / sqrt(2), so for |x| < w / 2."""
    x, y = _grid()
    radius = 1.5e-3

    phase = hermite_gaussian_phase(x, y, 2, 0, beam_radius=radius)

    np.testing.assert_array_equal(phase, np.where(np.abs(x) < radius / 2, np.pi, 0.0))


def test_orders_of_2_and_above_need_the_beam_radius():
    x, y = _grid()

    with pytest.raises(ValueError, match="beam_radius"):
        hermite_gaussian_phase(x, y, 0, 2)


def test_the_hg_phase_keeps_the_dtype_of_a_tensor():
    x, y = (torch.as_tensor(axis) for axis in _grid())

    phase = hermite_gaussian_phase(x, y, 1, 1)

    assert phase.dtype == torch.float64


# --- random_diversity_phase ---------------------------------------------------

DIVERSITY_SHAPE = (48, 64)


def _diversity_band_and_beam() -> tuple[torch.Tensor, torch.Tensor]:
    """A disc band of 0.6 rad per sample, and a Gaussian beam off the centre of the
    grid, so a tilt over the beam is not a tilt over the grid.
    """
    frequency_x, frequency_y = get_frequency_grid(DIVERSITY_SHAPE, (1.0, 1.0))
    band = torch.hypot(frequency_x, frequency_y) <= 0.6
    grid_x, grid_y = get_pixel_grid(DIVERSITY_SHAPE, dtype=torch.float64)
    beam = torch.exp(-2 * ((grid_x - 5.0) ** 2 + (grid_y + 3.0) ** 2) / 14.0**2)
    return band, beam


def _outside_the_band_up_to_a_tilt(phase: torch.Tensor, band: torch.Tensor) -> float:
    """The part of ``phase`` outside ``band`` that no tilt accounts for, as a fraction
    of the phase. A tilt is not band-limited on a finite grid, so the tilt's own part
    outside the band is fitted away first. A piston lies inside a band that holds the
    zero frequency.
    """

    def outside(image: torch.Tensor) -> torch.Tensor:
        return ifft_2d(fft_2d(image + 0j) * ~band).real.flatten()

    grid_x, grid_y = get_pixel_grid(tuple(phase.shape), dtype=torch.float64)
    tilts = torch.stack((outside(grid_x), outside(grid_y)), dim=1)
    residual = outside(phase)
    fit = torch.linalg.solve(tilts.T @ tilts, tilts.T @ residual)
    leftover = residual - tilts @ fit
    return float(leftover.norm() / phase.norm())


def test_random_diversity_phase_has_the_requested_rms_over_the_weights():
    band, beam = _diversity_band_and_beam()
    phase = random_diversity_phase(
        band, beam, 0.25, generator=torch.Generator().manual_seed(0)
    )

    rms = torch.sqrt((beam * phase**2).sum() / beam.sum())
    assert phase.dtype == torch.float64
    assert float(rms) == pytest.approx(0.25, rel=1e-12)


def test_random_diversity_phase_has_no_weighted_piston_or_tilt():
    band, beam = _diversity_band_and_beam()
    phase = random_diversity_phase(
        band, beam, 0.25, generator=torch.Generator().manual_seed(1)
    )

    grid_x, grid_y = get_pixel_grid(DIVERSITY_SHAPE, dtype=torch.float64)
    for basis in (torch.ones_like(grid_x), grid_x, grid_y):
        moment = (beam * phase * basis).sum() / (beam * phase.abs() * basis.abs()).sum()
        assert abs(float(moment)) < 1e-12


def test_random_diversity_phase_lies_in_the_band_up_to_a_tilt():
    band, beam = _diversity_band_and_beam()
    phase = random_diversity_phase(
        band, beam, 0.25, generator=torch.Generator().manual_seed(2)
    )

    assert band[DIVERSITY_SHAPE[0] // 2, DIVERSITY_SHAPE[1] // 2]
    assert _outside_the_band_up_to_a_tilt(phase, band) < 1e-10


def test_random_diversity_phase_draws_anew_from_its_generator():
    band, beam = _diversity_band_and_beam()
    generator = torch.Generator().manual_seed(3)
    first = random_diversity_phase(band, beam, 0.25, generator=generator)
    second = random_diversity_phase(band, beam, 0.25, generator=generator)
    repeated = random_diversity_phase(
        band, beam, 0.25, generator=torch.Generator().manual_seed(3)
    )

    assert not torch.allclose(first, second)
    # Threaded BLAS sums in an order that varies, so a repeat agrees to round-off.
    torch.testing.assert_close(first, repeated, rtol=0.0, atol=1e-12)


# --- band_limited_random_phase ------------------------------------------------

RANDOM_PHASE_SHAPE = (128, 128)
RANDOM_PHASE_BAND_RADIUS = 12.0


def _random_phase_far_field(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The far-field intensity of a uniformly lit pattern, as a fraction of its power,
    and the band. The pattern is drawn in a disc band of ``RANDOM_PHASE_BAND_RADIUS``
    frequency bins.
    """
    grid_x, grid_y = get_pixel_grid(RANDOM_PHASE_SHAPE, dtype=torch.float64)
    band = torch.hypot(grid_x, grid_y) <= RANDOM_PHASE_BAND_RADIUS
    phase = band_limited_random_phase(
        band, generator=torch.Generator().manual_seed(seed)
    )
    intensity = fft_2d(torch.exp(1j * phase.to(torch.float64))).abs() ** 2
    return intensity / intensity.sum(), band


def test_band_limited_random_phase_leaves_no_bright_undiffracted_spot():
    """The undiffracted light lands on the zero frequency, where the tilt of a speckle
    calibration carries it to the speckle centre. It stays below the brightest speckle
    grain in the band, which sets the exposure.
    """
    intensity, band = _random_phase_far_field(seed=0)
    centre = tuple(length // 2 for length in RANDOM_PHASE_SHAPE)
    speckle = band.clone()
    speckle[centre] = False

    assert band[centre]
    assert intensity[centre] < intensity[speckle].max()


def test_band_limited_random_phase_puts_pi_over_four_of_its_power_in_the_band():
    intensity, band = _random_phase_far_field(seed=1)

    assert float(intensity[band].sum()) > math.pi / 4
