"""Unwrapping a phase image inside a region of interest.

The failure these guard against is silent: the returned image looks like a wavefront,
just with whole columns displaced by 2 pi, so it only shows up as a hard seam in a plot
or as an RMS that is far too large.
"""

import numpy as np
import pytest

from hologradpy.analysis.unwrapping import (
    unwrap_2d_laplace,
    unwrap_2d_poisson,
    wrap,
)

# The two solve the same least-squares problem by different means, so every property
# below has to hold for both. Parametrised rather than duplicated so a new solver is
# one line here.
UNWRAPPERS = pytest.mark.parametrize(
    "unwrap", [unwrap_2d_laplace, unwrap_2d_poisson], ids=["laplace", "poisson"]
)


RESOLUTION = 96


def _grid(resolution: int = RESOLUTION):
    y, x = np.mgrid[0:resolution, 0:resolution] - (resolution - 1) / 2
    return x, y


def _disc(radius_fraction: float = 0.45):
    x, y = _grid()
    return (x**2 + y**2) < (radius_fraction * RESOLUTION) ** 2


def _residual(unwrapped, truth, mask):
    """The recovered phase against the truth, up to the constant nothing can fix."""
    residual = (unwrapped - truth)[mask]
    return residual - residual.mean()


@UNWRAPPERS
@pytest.mark.parametrize(
    "name, coefficients",
    [
        ("tilt in x", (1.5, 0.0, 0.0)),
        ("tilt in y", (0.0, 1.5, 0.0)),
        ("steep diagonal tilt", (2.0, 1.4, 0.0)),
        ("defocus", (0.0, 0.0, 0.008)),
    ],
)
def test_a_smooth_wavefront_comes_back_exactly(unwrap, name, coefficients) -> None:
    """Several wraps across a circular aperture, all well below the pi per pixel that
    makes unwrapping ambiguous. Row-then-column unwrapping fails every one of these
    except defocus, which happens to be symmetric enough to survive it.
    """
    x, y = _grid()
    tilt_x, tilt_y, curvature = coefficients
    truth = tilt_x * x + tilt_y * y + curvature * (x**2 - y**2)
    mask = _disc()

    unwrapped = unwrap(np.angle(np.exp(1j * truth)), mask)

    assert np.abs(_residual(unwrapped, truth, mask)).max() < 1e-6
    assert not unwrapped[~mask].any()


@UNWRAPPERS
def test_no_column_is_left_offset_by_two_pi(unwrap) -> None:
    """The specific artefact, stated the way it appears: the residual takes one value
    over the whole aperture rather than one value per column.
    """
    x, y = _grid()
    truth = 2.0 * x + 1.4 * y
    mask = _disc()

    unwrapped = unwrap(np.angle(np.exp(1j * truth)), mask)
    residual = _residual(unwrapped, truth, mask)
    offsets = np.unique(np.round(residual / (2 * np.pi)).astype(int))

    assert offsets.tolist() == [0]


@UNWRAPPERS
def test_noise_below_the_wrapping_limit_passes_straight_through(unwrap) -> None:
    """Measurement noise must come back as noise, not as unwrapping errors. Anything
    that failed would show up as a residual far larger than the noise it was given.
    """
    x, y = _grid()
    truth = 0.8 * x + 0.5 * y
    mask = _disc()
    noise = 0.1 * np.random.default_rng(0).standard_normal(truth.shape)

    unwrapped = unwrap(np.angle(np.exp(1j * (truth + noise))), mask)

    assert _residual(unwrapped, truth, mask).std() == pytest.approx(0.1, rel=0.2)


@UNWRAPPERS
def test_each_disconnected_region_is_unwrapped_on_its_own(unwrap) -> None:
    """Nothing ties one region to another, so each is internally consistent and carries
    its own offset. Joining them across the gap would invent a phase relationship the
    measurement does not contain.
    """
    x, y = _grid()
    left = (x + 24) ** 2 + y**2 < 14**2
    right = (x - 24) ** 2 + y**2 < 14**2
    truth = 0.6 * x + 0.4 * y

    unwrapped = unwrap(np.angle(np.exp(1j * truth)), left | right)

    for region in (left, right):
        assert np.abs(_residual(unwrapped, truth, region)).max() < 1e-6


@UNWRAPPERS
def test_an_annular_mask_is_unwrapped_around_the_obstruction(unwrap) -> None:
    """A row of an annulus is two separate runs of pixels, which is exactly what the
    row-by-row approach cannot handle.
    """
    x, y = _grid()
    radius = np.sqrt(x**2 + y**2)
    mask = (radius < 0.45 * RESOLUTION) & (radius > 0.2 * RESOLUTION)
    truth = 1.2 * x + 0.9 * y

    unwrapped = unwrap(np.angle(np.exp(1j * truth)), mask)

    assert np.abs(_residual(unwrapped, truth, mask)).max() < 1e-6


@UNWRAPPERS
def test_an_empty_mask_returns_zeros(unwrap) -> None:
    """A degenerate region of interest is not worth an exception, and the sparse solve
    would otherwise fail on an empty system.
    """
    phase = np.zeros((8, 8))
    mask = np.zeros((8, 8), dtype=bool)

    assert not unwrap(phase, mask).any()


@UNWRAPPERS
def test_isolated_pixels_keep_their_wrapped_value(unwrap) -> None:
    """A pixel with no masked neighbour has no difference to satisfy, so the only
    defensible answer is the value that was measured there.
    """
    phase = np.full((8, 8), 2.0)
    mask = np.zeros((8, 8), dtype=bool)
    mask[1, 1] = True
    mask[6, 6] = True

    unwrapped = unwrap(phase, mask)

    assert unwrapped[1, 1] == pytest.approx(2.0)
    assert unwrapped[6, 6] == pytest.approx(2.0)


def test_wrap_maps_onto_the_half_open_interval() -> None:
    """Half open at the top, so exactly pi comes back as -pi. numpy.angle rounds the
    other way, which is worth pinning since the two are used interchangeably here.
    """
    values = np.array([-3 * np.pi, 0.0, 7.0, 100.0, -0.3])

    wrapped = wrap(values)

    assert np.all(wrapped >= -np.pi)
    assert np.all(wrapped < np.pi)
    assert np.allclose(np.angle(np.exp(1j * values)), wrapped)
    assert wrap(np.array([np.pi])) == pytest.approx(-np.pi)


def test_the_two_solvers_agree() -> None:
    """The whole point of the Poisson solver is that it is the direct solve made cheap.
    If they ever disagreed, the fast one would be quietly returning a different
    wavefront, so pin them together on a case with real structure and several wraps.
    """
    x, y = _grid()
    truth = 1.8 * x + 1.2 * y + 0.004 * (x**2 - y**2) + 1e-5 * x**3
    mask = _disc()
    wrapped_phase = np.angle(np.exp(1j * truth))

    direct = unwrap_2d_laplace(wrapped_phase, mask)
    iterative = unwrap_2d_poisson(wrapped_phase, mask)

    difference = (iterative - direct)[mask]
    assert np.abs(difference - difference.mean()).max() < 1e-6


def test_the_poisson_solver_reports_a_failure_to_converge() -> None:
    """A half-solved field still looks like a plausible wavefront, so running out of
    iterations has to raise rather than return one.
    """
    x, y = _grid()
    mask = _disc()
    wrapped_phase = np.angle(np.exp(1j * (2.0 * x + 1.4 * y)))

    with pytest.raises(RuntimeError, match="failed to unwrap"):
        unwrap_2d_poisson(wrapped_phase, mask, tolerance=1e-16, max_iterations=1)
