"""Tests for the semi-analytical Fourier transform.

The point of the method is that a quadratic phase never gets sampled: it is carried
analytically, so only the smooth residual has to be resolved. These pin that the
transform is the one it claims to be, that the analytic bookkeeping matches the paper,
and that it reaches sample points a chirp-z cannot.
"""

from __future__ import annotations

import pytest
import torch

from hologradpy.fourier_transforms import (
    ifft_2d,
    SemiAnalyticalFourierTransform,
    transformed_curvature,
)

RESOLUTION = (24, 24)
SEPARABLE = (0.05, 0.0, 0.12)
WITH_CROSS = (0.05, 0.03, 0.12)


def _grid(resolution=RESOLUTION):
    rows = torch.arange(resolution[0], dtype=torch.float64) - resolution[0] // 2
    columns = torch.arange(resolution[1], dtype=torch.float64) - resolution[1] // 2
    return torch.meshgrid(columns, rows, indexing="xy")


def _compact_residual(resolution=RESOLUTION):
    """A field that has died away by the edge, so the array holds all of it."""
    x, y = _grid(resolution)
    return torch.exp(-(x**2 + y**2) / (2 * (resolution[0] / 16) ** 2)).to(
        torch.complex128
    )


def _direct(residual, curvature, frequencies):
    """The sum the transform is a fast way of computing."""
    x, y = _grid(residual.shape)
    curvature_x, cross, curvature_y = curvature
    whole = residual * torch.exp(
        1j * (curvature_x * x**2 + cross * x * y + curvature_y * y**2)
    )
    values = [
        (whole * torch.exp(-1j * (kx * x + ky * y))).sum()
        for kx, ky in zip(frequencies[0], frequencies[1])
    ]
    return torch.stack(values).reshape(residual.shape)


@pytest.mark.parametrize(
    "curvature", [SEPARABLE, WITH_CROSS], ids=["separable", "cross"]
)
def test_it_is_the_transform_of_the_field_times_the_chirp(curvature) -> None:
    """Exact, to floating point, against the sum written out."""
    residual = _compact_residual()
    transform = SemiAnalyticalFourierTransform(RESOLUTION, curvature)

    got = transform(residual)
    reference = _direct(residual, curvature, transform.frequencies)

    assert (got - reference).abs().max() / reference.abs().max() < 1e-12


def test_the_error_is_the_field_running_off_the_edge() -> None:
    """The one approximation in the method, and it belongs to the caller.

    The convolution is circular, so whatever the residual has left at the edge wraps.
    Nothing else about the transform is approximate.
    """
    transform = SemiAnalyticalFourierTransform(RESOLUTION, SEPARABLE)
    errors = []
    for width in (6, 10, 16):
        x, y = _grid()
        residual = torch.exp(
            -(x**2 + y**2) / (2 * (RESOLUTION[0] / width) ** 2)
        ).to(torch.complex128)
        reference = _direct(residual, SEPARABLE, transform.frequencies)
        got = transform(residual)
        edge = float(residual.abs()[0, RESOLUTION[1] // 2])
        errors.append(
            (edge, float((got - reference).abs().max() / reference.abs().max()))
        )

    for edge, error in errors:
        assert error < 10 * edge + 1e-13
    assert errors[-1][1] < 1e-12


def test_the_transformed_curvature_is_minus_a_inverse_over_four() -> None:
    """``A -> -A^-1 / 4``, which is where the sampling saving comes from."""
    curvature_x, cross, curvature_y = WITH_CROSS
    matrix = torch.tensor(
        [[curvature_x, cross / 2], [cross / 2, curvature_y]], dtype=torch.float64
    )
    wanted = -torch.linalg.inv(matrix) / 4

    got_x, got_cross, got_y = transformed_curvature(WITH_CROSS)

    assert got_x == pytest.approx(float(wanted[0, 0]))
    assert got_y == pytest.approx(float(wanted[1, 1]))
    assert got_cross == pytest.approx(float(2 * wanted[0, 1]))


def test_a_strong_curvature_becomes_a_weak_one() -> None:
    """The claim the method rests on, stated as a test."""
    weak = transformed_curvature((5.0, 0.0, 5.0))
    assert abs(weak[0]) < 0.06 and abs(weak[2]) < 0.06


def test_transforming_twice_returns_the_curvature() -> None:
    """An involution: ``-(-4A)/4`` is ``A``, so the form comes back as it went in."""
    there = transformed_curvature(WITH_CROSS)
    back = transformed_curvature(there)

    for original, returned in zip(WITH_CROSS, back):
        assert returned == pytest.approx(original)


def test_a_degenerate_quadratic_form_is_refused() -> None:
    """``Dx Dy == C**2 / 4`` is a one dimensional phase, which has no 2D transform."""
    with pytest.raises(ValueError, match="one dimensional"):
        transformed_curvature((0.05, 0.2, 0.2))

    with pytest.raises(ValueError, match="one dimensional"):
        SemiAnalyticalFourierTransform(RESOLUTION, (0.05, 0.2, 0.2))


def test_the_cross_term_shears_the_sample_points() -> None:
    """What a chirp-z cannot do: its window is rectangular, this one is not."""
    rectangular = SemiAnalyticalFourierTransform(RESOLUTION, SEPARABLE).frequencies
    sheared = SemiAnalyticalFourierTransform(RESOLUTION, WITH_CROSS).frequencies

    # Along a row of the lattice, x moves and y should stay put without a cross term.
    row = slice(0, RESOLUTION[1])
    assert rectangular[1][row].std() == pytest.approx(0.0, abs=1e-12)
    assert sheared[1][row].std() > 1e-3


def test_the_adjoint_is_the_conjugate_transpose() -> None:
    """So the transform composes into a model that has to differentiate through it."""
    transform = SemiAnalyticalFourierTransform(RESOLUTION, WITH_CROSS)
    generator = torch.Generator().manual_seed(0)
    a = torch.randn(RESOLUTION, generator=generator, dtype=torch.complex128)
    b = torch.randn(RESOLUTION, generator=generator, dtype=torch.complex128)

    left = (transform(a).conj() * b).sum()
    right = (a.conj() * transform.adjoint(b)).sum()

    assert abs(left - right) / abs(left) < 1e-12


def test_it_carries_a_batch() -> None:
    """Leading dimensions come through untouched, as every other transform allows."""
    transform = SemiAnalyticalFourierTransform(RESOLUTION, SEPARABLE)
    residual = _compact_residual()
    batched = torch.stack([residual, 2 * residual])

    got = transform(batched)

    assert got.shape == (2, *RESOLUTION)
    assert (got[1] - 2 * got[0]).abs().max() < 1e-12


def test_the_margin_says_how_far_the_lattice_reaches() -> None:
    """Below one, every sample point is a frequency of its own."""
    transform = SemiAnalyticalFourierTransform(RESOLUTION, (0.01, 0.0, 0.005))

    margin_x, margin_y = transform.sampling_margin()

    assert margin_x < 1.0 and margin_y < 1.0
    # Twice the curvature is twice the reach, per axis independently.
    doubled = SemiAnalyticalFourierTransform(RESOLUTION, (0.02, 0.0, 0.005))
    assert doubled.sampling_margin()[0] == pytest.approx(2 * margin_x)
    assert doubled.sampling_margin()[1] == pytest.approx(margin_y)


def test_a_margin_over_one_means_the_lattice_repeats_itself() -> None:
    """What the margin is warning about, demonstrated.

    The input sits on integer samples, so the sum is ``2 pi`` periodic in ``k``. A
    lattice reaching past ``pi`` walks into the next period and takes frequencies it
    already has, and those outputs are equal rather than merely similar.
    """
    # Spacing is 2 * curvature, so points eight apart differ by exactly 2 pi.
    step = torch.pi / 8
    transform = SemiAnalyticalFourierTransform(RESOLUTION, (step, 0.0, step))
    assert transform.sampling_margin()[0] > 1.0

    got = transform(_compact_residual())
    frequencies = transform.frequencies[0].reshape(RESOLUTION)
    row = RESOLUTION[0] // 2

    for column in (2, 4, 5):
        wrapped = column + 8
        difference = float(frequencies[row, wrapped] - frequencies[row, column])
        assert difference == pytest.approx(2 * torch.pi)
        assert got[row, wrapped] == pytest.approx(got[row, column], rel=1e-9)


@pytest.mark.parametrize(
    "curvature", [SEPARABLE, WITH_CROSS], ids=["separable", "cross"]
)
def test_the_inverse_sums_the_other_way(curvature) -> None:
    """The leg that comes back from a spectrum to a plane.

    Completing the square leaves ``(x + m)`` where the forward has ``(x - m)``, so
    the same convolution is read the other way up and nothing else changes.
    """
    residual = _compact_residual()
    transform = SemiAnalyticalFourierTransform(RESOLUTION, curvature, inverse=True)

    got = transform(residual)

    x, y = _grid(RESOLUTION)
    curvature_x, cross, curvature_y = curvature
    whole = residual * torch.exp(
        1j * (curvature_x * x**2 + cross * x * y + curvature_y * y**2)
    )
    wanted = torch.stack([
        (whole * torch.exp(1j * (kx * x + ky * y))).sum()
        for kx, ky in zip(transform.frequencies[0], transform.frequencies[1])
    ]).reshape(RESOLUTION)

    assert (got - wanted).abs().max() / wanted.abs().max() < 1e-12


def test_the_inverse_has_its_own_adjoint() -> None:
    """It reads the convolution elsewhere, so its transpose is not the forward's."""
    transform = SemiAnalyticalFourierTransform(
        RESOLUTION, WITH_CROSS, inverse=True
    )
    generator = torch.Generator().manual_seed(0)
    a = torch.randn(RESOLUTION, generator=generator, dtype=torch.complex128)
    b = torch.randn(RESOLUTION, generator=generator, dtype=torch.complex128)

    left = (transform(a).conj() * b).sum()
    right = (a.conj() * transform.adjoint(b)).sum()

    assert abs(left - right) / abs(left) < 1e-12


def test_the_inverse_is_the_unnormalised_inverse_transform() -> None:
    """With the chirp divided out first, it is a plain inverse DFT.

    Which is what lets a propagator use it as the leg back from a spectrum: the only
    thing it owes the caller afterwards is the transform size it did not divide by.
    """
    length = RESOLUTION[0]
    curvature = -torch.pi / length
    transform = SemiAnalyticalFourierTransform(
        (length, length), (curvature, 0.0, curvature), inverse=True
    )
    generator = torch.Generator().manual_seed(1)
    spectrum = torch.randn(length, length, generator=generator, dtype=torch.complex128)

    x, y = _grid((length, length))
    got = transform(spectrum * torch.exp(-1j * curvature * (x**2 + y**2)))
    wanted = ifft_2d(spectrum) * length * length

    assert (got - wanted).abs().max() / wanted.abs().max() < 1e-12
