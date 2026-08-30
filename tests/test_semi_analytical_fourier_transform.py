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
