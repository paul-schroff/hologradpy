"""Tests for the linear error propagation of a fit's covariance."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from hologradpy.analysis.uncertainties import propagate_covariance, standard_deviation


@dataclass(frozen=True, eq=False)
class Line:
    """A fitted line, ``slope * x + intercept``. It matches the Fit protocol without
    inheriting from it.
    """

    parameters: np.ndarray
    covariance: np.ndarray

    @property
    def slope(self) -> float:
        return float(self.parameters[0])

    @property
    def intercept(self) -> float:
        return float(self.parameters[1])


def test_a_linear_function_propagates_exactly() -> None:
    """For f(p) = A p, the covariance of f is A C A^T."""
    matrix = np.array([[1.0, 2.0, 0.0], [0.5, -1.0, 3.0]])
    covariance = np.array([[4.0, 1.0, 0.0], [1.0, 2.0, 0.5], [0.0, 0.5, 1.0]])
    parameters = np.array([1.0, -2.0, 0.5])

    propagated = propagate_covariance(lambda p: matrix @ p, parameters, covariance)

    np.testing.assert_allclose(propagated, matrix @ covariance @ matrix.T, rtol=1e-9)


def test_a_parameter_without_spread_adds_nothing() -> None:
    covariance = np.diag([1.0, 0.0])

    propagated = propagate_covariance(
        lambda p: p[0] + p[1] ** 2, np.array([1.0, 0.0]), covariance
    )

    np.testing.assert_allclose(propagated, [[1.0]])


def test_a_covariance_that_could_not_be_estimated_gives_nan() -> None:
    """curve_fit fills the covariance with infinities when it can't estimate it."""
    covariance = np.full((2, 2), np.inf)

    propagated = propagate_covariance(lambda p: p.sum(), np.zeros(2), covariance)

    assert propagated.shape == (1, 1)
    assert np.isnan(propagated).all()


def test_a_value_that_repeats_propagates_across_its_wrap() -> None:
    """A value wrapped into (-90, 90] jumps by 180 where it wraps. With the period, its
    variance is that of the value without the wrap.
    """

    def wrapped(p: np.ndarray) -> float:
        return 90.0 - (90.0 - p[0]) % 180.0

    propagated = propagate_covariance(
        wrapped, np.array([90.0]), np.array([[0.01]]), period=180.0
    )

    np.testing.assert_allclose(propagated, [[0.01]])


def test_the_standard_deviation_of_a_quantity_derived_from_a_fit() -> None:
    """The zero crossing of a line, -intercept / slope, against its known propagation."""
    covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
    line = Line(np.array([2.0, 3.0]), covariance)

    std = standard_deviation(line, lambda line: -line.intercept / line.slope)

    # The gradient of -b / m over (m, b) is (b / m^2, -1 / m).
    gradient = np.array([3.0 / 4.0, -1.0 / 2.0])
    assert std == pytest.approx(math.sqrt(gradient @ covariance @ gradient), rel=1e-6)
