"""Uncertainties of quantities derived from fitted parameters, by linear error
propagation of their covariance.
"""

from __future__ import annotations

import math
from dataclasses import Field, replace
from typing import Any, Callable, ClassVar, Protocol, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

_STEP_FRACTION = 1e-3
_MIN_RELATIVE_STEP = 1e-8


class Fit(Protocol):
    """A dataclass that holds a fit's parameters and their covariance."""

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

    @property
    def parameters(self) -> NDArray[np.float64]: ...

    @property
    def covariance(self) -> NDArray[np.float64]: ...


FitT = TypeVar("FitT", bound=Fit)


def standard_deviation(
    fit: FitT, quantity: Callable[[FitT], float], period: float | None = None
) -> float:
    """Return the standard deviation of ``quantity(fit)``.

    The covariance of the fit's parameters is propagated linearly to the quantity.
    ``quantity`` is any number derived from a fit, for example
    ``standard_deviation(fit, lambda fit: fit.beam.mean_radius)``. It is evaluated on
    copies of ``fit`` with stepped parameters.

    Args:
        fit: The fit.
        quantity: The quantity, as a function of the fit.
        period: The period of a quantity that repeats, such as 180 for the angle of an
            axis in degrees. None for a quantity that doesn't repeat.

    Returns:
        float: The standard deviation.

        - If the covariance holds an infinity or a NaN, return NaN.
    """
    variance = propagate_covariance(
        lambda parameters: quantity(replace(fit, parameters=parameters)),
        fit.parameters,
        fit.covariance,
        period,
    )[0, 0]
    return math.sqrt(variance) if variance >= 0 else math.nan


def propagate_covariance(
    function: Callable[[NDArray[np.float64]], ArrayLike],
    parameters: ArrayLike,
    covariance: ArrayLike,
    period: float | None = None,
) -> NDArray[np.float64]:
    """Return the covariance of ``function(parameters)`` by linear error propagation.

    The result is ``J C J^T``. ``C`` is ``covariance``, and ``J`` is the Jacobian of
    ``function`` at ``parameters``. Central differences give ``J``.

    Args:
        function: The values, as a function of the parameters. It returns one value or
            an array of values.
        parameters: The fitted parameters.
        covariance: The covariance of the parameters.
        period: The period of values that repeat, such as 2 pi for a phase. The
            differences of the steps wrap into (-period / 2, period / 2]. None for
            values that don't repeat.

    Returns:
        NDArray[np.float64]: The ``(n, n)`` covariance of the ``n`` values.

        - If ``covariance`` holds an infinity or a NaN, return NaNs.
    """
    parameters = np.asarray(parameters, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    size = np.size(function(parameters))
    if not np.isfinite(covariance).all():
        return np.full((size, size), np.nan)

    jacobian = np.zeros((size, parameters.size))
    deviations = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    for index, deviation in enumerate(deviations):
        if deviation == 0.0:
            continue
        step = max(
            _STEP_FRACTION * deviation, _MIN_RELATIVE_STEP * abs(parameters[index])
        )
        forward, backward = parameters.copy(), parameters.copy()
        forward[index] += step
        backward[index] -= step
        difference = np.ravel(function(forward)) - np.ravel(function(backward))
        if period is not None:
            difference = period / 2 - (period / 2 - difference) % period
        jacobian[:, index] = difference / (forward[index] - backward[index])
    return jacobian @ covariance @ jacobian.T
