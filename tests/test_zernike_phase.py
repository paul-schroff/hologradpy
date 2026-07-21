"""ZernikePhase-specific behaviour.

The phase plate is covered by the shared ND contract + square-propagator
adjoint suites via the registry; this module adds the distinguishing tests: it
learns per-wavelength Zernike coefficients, builds distinct phases per
wavelength, and backpropagates to the coefficients. It applies a pure phase
(no SLM modulo wrapping), so it should conserve power.
"""
from __future__ import annotations

import pytest
import torch

from hologradpy.optics.modules.diagonal_elements import ZernikePhase

from .registry import ZERNIKE_RADIAL_ORDERS, make_field


pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

NUMBER_OF_COEFFICIENTS = (
    ZERNIKE_RADIAL_ORDERS * (ZERNIKE_RADIAL_ORDERS + 1) // 2
)


def test_learns_per_wavelength_coefficients() -> None:
    module = ZernikePhase(
        number_of_radial_orders=ZERNIKE_RADIAL_ORDERS
    )
    module(make_field((2, 16, 16), 2))

    params = dict(module.named_parameters())
    assert "zernike_coefficients" in params
    assert params["zernike_coefficients"].shape == (
        2,
        NUMBER_OF_COEFFICIENTS,
    )
    assert params["zernike_coefficients"].requires_grad


def test_per_wavelength_coefficients_give_distinct_phases() -> None:
    coefficients = torch.stack(
        [
            torch.zeros(NUMBER_OF_COEFFICIENTS),
            torch.linspace(0.1, 1.0, NUMBER_OF_COEFFICIENTS),
        ]
    )
    module = ZernikePhase(
        number_of_radial_orders=ZERNIKE_RADIAL_ORDERS,
        initial_coefficients=coefficients,
    )
    module(make_field((2, 16, 16), 2))

    phase = module.get_phase()
    assert phase.shape == (2, 16, 16)
    assert phase[0].abs().max() < 1e-6  # zero coefficients -> zero phase
    assert (phase[0] - phase[1]).abs().max() > 1e-3


def test_pure_phase_conserves_power() -> None:
    """A phase-only element leaves the field amplitude unchanged."""
    module = ZernikePhase(
        number_of_radial_orders=ZERNIKE_RADIAL_ORDERS,
        initial_coefficients=torch.linspace(0.1, 1.0, NUMBER_OF_COEFFICIENTS),
    )
    field = make_field((2, 16, 16), 2)
    output = module(field)

    torch.testing.assert_close(output.amplitude, field.amplitude)


def test_gradient_flows_to_coefficients() -> None:
    module = ZernikePhase(
        number_of_radial_orders=ZERNIKE_RADIAL_ORDERS
    )
    output = module(make_field((2, 16, 16), 2))

    (output.abs() ** 2).sum().backward()

    grad = module.zernike_coefficients.grad
    assert grad is not None
    assert grad.shape == (2, NUMBER_OF_COEFFICIENTS)
    assert torch.isfinite(grad).all()


def test_bad_coefficient_shape_raises() -> None:
    module = ZernikePhase(
        number_of_radial_orders=ZERNIKE_RADIAL_ORDERS,
        initial_coefficients=torch.zeros(NUMBER_OF_COEFFICIENTS + 5),
    )
    with pytest.raises(ValueError):
        module(make_field((2, 16, 16), 2))
