"""ZernikeSLM-specific behaviour.

ZernikeSLM is covered by the shared ND contract suite via the module registry;
this module adds tests for what makes it distinct: it learns per-wavelength
Zernike *coefficients* (not a per-pixel phase), reconstructs a per-wavelength
phase, and feeds gradients back to those coefficients.
"""

from __future__ import annotations

import pytest
import torch

from hologradpy.propagation.complex_amplitude import ComplexAmplitude
from hologradpy.propagation.virtual_slms.zernike_slm import ZernikeSLM

from .registry import ZERNIKE_RADIAL_ORDERS, make_field


pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

NUMBER_OF_COEFFICIENTS = ZERNIKE_RADIAL_ORDERS * (ZERNIKE_RADIAL_ORDERS + 1) // 2


class _FakeSLM:
    shape = (16, 16)
    pitch_um = (10.0, 10.0)
    phase_scaling = 1.0


def _field(shape, n_wavelengths):
    return make_field(shape, n_wavelengths)


def test_learns_coefficients_not_phase() -> None:
    """The learnable parameter is the Zernike coefficient tensor, shaped
    ``(n_wavelengths, n_coefficients)`` — not a per-pixel phase param."""
    module = ZernikeSLM(
        phase_scaling=1.0, number_of_radial_orders=ZERNIKE_RADIAL_ORDERS
    )
    module(_field((2, 16, 16), 2))

    params = dict(module.named_parameters())
    assert "zernike_coefficients" in params
    assert params["zernike_coefficients"].shape == (2, NUMBER_OF_COEFFICIENTS)
    assert params["zernike_coefficients"].requires_grad
    # No inherited per-pixel phase parameter from VirtualSLM.
    assert "phase" not in params


def test_per_wavelength_coefficients_give_distinct_phases() -> None:
    coefficients = torch.stack(
        [
            torch.zeros(NUMBER_OF_COEFFICIENTS),
            torch.linspace(0.1, 1.0, NUMBER_OF_COEFFICIENTS),
        ]
    )
    module = ZernikeSLM(
        phase_scaling=1.0,
        number_of_radial_orders=ZERNIKE_RADIAL_ORDERS,
        initial_coefficients=coefficients,
    )
    module(_field((2, 16, 16), 2))

    phase = module.get_phase()
    assert phase.shape == (2, 16, 16)
    # First wavelength has zero coefficients -> zero phase; second does not.
    assert phase[0].abs().max() < 1e-6
    assert (phase[0] - phase[1]).abs().max() > 1e-3


def test_one_dimensional_coefficients_broadcast_across_wavelengths() -> None:
    coefficients = torch.linspace(0.1, 1.0, NUMBER_OF_COEFFICIENTS)
    module = ZernikeSLM(
        phase_scaling=1.0,
        number_of_radial_orders=ZERNIKE_RADIAL_ORDERS,
        initial_coefficients=coefficients,
    )
    module(_field((3, 16, 16), 3))

    assert module.zernike_coefficients.shape == (3, NUMBER_OF_COEFFICIENTS)
    # Broadcast means every wavelength starts from the same coefficients.
    torch.testing.assert_close(
        module.zernike_coefficients[0], module.zernike_coefficients[2]
    )


def test_gradient_flows_to_coefficients() -> None:
    module = ZernikeSLM(
        phase_scaling=1.0, number_of_radial_orders=ZERNIKE_RADIAL_ORDERS
    )
    output = module(_field((2, 16, 16), 2))

    # Loss must be computed on the ComplexAmplitude wrapper (not ._data) so the
    # autograd graph is preserved.
    loss = (output.abs() ** 2).sum()
    loss.backward()

    grad = module.zernike_coefficients.grad
    assert grad is not None
    assert grad.shape == (2, NUMBER_OF_COEFFICIENTS)
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_bad_coefficient_shape_raises() -> None:
    module = ZernikeSLM(
        phase_scaling=1.0,
        number_of_radial_orders=ZERNIKE_RADIAL_ORDERS,
        initial_coefficients=torch.zeros(NUMBER_OF_COEFFICIENTS + 5),
    )
    with pytest.raises(ValueError):
        module(_field((2, 16, 16), 2))


def test_set_phase_not_supported() -> None:
    module = ZernikeSLM(
        phase_scaling=1.0, number_of_radial_orders=ZERNIKE_RADIAL_ORDERS
    )
    with pytest.raises(NotImplementedError):
        module.set_phase(torch.zeros(16, 16))


def test_from_slm_constructs() -> None:
    module = ZernikeSLM.from_slm(
        _FakeSLM(), number_of_radial_orders=ZERNIKE_RADIAL_ORDERS
    )
    output = module(_field((16, 16), 1))
    assert output.shape == (16, 16)
    assert module.phase_scaling == _FakeSLM.phase_scaling
