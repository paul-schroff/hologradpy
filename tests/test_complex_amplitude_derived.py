"""Derived real quantities of ComplexAmplitude: intensity / amplitude / phase.

These must be **plain real ``torch.Tensor``s** (not ``ComplexAmplitude`` — an
intensity is not a complex field), have the correct physical values, and stay
attached to the autograd graph so they can drive optimization. The graph may
live either on the inner ``_data`` (a field built directly from graph-carrying
data) or on the wrapper (a field produced by an ``OpticsModule``); both paths
are covered.
"""
from __future__ import annotations

import pytest
import torch

from hologradpy.propagation.complex_amplitude import ComplexAmplitude

from .registry import make_field
from .registry import MODULE_FACTORIES, RANK_CASES


pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

DERIVED = ["intensity", "amplitude", "phase"]


@pytest.mark.parametrize("name", DERIVED)
def test_returns_plain_real_tensor(name: str) -> None:
    field = make_field((2, 16, 16), 2)
    value = getattr(field, name)

    # Exactly a torch.Tensor, not the ComplexAmplitude subclass.
    assert type(value) is torch.Tensor
    assert not value.is_complex()
    assert value.shape == field.shape


def test_values_match_reference() -> None:
    field = make_field((2, 16, 16), 2)
    raw = field._data

    torch.testing.assert_close(field.amplitude, raw.abs())
    torch.testing.assert_close(field.intensity, raw.abs() ** 2)
    torch.testing.assert_close(field.phase, torch.angle(raw))


def test_gradient_flows_when_data_carries_graph() -> None:
    """Field built directly from a leaf parameter: graph is on ``_data``."""
    coefficient = torch.zeros(16, 16, requires_grad=True)
    field = ComplexAmplitude(
        torch.exp(1j * coefficient), 800e-9, (10e-6, 10e-6)
    )

    field.intensity.sum().backward()

    assert coefficient.grad is not None
    assert torch.isfinite(coefficient.grad).all()


@pytest.mark.parametrize("name", DERIVED)
def test_gradient_flows_when_wrapper_carries_graph(name: str) -> None:
    """Field produced by an OpticsModule forward: graph is on the wrapper, so
    ``_data`` is detached and only ``as_tensor()`` preserves the gradient."""
    module = MODULE_FACTORIES["ZernikeSLM"]()
    field = make_field(*RANK_CASES["3d"])

    output = module(field)
    value = getattr(output, name)

    assert value.requires_grad
    value.sum().backward()

    grad = module.zernike_coefficients.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_numpy_is_detached_copy() -> None:
    field = make_field((16, 16), 1)
    array = field.numpy()
    assert array.shape == (16, 16)
    # numpy() must not require a grad-enabled tensor.
    assert not isinstance(array, torch.Tensor)
