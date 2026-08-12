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

from hologradpy.optics.complex_amplitude import ComplexAmplitude

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
    field = ComplexAmplitude(torch.exp(1j * coefficient), 800e-9, (10e-6, 10e-6))

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


def test_intensity_gradient_conjugation_is_correct() -> None:
    """The dispatch backward must apply the complex conjugation.

    Regression for a dropped conjugate/negative bit in the wrapper's
    ``__torch_dispatch__``: ``_make_wrapper_subclass`` does not carry those lazy
    bits, so a conj view produced by complex autograd's ``mul`` backward was
    re-wrapped without them, silently dropping the conjugation. ``|c * exp(i *
    phase)|**2 == |c|**2`` is independent of ``phase``, so its gradient must be
    zero; the bug instead produced a spurious ``2 * Im(c**2)``.
    """
    constant = torch.tensor(
        [[1 + 2j, 3 - 1j], [0.5 + 0.7j, -1 + 0.3j]], dtype=torch.complex128
    )
    phase = torch.tensor(
        [[0.3, -0.5], [1.1, 0.2]], dtype=torch.float64, requires_grad=True
    )

    field = ComplexAmplitude(constant, 800e-9, (10e-6, 10e-6)) * torch.exp(1j * phase)
    field.intensity.sum().backward()

    torch.testing.assert_close(phase.grad, torch.zeros_like(phase), atol=1e-9, rtol=0)


def test_gradient_matches_plain_torch_through_complex_multiply() -> None:
    """A gradient through a complex dispatch multiply must match plain torch.

    The wrapper autograd previously dropped the conjugation for complex operands
    (correct only when the operand was real-valued), corrupting the gradient of
    any field produced by a complex per-pixel multiply (e.g. a ``VirtualSLM`` or
    ``PixelwiseSLMField``)."""
    constant = torch.tensor(
        [[1 + 2j, 3 - 1j], [0.5 + 0.7j, -1 + 0.3j]], dtype=torch.complex128
    )
    weight = torch.tensor(
        [[1 + 0.5j, -2 - 1j], [0.5, 1.5 - 0.7j]], dtype=torch.complex128
    )
    start = torch.tensor([[0.3, -0.5], [1.1, 0.2]], dtype=torch.float64)

    phase = start.clone().requires_grad_(True)
    field = ComplexAmplitude(constant, 800e-9, (10e-6, 10e-6)) * torch.exp(1j * phase)
    (field.as_tensor() * weight).real.sum().backward()

    reference = start.clone().requires_grad_(True)
    (constant * torch.exp(1j * reference) * weight).real.sum().backward()

    torch.testing.assert_close(phase.grad, reference.grad)


def test_numpy_is_detached_copy() -> None:
    field = make_field((16, 16), 1)
    array = field.numpy()
    assert array.shape == (16, 16)
    # numpy() must not require a grad-enabled tensor.
    assert not isinstance(array, torch.Tensor)
