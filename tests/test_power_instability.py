"""Tests for the PowerInstability OpticsModule (a sampled optical-power scale)."""

import torch

from hologradpy.propagation.complex_amplitude import ComplexAmplitude, FieldGeometry
from hologradpy.propagation.power_instability import PowerInstability

GEOMETRY = FieldGeometry(
    resolution=(8, 8),
    pixel_size=torch.tensor([10e-6, 10e-6]),
    wavelength=torch.tensor(0.5e-6),
)


def _uniform_field() -> ComplexAmplitude:
    field = torch.ones(8, 8, dtype=torch.complex64)
    return ComplexAmplitude(field, GEOMETRY.wavelength, GEOMETRY.pixel_size)


def test_power_factor_is_mean_one_gaussian():
    std = 0.1
    module = PowerInstability(std, seed=0)
    samples = []
    for _ in range(3000):
        module(_uniform_field())
        samples.append(module.last_power_factor.item())
    samples = torch.tensor(samples)
    assert (samples.mean() - 1.0).abs() < 0.1 * std  # ~mean 1
    assert torch.isclose(samples.std(), torch.tensor(std), rtol=0.1)  # std matches


def test_scales_power_by_factor():
    field = _uniform_field()
    module = PowerInstability(0.2, seed=3)
    out = module(field)
    torch.testing.assert_close(
        out.power(),
        field.power() * module.last_power_factor.double(),
        rtol=1e-4,
        atol=0.0,
    )


def test_preserves_type_and_geometry():
    out = PowerInstability(0.1, seed=1)(_uniform_field())
    assert isinstance(out, ComplexAmplitude)
    assert out.shape == (8, 8)
    assert torch.allclose(out.pixel_size, GEOMETRY.pixel_size.reshape(1, 2))


def test_adjoint_reapplies_last_factor():
    # A real scalar scaling is self-adjoint: the adjoint re-applies sqrt(factor).
    field = _uniform_field()
    module = PowerInstability(0.3, seed=7)
    module(field)  # sets the last factor
    back = module.adjoint(field)
    torch.testing.assert_close(
        back.as_tensor(),
        field.as_tensor() * module.last_power_factor.sqrt(),
    )


def test_power_factor_clamped_non_negative():
    # A huge std would draw negatives; they are clamped so power stays >= 0.
    module = PowerInstability(5.0, seed=1)
    for _ in range(200):
        module(_uniform_field())
        assert module.last_power_factor.item() >= 0.0


def test_differentiable_with_respect_to_input():
    field = _uniform_field()
    field.requires_grad_(True)
    PowerInstability(0.1, seed=3)(field).abs().sum().backward()
    assert field.grad is not None and torch.isfinite(field.grad).all()


def test_reproducible_with_seed():
    first = PowerInstability(0.2, seed=42)(_uniform_field())
    second = PowerInstability(0.2, seed=42)(_uniform_field())
    assert torch.allclose(first.as_tensor(), second.as_tensor())


def test_recording_toggle_captures_power_factors():
    module = PowerInstability(0.1, seed=0)
    # Nothing recorded by default.
    module(_uniform_field())
    assert module.power_factor_history.shape == (0,)
    # Enabling clears and records each forward's factor.
    module.record()
    for _ in range(3):
        module(_uniform_field())
    history = module.power_factor_history
    assert history.shape == (3,)
    torch.testing.assert_close(history[-1], module.last_power_factor)
    # Disabling stops recording but keeps the history.
    module.record(False)
    module(_uniform_field())
    assert module.power_factor_history.shape == (3,)


def test_record_samples_context_manager():
    module = PowerInstability(0.1, seed=1)
    with module.record_samples():
        module(_uniform_field())
        module(_uniform_field())
    assert module.power_factor_history.shape == (2,)
    # Recording is off again after the block.
    module(_uniform_field())
    assert module.power_factor_history.shape == (2,)
