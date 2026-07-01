"""Tests for the PointingInstability OpticsModule (a sampled phase tilt)."""

import torch

from hologradpy.propagation.complex_amplitude import ComplexAmplitude, FieldGeometry
from hologradpy.propagation.pointing_instability import PointingInstability

GEOMETRY = FieldGeometry(
    resolution=(8, 8),
    pixel_size=torch.tensor([10e-6, 10e-6]),
    wavelength=torch.tensor(0.5e-6),
)


def _spot() -> ComplexAmplitude:
    field = torch.zeros(8, 8, dtype=torch.complex64)
    field[4, 4] = 1.0
    return ComplexAmplitude(field, GEOMETRY.wavelength, GEOMETRY.pixel_size)


def _uniform_field() -> ComplexAmplitude:
    field = torch.ones(8, 8, dtype=torch.complex64)
    return ComplexAmplitude(field, GEOMETRY.wavelength, GEOMETRY.pixel_size)


def test_tilt_is_zero_mean_gaussian():
    std = 2e-3
    module = PointingInstability(tilt_std=std, seed=0)
    samples = []
    for _ in range(3000):
        module(_uniform_field())
        samples.append(module.last_angle[0].item())  # sampled beam angle [rad]
    samples = torch.tensor(samples)
    assert samples.mean().abs() < 0.1 * std  # ~zero mean
    assert torch.isclose(samples.std(), torch.tensor(std), rtol=0.1)  # std matches


def test_preserves_type_and_geometry():
    out = PointingInstability(1e-3, seed=1)(_spot())
    assert isinstance(out, ComplexAmplitude)
    assert out.shape == (8, 8)
    assert torch.allclose(out.pixel_size, GEOMETRY.pixel_size.reshape(1, 2))


def test_tilt_is_phase_only():
    # The tilt is a unit-modulus phase ramp, so amplitudes are unchanged.
    field = _uniform_field()
    out = PointingInstability(5e-3, seed=2)(field)
    assert torch.allclose(out.abs(), field.abs(), atol=1e-5)


def test_adjoint_inverts_last_forward():
    field = _spot()
    module = PointingInstability(2e-3, seed=7)
    back = module.adjoint(module(field))
    assert torch.allclose(back, field, atol=1e-5)


def test_differentiable_with_respect_to_input():
    field = _spot()
    field.requires_grad_(True)
    PointingInstability(1e-3, seed=3)(field).abs().sum().backward()
    assert field.grad is not None and torch.isfinite(field.grad).all()


def test_reproducible_with_seed():
    first = PointingInstability(1e-3, seed=42)(_spot())
    second = PointingInstability(1e-3, seed=42)(_spot())
    assert torch.allclose(first, second)


def test_from_focal_shift_converts_to_angle():
    module = PointingInstability.from_focal_shift(
        focal_shift_std=250e-6, focal_length=0.25
    )
    assert module.tilt_std == (250e-6 / 0.25, 250e-6 / 0.25)


def test_recording_toggle_captures_sampled_angles():
    module = PointingInstability(1e-3, seed=0)
    # Nothing recorded by default.
    module(_uniform_field())
    assert module.angle_history.shape == (0, 2)
    # Enabling clears and records each forward's (angle_x, angle_y).
    module.record()
    for _ in range(3):
        module(_uniform_field())
    history = module.angle_history
    assert history.shape == (3, 2)
    torch.testing.assert_close(history[-1], torch.stack(module.last_angle))
    # Disabling stops recording but keeps the history.
    module.record(False)
    module(_uniform_field())
    assert module.angle_history.shape == (3, 2)


def test_record_samples_context_manager():
    module = PointingInstability(1e-3, seed=1)
    with module.record_samples():
        module(_uniform_field())
        module(_uniform_field())
    assert module.angle_history.shape == (2, 2)
    # Recording is off again after the block.
    module(_uniform_field())
    assert module.angle_history.shape == (2, 2)
