"""Tests for the wrap-safe wavefront error metrics.

The trap these exist to avoid: a recovered field's phase comes from
``numpy.angle`` and is therefore wrapped into ``(-pi, pi]``, while an aberration
of any size is not. Subtracting one from the other and fitting a plane, which is
what ``remove_tilt`` followed by ``rms_phase`` does, is skewed by the 2 pi
discontinuities. Scoring a *perfect* recovery that way returned 0.57 instead of
0, which silently corrupted every comparison built on it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hologradpy.analysis import error_metrics as em
from hologradpy.analysis.error_metrics import (
    remove_linear_phase,
    strehl_amplitude,
    wavefront_residual,
    wavefront_rms,
)

RESOLUTION = (192, 256)


def _aberration() -> np.ndarray:
    """A smooth aberration spanning several radians, so it genuinely wraps."""
    rows, columns = np.indices(RESOLUTION)
    x = (columns - RESOLUTION[1] / 2) / (RESOLUTION[1] / 2)
    y = (rows - RESOLUTION[0] / 2) / (RESOLUTION[0] / 2)
    return 4.5 * (x**2 - y**2) + 2.6 * x * y + 1.4 * x - 1.8 * y**3


def _mask() -> np.ndarray:
    rows, columns = np.indices(RESOLUTION)
    x = (columns - RESOLUTION[1] / 2) / (RESOLUTION[1] / 2)
    y = (rows - RESOLUTION[0] / 2) / (RESOLUTION[0] / 2)
    return x**2 + y**2 < 0.8**2


def _tilt(cycles_x: float, cycles_y: float) -> np.ndarray:
    rows, columns = np.indices(RESOLUTION)
    return 2 * np.pi * (
        cycles_x * columns / RESOLUTION[1] + cycles_y * rows / RESOLUTION[0]
    )


def test_the_aberration_used_here_really_does_wrap() -> None:
    """Guards the premise: a metric that is wrap-safe is only interesting if the
    test case wraps."""
    aberration = _aberration()[_mask()]
    assert np.ptp(aberration) > 2 * np.pi


def test_a_perfect_recovery_scores_zero_even_when_wrapped() -> None:
    """The case the previous metric failed, returning 0.57 for a perfect fit."""
    aberration, mask = _aberration(), _mask()
    wrapped = np.angle(np.exp(1j * aberration))

    for candidate in (aberration, wrapped):
        assert wavefront_residual(
            candidate, aberration, mask
        ) == pytest.approx(0, abs=1e-6)


def test_no_correction_scores_one() -> None:
    """Doing nothing must leave exactly what was there."""
    aberration, mask = _aberration(), _mask()
    assert wavefront_residual(
        np.zeros(RESOLUTION), aberration, mask
    ) == pytest.approx(1.0, rel=1e-6)


def test_the_sign_ambiguity_is_forgiven() -> None:
    """Intensity-only sensing recovers a wavefront up to a conjugate, so the
    global sign is not observable and must not be charged."""
    aberration, mask = _aberration(), _mask()
    assert wavefront_residual(
        -aberration, aberration, mask
    ) == pytest.approx(0, abs=1e-6)
    assert wavefront_residual(
        -aberration, aberration, mask, allow_sign_flip=False
    ) > 1.0


@pytest.mark.parametrize("cycles", [0.4, 3.0, 25.0])
def test_tilt_is_not_charged_as_aberration(cycles) -> None:
    """A ramp displaces the focus rather than degrading it.

    Sub-bin ramps are the awkward case: locating the tilt as a Fourier peak
    fails there, since the aperture's own spectrum is wider than the shift.
    """
    aberration, mask = _aberration(), _mask()
    recovered = np.angle(np.exp(1j * (aberration + _tilt(cycles, 0.6 * cycles))))
    assert wavefront_residual(recovered, aberration, mask) == pytest.approx(0, abs=1e-3)


def test_recovering_only_tilt_buys_nothing() -> None:
    """The complement: a fit that returns pure tilt has achieved nothing, and
    must not be flattered by how large that tilt looks."""
    aberration, mask = _aberration(), _mask()
    assert wavefront_residual(
        _tilt(3.0, 2.0), aberration, mask
    ) == pytest.approx(1.0, rel=1e-3)


def test_the_reference_is_not_inflated_by_tilt() -> None:
    """Tilt is discounted on both sides of the ratio, not just the residual."""
    aberration, mask = _aberration(), _mask()
    assert wavefront_rms(aberration, mask) == pytest.approx(
        wavefront_rms(aberration + _tilt(4.0, 2.0), mask), rel=1e-6
    )


def test_partial_recovery_lands_between() -> None:
    """Monotonic in how much was recovered, which is what ranking needs.

    The values are Marechal equivalents rather than exact ratios, so they are
    close to but not exactly the recovered fraction.
    """
    aberration, mask = _aberration(), _mask()
    half = wavefront_residual(0.5 * aberration, aberration, mask)
    quarter = wavefront_residual(0.25 * aberration, aberration, mask)
    assert 0.4 < half < 0.6
    assert 0.65 < quarter < 0.85
    assert half < quarter


def test_strehl_amplitude_is_one_for_a_flat_error_and_falls_with_it() -> None:
    mask = _mask()
    assert strehl_amplitude(np.zeros(RESOLUTION), mask) == pytest.approx(1.0, rel=1e-9)
    assert strehl_amplitude(0.5 * _aberration(), mask) < strehl_amplitude(
        0.2 * _aberration(), mask
    )


def test_strehl_amplitude_squares_to_the_intensity_strehl() -> None:
    """The reason for the name. For Gaussian phase the intensity Strehl is
    ``exp(-sigma ** 2)``, and this returns its square root."""
    rng = np.random.default_rng(0)
    mask = np.ones((256, 256), dtype=bool)
    sigma = 0.5
    phi = rng.normal(0.0, sigma, size=(256, 256))

    amplitude = strehl_amplitude(phi, mask, remove_ramp=False)
    assert amplitude == pytest.approx(np.exp(-(sigma**2) / 2), rel=2e-3)
    assert amplitude**2 == pytest.approx(np.exp(-(sigma**2)), rel=2e-3)


def test_every_metric_agrees_between_numpy_and_torch() -> None:
    """The point of dispatching through ``array_namespace``.

    Covers the intensity metrics as well as the wavefront ones, since the whole module
    is backend-agnostic and a torch-only regression in any of them would otherwise
    surface only in a calibration.
    """
    rng = np.random.default_rng(0)
    aberration, mask = _aberration(), _mask()
    i_target = rng.uniform(0.1, 1.0, size=RESOLUTION)
    i_out = i_target + rng.normal(0.0, 0.02, size=RESOLUTION)
    phase = rng.uniform(-np.pi, np.pi, size=RESOLUTION)

    def as_torch(array):
        return torch.as_tensor(array, dtype=torch.float64)

    torch_mask = torch.as_tensor(mask)

    scalar_cases = {
        "eff": (
            lambda: em.efficiency(mask, i_out),
            lambda: em.efficiency(torch_mask, as_torch(i_out)),
        ),
        "rms": (
            lambda: em.rms(mask, i_target, i_out),
            lambda: em.rms(torch_mask, as_torch(i_target), as_torch(i_out)),
        ),
        "psnr": (
            lambda: em.psnr(mask, i_target, i_out),
            lambda: em.psnr(torch_mask, as_torch(i_target), as_torch(i_out)),
        ),
        "rms_phase": (
            lambda: em.rms_phase(aberration),
            lambda: em.rms_phase(as_torch(aberration)),
        ),
        "fidelity": (
            lambda: em.fidelity(mask, i_target, phase, i_out, phase),
            lambda: em.fidelity(
                torch_mask,
                as_torch(i_target),
                as_torch(phase),
                as_torch(i_out),
                as_torch(phase),
            ),
        ),
        "strehl_amplitude": (
            lambda: em.strehl_amplitude(aberration, mask),
            lambda: em.strehl_amplitude(as_torch(aberration), torch_mask),
        ),
        "wavefront_rms": (
            lambda: em.wavefront_rms(aberration, mask),
            lambda: em.wavefront_rms(as_torch(aberration), torch_mask),
        ),
        "wavefront_residual": (
            lambda: em.wavefront_residual(0.5 * aberration, aberration, mask),
            lambda: em.wavefront_residual(
                as_torch(0.5 * aberration), as_torch(aberration), torch_mask
            ),
        ),
    }
    for name, (numpy_call, torch_call) in scalar_cases.items():
        assert float(numpy_call()) == pytest.approx(
            float(torch_call()), rel=1e-9
        ), name

    # The array-returning ones, which must also keep their input's backend.
    normalized_numpy = em.normalize(i_out, mask)
    normalized_torch = em.normalize(as_torch(i_out), torch_mask)
    assert isinstance(normalized_torch, torch.Tensor)
    assert np.allclose(normalized_numpy, normalized_torch.numpy())

    flattened_numpy = em.remove_linear_phase(np.exp(1j * aberration), mask)
    flattened_torch = em.remove_linear_phase(
        torch.exp(1j * as_torch(aberration)), torch_mask
    )
    assert isinstance(flattened_torch, torch.Tensor)
    assert np.allclose(flattened_numpy, flattened_torch.numpy())


def test_ramp_removal_is_idempotent() -> None:
    """Removing the ramp twice must be the same as removing it once.

    Note this aberration carries a genuine linear term, so the first pass is
    expected to change it. What must not happen is a second pass finding another
    ramp that was never there.
    """
    aberration, mask = _aberration(), _mask()
    once = remove_linear_phase(np.exp(1j * aberration), mask)
    twice = remove_linear_phase(once, mask)
    assert np.abs(twice[mask] - once[mask]).max() < 1e-3
