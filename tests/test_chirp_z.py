"""The chirp-z (Bluestein) Fourier zoom and the 3-shear FFT rotation.

``ChirpZZoom`` is the *exact* DFT sampled on a scaled + shifted (+ rotated) zoom
window, so it must match the full FFT (and a padded FFT for a true zoom), have a
correct conjugate-transpose adjoint, and be differentiable. ``shear_rotate`` must
rotate by the right angle in the right direction (validated with an *asymmetric*
pattern, since a symmetric one is rotation-invariant) and conserve power.
"""

from __future__ import annotations

import math

import pytest
import torch

from hologradpy.optics.fourier_transforms import (
    fft_2d,
    ChirpZZoom,
    shear_rotate,
)


pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _random_field(n: int) -> torch.Tensor:
    torch.manual_seed(0)
    return (torch.randn(n, n) + 1j * torch.randn(n, n)).to(torch.complex64)


def _elliptical_gaussian(n: int, sigma_x: float, sigma_y: float, theta: float):
    grid_y, grid_x = torch.meshgrid(
        torch.arange(n) - n // 2, torch.arange(n) - n // 2, indexing="ij"
    )
    cos, sin = math.cos(theta), math.sin(theta)
    rotated_x = cos * grid_x + sin * grid_y
    rotated_y = -sin * grid_x + cos * grid_y
    return torch.exp(
        -0.5 * ((rotated_x / sigma_x) ** 2 + (rotated_y / sigma_y) ** 2)
    ).to(torch.complex64)


def _relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max())


# %% Chirp-z zoom correctness
def test_czt_matches_fft_at_unit_magnification() -> None:
    field = _random_field(48)
    czt = ChirpZZoom((48, 48), (48, 48), (1.0, 1.0))
    reference = fft_2d(field, norm="backward", fft_shift=True)
    assert _relative_error(czt.forward(field), reference) < 1e-3


def test_czt_zoom_matches_padded_fft() -> None:
    n, magnification = 48, 2.0
    field = _random_field(n)
    czt = ChirpZZoom((n, n), (n, n), (magnification, magnification))

    padded = int(n * magnification)
    offset = (padded - n) // 2
    padded_field = torch.zeros(padded, padded, dtype=torch.complex64)
    padded_field[offset:offset + n, offset:offset + n] = field
    full = fft_2d(padded_field, norm="backward", fft_shift=True)
    centre = padded // 2
    window = full[centre - n // 2:centre + n - n // 2,
                  centre - n // 2:centre + n - n // 2]
    assert _relative_error(czt.forward(field), window) < 1e-3


def test_czt_adjoint_is_conjugate_transpose() -> None:
    field = _random_field(48)
    other = (torch.randn(48, 48) + 1j * torch.randn(48, 48)).to(torch.complex64)
    czt = ChirpZZoom(
        (48, 48), (48, 48), (1.5, 1.5), shift=(0.3, -0.2),
        angle=math.radians(20),
    )
    forward_inner = (czt.forward(field) * other.conj()).sum()
    adjoint_inner = (field * czt.adjoint(other).conj()).sum()
    assert float((forward_inner - adjoint_inner).abs() / forward_inner.abs()) < 1e-4


def test_czt_is_differentiable() -> None:
    field = _random_field(32).requires_grad_(True)
    czt = ChirpZZoom((32, 32), (32, 32), (1.5, 1.5))
    czt.forward(field).abs().sum().backward()
    assert field.grad is not None


# %% Shear rotation
def test_shear_rotate_round_trip() -> None:
    field = _elliptical_gaussian(96, 10.0, 6.0, 0.0)
    recovered = shear_rotate(shear_rotate(field, math.radians(25)), math.radians(-25))
    assert _relative_error(recovered, field) < 1e-4


def test_shear_rotate_conserves_power() -> None:
    field = _elliptical_gaussian(96, 10.0, 6.0, 0.0)
    rotated = shear_rotate(field, math.radians(25))
    power_in = float((field.abs() ** 2).sum())
    power_out = float((rotated.abs() ** 2).sum())
    assert abs(power_out / power_in - 1.0) < 1e-3


def test_shear_rotate_matches_analytic_rotation_with_correct_direction() -> None:
    # Asymmetric (elongated) Gaussian: rotating it must match the analytically
    # rotated ellipse and clearly NOT the opposite rotation.
    field = _elliptical_gaussian(96, 13.0, 5.0, 0.0)
    theta = math.radians(25)
    rotated = shear_rotate(field, theta)
    error_same = _relative_error(rotated, _elliptical_gaussian(96, 13.0, 5.0, theta))
    error_opposite = _relative_error(
        rotated, _elliptical_gaussian(96, 13.0, 5.0, -theta)
    )
    assert error_same < 1e-2
    assert error_opposite > 0.1


def test_chirpzzoom_rotation_equals_rotated_input_zoom() -> None:
    field = _random_field(48)
    theta = math.radians(20)
    rotated_zoom = ChirpZZoom((48, 48), (48, 48), (1.5, 1.5), angle=theta)
    plain_zoom = ChirpZZoom((48, 48), (48, 48), (1.5, 1.5))
    assert _relative_error(
        rotated_zoom.forward(field), plain_zoom.forward(shear_rotate(field, theta))
    ) < 1e-5
