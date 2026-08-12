"""Unit tests for ``KbNufftPartialAffine`` -- the scaled + shifted + rotated NUFFT
zoom that ``FourierLensNUFFT`` composes.

The two properties that matter at the transform level are: ``adjoint`` is the
true conjugate transpose of ``forward`` (the dot-product identity), and the
rotation is exactly a rotation of the k-space trajectory.
"""

from __future__ import annotations

import math

import pytest
import torch

from hologradpy.fourier_transforms import KbNufftPartialAffine


pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _complex(*shape: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.rand(*shape, generator=generator)
        + 1j * torch.rand(*shape, generator=generator)
    ).to(torch.complex64)


def test_adjoint_is_conjugate_transpose() -> None:
    transform = KbNufftPartialAffine(
        resolution=(12, 16),
        resolution_out=(12, 16),
        magnification=(1.3, 1.1),
        shift=(0.2, -0.1),
        angle=math.radians(15),
        grid_size=(24, 32),
    )
    x = _complex(1, 1, 12, 16, seed=0)
    y = _complex(1, 1, 12, 16, seed=1)

    forward_x = transform.forward(x)
    adjoint_y = transform.adjoint(y)

    lhs = torch.sum(forward_x.conj() * y)
    rhs = torch.sum(x.conj() * adjoint_y)
    torch.testing.assert_close(lhs, rhs, rtol=1e-3, atol=1e-2)


def test_rotation_rotates_the_trajectory() -> None:
    angle = math.radians(25)
    common = dict(
        resolution=(12, 16),
        resolution_out=(12, 16),
        magnification=(1.3, 1.1),
        shift=(0.2, -0.1),
        grid_size=(24, 32),
    )
    base = KbNufftPartialAffine(angle=0.0, **common)
    rotated = KbNufftPartialAffine(angle=angle, **common)

    cos, sin = math.cos(angle), math.sin(angle)
    expected_x = base.frequencies[0] * cos - base.frequencies[1] * sin
    expected_y = base.frequencies[0] * sin + base.frequencies[1] * cos

    torch.testing.assert_close(rotated.frequencies[0], expected_x)
    torch.testing.assert_close(rotated.frequencies[1], expected_y)


def test_per_wavelength_trajectory_shape() -> None:
    """A per-wavelength magnification yields one trajectory per wavelength."""
    magnification = torch.tensor([[1.0, 1.0], [1.5, 1.2], [2.0, 0.8]])
    transform = KbNufftPartialAffine(
        resolution=(8, 8),
        resolution_out=(8, 8),
        magnification=magnification,
        grid_size=(16, 16),
    )
    # frequencies: (2, n_wl, hw)
    assert transform.frequencies.shape == (2, 3, 64)

    field = _complex(2, 3, 8, 8, seed=0)
    output = transform.forward(field)
    assert output.shape == (2, 3, 8, 8)
