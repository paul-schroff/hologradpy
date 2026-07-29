"""Tests for ``FourierLensCZT`` -- the exact chirp-z Fourier lens with learnable
scale / shift / angle.

The key claims are: it is the *exact* DFT at identity parameters (so it matches a
plain FFT, unlike the interpolating NUFFT lens), its ``adjoint`` is the conjugate
transpose of ``forward``, and gradients flow to all three affine parameters
(including ``angle`` starting from 0, via the differentiable shear rotation).
"""

from __future__ import annotations

import pytest
import torch

from hologradpy.optics.complex_amplitude import ComplexAmplitude
from hologradpy.optics.modules.propagators import FourierLensCZT
from hologradpy.optics.fourier_transforms import fft_2d


pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

FOCAL_LENGTH = 0.1
PIXEL_IN = (10e-6, 12e-6)
RESOLUTION = (12, 16)


def make_field(shape, n_wl, seed=0, pixel_size=PIXEL_IN):
    generator = torch.Generator().manual_seed(seed)
    data = (
        torch.rand(*shape, generator=generator)
        + 1j * torch.rand(*shape, generator=generator)
    ).to(torch.complex64)
    wavelength = (
        torch.tensor(800e-9)
        if n_wl == 1
        else torch.linspace(800e-9, 900e-9, n_wl)
    )
    return ComplexAmplitude(data, wavelength, pixel_size)


def _identity_pixel_out():
    """The output pixel size for which the base magnification is exactly 1, i.e.
    the chirp-z window equals the full FFT grid."""
    return (
        800e-9 * FOCAL_LENGTH / (PIXEL_IN[0] * RESOLUTION[0]),
        800e-9 * FOCAL_LENGTH / (PIXEL_IN[1] * RESOLUTION[1]),
    )


def test_czt_lens_is_exact_at_identity_parameters() -> None:
    field = make_field(RESOLUTION, 1, seed=0)
    lens = FourierLensCZT(
        FOCAL_LENGTH, RESOLUTION, _identity_pixel_out(), power_normalized=False
    )
    out = lens(field)._data
    reference = fft_2d(field._data, norm="backward", fft_shift=True)
    assert float((out - reference).abs().max() / reference.abs().max()) < 1e-4


def test_czt_lens_power_normalized_conserves_power() -> None:
    """With power_normalized the (du*dv)/(lambda*f) prefactor makes the exact
    chirp-z conserve optical power: at magnification 1 over the full focal plane,
    focal power == input power (Parseval)."""
    field = make_field(RESOLUTION, 1, seed=0)
    lens = FourierLensCZT(
        FOCAL_LENGTH, RESOLUTION, _identity_pixel_out(),
        learnable=False, power_normalized=True,
    )
    out = lens(field)
    torch.testing.assert_close(out.power(), field.power(), rtol=1e-3, atol=0.0)


def test_czt_lens_adjoint_is_conjugate_transpose() -> None:
    lens = FourierLensCZT(
        FOCAL_LENGTH, RESOLUTION, (5e-6, 8e-6),
        shift=(1.0, 2.0), angle=8.0, learnable=False,
    )
    x = make_field((2, *RESOLUTION), 2, seed=0)
    lens(x)  # lazily initialise

    y = make_field((2, *RESOLUTION), 2, seed=1, pixel_size=(5e-6, 8e-6))
    forward_x = lens(x)._data
    adjoint_y = lens.adjoint(y)._data

    lhs = torch.sum(forward_x.conj() * y._data)
    rhs = torch.sum(x._data.conj() * adjoint_y)
    torch.testing.assert_close(lhs, rhs, rtol=1e-3, atol=1e-3)


def test_czt_lens_parameters_are_learnable() -> None:
    field = make_field((2, *RESOLUTION), 2, seed=2)
    lens = FourierLensCZT(FOCAL_LENGTH, RESOLUTION, (5e-6, 8e-6))

    # as_tensor(), not _data: the field returned by a resampling module keeps
    # its autograd graph on the wrapper and its inner tensor is detached.
    lens(field).as_tensor().abs().pow(2).sum().backward()

    # angle starts at 0 but still receives a gradient (differentiable shear).
    for name in ("scale_factor", "shift", "angle"):
        grad = getattr(lens, name).grad
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert float(grad.abs().sum()) > 0.0


def test_czt_lens_static_has_no_grad_parameters() -> None:
    lens = FourierLensCZT(
        FOCAL_LENGTH, RESOLUTION, (5e-6, 8e-6), learnable=False
    )
    lens(make_field((2, *RESOLUTION), 2, seed=3))
    assert lens.scale_factor.requires_grad is False
    assert lens.shift.requires_grad is False
    assert lens.angle.requires_grad is False


def test_czt_lens_preserves_batch_rank_and_geometry() -> None:
    field = make_field((3, 2, *RESOLUTION), 2, seed=4)
    lens = FourierLensCZT(
        FOCAL_LENGTH, (10, 14), (5e-6, 8e-6), shift=(1.5, -2.0),
        angle=12.0,
    )
    out = lens(field)
    assert out._data.shape == (3, 2, 10, 14)
    assert out.resolution == (10, 14)
    assert out.number_of_wavelengths == 2
    assert torch.isfinite(out._data).all()
