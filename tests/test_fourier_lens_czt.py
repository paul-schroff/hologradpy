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
from hologradpy.fourier_transforms import fft_2d, padded_resolution_for_rotation


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
    lens(x)  # lazily initialize

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


def test_padding_leaves_the_focal_plane_alone() -> None:
    """Padding exists to give the rotation room, not to change the optics. The focal
    sampling only stays put because the base magnification and the chirp-z step both
    carry the input resolution, so this is the assertion that catches taking one of
    them from the unpadded frame and the other from the padded one. It would also catch
    an off-center pad, which shows up as a tilt across the focal plane.
    """
    field = make_field(RESOLUTION, 1, seed=3)
    settings = dict(
        focal_length=FOCAL_LENGTH,
        resolution_out=RESOLUTION,
        pixel_size_out=_identity_pixel_out(),
    )

    plain = FourierLensCZT(**settings)(field).as_tensor()
    padded = FourierLensCZT(**settings, padded_resolution=(28, 32))(field).as_tensor()

    assert padded.shape == plain.shape
    # Loose only because the field is complex64 and the padded transform is larger, so
    # it accumulates more round-off. The same tolerance the exactness test above uses.
    assert float((padded - plain).abs().max() / plain.abs().max()) < 1e-4


def _power_reaching_the_focal_plane(field, resolution, padded_resolution, angle):
    """Input power that survives the lens, by Parseval.

    The output window is the full FFT grid of whatever the chirp-z transforms, so the
    focal sum is exactly the input power that made it through the rotation.
    """
    lens = FourierLensCZT(
        FOCAL_LENGTH,
        resolution,
        (
            800e-9 * FOCAL_LENGTH / (PIXEL_IN[0] * resolution[0]),
            800e-9 * FOCAL_LENGTH / (PIXEL_IN[1] * resolution[1]),
        ),
        angle=angle,
        power_normalized=False,
        padded_resolution=padded_resolution,
    )
    transformed = lens(field).as_tensor().abs().pow(2).sum()
    return float(transformed) / (resolution[0] * resolution[1])


def _smooth_field_filling_the_frame():
    """A beam that is still bright at the corners, but without the hard pixel-to-pixel
    edges of a random field. The band-limited shear leaks a little of a sharp edge past
    the frame however much room it is given, and that leakage would otherwise be
    confused with the geometric clipping this is about.
    """
    rows, columns = torch.meshgrid(
        torch.arange(RESOLUTION[0], dtype=torch.float32),
        torch.arange(RESOLUTION[1], dtype=torch.float32),
        indexing="ij",
    )
    y = (rows - (RESOLUTION[0] - 1) / 2) / RESOLUTION[0]
    x = (columns - (RESOLUTION[1] - 1) / 2) / RESOLUTION[1]
    return ComplexAmplitude(
        torch.exp(-(x**2 + y**2) / 0.5).to(torch.complex64),
        torch.tensor(800e-9),
        PIXEL_IN,
    )


def test_padding_keeps_the_corners_a_rotation_would_otherwise_clip() -> None:
    """The reason the padding exists, and the reason it is the default.

    The shear crops back to the frame it is given, so a field that fills its frame loses
    its corners, which is unphysical: rotating the camera does not vignette the SLM. The
    only way to get that back now is to ask for it, by pinning the frame to the input.
    """
    angle = 8.0
    field = _smooth_field_filling_the_frame()
    power = float(field.as_tensor().abs().pow(2).sum())
    padded = padded_resolution_for_rotation(RESOLUTION, angle)

    # padded_resolution=RESOLUTION pins the frame to the input, which is what the lens
    # used to do by default and what loses the corners.
    clipped = (
        _power_reaching_the_focal_plane(field, RESOLUTION, RESOLUTION, angle) / power
    )
    kept = _power_reaching_the_focal_plane(field, padded, padded, angle) / power
    by_default = _power_reaching_the_focal_plane(field, padded, None, angle) / power

    assert clipped < 0.99
    assert kept > 0.999
    # Asking for nothing gives the same as asking for the right frame.
    assert by_default == pytest.approx(kept, abs=1e-6)


def test_padding_smaller_than_the_field_is_refused() -> None:
    """Silently cropping the field would look like a badly converged calibration rather
    than a configuration error."""
    field = make_field(RESOLUTION, 1, seed=5)
    lens = FourierLensCZT(
        FOCAL_LENGTH, RESOLUTION, _identity_pixel_out(), padded_resolution=(8, 8)
    )

    with pytest.raises(ValueError, match="smaller than the input"):
        lens(field)
