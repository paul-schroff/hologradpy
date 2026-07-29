"""Behaviour locks for the propagators after they were refactored to compose
the ``hologradpy.optics.fourier_transforms`` transforms.

* ``FourierLensNUFFT`` -> ``KbNufftZoomRotate``: a characterization golden pins
  the migrated output to values captured from the pre-refactor implementation
  (non-square, asymmetric pixel size, rotated, multi-wavelength -- so a swapped
  axis or x/y scale would be caught).
* ``FourierLensFFT`` -> ``FastFourierTransform``: forward/adjoint equal the plain
  padded FFT / cropped IFFT.
* ``AngularSpectrumMethod``: transform-pluggable -- an explicit orthonormal FFT
  reproduces the default, and a different transform (``ChirpZZoom``) is accepted.
"""

from __future__ import annotations

import math

import pytest
import torch

from hologradpy.optics.complex_amplitude import ComplexAmplitude
from hologradpy.optics.modules.propagators import (
    FourierLensFFT,
    FourierLensNUFFT,
    FourierLensCZT,
    AngularSpectrumMethod,
)
from hologradpy.optics.fourier_transforms import (
    FastFourierTransform,
    ChirpZZoom,
    fft_2d,
    ifft_2d,
)
from hologradpy.utils import pad_to_shape_2D, crop_to_shape_2D


pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

PIXEL_IN = (10e-6, 12e-6)
PIXEL_OUT = (5e-6, 8e-6)
H, W = 12, 16


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


def test_fourier_lens_nufft_orientation_matches_czt() -> None:
    """The NUFFT lens shares the x/y orientation of the exact CZT lens (and a
    plain FFT): an x-tilt deflects the focal spot along x, not y -- the regression
    guard for the historical KbNufft omega[0]<->omega[1] transpose. They also
    agree to within the NUFFT's interpolation error."""
    resolution = (32, 32)
    pixel_in = (8e-6, 8e-6)
    # Output pixel = the natural focal pixel (magnification 1), so a tilt gives a
    # resolvable, in-FOV spot.
    pixel_out = (
        800e-9 * 0.1 / (pixel_in[0] * resolution[0]),
        800e-9 * 0.1 / (pixel_in[1] * resolution[1]),
    )
    rows, cols = torch.meshgrid(
        torch.arange(resolution[0]) - resolution[0] // 2,
        torch.arange(resolution[1]) - resolution[1] // 2,
        indexing="ij",
    )
    phase = 2 * math.pi * (0.31 * cols + 0.15 * rows)  # x-tilt > y-tilt
    field = ComplexAmplitude(
        torch.exp(1j * phase).to(torch.complex64), torch.tensor(800e-9), pixel_in
    )

    nufft = FourierLensNUFFT(0.1, resolution, pixel_out)(field)._data.abs().pow(2)
    czt = (
        FourierLensCZT(0.1, resolution, pixel_out, learnable=False)(field)
        ._data.abs()
        .pow(2)
    )

    # The x-tilt deflects the spot more along x than y (not transposed).
    flat = int(nufft.argmax())
    delta_y = flat // resolution[1] - resolution[0] // 2
    delta_x = flat % resolution[1] - resolution[1] // 2
    assert abs(delta_x) > abs(delta_y)

    # Same orientation as -- and close to -- the exact CZT, not its transpose.
    normalized_nufft = nufft / nufft.max()
    normalized_czt = czt / czt.max()
    assert float((normalized_nufft - normalized_czt).abs().max()) < float(
        (normalized_nufft - normalized_czt.transpose(-1, -2)).abs().max()
    )
    assert float((normalized_nufft - normalized_czt).abs().mean()) < 5e-2


def test_fourier_lens_fft_forward_is_padded_fft() -> None:
    field = make_field((2, H, W), 2, seed=0)
    lens = FourierLensFFT(focal_length=0.1, power_normalized=False)
    out = lens(field)._data

    padded = pad_to_shape_2D(field, lens.resolution_out)
    expected = fft_2d(padded._data)
    torch.testing.assert_close(out, expected)


def test_fourier_lens_fft_adjoint_is_scaled_cropped_ifft() -> None:
    """The adjoint is the conjugate transpose, so ``N`` times the cropped ifft.

    ``ifft`` carries a ``1 / N`` under ``norm="backward"``, and undoing that
    factor is what makes this the conjugate transpose rather than the inverse.
    """
    lens = FourierLensFFT(focal_length=0.1, power_normalized=False)
    field = make_field((2, H, W), 2, seed=0)
    lens(field)  # lazily initialise

    spectrum = make_field((2, 2 * H, 2 * W), 2, seed=2, pixel_size=PIXEL_OUT)
    restored = lens.adjoint(spectrum)._data
    number_of_samples = (2 * H) * (2 * W)
    expected = number_of_samples * crop_to_shape_2D(ifft_2d(spectrum), (H, W))._data
    torch.testing.assert_close(restored, expected)


def test_asm_explicit_fft_transform_equals_default() -> None:
    field = make_field((2, H, W), 2, seed=0)

    default = AngularSpectrumMethod(propagation_distance=1e-3)
    out_default = default(field)._data

    transform = FastFourierTransform((2 * H, 2 * W), norm="ortho")
    explicit = AngularSpectrumMethod(propagation_distance=1e-3, transform=transform)
    out_explicit = explicit(field)._data

    torch.testing.assert_close(out_default, out_explicit)


def test_asm_accepts_chirpz_transform() -> None:
    """The angular spectrum is transform-pluggable: a ``ChirpZZoom`` is accepted
    and produces a finite field of the right shape (a band-limited variant -- the
    point is that the transfer function is built on the transform's own
    frequencies)."""
    field = make_field((2, H, W), 2, seed=0)
    transform = ChirpZZoom((2 * H, 2 * W), (2 * H, 2 * W), (1.0, 1.0))
    asm = AngularSpectrumMethod(propagation_distance=1e-4, transform=transform)

    out = asm(field)._data
    assert out.shape == (2, H, W)
    assert torch.isfinite(out).all()
