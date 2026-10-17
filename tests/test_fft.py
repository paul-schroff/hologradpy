"""fft_2d and ifft_2d in the centred layout."""

from __future__ import annotations

import pytest
import torch

from hologradpy.fourier_transforms import fft_2d, ifft_2d


@pytest.mark.parametrize("shape", [(6, 8), (5, 7), (21, 20)])
def test_ifft_2d_inverts_fft_2d(shape: tuple[int, int]) -> None:
    """For an odd length, ``fftshift`` and ``ifftshift`` move the samples by different
    amounts, so the inverse has to take them in the reverse order.
    """
    generator = torch.Generator().manual_seed(0)
    field = torch.randn(shape, generator=generator, dtype=torch.complex128)

    assert torch.allclose(ifft_2d(fft_2d(field)), field, atol=1e-12)


def test_the_zero_frequency_sits_at_the_centre_sample() -> None:
    """A constant puts all of its power into bin ``(H // 2, W // 2)``."""
    spectrum = fft_2d(torch.ones((5, 6), dtype=torch.complex128))

    assert spectrum[2, 3].abs() == pytest.approx(30.0)
    assert float(spectrum.abs().sum()) == pytest.approx(30.0)
