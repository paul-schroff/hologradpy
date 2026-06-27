"""Tests for Zernike.fit (the inverse of Zernike.get_phase).

``fit`` solves a masked, weighted least-squares problem for the coefficients
that best reconstruct a measured phase. The defining property is round-trip
consistency: ``fit(get_phase(c)) == c``, including for batched inputs and
per-sample (per-wavelength) masks.
"""
from __future__ import annotations

import pytest
import torch

from hologradpy.propagation.utils.zernike import Zernike


def _zernike(resolution=(40, 52), orders=5, mode="fit"):
    return Zernike(
        resolution=resolution,
        number_of_radial_orders=orders,
        unit_disk_mode=mode,
    )


def test_single_round_trip() -> None:
    z = _zernike()
    coefficients = torch.randn(z.number_of_zernikes, generator=_gen())
    recovered = z.fit(z.get_phase(coefficients))
    torch.testing.assert_close(recovered, coefficients, rtol=0, atol=1e-5)


def test_batched_round_trip() -> None:
    """Leading (batch, wavelength) dimensions are fitted independently."""
    z = _zernike()
    coefficients = torch.randn(2, 3, z.number_of_zernikes, generator=_gen())

    phase = z.get_phase(coefficients)
    assert phase.shape == (2, 3, *z.resolution)

    recovered = z.fit(phase)
    assert recovered.shape == coefficients.shape
    torch.testing.assert_close(recovered, coefficients, rtol=0, atol=1e-4)


def test_per_wavelength_mask_round_trip() -> None:
    """A per-wavelength mask fits each wavelength over its own region."""
    z = _zernike()
    number_of_wavelengths = 3
    coefficients = torch.randn(
        number_of_wavelengths, z.number_of_zernikes, generator=_gen()
    )
    phase = z.get_phase(coefficients)

    # Distinct mask per wavelength (still covering most of the disk).
    mask = z.mask.unsqueeze(0).repeat(number_of_wavelengths, 1, 1).clone()
    mask[0, :15, :] = False
    mask[1, :, :18] = False

    recovered = z.fit(phase, mask=mask)
    assert recovered.shape == coefficients.shape
    torch.testing.assert_close(recovered, coefficients, rtol=0, atol=1e-4)


def test_shared_mask_ignores_outside_pixels() -> None:
    """A shared mask restricts the fit; pixels outside it must not matter."""
    z = _zernike()
    coefficients = torch.randn(z.number_of_zernikes, generator=_gen())
    phase = z.get_phase(coefficients)

    mask = z.mask.clone()
    mask[:10, :] = False

    # Corrupting the masked-out region must not change the fit.
    corrupted = phase.clone()
    corrupted[:10, :] += 100.0

    torch.testing.assert_close(
        z.fit(phase, mask=mask), z.fit(corrupted, mask=mask), rtol=0, atol=1e-6
    )


def test_get_phase_rejects_wrong_coefficient_count() -> None:
    z = _zernike()
    with pytest.raises(ValueError):
        z.get_phase(torch.zeros(z.number_of_zernikes + 3))


def _gen() -> torch.Generator:
    return torch.Generator().manual_seed(0)
