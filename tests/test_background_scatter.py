"""Tests for BackgroundScatter and the laser_speckle_intensity generator."""

from __future__ import annotations

import pytest
import torch

from hologradpy.profiles.amplitude import laser_speckle_intensity
from hologradpy.optics.modules.hardware_models import BackgroundScatter
from hologradpy.optics.complex_amplitude import ComplexAmplitude

DEVICE = torch.device("cpu")
PITCH = 3.45e-6
WAVELENGTH = 0.63e-6


def _field(resolution=(24, 32)) -> ComplexAmplitude:
    # A field with a bright block and (mostly) dark surroundings.
    data = torch.zeros(resolution, dtype=torch.complex64, device=DEVICE)
    data[8:12, 12:16] = 2.0 * torch.exp(1j * torch.tensor(0.7))
    return ComplexAmplitude(data, wavelength=WAVELENGTH, pixel_size=(PITCH, PITCH))


def test_laser_speckle_intensity_stats():
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    # 1 um/px pixel, 3 um grain -> 3 px grain.
    speckle = laser_speckle_intensity((256, 256), 1e-6, 3e-6, generator=gen)
    assert speckle.shape == (256, 256)
    assert speckle.min() >= 0.0
    assert speckle.mean().item() == pytest.approx(1.0, abs=1e-5)  # unit mean
    # Fully-developed speckle: intensity std ~= mean (negative-exponential).
    assert speckle.std().item() == pytest.approx(1.0, abs=0.1)


def test_laser_speckle_intensity_reproducible():
    a = laser_speckle_intensity(
        (64, 64), 1e-6, 2e-6, generator=torch.Generator(device=DEVICE).manual_seed(7)
    )
    b = laser_speckle_intensity(
        (64, 64), 1e-6, 2e-6, generator=torch.Generator(device=DEVICE).manual_seed(7)
    )
    assert torch.equal(a, b)


def test_background_scatter_adds_intensity_everywhere():
    field = _field()
    module = BackgroundScatter(power=1e-6, seed=1)
    out = module(field)

    added = out.intensity - field.intensity
    # Background must be strictly positive everywhere, including dark pixels.
    assert added.min().item() > 0.0
    assert torch.allclose(added, module.background, atol=1e-4)


def test_background_scatter_added_power_matches():
    field = _field()
    power = 5e-7
    out = BackgroundScatter(power=power, seed=2)(field)
    added_power = float((out.intensity - field.intensity).sum()) * PITCH * PITCH
    assert added_power == pytest.approx(power, rel=1e-4)


def test_background_scatter_preserves_geometry_and_phase():
    field = _field()
    out = BackgroundScatter(power=1e-6, seed=0)(field)
    assert tuple(out.geometry.resolution) == (24, 32)
    # Phase is preserved where the field is non-zero.
    signal = field.amplitude > 0
    assert torch.allclose(out.phase[signal], field.phase[signal], atol=1e-5)
