"""Tests for drawing a GaussianZernikeBeam, and for a calibration made from one.

The fit of a beam is tested with the calibrators, in test_raster_calibrator.py.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import pytest
import torch

from hologradpy.calibration.wavefront.abstract import WavefrontCalibrationData
from hologradpy.optics.complex_amplitude import ComplexAmplitude, FieldGeometry
from hologradpy.profiles.gaussian import Gaussian, GaussianBeam
from hologradpy.profiles.gaussian_zernike import GaussianZernikeBeam

RESOLUTION = (64, 80)
PITCH = (12.5e-6, 12.5e-6)
GAUSSIAN = GaussianBeam(300e-6, 200e-6, 100e-6, -50e-6, angle_degrees=30.0)


def test_the_cutoff_cuts_the_beam_beyond_its_radius_from_the_centre():
    uncut = GaussianZernikeBeam(GAUSSIAN, 400e-6).get_wavefront(RESOLUTION, PITCH)
    cut = GaussianZernikeBeam(GAUSSIAN, 400e-6, cutoff_radius=250e-6).get_wavefront(
        RESOLUTION, PITCH
    )

    grid = Gaussian(RESOLUTION, PITCH)
    inside = torch.hypot(grid.x - 100e-6, grid.y + 50e-6) <= 250e-6
    assert torch.equal(cut[inside], uncut[inside])
    assert torch.all(cut[~inside] == 0)


def test_a_beam_without_zernike_terms_has_a_flat_phase():
    """Its amplitude is the square root of the Gaussian's intensity without the
    offset.
    """
    gaussian = GaussianBeam(300e-6, None, 100e-6, -50e-6, intensity=4.0, offset=0.1)

    field = GaussianZernikeBeam(gaussian, 400e-6).get_wavefront(RESOLUTION, PITCH)

    assert torch.all(field.imag == 0)
    without_offset = GaussianBeam(300e-6, None, 100e-6, -50e-6, intensity=4.0)
    expected = Gaussian(RESOLUTION, PITCH).get_intensity(without_offset).sqrt()
    torch.testing.assert_close(field.real, expected)


def test_a_calibration_from_a_beam_holds_the_beam_and_its_field():
    beam = GaussianZernikeBeam(
        GAUSSIAN, 400e-6, ((2, 0), (1, -1)), (0.5, 1.0), cutoff_radius=350e-6
    )
    geometry = FieldGeometry(
        resolution=RESOLUTION,
        pixel_size=torch.tensor(PITCH),
        wavelength=torch.tensor(1064e-9),
    )

    calibration = WavefrontCalibrationData.from_beam(beam, geometry)

    assert calibration.name == "gaussian"
    assert calibration.beam == beam
    field = calibration.complex_amplitude
    torch.testing.assert_close(
        field.as_tensor().reshape(RESOLUTION), beam.get_wavefront(RESOLUTION, PITCH)
    )
    torch.testing.assert_close(field.wavelength, geometry.wavelength)
    torch.testing.assert_close(field.pixel_size, geometry.pixel_size)


GEOMETRY = FieldGeometry(
    resolution=RESOLUTION,
    pixel_size=torch.tensor(PITCH),
    wavelength=torch.tensor(1064e-9),
)


def test_without_defocus_a_calibration_keeps_its_other_terms():
    """Inside the Zernike disk, the phase is then the astigmatism-only beam's.

    Outside it, the beam's phase was 0, and the defocus comes off as its polynomial
    continues, sqrt(3) (2 rho^2 - 1), so no step forms at the edge. The amplitude is
    unchanged, the beam's defocus is 0, and the metadata holds the defocus removed.
    """
    beam = GaussianZernikeBeam(GAUSSIAN, 400e-6, ((2, 0), (2, 2)), (1.5, 0.7))
    calibration = WavefrontCalibrationData.from_beam(beam, GEOMETRY)
    astigmatism = GaussianZernikeBeam(GAUSSIAN, 400e-6, ((2, 2),), (0.7,))

    without = calibration.without_defocus()

    field = without.complex_amplitude.as_tensor().reshape(RESOLUTION)
    expected = astigmatism.get_wavefront(RESOLUTION, PITCH)
    grid = Gaussian(RESOLUTION, PITCH)
    rho = torch.hypot(grid.x - 100e-6, grid.y + 50e-6) / 400e-6
    inside = rho < 0.95
    difference = torch.angle(field[inside] * expected[inside].conj())
    assert difference.abs().max() < 1e-5
    outside = (rho > 1.05) & (rho < 1.2)
    continued = -1.5 * 3**0.5 * (2 * rho[outside] ** 2 - 1)
    step = torch.angle(field[outside] * torch.exp(-1j * continued))
    assert step.abs().max() < 1e-4
    torch.testing.assert_close(field.abs(), expected.abs())
    assert without.beam.zernike_coefficients == (0.0, 0.7)
    assert without.metadata["removed_defocus"] == 1.5


def test_a_calibration_without_a_fitted_beam_has_no_defocus_to_remove():
    calibration = WavefrontCalibrationData(
        timestamp=datetime(2026, 10, 13),
        name="speckle",
        complex_amplitude=ComplexAmplitude.from_geometry(GEOMETRY),
    )

    with pytest.raises(ValueError, match="no fitted beam"):
        calibration.without_defocus()


def test_with_defocus_a_loaded_calibration_gets_its_defocus_back(tmp_path):
    """Saved without its defocus and loaded again, a calibration gets its field and its
    beam back, and its metadata no longer holds a removed defocus.
    """
    beam = GaussianZernikeBeam(GAUSSIAN, 400e-6, ((2, 0), (2, 2)), (1.5, 0.7))
    calibration = WavefrontCalibrationData.from_beam(beam, GEOMETRY)
    path = tmp_path / "without_defocus.asdf"
    calibration.without_defocus().save(path)

    restored = WavefrontCalibrationData.load(path).with_defocus()

    torch.testing.assert_close(
        restored.complex_amplitude.as_tensor(),
        calibration.complex_amplitude.as_tensor(),
    )
    assert restored.beam == beam
    assert "removed_defocus" not in restored.metadata


def test_a_lean_calibration_keeps_all_but_its_visualization_data():
    beam = GaussianZernikeBeam(GAUSSIAN, 400e-6, ((2, 0),), (1.5,))
    calibration = replace(
        WavefrontCalibrationData.from_beam(beam, GEOMETRY), visualization_data=object()
    )

    lean = calibration.lean()

    assert lean.visualization_data is None
    assert lean.beam == calibration.beam
    assert lean.complex_amplitude is calibration.complex_amplitude
    assert (lean.name, lean.timestamp) == (calibration.name, calibration.timestamp)


def test_a_calibration_without_a_removed_defocus_has_none_to_add_back():
    calibration = WavefrontCalibrationData.from_beam(
        GaussianZernikeBeam(GAUSSIAN, 400e-6, ((2, 0),), (1.5,)), GEOMETRY
    )

    with pytest.raises(ValueError, match="no removed defocus"):
        calibration.with_defocus()
