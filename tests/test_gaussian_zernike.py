"""Tests for drawing a GaussianZernikeBeam, and for a calibration made from one.

The fit of a beam is tested with the calibrators, in test_raster_calibrator.py.
"""

from __future__ import annotations

import torch

from hologradpy.calibration.wavefront.abstract import WavefrontCalibrationData
from hologradpy.optics.complex_amplitude import FieldGeometry
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
