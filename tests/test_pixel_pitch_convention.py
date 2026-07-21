"""Non-square-pixel regression tests for the pitch-axis convention.

slmsuite exposes camera ``pitch_um`` as ``(x, y)`` while hologradpy stores
``pixel_size`` / ``resolution`` as ``(y, x) = (height, width)``. These orders are
indistinguishable for square pixels, so the rest of the suite (square pixels) cannot
catch an axis swap. These tests use deliberately non-square camera pixels.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from hologradpy.hardware import SimulatedSLMTorch, SimulatedCameraTorch  # noqa: E402
from hologradpy.propagation.complex_amplitude import (  # noqa: E402
    ComplexAmplitude,
    FieldGeometry,
)
from hologradpy.propagation.optical_systems import SLMCZT, SLMFFT  # noqa: E402
from hologradpy.propagation.diagonal_elements import StaticSLMField  # noqa: E402
from hologradpy.propagation.virtual_slms import VirtualSLM  # noqa: E402
from hologradpy.propagation.amplitude_profiles import (  # noqa: E402
    gaussian_beam_intensity,
)
from hologradpy.calibration.camera_mapping import CoarseMapper  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

DEVICE = torch.device("cpu")
FOCAL = 0.25
WAVELENGTH = 0.630e-6
SLM_PITCH = 12.5e-6
PADDED = 512
# Non-square camera: pixel_size (y, x) = (30, 20) um -> pitch_um (x, y) = (20, 30) um.
CAMERA_PIXEL_SIZE = (30e-6, 20e-6)
# The reference SLMFFT focal-plane pixel (square) = lambda * f / (slm_pitch * padded).
MODEL_PIXEL = WAVELENGTH * FOCAL / (SLM_PITCH * PADDED)


def _build(camera_angle=0.0, camera_shift=(0.0, 0.0)):
    torch.manual_seed(0)
    geometry = FieldGeometry(
        resolution=(256, 320),
        pixel_size=torch.tensor([SLM_PITCH, SLM_PITCH], device=DEVICE),
        wavelength=torch.tensor(WAVELENGTH, device=DEVICE),
    )
    slm = SimulatedSLMTorch(input_geometry=geometry, bitdepth=8)
    intensity = gaussian_beam_intensity(*geometry.get_spatial_grid(), beam_radius=1e-3)
    beam = ComplexAmplitude(
        intensity.sqrt() + 0j,
        wavelength=geometry.wavelength,
        pixel_size=geometry.pixel_size,
    )
    sim_model = SLMCZT(
        input_geometry=geometry,
        virtual_slm=slm.virtual_slm,
        camera_resolution=(240, 320),
        camera_pixel_size=CAMERA_PIXEL_SIZE,
        focal_length=FOCAL,
        static_slm_field=StaticSLMField(beam),
        camera_angle=camera_angle,
        camera_shift=camera_shift,
    )
    camera = SimulatedCameraTorch(sim_model)
    camera.set_exposure(1e-3)
    camera.get_image()
    reference = SLMFFT(
        input_geometry=geometry,
        virtual_slm=VirtualSLM(phase_scaling=1.0),
        static_slm_field=StaticSLMField(beam),
        focal_length=FOCAL,
        padded_resolution=(PADDED, PADDED),
    )
    return slm, camera, reference


def test_simulated_camera_exposes_pixel_size_in_yx():
    """A non-square simulated camera exposes pixel_size as (y, x) in metres, the native
    convention taken straight from the model (fails on an axis swap)."""
    _, camera, _ = _build()
    # camera_pixel_size (y, x) = (30, 20) um.
    np.testing.assert_allclose(camera.pixel_size, (30e-6, 20e-6), rtol=1e-6)


def test_coarse_mapper_recovers_anisotropic_scales_nonsquare_camera():
    """With a non-square camera the coarse mapper must recover the correct anisotropic
    scale (camera pitch / model pixel per axis). A pitch axis swap anywhere in the
    detection chain would invert the anisotropy or raise the reprojection error."""
    slm, camera, reference = _build(camera_angle=8.0, camera_shift=(15, -10))
    coarse = CoarseMapper(slm, camera, reference).map_camera()

    # pitch_um (x, y) = (20, 30) um over the ~24.6 um square model pixel.
    expected = sorted((20e-6 / MODEL_PIXEL, 30e-6 / MODEL_PIXEL), reverse=True)
    assert sorted(coarse.scales, reverse=True) == pytest.approx(expected, rel=0.03)
    assert coarse.reprojection_rms < 1.0
