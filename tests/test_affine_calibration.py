"""Tests for the Fourier-lens bridge: seeding a system's focal-plane affine from a
fitted camera mapping.

The oracle tests are round trips: build a system with known focal-plane
``(scale, angle, shift)``, wrap it as a camera, fit it with the coarse mapper
against an identity reference, and assert ``calibrate_from_mapping`` reproduces the
known parameters in the reference. This pins the sign / axis / centre conventions
empirically rather than by derivation.
"""

from __future__ import annotations

from datetime import datetime

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
from hologradpy.propagation.optical_systems import (  # noqa: E402
    SLMCZT,
    SLMFFT,
    SLMFFTAffine,
)
from hologradpy.propagation.diagonal_elements import StaticSLMField  # noqa: E402
from hologradpy.propagation.virtual_slms import VirtualSLM  # noqa: E402
from hologradpy.propagation.amplitude_profiles import (  # noqa: E402
    gaussian_beam_intensity,
)
from hologradpy.geometry import recalibrated_partial_affine  # noqa: E402
from hologradpy.calibration.camera_mapping import CameraMapping, CoarseMapper  # noqa: E402
from hologradpy.geometry import PartialAffineTransform  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

DEVICE = torch.device("cpu")
CAM_RES = (240, 320)
CAM_PIX = (30e-6, 30e-6)
FOCAL = 0.25
TRUTH_ANGLE = 7.0
TRUTH_SHIFT = (15.0, -8.0)


# --- mapping.partial_affine -----------------------------------------------------


def _synthetic_mapping(transform: PartialAffineTransform) -> CameraMapping:
    """A CameraMapping whose detected / calculated points realise ``transform``."""
    detected = np.random.default_rng(0).uniform(-50, 50, size=(12, 2))
    calculated = transform.transform_points(detected)
    return CameraMapping(
        timestamp=datetime.now(),
        name="synthetic",
        transform=transform.as_matrix(homogeneous=False),
        inverse_transform=transform.inverse().as_matrix(homogeneous=False),
        detected_points=detected.tolist(),
        calculated_points=calculated.tolist(),
        camera_images=[np.zeros((2, 2))],
        simulated_images=[np.zeros((2, 2))],
        zeroth_order_position=(0.0, 0.0),
        focal_spot_radius=1.0,
    )


def test_partial_affine_refits_similarity_from_correspondences():
    truth = PartialAffineTransform.from_components(
        scale=1.4, angle_deg=18.0, shift=(6.0, -3.0)
    )
    mapping = _synthetic_mapping(truth)
    refit = mapping.partial_affine
    assert isinstance(refit, PartialAffineTransform)
    assert refit.scale == pytest.approx(1.4, rel=1e-4)
    assert refit.angle_degrees == pytest.approx(18.0, abs=1e-3)
    np.testing.assert_allclose(refit.translation, [6.0, -3.0], atol=1e-3)


# --- recalibrated_partial_affine (pure-numpy core) ------------------------------


def test_recalibrated_core_identity_residual_is_noop():
    identity = PartialAffineTransform.from_components()
    scale, angle, shift = recalibrated_partial_affine(
        1.3, 12.0, (4.0, -5.0), identity, center_xy=(160, 120)
    )
    assert scale == pytest.approx(1.3)
    assert angle == pytest.approx(12.0)
    np.testing.assert_allclose(shift, (4.0, -5.0), atol=1e-9)


def test_recalibrated_core_composes_residual_about_centre():
    # A pure rotation about the centre with no current transform: the residual's
    # inverse rotation is what lands in the parameters.
    residual = PartialAffineTransform.from_components(
        angle_deg=-9.0, center=(160, 120)
    )
    scale, angle, shift = recalibrated_partial_affine(
        1.0, 0.0, (0.0, 0.0), residual, center_xy=(160, 120)
    )
    assert scale == pytest.approx(1.0)
    assert angle == pytest.approx(9.0)
    np.testing.assert_allclose(shift, (0.0, 0.0), atol=1e-9)


# --- oracle round trips ---------------------------------------------------------


def _geometry_and_beam():
    torch.manual_seed(0)
    geometry = FieldGeometry(
        resolution=(256, 320),
        pixel_size=torch.tensor([12.5e-6, 12.5e-6], device=DEVICE),
        wavelength=torch.tensor(0.630e-6, device=DEVICE),
    )
    slm = SimulatedSLMTorch(input_geometry=geometry, bitdepth=8)
    intensity = gaussian_beam_intensity(*geometry.get_spatial_grid(), beam_radius=1e-3)
    beam = ComplexAmplitude(
        intensity.sqrt() + 0j,
        wavelength=geometry.wavelength,
        pixel_size=geometry.pixel_size,
    )
    return geometry, slm, beam


def _czt(geometry, beam, virtual_slm, angle, shift):
    return SLMCZT(
        input_geometry=geometry,
        virtual_slm=virtual_slm,
        camera_resolution=CAM_RES,
        camera_pixel_size=CAM_PIX,
        focal_length=FOCAL,
        static_slm_field=StaticSLMField(beam),
        camera_angle=angle,
        camera_shift=shift,
    )


def _fft_affine(geometry, beam, virtual_slm, angle, shift):
    return SLMFFTAffine(
        input_geometry=geometry,
        virtual_slm=virtual_slm,
        camera_resolution=CAM_RES,
        camera_pixel_size=CAM_PIX,
        focal_length=FOCAL,
        static_slm_field=StaticSLMField(beam),
        padded_resolution=(1024, 1024),
        camera_angle=angle,
        camera_shift=shift,
    )


def _assert_reproduces_truth(module):
    angle = module.angle if module.angle.ndim == 0 else module.angle[0]
    assert float(angle) == pytest.approx(TRUTH_ANGLE, abs=0.3)
    np.testing.assert_allclose(module.shift.tolist(), TRUTH_SHIFT, atol=1.0)
    np.testing.assert_allclose(module.scale_factor.tolist(), (1.0, 1.0), atol=0.02)


def _run_oracle(build, module_of, seed_angle, seed_shift):
    geometry, slm, beam = _geometry_and_beam()
    truth = build(geometry, beam, slm.virtual_slm, TRUTH_ANGLE, TRUTH_SHIFT)
    camera = SimulatedCameraTorch(truth, rot="0", fliplr=False)
    camera.set_exposure(1e-3)
    camera.get_image()
    reference = build(
        geometry, beam, VirtualSLM(phase_scaling=1.0), seed_angle, seed_shift
    )
    mapping = CoarseMapper(slm, camera, reference).map_camera()
    reference.calibrate_from_mapping(mapping)
    _assert_reproduces_truth(module_of(reference))


def test_czt_identity_reference_reproduces_truth():
    _run_oracle(_czt, lambda model: model.fourier_lens, 0.0, (0.0, 0.0))


def test_czt_warm_start_reference_reproduces_truth():
    # A rough non-identity seed must still be corrected onto the true parameters,
    # exercising the residual composition (not just an identity start).
    _run_oracle(_czt, lambda model: model.fourier_lens, 4.0, (9.0, -4.0))


def test_fft_affine_identity_reference_reproduces_truth():
    _run_oracle(_fft_affine, lambda model: model.affine_transform, 0.0, (0.0, 0.0))


# --- guards ---------------------------------------------------------------------


def test_calibrate_from_mapping_rejects_system_without_affine_module():
    geometry, _, beam = _geometry_and_beam()
    model = SLMFFT(
        input_geometry=geometry,
        virtual_slm=VirtualSLM(phase_scaling=1.0),
        static_slm_field=StaticSLMField(beam),
        focal_length=FOCAL,
        padded_resolution=(512, 512),
    )
    mapping = _synthetic_mapping(PartialAffineTransform.from_components())
    with pytest.raises(TypeError, match="no affine module"):
        model.calibrate_from_mapping(mapping)
