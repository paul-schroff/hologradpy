"""Tests for the Fourier-lens bridge: seeding a system's focal-plane affine from a
fitted camera mapping.

The oracle tests are round trips: build a system with known focal-plane
``(scale, angle, shift)``, wrap it as a camera, fit it with the coarse mapper
against an identity reference, and assert ``calibrate_from_mapping`` reproduces the
known parameters in the reference. This pins the sign, axis and centre conventions
empirically.

The calibration tests check that ``calibrate_from_mapping`` sets the affine from the
mapping alone, starting from identity. They also check that the mappers measure with the
affine held at identity, and that a model and its mapping reload from separate files to
the same calibrated affine.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
import torch

from hologradpy.hardware import (
    CameraOrientation,
    SimulatedCameraTorch,
    SimulatedSLMTorch,
)
from hologradpy.optics.complex_amplitude import (
    ComplexAmplitude,
    FieldGeometry,
)
from hologradpy.optics.systems import (
    SLMCZT,
    SLMFFT,
    SLMFFTAffine,
    load_optical_system,
)
from hologradpy.optics.modules.slm_fields import PixelwiseSLMField
from hologradpy.optics.modules.virtual_slms import VirtualSLM
from hologradpy.profiles.amplitude import (
    gaussian_beam_intensity,
)
from hologradpy.geometry import inverse_partial_affine_parameters
from hologradpy.calibration.camera_mapping import (
    CameraMapping,
    CoarseMapper,
    FocalSpotFit,
)
from hologradpy.geometry import PartialAffineTransform
from hologradpy.grids import plane_center

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
        detected_points=detected.tolist(),
        calculated_points=calculated.tolist(),
        zeroth_order_position=(0.0, 0.0),
        spot_fit=FocalSpotFit(waist=1.0),
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


# --- inverse_partial_affine_parameters (pure-numpy core) ------------------------


def test_the_identity_inverts_to_identity_parameters():
    scale, angle, shift = inverse_partial_affine_parameters(
        PartialAffineTransform.from_components(), center_xy=(160, 120)
    )
    assert scale == pytest.approx(1.0)
    assert angle == pytest.approx(0.0)
    np.testing.assert_allclose(shift, (0.0, 0.0), atol=1e-9)


def test_a_rotation_about_the_center_inverts_to_the_opposite_angle():
    """The parameters describe the inverse of the camera -> model similarity, so a
    rotation about the centre lands in the parameters with its sign reversed.
    """
    transform = PartialAffineTransform.from_components(
        angle_deg=-9.0, center=(160, 120)
    )
    scale, angle, shift = inverse_partial_affine_parameters(
        transform, center_xy=(160, 120)
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
        slm_field=PixelwiseSLMField(beam),
        camera_angle=angle,
        camera_shift=tuple(s * p for s, p in zip(shift, CAM_PIX)),
    )


def _fft_affine(geometry, beam, virtual_slm, angle, shift):
    return SLMFFTAffine(
        input_geometry=geometry,
        virtual_slm=virtual_slm,
        camera_resolution=CAM_RES,
        camera_pixel_size=CAM_PIX,
        focal_length=FOCAL,
        slm_field=PixelwiseSLMField(beam),
        padded_resolution=(1024, 1024),
        camera_angle=angle,
        camera_shift=tuple(s * p for s, p in zip(shift, CAM_PIX)),
    )


def _assert_reproduces_truth(partial_affine):
    assert float(partial_affine.angle) == pytest.approx(TRUTH_ANGLE, abs=0.3)
    np.testing.assert_allclose(partial_affine.shift.tolist(), TRUTH_SHIFT, atol=1.0)
    np.testing.assert_allclose(
        partial_affine.scale_factor.tolist(), (1.0, 1.0), atol=0.02
    )


def _oracle_bench(build):
    """The simulated SLM and a camera watching a truth system at ``TRUTH_ANGLE`` and
    ``TRUTH_SHIFT``, with the geometry and the beam for building a reference.
    """
    geometry, slm, beam = _geometry_and_beam()
    truth = build(geometry, beam, slm.virtual_slm, TRUTH_ANGLE, TRUTH_SHIFT)
    camera = SimulatedCameraTorch(truth, orientation=CameraOrientation())
    camera.set_exposure(1e-3)
    camera.get_image()
    return geometry, slm, beam, camera


def _run_oracle(build, seed_angle, seed_shift):
    geometry, slm, beam, camera = _oracle_bench(build)
    reference = build(
        geometry, beam, VirtualSLM(phase_scaling=1.0), seed_angle, seed_shift
    )
    mapping = CoarseMapper(slm, camera, reference).map_camera()
    reference.calibrate_from_mapping(mapping)
    _assert_reproduces_truth(reference.focal_plane_partial_affine)


def test_czt_identity_reference_reproduces_truth():
    _run_oracle(_czt, 0.0, (0.0, 0.0))


def test_czt_seeded_reference_reproduces_truth():
    # The mapper measures the seeded reference with its affine held at identity, and
    # the calibration replaces the seed with the true parameters.
    _run_oracle(_czt, 4.0, (9.0, -4.0))


def test_fft_affine_identity_reference_reproduces_truth():
    _run_oracle(_fft_affine, 0.0, (0.0, 0.0))


# --- guards ---------------------------------------------------------------------


def _slm_fft() -> SLMFFT:
    """A model of the oracle geometry, without a focal-plane partial affine."""
    geometry, _, beam = _geometry_and_beam()
    return SLMFFT(
        input_geometry=geometry,
        virtual_slm=VirtualSLM(phase_scaling=1.0),
        slm_field=PixelwiseSLMField(beam),
        focal_length=FOCAL,
        padded_resolution=(512, 512),
    )


def test_calibrate_from_mapping_rejects_system_without_focal_plane_partial_affine():
    model = _slm_fft()
    mapping = _synthetic_mapping(PartialAffineTransform.from_components())
    with pytest.raises(TypeError, match="no focal-plane partial affine"):
        model.calibrate_from_mapping(mapping)


# --- calibrating from a mapping -------------------------------------------------

SEED_ANGLE = 4.0
SEED_SHIFT = (9.0, -4.0)
BUILDERS = {"czt": _czt, "fft_affine": _fft_affine}


def _reference(build, angle: float = 0.0, shift: tuple[float, float] = (0.0, 0.0)):
    """A reference model of the oracle geometry, run once so its affine exists."""
    geometry, _, beam = _geometry_and_beam()
    model = build(geometry, beam, VirtualSLM(phase_scaling=1.0), angle, shift)
    model()
    return model


def _calibration_mapping(
    angle_deg: float = -3.0,
    shift: tuple[float, float] = (2.0, -1.0),
) -> CameraMapping:
    """A synthetic camera -> model similarity about the centre of the camera plane."""
    return _synthetic_mapping(
        PartialAffineTransform.from_components(
            scale=1.01,
            angle_deg=angle_deg,
            shift=shift,
            center=plane_center(CAM_RES),
        )
    )


def _affine_state(model) -> dict[str, torch.Tensor]:
    """A detached copy of the tensors of the model's focal-plane partial affine."""
    return {
        name: value.detach().clone()
        for name, value in model.focal_plane_partial_affine.state_dict().items()
    }


def _assert_affine_equal(model, expected: dict[str, torch.Tensor]) -> None:
    actual = model.focal_plane_partial_affine.state_dict()
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        assert torch.equal(actual[name], value), name


def _assert_at_identity(model) -> None:
    partial_affine = model.focal_plane_partial_affine
    assert torch.all(partial_affine.scale_factor == 1.0)
    assert torch.all(partial_affine.shift == 0.0)
    assert torch.all(partial_affine.angle == 0.0)


def _move_affine(model) -> None:
    """Move the focal-plane partial affine by a small step."""
    partial_affine = model.focal_plane_partial_affine
    with torch.no_grad():
        partial_affine.angle.add_(0.2)
        partial_affine.shift.add_(torch.tensor([0.5, -0.3]))


@pytest.mark.parametrize("build", list(BUILDERS.values()), ids=list(BUILDERS))
def test_calibrating_twice_with_one_mapping_sets_the_same_affine(build):
    """A model handed from a calibrator to camera feedback meets its mapping twice, and
    the second calibration leaves the affine exactly where the first put it.
    """
    model = _reference(build, SEED_ANGLE, SEED_SHIFT)
    mapping = _calibration_mapping()

    model.calibrate_from_mapping(mapping)
    calibrated = _affine_state(model)
    model.calibrate_from_mapping(mapping)

    _assert_affine_equal(model, calibrated)


@pytest.mark.parametrize("build", list(BUILDERS.values()), ids=list(BUILDERS))
def test_the_calibration_is_absolute_whatever_the_seed(build):
    """The calibration replaces the camera_angle and camera_shift seeds, so a seeded
    and an unseeded model end with the same affine.
    """
    mapping = _calibration_mapping()
    unseeded = _reference(build)
    seeded = _reference(build, SEED_ANGLE, SEED_SHIFT)

    unseeded.calibrate_from_mapping(mapping)
    seeded.calibrate_from_mapping(mapping)

    expected = _affine_state(unseeded)
    for name, value in _affine_state(seeded).items():
        torch.testing.assert_close(value, expected[name], rtol=0.0, atol=1e-4)


@pytest.mark.parametrize("build", list(BUILDERS.values()), ids=list(BUILDERS))
def test_calibrating_again_resets_a_moved_affine(build):
    """The calibration depends on the mapping alone, so calibrating again returns a
    moved affine to the values of the mapping.
    """
    model = _reference(build)
    mapping = _calibration_mapping()
    model.calibrate_from_mapping(mapping)
    calibrated = _affine_state(model)
    _move_affine(model)

    model.calibrate_from_mapping(mapping)

    _assert_affine_equal(model, calibrated)


def test_a_different_mapping_replaces_the_calibration():
    """A new mapping replaces the calibration, whatever the affine held before."""
    first = _calibration_mapping()
    second = _calibration_mapping(angle_deg=2.0, shift=(-3.0, 4.0))
    fresh = _reference(_czt)
    fresh.calibrate_from_mapping(second)
    expected = _affine_state(fresh)

    model = _reference(_czt)
    model.calibrate_from_mapping(first)
    _move_affine(model)
    model.calibrate_from_mapping(second)

    _assert_affine_equal(model, expected)


@pytest.mark.parametrize("build", list(BUILDERS.values()), ids=list(BUILDERS))
def test_bypass_partial_affine_holds_identity_and_restores(build):
    """The affine sits at identity inside the block. The calibrated values are
    restored when the block ends, whether it held a nested block or ended with an
    exception.
    """
    model = _reference(build)
    model.calibrate_from_mapping(_calibration_mapping())
    calibrated = _affine_state(model)

    with model.bypass_partial_affine() as held:
        assert held is model
        _assert_at_identity(model)
        with model.bypass_partial_affine():
            _assert_at_identity(model)
        _assert_at_identity(model)
    _assert_affine_equal(model, calibrated)

    with pytest.raises(ValueError, match="inside the block"):
        with model.bypass_partial_affine():
            raise ValueError("raised inside the block")
    _assert_affine_equal(model, calibrated)


def test_bypass_partial_affine_leaves_a_model_without_one_alone():
    model = _slm_fft()
    model()
    expected = {name: value.clone() for name, value in model.state_dict().items()}

    with model.bypass_partial_affine() as held:
        assert held is model
        for name, value in model.state_dict().items():
            assert torch.equal(value, expected[name]), name


def test_resetting_the_focal_plane_partial_affine_returns_the_warp_to_identity():
    model = _reference(_fft_affine)
    warp = model.affine_transform
    identity_matrix = warp.get_affine_matrix().detach().clone()
    with torch.no_grad():
        warp.focal_plane_partial_affine.scale_factor.fill_(1.02)
        warp.focal_plane_partial_affine.angle.fill_(3.0)
        warp.focal_plane_partial_affine.shift.fill_(4.0)

    warp.focal_plane_partial_affine.reset()

    torch.testing.assert_close(
        warp.get_affine_matrix().detach(), identity_matrix, rtol=0.0, atol=1e-6
    )


def test_a_seeded_reference_finds_the_true_zeroth_order():
    """Without its partial affine, the model has its zeroth order at the centre of its
    plane. The mapper measures a seeded reference with its affine at identity, so it
    finds the true position of the zeroth order.
    """
    geometry, slm, beam, camera = _oracle_bench(_czt)
    reference = _czt(
        geometry, beam, VirtualSLM(phase_scaling=1.0), SEED_ANGLE, SEED_SHIFT
    )

    mapping = CoarseMapper(slm, camera, reference).map_camera()

    expected = (CAM_RES[0] // 2 + TRUTH_SHIFT[1], CAM_RES[1] // 2 + TRUTH_SHIFT[0])
    np.testing.assert_allclose(mapping.zeroth_order_position, expected, atol=1.0)


def test_mapping_a_calibrated_model_gives_the_same_mapping():
    """A mapping measured on a calibrated model describes the camera against the model
    without its partial affine, as the first mapping does. Measuring it leaves the
    calibrated values in place.
    """
    geometry, slm, beam, camera = _oracle_bench(_czt)
    reference = _czt(geometry, beam, VirtualSLM(phase_scaling=1.0), 0.0, (0.0, 0.0))

    first = CoarseMapper(slm, camera, reference).map_camera()
    reference.calibrate_from_mapping(first)
    calibrated = _affine_state(reference)
    second = CoarseMapper(slm, camera, reference).map_camera()

    _assert_affine_equal(reference, calibrated)
    assert second.rotation_degrees == pytest.approx(-TRUTH_ANGLE, abs=0.3)
    assert second.rotation_degrees == pytest.approx(first.rotation_degrees, abs=0.05)
    np.testing.assert_allclose(second.scales, first.scales, atol=1e-3)
    np.testing.assert_allclose(
        second.zeroth_order_position, first.zeroth_order_position, atol=0.5
    )


@pytest.mark.parametrize("build", list(BUILDERS.values()), ids=list(BUILDERS))
def test_the_calibration_survives_save_and_load(build, tmp_path):
    """The model and its lean mapping are saved to separate files. The reloaded model
    holds the calibrated affine, and calibrating it from the reloaded mapping leaves the
    affine as it is.
    """
    model = _reference(build)
    mapping = _calibration_mapping()
    model.calibrate_from_mapping(mapping)
    calibrated = _affine_state(model)

    model_path = tmp_path / "calibrated.pt"
    mapping_path = tmp_path / "camera_mapping.asdf"
    model.save(str(model_path))
    mapping.lean().save(mapping_path)
    loaded = load_optical_system(model_path)
    loaded_mapping = CameraMapping.load(mapping_path)

    _assert_affine_equal(loaded, calibrated)

    loaded.calibrate_from_mapping(loaded_mapping)

    _assert_affine_equal(loaded, calibrated)
