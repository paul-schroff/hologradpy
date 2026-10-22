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

from dataclasses import replace
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
from hologradpy.calibration.camera_mapping import (
    CameraMapping,
    CoarseMapper,
    FocalSpotFit,
)
from hologradpy.geometry import AffineTransform, PartialAffineTransform
from hologradpy.grids import (
    metres_to_pixels,
    pixel_frame_angle,
    pixel_frame_vector,
    plane_center,
)
from hologradpy.utils import as_image

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
        spot_fit=FocalSpotFit(waist=1.0),
    )


def test_a_mirrored_mapping_places_its_zeroth_order_through_its_full_affine():
    """A partial affine describes no mirror."""
    resolution = (64, 80)
    mirrored = AffineTransform.from_components(
        scale=1.2, angle_deg=10.0, shift=(5.0, -3.0), mirror=True
    )
    detected = np.random.default_rng(0).uniform(0.0, 64.0, size=(12, 2))
    mapping = CameraMapping(
        timestamp=datetime.now(),
        name="mirrored",
        transform=mirrored.as_matrix(homogeneous=False),
        detected_points=detected.tolist(),
        calculated_points=mirrored.transform_points(detected).tolist(),
        spot_fit=FocalSpotFit(waist=1.0),
        output_pixel_size=CAM_PIX,
        output_resolution=resolution,
    )

    column, row = mirrored.inverse().transform_points([plane_center(resolution)])[0]

    assert mapping.is_mirrored
    assert mapping.zeroth_order_position == pytest.approx((column, row))


def test_partial_affine_refits_similarity_from_correspondences():
    truth = PartialAffineTransform.from_components(
        scale=1.4, angle_deg=18.0, shift=(6.0, -3.0)
    )
    mapping = _synthetic_mapping(truth)
    refit = mapping.partial_affine
    assert isinstance(refit, PartialAffineTransform)
    assert refit.scale == pytest.approx(1.4, rel=1e-4)
    assert refit.rotation_degrees == pytest.approx(18.0, abs=1e-3)
    np.testing.assert_allclose(refit.translation, [6.0, -3.0], atol=1e-3)


# --- the components of a similarity --------------------------------------------


def test_the_identity_inverts_to_identity_parameters():
    scale, angle, shift = (
        PartialAffineTransform.from_components().inverse().components((160, 120))
    )
    assert scale == pytest.approx(1.0)
    assert angle == pytest.approx(0.0)
    np.testing.assert_allclose(shift, (0.0, 0.0), atol=1e-9)


def test_a_rotation_about_the_center_inverts_to_the_opposite_angle():
    """The partial affine of a model applies the inverse of the camera -> model
    similarity.
    """
    transform = PartialAffineTransform.from_components(
        angle_deg=-9.0, center=(160, 120)
    )
    scale, angle, shift = transform.inverse().components((160, 120))
    assert scale == pytest.approx(1.0)
    assert angle == pytest.approx(9.0)
    np.testing.assert_allclose(shift, (0.0, 0.0), atol=1e-9)


def test_the_components_rebuild_the_similarity():
    transform = PartialAffineTransform.from_components(
        scale=1.3, angle_deg=25.0, shift=(4.0, -7.0), center=(160, 120)
    )

    scale, angle, shift = transform.components((160, 120))

    assert scale == pytest.approx(1.3)
    assert angle == pytest.approx(25.0)
    np.testing.assert_allclose(shift, (4.0, -7.0), atol=1e-9)


# --- oracle round trips ---------------------------------------------------------


def _geometry_and_beam():
    torch.manual_seed(0)
    geometry = FieldGeometry(
        resolution=(256, 320),
        pixel_size=torch.tensor([12.5e-6, 12.5e-6], device=DEVICE),
        wavelength=torch.tensor(0.630e-6, device=DEVICE),
    )
    slm = SimulatedSLMTorch(input_geometry=geometry, bitdepth=8)
    intensity = gaussian_beam_intensity(*geometry.get_spatial_grid(), 1e-3)
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
        geometry, beam, VirtualSLM(full_scale_cycles=1.0), seed_angle, seed_shift
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
        virtual_slm=VirtualSLM(full_scale_cycles=1.0),
        slm_field=PixelwiseSLMField(beam),
        focal_length=FOCAL,
        padded_resolution=(512, 512),
    )


def test_calibrate_from_mapping_rejects_system_without_focal_plane_partial_affine():
    model = _slm_fft()
    mapping = _synthetic_mapping(PartialAffineTransform.from_components())
    with pytest.raises(TypeError, match="no focal-plane partial affine"):
        model.calibrate_from_mapping(mapping)


def test_a_mapping_of_another_output_plane_is_refused():
    """A mapping records the output plane of the model it was measured against, and a
    model with another output plane refuses it.
    """
    geometry, _, beam = _geometry_and_beam()
    model = _czt(geometry, beam, VirtualSLM(full_scale_cycles=1.0), 0.0, (0.0, 0.0))
    model()
    output = model[-1]
    pixel_size = tuple(output.pixel_size_out.tolist()[0])
    resolution = tuple(output.resolution_out)
    mapping = replace(
        _calibration_mapping(),
        output_pixel_size=pixel_size,
        output_resolution=resolution,
    )

    model.calibrate_from_mapping(mapping)

    with pytest.raises(ValueError, match="output plane"):
        model.calibrate_from_mapping(
            replace(mapping, output_pixel_size=(2 * pixel_size[0], 2 * pixel_size[1]))
        )
    with pytest.raises(ValueError, match="output plane"):
        model.calibrate_from_mapping(
            replace(mapping, output_resolution=(resolution[0] // 2, resolution[1]))
        )


# --- calibrating from a mapping -------------------------------------------------

SEED_ANGLE = 4.0
SEED_SHIFT = (9.0, -4.0)
BUILDERS = {"czt": _czt, "fft_affine": _fft_affine}


def _reference(build, angle: float = 0.0, shift: tuple[float, float] = (0.0, 0.0)):
    """A reference model of the oracle geometry, run once so its affine exists."""
    geometry, _, beam = _geometry_and_beam()
    model = build(geometry, beam, VirtualSLM(full_scale_cycles=1.0), angle, shift)
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


def test_calibrating_to_a_larger_angle_keeps_the_corners():
    """A calibration can set a larger angle than the seed, and the padding of the
    chirp-z lens grows to hold it. A model seeded at 0 degrees and calibrated to about
    20 degrees matches a model seeded at the calibrated angle, and so keeps the corners
    of a beam that fills the SLM.
    """
    geometry, _, _ = _geometry_and_beam()
    uniform_beam = ComplexAmplitude(
        torch.ones(geometry.resolution, dtype=torch.complex64),
        wavelength=geometry.wavelength,
        pixel_size=geometry.pixel_size,
    )
    mapping = _calibration_mapping(angle_deg=-20.0)

    def calibrated_model(seed_angle: float) -> SLMCZT:
        model = _czt(
            geometry,
            uniform_beam,
            VirtualSLM(full_scale_cycles=1.0),
            seed_angle,
            (0.0, 0.0),
        )
        model()
        model.calibrate_from_mapping(mapping)
        return model

    seeded_at_zero = calibrated_model(0.0)
    angle = float(seeded_at_zero.focal_plane_partial_affine.angle)
    expected = calibrated_model(angle)().intensity.detach()

    # The mapping turns the pixels by -20 degrees, and the lens turns them back.
    assert pixel_frame_angle(angle) == pytest.approx(20.0, abs=0.5)
    torch.testing.assert_close(
        seeded_at_zero().intensity.detach(),
        expected,
        rtol=1e-5,
        atol=1e-5 * float(expected.max()),
    )


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
        geometry, beam, VirtualSLM(full_scale_cycles=1.0), SEED_ANGLE, SEED_SHIFT
    )

    mapping = CoarseMapper(slm, camera, reference).map_camera()

    # The truth shift is physical, and the position is (column, row).
    column_shift, row_shift = pixel_frame_vector(TRUTH_SHIFT)
    center_column, center_row = plane_center(CAM_RES)
    expected = (center_column + column_shift, center_row + row_shift)
    np.testing.assert_allclose(mapping.zeroth_order_position, expected, atol=1.0)


def _spot_centroid(image: np.ndarray, half_window: int = 4) -> tuple[float, float]:
    """The ``(column, row)`` centroid of the window around the brightest pixel."""
    row, column = np.unravel_index(int(np.argmax(image)), image.shape)
    window = image[
        row - half_window : row + half_window + 1,
        column - half_window : column + half_window + 1,
    ]
    rows, columns = np.indices(window.shape)
    return (
        column - half_window + float((columns * window).sum() / window.sum()),
        row - half_window + float((rows * window).sum() / window.sum()),
    )


def test_the_zeroth_order_of_a_mapping_is_where_the_calibrated_model_puts_it():
    """A camera with pixels 5 % wider than tall is no similarity of the model
    plane. The full affine of its mapping and the partial affine then place the zeroth
    order 7 pixels apart. The mapping places it through the partial affine.
    """
    geometry, slm, beam = _geometry_and_beam()
    wide_pixels = (CAM_PIX[0], 1.05 * CAM_PIX[1])
    shift = (130.0, 90.0)  # (x, y) sensor pixels from the centre to the zeroth order
    truth = SLMCZT(
        input_geometry=geometry,
        virtual_slm=slm.virtual_slm,
        camera_resolution=CAM_RES,
        camera_pixel_size=wide_pixels,
        focal_length=FOCAL,
        slm_field=PixelwiseSLMField(beam),
        camera_shift=(shift[0] * wide_pixels[1], shift[1] * wide_pixels[0]),
    )
    camera = SimulatedCameraTorch(truth, orientation=CameraOrientation())
    camera.set_exposure(1e-3)
    camera.get_image()
    reference = _czt(geometry, beam, VirtualSLM(full_scale_cycles=1.0), 0.0, (0.0, 0.0))

    mapping = CoarseMapper(slm, camera, reference).map_camera()
    reference.calibrate_from_mapping(mapping)
    with torch.no_grad():
        reference.virtual_slm.set_phase(np.zeros(tuple(geometry.resolution)))
        image = as_image(reference().intensity).cpu().numpy()

    np.testing.assert_allclose(
        _spot_centroid(image), mapping.zeroth_order_position, atol=0.5
    )


def test_mapping_a_calibrated_model_gives_the_same_mapping():
    """A mapping measured on a calibrated model describes the camera against the model
    without its partial affine, as the first mapping does. Measuring it leaves the
    calibrated values in place.
    """
    geometry, slm, beam, camera = _oracle_bench(_czt)
    reference = _czt(geometry, beam, VirtualSLM(full_scale_cycles=1.0), 0.0, (0.0, 0.0))

    first = CoarseMapper(slm, camera, reference).map_camera()
    reference.calibrate_from_mapping(first)
    calibrated = _affine_state(reference)
    second = CoarseMapper(slm, camera, reference).map_camera()

    _assert_affine_equal(reference, calibrated)
    assert second.camera_angle == pytest.approx(TRUTH_ANGLE, abs=0.3)
    assert second.camera_angle == pytest.approx(first.camera_angle, abs=0.05)
    assert second.camera_zoom == pytest.approx(first.camera_zoom, rel=1e-3)
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


# --- The camera's numbers read back as written -------------------------------------

READ_BACK_ANGLE = 15.0
READ_BACK_SHIFT = (20.0, -10.0)  # camera pitches
READ_BACK_ZOOM = 1.1


def _zoomed_truth_camera(build):
    """The SLM and a camera watching a truth at the read-back angle, shift and zoom."""
    geometry, slm, beam = _geometry_and_beam()
    truth = build(geometry, beam, slm.virtual_slm, READ_BACK_ANGLE, READ_BACK_SHIFT)
    with torch.no_grad():
        truth.focal_plane_partial_affine.scale_factor.fill_(READ_BACK_ZOOM)
    camera = SimulatedCameraTorch(truth, orientation=CameraOrientation())
    camera.set_exposure(1e-3)
    camera.get_image()
    return geometry, slm, beam, camera


@pytest.mark.parametrize("build", list(BUILDERS.values()), ids=list(BUILDERS))
def test_a_simulated_camera_reads_back_its_camera_angle_shift_and_zoom(build):
    """A camera simulated with a camera_angle, a camera_shift and a zoom maps back to
    the same three numbers. The calibration writes them into the reference, and a model
    built from them shows the calibrated model's image.
    """
    geometry, slm, beam, camera = _zoomed_truth_camera(build)
    reference = _reference(build)
    shift = tuple(s * p for s, p in zip(READ_BACK_SHIFT, CAM_PIX))

    mapping = CoarseMapper(slm, camera, reference).map_camera()

    assert mapping.camera_angle == pytest.approx(READ_BACK_ANGLE, abs=0.05)
    np.testing.assert_allclose(mapping.camera_shift, shift, atol=0.1 * CAM_PIX[0])
    assert mapping.camera_zoom == pytest.approx(READ_BACK_ZOOM, rel=1e-3)

    reference.calibrate_from_mapping(mapping)
    partial_affine = reference.focal_plane_partial_affine
    assert float(partial_affine.angle) == pytest.approx(mapping.camera_angle, abs=1e-5)
    np.testing.assert_allclose(
        partial_affine.shift.tolist(),
        metres_to_pixels(mapping.camera_shift, CAM_PIX),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        partial_affine.scale_factor.tolist(), (mapping.camera_zoom,) * 2, rtol=1e-6
    )

    rebuilt = build(
        geometry,
        beam,
        VirtualSLM(full_scale_cycles=1.0),
        mapping.camera_angle,
        metres_to_pixels(mapping.camera_shift, CAM_PIX),
    )
    with torch.no_grad():
        rebuilt.focal_plane_partial_affine.scale_factor.fill_(mapping.camera_zoom)
        rebuilt()
        phase = reference.steering_phase((300e-6, -200e-6))
        reference.virtual_slm.set_phase(phase.to(reference.device))
        rebuilt.virtual_slm.set_phase(phase.to(rebuilt.device))
        expected = reference().intensity.squeeze()
        actual = rebuilt().intensity.squeeze()
    assert float((actual - expected).abs().max()) <= 1e-5 * float(expected.max())


def test_a_mapping_against_an_slmfft_reads_back_the_same_camera():
    """The same camera mapped against an SLMFFT reads back the same three numbers,
    while the scales of the pixel fit hold the ratio of the two pitches.
    """
    _, slm, _, camera = _zoomed_truth_camera(_czt)
    reference = _slm_fft()
    reference()
    shift = tuple(s * p for s, p in zip(READ_BACK_SHIFT, CAM_PIX))

    mapping = CoarseMapper(slm, camera, reference).map_camera()

    assert mapping.camera_angle == pytest.approx(READ_BACK_ANGLE, abs=0.05)
    np.testing.assert_allclose(mapping.camera_shift, shift, atol=0.1 * CAM_PIX[0])
    assert mapping.camera_zoom == pytest.approx(READ_BACK_ZOOM, rel=1e-3)
    assert mapping.affine.scales[0] == pytest.approx(1.108, abs=0.005)


def test_a_mapping_on_pixels_that_are_not_square_gives_no_camera_angle():
    """A mapping measured against an output plane of 24.6 x 19.7 um pixels gives no
    camera angle, shift or zoom.
    """
    geometry, slm, beam, camera = _zoomed_truth_camera(_czt)
    reference = SLMFFT(
        input_geometry=geometry,
        virtual_slm=VirtualSLM(full_scale_cycles=1.0),
        slm_field=PixelwiseSLMField(beam),
        focal_length=FOCAL,
        padded_resolution=(512, 640),
    )
    reference()

    mapping = CoarseMapper(slm, camera, reference).map_camera()

    for name in ("camera_angle", "camera_shift", "camera_zoom"):
        with pytest.raises(ValueError, match="square"):
            getattr(mapping, name)


def test_a_mirrored_mapping_gives_no_camera_angle():
    """A camera mounted mirrored maps to a mirrored mapping, which gives no camera
    angle.
    """
    geometry, slm, beam = _geometry_and_beam()
    truth = _czt(geometry, beam, slm.virtual_slm, 0.0, (0.0, 0.0))
    camera = SimulatedCameraTorch(truth, orientation=CameraOrientation(fliplr=True))
    camera.set_exposure(1e-3)
    camera.get_image()
    reference = _reference(_czt)

    mapping = CoarseMapper(slm, camera, reference).map_camera()

    assert mapping.is_mirrored
    with pytest.raises(ValueError, match="mirrored"):
        mapping.camera_angle


def test_a_mapping_without_a_camera_gives_no_camera_angle():
    """A mapping that records no camera gives no camera angle."""
    mapping = _synthetic_mapping(PartialAffineTransform.from_components())
    with pytest.raises(ValueError, match="records no camera"):
        mapping.camera_angle


@pytest.mark.parametrize(
    ("rot", "expected_angle", "expected_shift"),
    [
        ("90", 90.0, (0.0, 1.0)),
        ("180", 180.0, (-1.0, 1.0)),
        ("270", -90.0, (-1.0, 0.0)),
    ],
)
def test_a_turned_mounting_reads_its_turn_as_the_camera_angle(
    rot, expected_angle, expected_shift
):
    """A camera mounted with ``rot`` shows the unturned focal plane turned, and the
    mapping reads the turn as its camera_angle. On the even sensor, the centre pixel
    moves by one pitch along each reversed axis.
    """
    geometry, slm, beam = _geometry_and_beam()
    truth = _czt(geometry, beam, slm.virtual_slm, 0.0, (0.0, 0.0))
    camera = SimulatedCameraTorch(truth, orientation=CameraOrientation(rot=rot))
    camera.set_exposure(1e-3)
    camera.get_image()
    reference = SLMCZT(
        input_geometry=geometry,
        virtual_slm=VirtualSLM(full_scale_cycles=1.0),
        camera_resolution=tuple(camera.sensor_resolution),
        camera_pixel_size=CAM_PIX,
        focal_length=FOCAL,
        slm_field=PixelwiseSLMField(beam),
    )
    reference()

    mapping = CoarseMapper(slm, camera, reference).map_camera()

    # The reading at the half turn can take either sign, so the angle is compared
    # wrapped.
    wrapped = (mapping.camera_angle - expected_angle + 180.0) % 360.0 - 180.0
    assert wrapped == pytest.approx(0.0, abs=0.05)
    np.testing.assert_allclose(
        [s / CAM_PIX[0] for s in mapping.camera_shift], expected_shift, atol=0.05
    )


def test_the_tilt_to_the_centre_pixel_undoes_the_camera_angle_shift_and_zoom():
    """The tilt to the sensor's centre pixel is minus camera_shift, turned by minus
    camera_angle and divided by camera_zoom.
    """
    _, slm, _, camera = _zoomed_truth_camera(_czt)
    reference = _reference(_czt)
    mapping = CoarseMapper(slm, camera, reference).map_camera()

    angle = np.radians(mapping.camera_angle)
    turn_back = np.array(
        [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
    )
    expected = -turn_back @ np.asarray(mapping.camera_shift) / mapping.camera_zoom

    tilt = mapping.tilt_to_sensor_center(camera)
    np.testing.assert_allclose(tilt, expected, rtol=0.0, atol=1e-15)


def test_the_model_and_its_mapping_place_points_alike_after_calibration():
    """Right after the calibration, the model's focal_plane_positions and the
    mapping's sensor_to_image_plane put the same pixels at the same focal-plane
    positions.
    """
    _, slm, _, camera = _zoomed_truth_camera(_czt)
    reference = _reference(_czt)
    mapping = CoarseMapper(slm, camera, reference).map_camera()
    reference.calibrate_from_mapping(mapping)

    grid_x, grid_y = reference[-1].get_spatial_grid_output()
    model_x, model_y = reference.focal_plane_positions(grid_x, grid_y)
    pixels = [(0, 0), (319, 239), (100, 50), (160, 120)]  # (column, row)
    from_model = np.array(
        [
            [float(model_x[row, column]), float(model_y[row, column])]
            for column, row in pixels
        ]
    )
    from_mapping = mapping.sensor_to_image_plane(pixels)
    np.testing.assert_allclose(
        from_model, from_mapping, rtol=0.0, atol=1e-4 * CAM_PIX[0]
    )
