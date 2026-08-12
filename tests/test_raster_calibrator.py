"""Tests for RasterCalibrator's coarse-mapping-driven linear-phase placement.

Covers the model-free tilt/orientation maths on a synthetic CameraMapping (fast)
and an end-to-end check that the auto-computed tilt actually lands the spot where
intended on a small simulated setup, for an aligned and a rotated camera.
"""

from __future__ import annotations

from datetime import datetime

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from hologradpy.hardware import SimulatedSLMTorch, SimulatedCameraTorch  # noqa: E402
from hologradpy.optics.complex_amplitude import (  # noqa: E402
    ComplexAmplitude,
    FieldGeometry,
)
from hologradpy.optics.systems import SLMFFTAffine  # noqa: E402
from hologradpy.optics.modules.slm_fields import PixelwiseSLMField  # noqa: E402
from hologradpy.profiles.amplitude import (  # noqa: E402
    gaussian_beam_intensity,
)
from hologradpy.calibration import (  # noqa: E402
    RasterCalibrator,
    get_diffraction_spot_position,
)
from hologradpy.calibration.camera_mapping import CameraMapping  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

DEVICE = torch.device("cpu")
FOCAL_LENGTH = 0.25


def _build_setup(
    camera_angle: float = 0.0,
    noise_level: float = 0.0,
    power_std: float | None = None,
    power_seed: int | None = None,
):
    """A small simulated SLM + camera (the camera is the 'hardware')."""
    geometry = FieldGeometry(
        resolution=(256, 320),
        pixel_size=torch.tensor([12.5e-6, 12.5e-6], device=DEVICE),
        wavelength=torch.tensor(0.630e-6, device=DEVICE),
    )
    slm = SimulatedSLMTorch(input_geometry=geometry, bitdepth=8)
    intensity = gaussian_beam_intensity(
        *geometry.get_spatial_grid(), beam_radius=2.0e-3
    )
    beam = ComplexAmplitude(
        intensity.sqrt() + 0j,
        wavelength=geometry.wavelength,
        pixel_size=geometry.pixel_size,
        power=1e-3,
    )
    hardware = SLMFFTAffine(
        input_geometry=geometry,
        virtual_slm=slm.virtual_slm,
        camera_resolution=(240, 320),
        camera_pixel_size=(30e-6, 30e-6),
        focal_length=FOCAL_LENGTH,
        slm_field=PixelwiseSLMField(beam),
        padded_resolution=(512, 512),
        camera_angle=camera_angle,
        camera_shift=(0, 0),
    )
    camera = SimulatedCameraTorch(
        hardware,
        noise_level=noise_level,
        power_std=power_std,
        power_seed=power_seed,
    )
    camera.set_exposure(1e-3)
    camera.get_image()
    return slm, camera


def _synthetic_mapping(
    rotation_deg=0.0, scale=1.5, mirror=False, zeroth=(160.0, 120.0)
):
    """A CameraMapping with a known camera-px -> model-px transform.

    ``zeroth`` is the stored ``(y, x)`` zeroth-order position.
    """
    angle = np.radians(rotation_deg)
    linear = scale * np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    if mirror:
        linear = linear @ np.array([[1.0, 0.0], [0.0, -1.0]])
    transform = np.hstack([linear, np.zeros((2, 1))])
    inverse = np.hstack([np.linalg.inv(linear), np.zeros((2, 1))])
    return CameraMapping(
        timestamp=datetime.now(),
        name="coarse",
        transform=transform,
        inverse_transform=inverse,
        detected_points=[],
        calculated_points=[],
        camera_images=[],
        simulated_images=[],
        zeroth_order_position=zeroth,
        focal_spot_radius=8e-6,
    )


# --- fast unit tests (synthetic mapping, model-free path) ----------------------


def _zeroth_xy(mapping):
    return np.array(
        [mapping.zeroth_order_position[1], mapping.zeroth_order_position[0]]
    )


def test_rotation_matrix_orthonormal_and_identity_when_aligned():
    slm, camera = _build_setup()
    calibrator = RasterCalibrator(slm, camera, focal_length=FOCAL_LENGTH)
    calibrator.camera_mapping = _synthetic_mapping(rotation_deg=30.0, scale=2.0)
    rotation = calibrator._rotation_matrix()
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(2), atol=1e-9)
    # An aligned camera (pure scale, no rotation) gives the identity, so the fit
    # grids are left untouched.
    calibrator.camera_mapping = _synthetic_mapping(rotation_deg=0.0, scale=1.7)
    np.testing.assert_allclose(calibrator._rotation_matrix(), np.eye(2), atol=1e-9)


@pytest.mark.parametrize("mirror", [False, True])
def test_main_placement_is_diagonal_and_clears_dc(mirror):
    slm, camera = _build_setup()
    calibrator = RasterCalibrator(slm, camera, focal_length=FOCAL_LENGTH)
    mapping = _synthetic_mapping(
        rotation_deg=20.0, scale=1.5, mirror=mirror, zeroth=(90.0, 70.0)
    )
    calibrator.camera_mapping = mapping
    roi = (20, 20)
    tilt, target, _ = calibrator._ensure_and_place_main(roi, None, None)

    height, width = camera.shape
    zeroth = _zeroth_xy(mapping)  # (x, y)
    offset = np.asarray(target) - zeroth
    assert 0 < target[0] < width and 0 < target[1] < height  # on sensor
    assert np.hypot(*offset) >= 2 * max(roi) - 1e-6  # clears DC by 2 * roi
    assert abs(abs(offset[0]) - abs(offset[1])) < 1e-6  # on a 45 deg diagonal
    addressable = np.asarray(calibrator._addressable_half_extent())
    assert np.all(np.abs(tilt) <= 0.9 * addressable + 1e-9)  # reachable


def test_lattice_placement_same_diagonal_further_out():
    slm, camera = _build_setup()
    calibrator = RasterCalibrator(slm, camera, focal_length=FOCAL_LENGTH)
    mapping = _synthetic_mapping(zeroth=(90.0, 70.0))
    calibrator.camera_mapping = mapping
    zeroth = _zeroth_xy(mapping)
    roi = (20, 20)
    _, main_target, direction = calibrator._ensure_and_place_main(roi, None, None)
    main_distance = float(np.hypot(*(np.asarray(main_target) - zeroth)))
    clearance = main_distance + max(roi) + max(roi)
    _, lattice_target = calibrator._auto_phase_tilt(roi, clearance, direction)

    lattice_offset = np.asarray(lattice_target) - zeroth
    assert np.hypot(*lattice_offset) > main_distance  # further from the DC
    assert abs(abs(lattice_offset[0]) - abs(lattice_offset[1])) < 1e-6  # same diagonal


# --- end-to-end placement (real coarse mapping) --------------------------------


@pytest.mark.parametrize("camera_angle", [0.0, 15.0])
def test_auto_tilt_lands_spot_on_target(camera_angle):
    """The auto tilt (built from a real coarse mapping) places the diffraction spot
    at the intended camera pixel, even for a rotated camera."""
    slm, camera = _build_setup(camera_angle=camera_angle)
    calibrator = RasterCalibrator(slm, camera, focal_length=FOCAL_LENGTH)

    (tilt, target, _) = calibrator._ensure_and_place_main((40, 40), None, None)
    (spot_x, spot_y), _, _, _ = get_diffraction_spot_position(
        slm, camera, tilt, focal_length=FOCAL_LENGTH, units="pixels", verbose=False
    )
    # The recovered camera rotation matches the injected angle.
    assert calibrator.camera_mapping.rotation_degrees == pytest.approx(
        -camera_angle, abs=1.5
    )
    # The spot lands at the placement target (a few pixels of detection precision).
    assert np.hypot(spot_x - target[0], spot_y - target[1]) < 6.0


# --- lattice steering / fitting robustness to camera noise ---------------------


def _corner_setup():
    """A calibrator plus the corner slices, ROI and centred detection spot used by
    ``calibrate_lattice_corner_tilts``."""
    slm, camera = _build_setup(noise_level=6.0)
    calibrator = RasterCalibrator(slm, camera, focal_length=FOCAL_LENGTH)
    corner = 32
    height, width = slm.resolution
    corner_slices = [
        (slice(0, corner), slice(0, corner)),
        (slice(0, corner), slice(width - corner, width)),
        (slice(height - corner, height), slice(0, corner)),
        (slice(height - corner, height), slice(width - corner, width)),
    ]
    roi = calibrator.get_roi_size(corner, corner)
    sensor_height, sensor_width = camera.shape
    window = (
        min(4 * roi[0], sensor_height),  # height
        min(4 * roi[1], sensor_width),  # width
    )
    spot_center = (sensor_width // 2, sensor_height // 2)  # unclamped window
    return calibrator, corner_slices, roi, spot_center, window


def test_capture_averaged_reduces_noise(monkeypatch):
    slm, camera = _build_setup()
    calibrator = RasterCalibrator(slm, camera, focal_length=FOCAL_LENGTH)
    rng = np.random.default_rng(0)
    shape = camera.shape
    # Fresh per-frame noise about a constant signal, as a real sensor delivers.
    monkeypatch.setattr(
        camera, "get_image", lambda *a, **k: rng.normal(100.0, 10.0, size=shape)
    )
    single = np.asarray(camera.get_image(), dtype=float)
    averaged = calibrator._capture_averaged(1e-3, 25)
    # Averaging 25 frames should cut the noise spread by roughly 1/5.
    assert averaged.std() < 0.5 * single.std()


def test_corner_steering_recovers_offset_under_noise(monkeypatch):
    calibrator, corner_slices, roi, spot_center, window = _corner_setup()
    window_height, window_width = window
    pitch_x = calibrator.camera.pixel_size[1]
    pitch_y = calibrator.camera.pixel_size[0]
    lattice_tilt = (300e-6, 300e-6)
    shift_px = (7, -5)  # spot offset from the window centre, in pixels

    rng = np.random.default_rng(1)

    def fake_capture(exposure, frame_averages):
        yy, xx = np.mgrid[0:window_height, 0:window_width]
        centre_x = window_width / 2 + shift_px[0]
        centre_y = window_height / 2 + shift_px[1]
        spot = 300.0 * np.exp(
            -((xx - centre_x) ** 2 + (yy - centre_y) ** 2) / (2 * 4.0**2)
        )
        return spot + rng.normal(16.0, 4.0, size=(window_height, window_width))

    monkeypatch.setattr(calibrator, "_capture_averaged", fake_capture)
    tilts = calibrator.calibrate_lattice_corner_tilts(
        corner_slices, lattice_tilt, roi, spot_center, exposure_time=1e-3
    )
    # Steering removes the measured offset, so every corner lands on the same tilt.
    expected = (
        lattice_tilt[0] - shift_px[0] * pitch_x,
        lattice_tilt[1] - shift_px[1] * pitch_y,
    )
    assert len(tilts) == 4
    for tilt in tilts:
        assert tilt == pytest.approx(expected, abs=2 * pitch_x)


def test_plot_full_frame_renders_with_markers():
    import matplotlib.pyplot as plt

    from hologradpy.calibration.wavefront.raster_calibration.visualizer import (
        RasterVisualizationData,
        RasterCalibratorVisualizer,
    )

    def _data(**extra):
        return RasterVisualizationData(
            camera_images=np.zeros((1, 1, 1)),
            fitted_images=np.zeros((1, 1, 1)),
            measured_phase=np.zeros((1, 1)),
            superpixel_coordinates=np.zeros((2, 1)),
            **extra,
        )

    image = np.zeros((40, 60))
    image[20, 30] = 100.0
    data = _data(
        full_frame_image=image,
        full_frame_marker_positions={
            "interference pattern": (30.0, 20.0),
            "optical lattice": (10.0, 35.0),
            "zeroth order": (50.0, 5.0),
        },
    )
    figure = RasterCalibratorVisualizer(data).plot_full_frame()
    assert figure is not None
    plt.close(figure)

    # No snapshot recorded -> a clear error rather than a blank plot.
    with pytest.raises(RuntimeError):
        RasterCalibratorVisualizer(_data()).plot_full_frame()


def test_corner_steering_gates_pure_noise(monkeypatch):
    calibrator, corner_slices, roi, spot_center, window = _corner_setup()
    window_height, window_width = window
    lattice_tilt = (300e-6, 300e-6)
    rng = np.random.default_rng(2)

    def fake_capture(exposure, frame_averages):
        # No spot: the validity gate must reject every corner fit and steer none.
        return rng.normal(16.0, 2.0, size=(window_height, window_width))

    monkeypatch.setattr(calibrator, "_capture_averaged", fake_capture)
    tilts = calibrator.calibrate_lattice_corner_tilts(
        corner_slices, lattice_tilt, roi, spot_center, exposure_time=1e-3
    )
    assert len(tilts) == 4
    for tilt in tilts:
        assert tilt == pytest.approx(lattice_tilt)  # gated -> shared lattice tilt


# --- power normalization corrects laser drift ----------------------------------


def test_normalize_power_removes_laser_fluctuation(monkeypatch):
    """A PowerInstability fluctuates the laser power per frame. With normalize_power
    the reference spot shares each frame, so the recovered map is (almost) the same as
    with a stable laser; without it the fluctuation clearly changes the map."""
    # Noise-free cameras (deterministic sim): one with a fluctuating laser, one
    # stable. Same optics, so their coarse mappings and placements match.
    slm_f, camera_f = _build_setup(power_std=0.04, power_seed=0)
    slm_s, camera_s = _build_setup()
    calibrator_f = RasterCalibrator(slm_f, camera_f, focal_length=FOCAL_LENGTH)
    calibrator_s = RasterCalibrator(slm_s, camera_s, focal_length=FOCAL_LENGTH)

    def scan(calibrator, *, normalize):
        intensity, _ = calibrator.measure_intensity(
            number_of_superpixels_x=6,
            number_of_superpixels_y=5,
            superpixel_width=32,
            superpixel_height=32,
            normalize_power=normalize,
            verbose=False,
        )
        return np.asarray(intensity)

    norm_fluctuating = scan(calibrator_f, normalize=True)
    norm_stable = scan(calibrator_s, normalize=True)
    plain_fluctuating = scan(calibrator_f, normalize=False)
    plain_stable = scan(calibrator_s, normalize=False)

    def relative_difference(a, b):
        return float(np.abs(a - b).sum() / np.abs(b).sum())

    norm_difference = relative_difference(norm_fluctuating, norm_stable)
    plain_difference = relative_difference(plain_fluctuating, plain_stable)

    # The fluctuation is real: it clearly changes the un-normalized map ...
    assert plain_difference > 0.01
    # ... but normalization makes the map nearly independent of it.
    assert norm_difference < 0.5 * plain_difference


def test_calibration_returns_a_complex_amplitude_its_consumers_accept():
    """The returned field must be a ComplexAmplitude, not a bare numpy array.

    WavefrontCalibrationData declares the field as one, and both consumers rely
    on that: PixelwiseSLMField.from_calibration_data hands it straight to the
    module, and the speckle calibrator reads a benchmark calibration through
    .as_tensor(). A bare array raised AttributeError in both, so neither
    applying a raster calibration to a model nor benchmarking against one
    worked. No existing test built a WavefrontCalibrationData, so this went
    unnoticed.
    """
    slm, camera = _build_setup()
    calibrator = RasterCalibrator(slm, camera, focal_length=FOCAL_LENGTH)

    record = calibrator.calibrate(
        number_of_superpixels=(4, 4),
        camera_mapping=_synthetic_mapping(),
        verbose=False,
    )

    assert isinstance(record.complex_amplitude, ComplexAmplitude)
    assert tuple(record.complex_amplitude.shape) == tuple(slm.resolution)
    assert torch.isfinite(record.complex_amplitude.as_tensor()).all()

    # The two paths that a bare array broke.
    field_module = PixelwiseSLMField.from_calibration_data(record)
    assert field_module.init_field is record.complex_amplitude
    assert record.complex_amplitude.as_tensor().shape == tuple(slm.resolution)


def test_a_supplied_model_must_match_the_calibrator_focal_length() -> None:
    """A model with a different focal length is rejected rather than used.

    The model is not only a coordinate reference for the coarse mapping:
    ``_orientation_matrix`` reads ``pixel_size_out`` off its output layer, and that
    spacing scales with the focal length. A mismatched model therefore biases the
    camera to focal-plane scale and misplaces every tilt, with no error to point at.
    """
    slm, camera = _build_setup()
    calibrator = RasterCalibrator(slm, camera, focal_length=FOCAL_LENGTH)
    model = calibrator._build_slm_camera_model()
    assert np.isclose(model.focal_length, FOCAL_LENGTH)

    mismatched = RasterCalibrator(slm, camera, focal_length=FOCAL_LENGTH * 1.2)
    with pytest.raises(ValueError, match="focal_length"):
        mismatched._ensure_camera_mapping(_synthetic_mapping(), model)

    # The matching case still goes through.
    calibrator._ensure_camera_mapping(_synthetic_mapping(), model)
    assert calibrator.camera_mapping is not None


def test_addressable_half_extent_needs_no_model() -> None:
    """The addressable extent is analytic, so it builds nothing.

    It is ``wavelength * focal_length / (2 * pitch)`` per axis, which the calibrator can
    evaluate from the SLM alone. It used to construct a whole SLMFFT on the fly to ask
    the model for it.
    """
    slm, camera = _build_setup()
    calibrator = RasterCalibrator(slm, camera, focal_length=FOCAL_LENGTH)

    extent = calibrator._addressable_half_extent()

    assert calibrator._slm_camera_model is None  # nothing was built
    pitch_y, pitch_x = (float(pitch) for pitch in slm.pixel_size)
    wavelength = float(slm.wavelength)
    assert np.isclose(extent[0], wavelength * FOCAL_LENGTH / (2 * pitch_x))
    assert np.isclose(extent[1], wavelength * FOCAL_LENGTH / (2 * pitch_y))
    # and it agrees with what the model would have said
    assert np.allclose(
        extent, calibrator._build_slm_camera_model().addressable_half_extent()
    )
