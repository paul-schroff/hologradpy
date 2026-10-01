"""Tests for RasterCalibrator's coarse-mapping-driven linear-phase placement.

Covers the model-free tilt/orientation maths on a synthetic CameraMapping (fast)
and an end-to-end check that the auto-computed tilt actually lands the spot where
intended on a small simulated setup, for an aligned and a rotated camera.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest
import torch

from hologradpy.analysis.fitting import fit_interferometric_fringes, remove_tilt
from hologradpy.analysis.unwrapping import unwrap_nonuniform
from hologradpy.hardware import SimulatedSLMTorch, SimulatedCameraTorch
from hologradpy.optics.complex_amplitude import (
    ComplexAmplitude,
    FieldGeometry,
)
from hologradpy.optics.systems import SLMFFTAffine
from hologradpy.optics.modules.slm_fields import PixelwiseSLMField
from hologradpy.profiles.amplitude import (
    gaussian_beam_intensity,
)
from hologradpy.calibration import (
    RasterCalibrator,
    get_diffraction_spot_position,
)
from hologradpy.calibration.camera_mapping import (
    CameraMapping,
    FocalSpotFit,
)
from hologradpy.calibration.wavefront.raster_calibration import (
    SuperpixelSlicer,
)
from hologradpy.roi import ROI
from hologradpy.utils import as_image

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

DEVICE = torch.device("cpu")
FOCAL_LENGTH = 0.25
RASTER_MODULE = "hologradpy.calibration.wavefront.raster_calibration.raster_calibrator"


def _build_setup(
    camera_angle: float = 0.0,
    read_noise: float = 0.0,
    power_std: float | None = None,
    power_seed: int | None = None,
    exposure_bounds: tuple[float, float] = (0.0, 1.0),
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
    # The ND filter brings the full-aperture spot to half full scale at about 4 us, and
    # a 32 x 32 superpixel at the edge of the beam within the 1 s bound.
    camera = SimulatedCameraTorch(
        hardware,
        exposure_bounds=exposure_bounds,
        read_noise=read_noise,
        power_std=power_std,
        power_seed=power_seed,
        nd_filter_optical_density=6.0,
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
    return CameraMapping(
        timestamp=datetime.now(),
        name="coarse",
        transform=transform,
        detected_points=[],
        calculated_points=[],
        zeroth_order_position=zeroth,
        spot_fit=FocalSpotFit(waist=8e-6),
    )


# --- fast unit tests (synthetic mapping, model-free path) ----------------------


def _zeroth_xy(mapping):
    return np.array(
        [mapping.zeroth_order_position[1], mapping.zeroth_order_position[0]]
    )


def test_without_an_output_pitch_the_fit_grids_are_only_turned_and_mirrored():
    """With neither a model nor a recorded output pitch, the map from camera metres to
    focal-plane metres is the rotation and mirror of the mapping, at unit
    magnification.
    """
    slm, camera = _build_setup()
    calibrator = RasterCalibrator(slm, camera, focal_length=FOCAL_LENGTH)
    calibrator.camera_mapping = _synthetic_mapping(rotation_deg=30.0, scale=2.0)
    matrix = calibrator._orientation_matrix()
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(2), atol=1e-9)
    # An aligned camera gives the identity, so the fit grids are left untouched.
    calibrator.camera_mapping = _synthetic_mapping(rotation_deg=0.0, scale=1.7)
    np.testing.assert_allclose(calibrator._orientation_matrix(), np.eye(2), atol=1e-9)


def test_the_fit_grids_carry_camera_metres_into_focal_plane_metres():
    """One camera pixel along x moves the focal-plane coordinate by the model pixels the
    mapping sends it to, at the recorded output pitch.
    """
    slm, camera = _build_setup()
    calibrator = RasterCalibrator(slm, camera, focal_length=FOCAL_LENGTH)
    output_pitch = 25e-6
    calibrator.camera_mapping = replace(
        _synthetic_mapping(rotation_deg=20.0, scale=1.5),
        output_pixel_size=(output_pitch, output_pitch),
    )
    linear = calibrator.camera_mapping.affine.linear
    camera_pitch_x = camera.pixel_size[1]

    focal_x, focal_y = calibrator._orient_grid(
        [np.array([camera_pitch_x]), np.array([0.0])]
    )

    assert float(focal_x[0]) == pytest.approx(linear[0, 0] * output_pitch)
    assert float(focal_y[0]) == pytest.approx(linear[1, 0] * output_pitch)


def test_a_supplied_model_that_has_not_run_sets_the_scale():
    """A model whose output geometry is not built yet runs once, and its output pitch
    scales the map from camera metres to focal-plane metres.
    """
    slm, camera = _build_setup()
    calibrator = RasterCalibrator(slm, camera, focal_length=FOCAL_LENGTH)
    model = calibrator._build_slm_camera_model()
    assert model[-1].pixel_size_out is None

    calibrator._ensure_camera_mapping(_synthetic_mapping(), model)

    pitch_y, pitch_x = model[-1].pixel_size_out.tolist()[0]
    camera_pitch_y, camera_pitch_x = camera.pixel_size
    expected = (
        np.diag([pitch_x, pitch_y])
        @ calibrator.camera_mapping.affine.linear
        @ np.diag([1.0 / camera_pitch_x, 1.0 / camera_pitch_y])
    )
    np.testing.assert_allclose(calibrator._orientation_matrix(), expected)


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
    at the intended camera pixel, even for a rotated camera.
    """
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


def _corner_setup(exposure_bounds: tuple[float, float] = (0.0, 1.0)):
    """A calibrator with an aligned synthetic camera mapping, plus the corner slices,
    ROI and centered detection spot used by ``calibrate_lattice_corner_tilts``.
    """
    slm, camera = _build_setup(read_noise=6.0, exposure_bounds=exposure_bounds)
    calibrator = RasterCalibrator(slm, camera, focal_length=FOCAL_LENGTH)
    calibrator.camera_mapping = _synthetic_mapping()
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


def test_lattice_corner_slices_span_the_slm_corners():
    """The four corner regions come from the SLM shape the slicer was built with."""
    slm_shape = (1024, 1280)
    slicer = SuperpixelSlicer(slm_shape, 4, 4, 32, 32)
    size = 16
    height, width = slm_shape

    assert slicer.lattice_corner_slices(size) == [
        (slice(0, size), slice(0, size)),
        (slice(0, size), slice(width - size, width)),
        (slice(height - size, height), slice(0, size)),
        (slice(height - size, height), slice(width - size, width)),
    ]


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


@pytest.mark.parametrize(
    ("rotation_deg", "mirror", "camera_to_focal_plane"),
    [
        (0.0, False, [[1.0, 0.0], [0.0, 1.0]]),
        (180.0, False, [[-1.0, 0.0], [0.0, -1.0]]),
        (0.0, True, [[1.0, 0.0], [0.0, -1.0]]),
    ],
)
def test_corner_steering_recovers_offset_under_noise(
    monkeypatch, rotation_deg, mirror, camera_to_focal_plane
):
    """The spot offset is measured on the camera and removed in focal-plane axes,
    for an aligned, a rotated and a mirrored camera.
    """
    calibrator, corner_slices, roi, spot_center, window = _corner_setup()
    calibrator.camera_mapping = _synthetic_mapping(
        rotation_deg=rotation_deg, mirror=mirror
    )
    window_height, window_width = window
    pitch_x = calibrator.camera.pixel_size[1]
    pitch_y = calibrator.camera.pixel_size[0]
    lattice_tilt = (300e-6, 300e-6)
    shift_px = (7, -5)  # spot offset from the window center, in pixels

    rng = np.random.default_rng(1)

    def fake_capture(exposure, frame_averages):
        yy, xx = np.mgrid[0:window_height, 0:window_width]
        center_x = window_width / 2 + shift_px[0]
        center_y = window_height / 2 + shift_px[1]
        spot = 300.0 * np.exp(
            -((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * 4.0**2)
        )
        return spot + rng.normal(16.0, 4.0, size=(window_height, window_width))

    monkeypatch.setattr(calibrator, "_capture_averaged", fake_capture)
    tilts = calibrator.calibrate_lattice_corner_tilts(
        corner_slices, lattice_tilt, roi, spot_center, exposure_time=1e-3
    )
    # Steering removes the measured offset, so every corner lands on the same tilt.
    camera_offset = np.array([shift_px[0] * pitch_x, shift_px[1] * pitch_y])
    expected = tuple(
        np.asarray(lattice_tilt) - np.asarray(camera_to_focal_plane) @ camera_offset
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


def test_a_corner_overexposed_at_the_shortest_exposure_is_captured_there(monkeypatch):
    """A corner spot that overexposes the camera at every exposure is captured at the
    shortest exposure the camera allows.
    """
    calibrator, corner_slices, roi, spot_center, window = _corner_setup(
        exposure_bounds=(1e-3, 1.0)
    )
    camera = calibrator.camera
    camera.set_exposure(0.01)
    overexposed_frame = np.full(
        tuple(camera.sensor_resolution), camera.max_pixel_value
    )
    monkeypatch.setattr(camera, "get_image", lambda *args, **kwargs: overexposed_frame)
    corner_exposures = []
    rng = np.random.default_rng(3)

    def capture(exposure, frame_averages):
        corner_exposures.append(exposure)
        return rng.normal(16.0, 2.0, size=window)

    monkeypatch.setattr(calibrator, "_capture_averaged", capture)
    calibrator.calibrate_lattice_corner_tilts(
        corner_slices, (300e-6, 300e-6), roi, spot_center
    )

    assert corner_exposures == pytest.approx([1e-3] * 4)


# --- power normalization corrects laser drift ----------------------------------


def test_normalize_power_removes_laser_fluctuation(monkeypatch):
    """A PowerInstability fluctuates the laser power per frame. With normalize_power
    the reference spot shares each frame, so the recovered map is (almost) the same as
    with a stable laser; without it the fluctuation clearly changes the map.
    """
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
    assert record.complex_amplitude.resolution == tuple(slm.resolution)
    assert torch.isfinite(record.complex_amplitude.as_tensor()).all()

    # The two paths that a bare array broke.
    field_module = PixelwiseSLMField.from_calibration_data(record)
    assert field_module.init_field is record.complex_amplitude
    assert as_image(record.complex_amplitude).shape == tuple(slm.resolution)


def test_calibrate_records_the_scan_for_the_animation(tmp_path) -> None:
    """The animation and the full-frame snapshot read the displayed phases, which
    only measure_phase recorded, so a calibration run through calibrate() could not be
    animated.
    """
    from hologradpy.calibration.wavefront.raster_calibration import (
        RasterCalibratorVisualizer,
    )

    slm, camera = _build_setup()
    calibrator = RasterCalibrator(slm, camera, focal_length=FOCAL_LENGTH)
    record = calibrator.calibrate(
        number_of_superpixels=(4, 4),
        camera_mapping=_synthetic_mapping(),
        verbose=False,
        record_displayed_phases=True,
    )

    data = record.visualization_data
    assert data.displayed_slm_phases is not None
    assert len(data.displayed_slm_phases) == data.camera_images.shape[0]
    assert data.full_frame_image is not None

    visualizer = RasterCalibratorVisualizer(data)
    gif = visualizer.save_gif(str(tmp_path / "scan.gif"), max_frames=2)
    assert (tmp_path / "scan.gif").exists(), gif


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


# --- measured phase and fringe fits ----------------------------------------------


def _aberrated_setup(camera_angle: float = 0.0, magnification: float = 1.0):
    """A setup whose beam carries a known phase, with a camera fine enough to resolve
    the fringes of the most distant superpixels.

    A relay of ``magnification`` sits between the focal plane and the sensor, so each
    camera pixel is that many times the focal-plane distance it samples.

    Returns the SLM, the camera, the injected phase in radians and the 1/e^2 region of
    the beam.
    """
    beam_radius = 2.0e-3
    geometry = FieldGeometry(
        resolution=(256, 320),
        pixel_size=torch.tensor([12.5e-6, 12.5e-6], device=DEVICE),
        wavelength=torch.tensor(0.630e-6, device=DEVICE),
    )
    slm = SimulatedSLMTorch(input_geometry=geometry, bitdepth=8)
    grid_x, grid_y = geometry.get_spatial_grid()
    intensity = gaussian_beam_intensity(grid_x, grid_y, beam_radius=beam_radius)
    x = grid_x / beam_radius
    y = grid_y / beam_radius
    injected_phase = 3.0 * (x**2 + y**2) + 2.0 * x * y + 1.5 * x**3
    beam = ComplexAmplitude(
        intensity.sqrt() * torch.exp(1j * injected_phase),
        wavelength=geometry.wavelength,
        pixel_size=geometry.pixel_size,
        power=1e-3,
    )
    hardware = SLMFFTAffine(
        input_geometry=geometry,
        virtual_slm=slm.virtual_slm,
        camera_resolution=(240, 320),
        camera_pixel_size=(12e-6, 12e-6),
        focal_length=FOCAL_LENGTH,
        slm_field=PixelwiseSLMField(beam),
        padded_resolution=(1024, 1024),
        camera_angle=camera_angle,
        camera_shift=(0, 0),
    )
    # The ND filter of _build_setup, which keeps the exposures realistic.
    camera = SimulatedCameraTorch(hardware, nd_filter_optical_density=6.0)
    # The simulator samples the focal plane at its model's pitch, so the relay only
    # changes the pitch the camera reports.
    camera._pixel_size = camera._pixel_size * magnification
    camera.set_exposure(1e-3)
    beam_region = (intensity >= intensity.max() * np.exp(-2)).numpy()
    return slm, camera, injected_phase.numpy(), beam_region


@pytest.mark.parametrize("camera_angle", [0.0, 180.0])
def test_the_measured_phase_is_that_of_the_incident_field(camera_angle):
    """measure_phase returns the phase of the incident field, the convention of every
    calibration record, so it follows the phase injected on the beam.

    No mapping is supplied, so a coarse mapping is measured and used to orient the
    fringe fits, also on a camera rotated by 180 degrees.
    """
    slm, camera, injected_phase, beam_region = _aberrated_setup(camera_angle)
    calibrator = RasterCalibrator(slm, camera, focal_length=FOCAL_LENGTH)

    phase, _, _ = calibrator.measure_phase(
        number_of_superpixels_x=8,
        number_of_superpixels_y=6,
        superpixel_width=40,
        superpixel_height=42,
        linear_phase_tilt=(1.0e-3, 0.7e-3),
        verbose=False,
    )

    # The piston and tilt depend on the reference superpixel and on the centring of
    # the camera window, so both are removed.
    measured = remove_tilt(phase, mask=beam_region)[beam_region]
    injected = remove_tilt(injected_phase, mask=beam_region)[beam_region]
    assert np.corrcoef(measured, injected)[0, 1] > 0.7


@pytest.mark.parametrize("magnification", [1.5, 0.6])
def test_the_measured_phase_follows_the_injected_phase_through_a_magnifying_relay(
    magnification,
):
    """The coarse mapping records the output pitch of its model, so the fringe fits
    take camera distances back into the focal plane through the relay.
    """
    slm, camera, injected_phase, beam_region = _aberrated_setup(
        magnification=magnification
    )
    calibrator = RasterCalibrator(slm, camera, focal_length=FOCAL_LENGTH)

    phase, _, _ = calibrator.measure_phase(
        number_of_superpixels_x=8,
        number_of_superpixels_y=6,
        superpixel_width=40,
        superpixel_height=42,
        linear_phase_tilt=(1.0e-3, 0.7e-3),
        verbose=False,
    )

    measured = remove_tilt(phase, mask=beam_region)[beam_region]
    injected = remove_tilt(injected_phase, mask=beam_region)[beam_region]
    assert np.corrcoef(measured, injected)[0, 1] > 0.9


def _fail_fringe_fits(monkeypatch, number_of_failures: int | None = None) -> list[int]:
    """Make the first ``number_of_failures`` fringe fits of scanned superpixels raise,
    or all of them when None.

    The reference superpixel is fitted at zero separation and every scanned superpixel
    at a nonzero one. Without pointing compensation, the scan fits each superpixel
    once, in scan order.

    Returns the scan indices of the failed fits, filled in as the scan runs.
    """
    fit_count = 0
    failed_indices: list[int] = []

    def fit_or_fail(x, y, image, separation_x, separation_y, *args, **kwargs):
        nonlocal fit_count
        index = fit_count
        fit_count += 1
        is_reference = separation_x == 0 and separation_y == 0
        if not is_reference and (
            number_of_failures is None or len(failed_indices) < number_of_failures
        ):
            failed_indices.append(index)
            raise RuntimeError("Optimal parameters not found.")
        return fit_interferometric_fringes(
            x, y, image, separation_x, separation_y, *args, **kwargs
        )

    monkeypatch.setattr(f"{RASTER_MODULE}.fit_interferometric_fringes", fit_or_fail)
    return failed_indices


def test_a_failed_fringe_fit_is_left_out_of_the_unwrapping(monkeypatch):
    """A superpixel whose fringe fit fails has no phase, so the unwrapping runs on the
    other superpixels and the phase map stays finite.
    """
    slm, camera = _build_setup()
    calibrator = RasterCalibrator(slm, camera, focal_length=FOCAL_LENGTH)
    failed_indices = _fail_fringe_fits(monkeypatch, number_of_failures=1)
    unwrapped_centers = []

    def record_centers(x, y, phase):
        unwrapped_centers.extend(zip(x.tolist(), y.tolist()))
        return unwrap_nonuniform(x, y, phase)

    monkeypatch.setattr(f"{RASTER_MODULE}.unwrap_nonuniform", record_centers)

    with pytest.warns(UserWarning, match=r"1 of \d+ fringe fits failed"):
        phase, _, _ = calibrator.measure_phase(
            number_of_superpixels_x=4,
            number_of_superpixels_y=4,
            superpixel_width=40,
            superpixel_height=32,
            linear_phase_tilt=(1.5e-3, 1.5e-3),
            camera_mapping=_synthetic_mapping(),
            verbose=False,
        )

    centers = calibrator.visualization_data.superpixel_coordinates
    failed_center = tuple(centers[:, failed_indices[0]].tolist())
    assert len(unwrapped_centers) == centers.shape[1] - 1
    assert failed_center not in unwrapped_centers
    assert np.isfinite(phase).all()


def test_a_scan_with_too_few_valid_fringe_fits_raises(monkeypatch):
    """The unwrapping needs three superpixels with a phase, and a scan in which every
    fit but the reference fails has one.
    """
    slm, camera = _build_setup()
    calibrator = RasterCalibrator(slm, camera, focal_length=FOCAL_LENGTH)
    _fail_fringe_fits(monkeypatch)

    with pytest.raises(RuntimeError, match="fringe fits are valid"):
        calibrator.measure_phase(
            number_of_superpixels_x=4,
            number_of_superpixels_y=4,
            superpixel_width=40,
            superpixel_height=32,
            linear_phase_tilt=(1.5e-3, 1.5e-3),
            camera_mapping=_synthetic_mapping(),
            verbose=False,
        )


# --- camera settings -------------------------------------------------------------


def test_calibrate_passes_the_camera_roi_as_height_and_width(monkeypatch):
    """Both scans take the camera ROI as ``(height, width)``, while get_roi_size
    returns ``(width, height)``. Non-square superpixels tell the two orders apart.
    """
    slm, camera = _build_setup()
    calibrator = RasterCalibrator(slm, camera, focal_length=FOCAL_LENGTH)
    roi_sizes = {}

    def measure_intensity(*args, **kwargs):
        roi_sizes["intensity"] = args[5]
        return np.ones(tuple(slm.resolution)), np.zeros((1, 1, 1))

    def measure_phase(*args, **kwargs):
        roi_sizes["phase"] = args[5]
        calibrator.visualization_data = None
        return np.zeros(tuple(slm.resolution)), None, None

    monkeypatch.setattr(calibrator, "measure_intensity", measure_intensity)
    monkeypatch.setattr(calibrator, "measure_phase", measure_phase)
    calibrator.calibrate(
        superpixel_size=(40, 32), camera_mapping=_synthetic_mapping(), verbose=False
    )

    roi_width, roi_height = calibrator.get_roi_size(40, 32)
    assert roi_width != roi_height
    assert roi_sizes == {
        "intensity": (roi_height, roi_width),
        "phase": (roi_height, roi_width),
    }


def test_the_spot_search_autoexposes(monkeypatch):
    """measure_intensity finds its spot on an autoexposed frame, below full scale and
    where the tilt places it.
    """
    slm, camera = _build_setup()
    calibrator = RasterCalibrator(slm, camera, focal_length=FOCAL_LENGTH)
    tilt = (1.5e-3, 1.5e-3)
    found_spots = []

    def find_spot(*args, **kwargs):
        result = get_diffraction_spot_position(*args, **kwargs)
        position, _, crop, _ = result
        found_spots.append((position, float(np.max(crop))))
        return result

    monkeypatch.setattr(f"{RASTER_MODULE}.get_diffraction_spot_position", find_spot)
    calibrator.measure_intensity(
        number_of_superpixels_x=2,
        number_of_superpixels_y=2,
        superpixel_width=32,
        superpixel_height=32,
        linear_phase_tilt=tilt,
        verbose=False,
    )

    height, width = camera.sensor_resolution
    pitch_y, pitch_x = camera.pixel_size
    assert len(found_spots) == 1
    (spot_x, spot_y), peak = found_spots[0]
    assert peak < camera.max_pixel_value
    assert np.hypot(
        spot_x - (width / 2 + tilt[0] / pitch_x),
        spot_y - (height / 2 + tilt[1] / pitch_y),
    ) < 3.0


def test_the_scans_put_the_camera_settings_back():
    """The intensity scan, the phase scan and the corner steering each leave the
    camera's exposure and region of interest as they found them.
    """
    calibrator, corner_slices, roi, spot_center, _ = _corner_setup()
    camera = calibrator.camera
    region = ROI(20, 30, 120, 200)
    camera.set_roi(region)
    camera.set_exposure(2e-3)

    calibrator.measure_intensity(
        number_of_superpixels_x=2,
        number_of_superpixels_y=2,
        superpixel_width=32,
        superpixel_height=32,
        linear_phase_tilt=(1.5e-3, 1.5e-3),
        verbose=False,
    )
    assert (camera.roi, camera.get_exposure()) == (region, 2e-3)

    calibrator.measure_phase(
        number_of_superpixels_x=4,
        number_of_superpixels_y=4,
        superpixel_width=40,
        superpixel_height=32,
        linear_phase_tilt=(1.5e-3, 1.5e-3),
        verbose=False,
    )
    assert (camera.roi, camera.get_exposure()) == (region, 2e-3)

    calibrator.calibrate_lattice_corner_tilts(
        corner_slices, (300e-6, 300e-6), roi, spot_center
    )
    assert (camera.roi, camera.get_exposure()) == (region, 2e-3)
