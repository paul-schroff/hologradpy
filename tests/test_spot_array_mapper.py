"""Tests for the SpotArrayMapper camera-mapping pipeline.

Covers the random-position sampler, the linear-superposition retriever's
default-argument regression, the inverse-variance waist average, an end-to-end
smoke test on a small simulated SLM + camera, and the general
CameraMapperVisualizer (spot-array and no-fit mappings).
"""

from __future__ import annotations

from datetime import datetime

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from hologradpy.hardware import SimulatedSLMTorch, SimulatedCameraTorch  # noqa: E402
from hologradpy.optics.complex_amplitude import (  # noqa: E402
    ComplexAmplitude,
    FieldGeometry,
)
from hologradpy.optics.systems import SLMFFT, SLMFFTAffine  # noqa: E402
from hologradpy.optics.modules.slm_fields import PixelwiseSLMField  # noqa: E402
from hologradpy.optics.modules.virtual_slms import VirtualSLM  # noqa: E402
from hologradpy.profiles.amplitude import (  # noqa: E402
    gaussian_beam_intensity,
)
from hologradpy.holography.phase_retrieval import (  # noqa: E402
    LinearSuperpositionPhaseRetriever,
)
from hologradpy.calibration.camera_mapping import (  # noqa: E402
    CameraMapperVisualizer,
    CameraMapping,
    CameraMappingVisualizationData,
    CoarseMapper,
    FocalSpotFit,
    MappingFit,
    SpotArrayMapper,
)
from hologradpy.calibration.spot_detection import disc_mask  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

DEVICE = torch.device("cpu")


def _build_setup(camera_angle: float = 0.0):
    """A small simulated SLM + camera + hologram model."""
    slm_geometry = FieldGeometry(
        resolution=(256, 320),
        pixel_size=torch.tensor([12.5e-6, 12.5e-6], device=DEVICE),
        wavelength=torch.tensor(0.630e-6, device=DEVICE),
    )
    slm = SimulatedSLMTorch(input_geometry=slm_geometry, bitdepth=8)

    gaussian_intensity = gaussian_beam_intensity(
        *slm_geometry.get_spatial_grid(), beam_radius=1e-3
    )
    gaussian_beam = ComplexAmplitude(
        gaussian_intensity.sqrt() + 0j,
        wavelength=slm_geometry.wavelength,
        pixel_size=slm_geometry.pixel_size,
    )

    simulated_camera_model = SLMFFTAffine(
        input_geometry=slm_geometry,
        virtual_slm=slm.virtual_slm,
        camera_resolution=(240, 320),
        camera_pixel_size=(30e-6, 30e-6),
        focal_length=0.25,
        slm_field=PixelwiseSLMField(gaussian_beam),
        padded_resolution=(512, 512),
        camera_angle=camera_angle,
        camera_shift=(0, 0),
    )
    camera = SimulatedCameraTorch(simulated_camera_model)
    camera.set_exposure(1e-3)
    # Forward-initialize the shared slm.virtual_slm so set_phase() works later.
    camera.get_image()

    slm_camera_model = SLMFFT(
        input_geometry=slm_geometry,
        virtual_slm=VirtualSLM(phase_scaling=1.0),
        slm_field=PixelwiseSLMField(gaussian_beam),
        focal_length=0.25,
        padded_resolution=(512, 512),
    )
    return slm, camera, slm_camera_model


@pytest.fixture(scope="module")
def simulated_setup():
    return _build_setup()


@pytest.fixture(scope="module")
def mapper(simulated_setup):
    slm, camera, slm_camera_model = simulated_setup
    return SpotArrayMapper(slm, camera, slm_camera_model)


@pytest.fixture(scope="module")
def spot_array_mapping(mapper):
    """Run the mapper once; reused by the e2e and visualizer tests."""
    return mapper.map_camera(number_of_spots=8, seed=1)


# --- _sample_positions --------------------------------------------------------


def test_sample_positions_inside_extent_and_separated(mapper):
    extent = (2.0e-3, 1.0e-3)
    minimum_separation = 1.0e-4
    generator = torch.Generator(device=DEVICE).manual_seed(7)
    positions = mapper._sample_positions(12, extent, minimum_separation, generator)

    assert positions.shape == (12, 2)
    xs = positions[:, 0].abs()
    ys = positions[:, 1].abs()
    assert torch.all(xs <= extent[0] / 2 + 1e-12)
    assert torch.all(ys <= extent[1] / 2 + 1e-12)

    distances = torch.cdist(positions, positions)
    distances.fill_diagonal_(float("inf"))
    assert torch.all(distances >= minimum_separation)


def test_sample_positions_is_reproducible(mapper):
    extent = (2.0e-3, 2.0e-3)
    args = (10, extent, 1.0e-4)
    first = mapper._sample_positions(
        *args, torch.Generator(device=DEVICE).manual_seed(3)
    )
    second = mapper._sample_positions(
        *args, torch.Generator(device=DEVICE).manual_seed(3)
    )
    assert torch.allclose(first, second)


def test_sample_positions_raises_when_infeasible(mapper):
    generator = torch.Generator(device=DEVICE).manual_seed(0)
    with pytest.raises(ValueError):
        # 100 spots each >= 1 mm apart cannot fit in a 1 mm x 1 mm box.
        mapper._sample_positions(100, (1.0e-3, 1.0e-3), 1.0e-3, generator)


# --- LinearSuperpositionPhaseRetriever (default-argument regression) ----------


def test_retriever_default_intensities_and_phases(simulated_setup):
    _, _, slm_camera_model = simulated_setup
    target_positions = torch.tensor(
        [[5.0e-4, 3.0e-4], [-2.0e-4, 1.0e-4], [0.0, -4.0e-4]], device=DEVICE
    )
    # No target_intensities / target_phases: exercises the None defaults that were
    # previously overwritten with None (and crashed).
    phase = LinearSuperpositionPhaseRetriever(
        slm_camera_model, target_positions
    ).retrieve_phase()

    assert phase.shape == slm_camera_model.input_geometry.resolution
    assert torch.isfinite(phase).all()


# --- inverse-variance waist average -------------------------------------------


def test_weighted_average_matches_hand_computed():
    values = np.array([2.0, 4.0])
    variances = np.array([1.0, 3.0])
    mean, uncertainty = SpotArrayMapper._weighted_average(values, variances)
    # weights = [1, 1/3]; mean = (2 + 4/3) / (4/3) = 2.5; sigma = sqrt(1 / (4/3)).
    assert mean == pytest.approx(2.5)
    assert uncertainty == pytest.approx(np.sqrt(3.0 / 4.0))


# --- end-to-end smoke test ----------------------------------------------------


def test_map_camera_returns_populated_mapping(spot_array_mapping):
    mapping = spot_array_mapping
    assert mapping.name == "spot_array"

    detected = np.asarray(mapping.detected_points, dtype=float)
    calculated = np.asarray(mapping.calculated_points, dtype=float)
    assert len(detected) >= 3
    assert len(detected) == len(calculated)

    # Per-spot Gaussian fits are populated and consistent in length.
    spot_fit = mapping.spot_fit
    assert spot_fit.parameters is not None
    assert spot_fit.covariances is not None
    assert len(spot_fit.parameters) == len(detected)
    assert spot_fit.waist > 0
    assert np.isfinite(spot_fit.waist_uncertainty)

    # The zeroth-order mask is a boolean full-sensor image.
    frames = mapping.visualization_data
    mask = np.asarray(frames.zeroth_order_mask)
    assert mask.dtype == bool
    assert mask.shape == tuple(frames.camera_image.shape)

    # The calculated points sit on the bright spots of the simulated image as
    # actually rendered (regression: they used to be analytic positions that
    # ignored the model's phase-convention mirror and landed on the ghost).
    simulated = np.asarray(frames.simulated_image)
    threshold = 0.1 * simulated.max()
    for x, y in calculated:
        xi, yi = int(round(x)), int(round(y))
        assert simulated[yi - 3:yi + 4, xi - 3:xi + 4].max() > threshold

    # The fitted affine maps detected (camera) points onto the analytic
    # (simulated-plane) points within a fraction of a pixel.
    transform = np.asarray(mapping.transform, dtype=float)
    mapped = detected @ transform[:, :2].T + transform[:, 2]
    rms = np.sqrt(np.mean(np.sum((mapped - calculated) ** 2, axis=1)))
    assert rms < 2.0

    # The reprojection error is computed by the mapper and stored on the
    # mapping (the visualizer only reads it).
    errors = np.asarray(mapping.fit.reprojection_errors, dtype=float)
    assert errors.shape == detected.shape
    np.testing.assert_allclose(errors, mapped - calculated, atol=1e-9)
    assert mapping.fit.reprojection_rms == pytest.approx(rms)

    # Detections excluded from the transform are recorded, possibly none for a
    # clean setup.
    assert mapping.fit.excluded_points is not None


def test_map_camera_recovers_rotated_camera():
    """Regression: a rotated camera must yield the rotated transform, not bogus
    background fits at the unrotated positions (which silently produced a
    rotation-free transform).
    """
    slm, camera, slm_camera_model = _build_setup(camera_angle=10.0)
    mapper = SpotArrayMapper(slm, camera, slm_camera_model)
    mapping = mapper.map_camera(number_of_spots=8, seed=1)

    detected = np.asarray(mapping.detected_points, dtype=float)
    calculated = np.asarray(mapping.calculated_points, dtype=float)
    transform = np.asarray(mapping.transform, dtype=float)

    # The camera view is rotated by +10 deg, so the camera->simulated transform
    # carries the inverse rotation. No 180-degree component: hardware and model
    # paths share the same (desired-phase) sign convention.
    angle = np.degrees(np.arctan2(transform[1, 0], transform[0, 0]))
    assert angle == pytest.approx(-10.0, abs=0.5)

    mapped = detected @ transform[:, :2].T + transform[:, 2]
    rms = np.sqrt(np.mean(np.sum((mapped - calculated) ** 2, axis=1)))
    assert rms < 1.0


def test_map_camera_fills_sensor_and_respects_border():
    """With no coarse mapping supplied, the mapper builds one internally and the
    detected spots fill the (rotated) sensor while keeping a border.
    """
    slm, camera, model = _build_setup(camera_angle=10.0)
    mapping = SpotArrayMapper(slm, camera, model).map_camera(
        number_of_spots=30, seed=1
    )
    detected = np.asarray(mapping.detected_points, dtype=float)
    height, width = camera.shape
    # Spots span most of each axis (fill the sensor) ...
    assert detected[:, 0].max() - detected[:, 0].min() > 0.5 * width
    assert detected[:, 1].max() - detected[:, 1].min() > 0.5 * height
    # ... and stay inside a border (never right at the sensor edge).
    assert detected[:, 0].min() > 5 and detected[:, 0].max() < width - 5
    assert detected[:, 1].min() > 5 and detected[:, 1].max() < height - 5


def test_map_camera_accepts_explicit_coarse_mapping():
    """A supplied coarse mapping is used directly (no internal coarse build)."""
    slm, camera, model = _build_setup(camera_angle=10.0)
    coarse = CoarseMapper(slm, camera, model).map_camera()
    mapping = SpotArrayMapper(slm, camera, model).map_camera(
        number_of_spots=20, seed=1, coarse_mapping=coarse
    )
    assert mapping.name == "spot_array"
    assert mapping.fit.reprojection_rms < 2.0


def test_disc_mask_membership():
    mask = disc_mask((10, 12), center=(5.0, 4.0), radius=3.0)
    assert mask.shape == (10, 12) and mask.dtype == bool
    assert mask[4, 5]  # center (row = y = 4, col = x = 5)
    assert mask[4, 8]  # exactly on the radius (<=)
    assert not mask[4, 9]  # just outside
    assert not mask[0, 0]


# --- _match_targets -------------------------------------------------------------


def _random_targets(number: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform([-1.5e-3, -1.0e-3], [1.5e-3, 1.0e-3], size=(number, 2))


def test_match_targets_recovers_large_rotation():
    targets = _random_targets(15, seed=2)
    pitch = 30e-6
    scale = 1.1 / pitch  # within the pruning tolerance of the expected scale
    angle = np.radians(120.0)
    rotation = scale * np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    translation = np.array([160.0, 120.0])

    rng = np.random.default_rng(3)
    detected_full = targets @ rotation.T + translation
    permutation = rng.permutation(len(targets))[:-2]  # 2 spots go undetected
    detected = detected_full[permutation] + rng.normal(0, 0.1, (len(permutation), 2))

    detected_indices, target_indices = SpotArrayMapper._match_targets(
        detected, targets, expected_scale=1.0 / pitch, tolerance=3.0
    )
    assert len(detected_indices) == len(permutation)
    # Every match pairs a detected point with the target it was generated from.
    assert np.array_equal(permutation[detected_indices], target_indices)


def test_match_targets_handles_mirrored_camera():
    targets = _random_targets(12, seed=4)
    pitch = 30e-6
    angle = np.radians(-35.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    mirror = np.array([[1.0, 0.0], [0.0, -1.0]])
    matrix = (1.0 / pitch) * rotation @ mirror
    detected = targets @ matrix.T + np.array([100.0, 80.0])

    detected_indices, target_indices = SpotArrayMapper._match_targets(
        detected, targets, expected_scale=1.0 / pitch, tolerance=3.0
    )
    assert len(detected_indices) == len(targets)
    # detected was built from targets in order, so the match is the identity.
    assert np.array_equal(detected_indices, target_indices)


def test_match_targets_raises_without_consensus():
    targets = _random_targets(10, seed=5)
    rng = np.random.default_rng(6)
    detected = rng.uniform(0, 300, size=(10, 2))  # unrelated point cloud
    with pytest.raises(RuntimeError):
        SpotArrayMapper._match_targets(
            detected, targets, expected_scale=1.0 / 30e-6, tolerance=1.0
        )


# --- _detect_peaks --------------------------------------------------------------


def test_detect_peaks_finds_spots_brightest_first():
    image = np.zeros((60, 80))
    spots = [((20.0, 15.0), 100.0), ((60.0, 40.0), 80.0), ((10.0, 50.0), 60.0)]
    ys, xs = np.indices(image.shape)
    for (x, y), amplitude in spots:
        image += amplitude * np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * 1.5**2))

    peaks = SpotArrayMapper._detect_peaks(
        image, number_of_peaks=5, blank_radius=6, threshold=10.0
    )
    assert len(peaks) == 3  # stops at the background threshold, not at 5
    for peak, ((x, y), _) in zip(peaks, spots):
        assert peak[0] == pytest.approx(x, abs=1.0)
        assert peak[1] == pytest.approx(y, abs=1.0)


# --- CameraMapperVisualizer ---------------------------------------------------


def test_visualizer_renders_spot_array_mapping(spot_array_mapping):
    figure = CameraMapperVisualizer(spot_array_mapping).render()
    titles = [axes.get_title() for axes in figure.axes]
    # The spot-array mapping carries fits, so the per-spot waist panel is drawn.
    assert any("waist" in title.lower() for title in titles)
    plt.close(figure)


def test_visualizer_marks_excluded_detections(spot_array_mapping):
    import dataclasses

    mapping = dataclasses.replace(
        spot_array_mapping,
        fit=dataclasses.replace(
            spot_array_mapping.fit, excluded_points=[(5.0, 5.0), (10.0, 12.0)]
        ),
    )
    figure = CameraMapperVisualizer(mapping).render()
    camera_axes = next(
        axes for axes in figure.axes if "Camera image" in axes.get_title()
    )
    # Matched markers plus the excluded-detection markers, with a legend.
    assert len(camera_axes.lines) == 2
    assert camera_axes.get_legend() is not None
    plt.close(figure)


def _no_fit_mapping() -> CameraMapping:
    """A minimal mapping without per-spot fits (mimics CheckerboardMapper)."""
    rng = np.random.default_rng(0)
    detected = [(2.0, 3.0), (8.0, 3.0), (8.0, 9.0), (2.0, 9.0)]
    calculated = [(4.0, 6.0), (16.0, 6.0), (16.0, 18.0), (4.0, 18.0)]
    transform = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    return CameraMapping(
        timestamp=datetime.now(),
        name="checkerboard",
        transform=transform,
        detected_points=detected,
        calculated_points=calculated,
        zeroth_order_position=(0.0, 0.0),
        spot_fit=FocalSpotFit(waist=1.0),
        fit=MappingFit(
            reprojection_errors=np.zeros((4, 2)), reprojection_rms=0.0
        ),
        visualization_data=CameraMappingVisualizationData(
            camera_image=rng.random((12, 16)),
            simulated_image=rng.random((24, 24)),
        ),
    )


def test_visualizer_renders_mapping_without_spot_fits():
    figure = CameraMapperVisualizer(_no_fit_mapping()).render()
    titles = [axes.get_title() for axes in figure.axes]
    # No fits -> no waist panel, but the residual panel is still drawn.
    assert not any("waist" in title.lower() for title in titles)
    assert any("residual" in title.lower() for title in titles)
    plt.close(figure)
