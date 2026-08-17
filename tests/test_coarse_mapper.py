"""Tests for CoarseMapper and the CameraMapping orientation properties.

The coarse mapper bootstraps the camera position / rotation / flip from
sequential probe spots; it must work when the camera is rotated, flipped, or
positioned so the zeroth order misses the sensor entirely.
"""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from hologradpy.hardware import (  # noqa: E402
    CameraOrientation,
    SimulatedCameraTorch,
    SimulatedSLMTorch,
)
from hologradpy.hardware import as_slm  # noqa: E402
from hologradpy.optics.complex_amplitude import (  # noqa: E402
    ComplexAmplitude,
    FieldGeometry,
)
from hologradpy.optics.systems import SLMFFT, SLMCZT  # noqa: E402
from hologradpy.optics.modules.slm_fields import PixelwiseSLMField  # noqa: E402
from hologradpy.optics.modules.virtual_slms import VirtualSLM  # noqa: E402
from hologradpy.profiles.amplitude import (  # noqa: E402
    gaussian_beam_intensity,
    get_focal_spot_radius,
)
from hologradpy.calibration.camera_mapping import (  # noqa: E402
    CameraMapping,
    FocalSpotFit,
    CoarseMapper,
    CoarseMapperVisualizer,
    CoarseVisualizationData,
    SpotArrayMapper,
)
from hologradpy.calibration.camera_mapping.coarse_mapping.coarse_mapper import (  # noqa: E402
    _PROBE_SPACING_FRACTION,
)
from hologradpy.calibration.spot_detection import (  # noqa: E402
    _WINDOW_SPOT_RADII,
    background_noise,
    detect_spot,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

DEVICE = torch.device("cpu")


def _build_setup(
    camera_angle: float = 0.0,
    camera_shift: tuple[float, float] = (0, 0),
    camera_resolution: tuple[int, int] = (240, 320),
    rot: str = "0",
    fliplr: bool = False,
    pointing_focal_shift_std: float | None = None,
    background_scatter_power: float | None = None,
    background_scatter_grain_radius: float = 5e-6,
):
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
    simulated_camera_model = SLMCZT(
        input_geometry=geometry,
        virtual_slm=slm.virtual_slm,
        camera_resolution=camera_resolution,
        camera_pixel_size=(30e-6, 30e-6),
        focal_length=0.25,
        slm_field=PixelwiseSLMField(beam),
        camera_angle=camera_angle,
        camera_shift=camera_shift,
        pointing_focal_shift_std=pointing_focal_shift_std,
        pointing_seed=1,
    )
    camera = SimulatedCameraTorch(
        simulated_camera_model,
        orientation=CameraOrientation(rot, fliplr=fliplr),
        background_scatter_power=background_scatter_power,
        background_scatter_grain_radius=background_scatter_grain_radius,
        background_scatter_seed=0,
    )
    camera.set_exposure(1e-3)
    camera.get_image()
    model = SLMFFT(
        input_geometry=geometry,
        virtual_slm=VirtualSLM(phase_scaling=1.0),
        slm_field=PixelwiseSLMField(beam),
        focal_length=0.25,
        padded_resolution=(512, 512),
    )
    return slm, camera, model


# --- CameraMapping orientation properties --------------------------------------


def _mapping_with_transform(linear: np.ndarray) -> CameraMapping:
    from datetime import datetime

    transform = np.hstack([linear, np.zeros((2, 1))])
    return CameraMapping(
        timestamp=datetime.now(),
        name="synthetic",
        transform=transform,
        detected_points=[],
        calculated_points=[],
        zeroth_order_position=(0.0, 0.0),
        spot_fit=FocalSpotFit(waist=1.0),
    )


def _rotation(angle_degrees: float) -> np.ndarray:
    angle = np.radians(angle_degrees)
    return np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )


def test_orientation_properties_pure_rotation():
    mapping = _mapping_with_transform(2.0 * _rotation(35.0))
    assert not mapping.is_mirrored
    assert mapping.rotation_degrees == pytest.approx(35.0)
    assert mapping.scales == pytest.approx((2.0, 2.0))


def test_orientation_properties_mirrored():
    mirror = np.array([[1.0, 0.0], [0.0, -1.0]])
    mapping = _mapping_with_transform(1.5 * _rotation(-20.0) @ mirror)
    assert mapping.is_mirrored
    assert mapping.scales == pytest.approx((1.5, 1.5))


def test_orientation_properties_anisotropic_scale():
    mapping = _mapping_with_transform(np.diag([3.0, 1.0]))
    assert not mapping.is_mirrored
    assert mapping.rotation_degrees == pytest.approx(0.0)
    assert mapping.scales == pytest.approx((3.0, 1.0))


def test_addressable_half_extent_is_nyquist_deflection():
    _, _, model = _build_setup()
    half = model.addressable_half_extent()
    expected = 0.630e-6 * 0.25 / (2 * 12.5e-6)
    assert half[0] == pytest.approx(expected)
    assert half[1] == pytest.approx(expected)


# --- background_noise -----------------------------------------------------------


def test_background_noise_recovers_gaussian_sigma():
    rng = np.random.default_rng(0)
    sample = rng.normal(10.0, 3.0, size=(400, 400))
    # MAD -> sigma with the analytic 1/Phi^-1(0.75) factor recovers the true
    # standard deviation of a Gaussian sample.
    assert background_noise(sample) == pytest.approx(3.0, rel=0.03)


# --- detect_spot ----------------------------------------------------------------

PIXEL_UM = 3.45
SPOT_RADIUS = 10e-6           # 1/e^2 radius -> ~2.9 px at 3.45 um pitch
BITRESOLUTION = 1024


def _fake_camera(pixel_um=PIXEL_UM, bitresolution=BITRESOLUTION):
    # detect_spot reads native .pixel_size (y, x) metres and .max_pixel_value.
    pitch_m = pixel_um * 1e-6
    return SimpleNamespace(
        pixel_size=np.array([pitch_m, pitch_m]),
        max_pixel_value=bitresolution - 1,
    )


def _gaussian_frame(shape, center, amplitude, sigma_px, *, background=5.0,
                    noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    rows, columns = np.indices(shape)
    column0, row0 = center
    frame = background + amplitude * np.exp(
        -((columns - column0) ** 2 + (rows - row0) ** 2) / (2 * sigma_px**2)
    )
    return frame + rng.normal(0.0, noise, shape)


def test_detect_spot_finds_clean_gaussian():
    # 1/e^2 radius w0 -> intensity sigma = w0/2 in the exp(-r^2/2s^2) sense.
    sigma_px = (SPOT_RADIUS / (PIXEL_UM * 1e-6)) / 2
    frame = _gaussian_frame((80, 120), (40, 30), amplitude=800.0, sigma_px=sigma_px)
    peak = detect_spot(frame, SPOT_RADIUS, _fake_camera())
    assert peak is not None
    row, column = peak
    assert (column, row) == pytest.approx((40, 30), abs=1)


def test_detect_spot_rejects_pure_noise():
    rng = np.random.default_rng(1)
    frame = 5.0 + rng.normal(0.0, 1.0, (80, 120))
    assert detect_spot(frame, SPOT_RADIUS, _fake_camera()) is None


def test_detect_spot_rejects_spot_below_dynamic_range_floor():
    sigma_px = (SPOT_RADIUS / (PIXEL_UM * 1e-6)) / 2
    # Amplitude 50 is well above the noise but below 0.1 * 1024 = 102.4.
    frame = _gaussian_frame((80, 120), (40, 30), amplitude=50.0, sigma_px=sigma_px)
    assert detect_spot(frame, SPOT_RADIUS, _fake_camera()) is None


def test_detect_spot_accepts_sub_pixel_spot():
    """A legitimately sub-pixel spot is a single bright pixel; the min-core
    floor of 1 must not reject it."""
    rng = np.random.default_rng(2)
    frame = 5.0 + rng.normal(0.0, 1.0, (80, 120))
    frame[30, 40] += 800.0
    peak = detect_spot(frame, spot_radius=2e-6, camera=_fake_camera())  # ~0.58 px
    assert peak == (30, 40)


def test_detect_spot_rejects_peak_at_border():
    sigma_px = (SPOT_RADIUS / (PIXEL_UM * 1e-6)) / 2
    # Peak at column 1, within border_margin (round(2.9)=3) of the left edge.
    frame = _gaussian_frame((80, 120), (1, 30), amplitude=800.0, sigma_px=sigma_px)
    assert detect_spot(frame, SPOT_RADIUS, _fake_camera()) is None


def test_detect_spot_rejects_unenclosed_broad_blob():
    """A broad plateau still bright at the window edge (the shoulder of an
    order sitting off the sensor) is rejected by the enclosure gate."""
    frame = _gaussian_frame((120, 120), (60, 60), amplitude=800.0, sigma_px=100.0)
    assert detect_spot(frame, SPOT_RADIUS, _fake_camera()) is None


# --- coarse mapping e2e ---------------------------------------------------------


def test_coarse_mapping_recovers_rotated_camera():
    slm, camera, model = _build_setup(camera_angle=10.0, camera_shift=(20, -10))
    coarse = CoarseMapper(slm, camera, model).map_camera()

    assert coarse.name == "coarse"
    assert not coarse.is_mirrored
    assert coarse.rotation_degrees == pytest.approx(-10.0, abs=0.3)
    # Camera pitch (30 um) over simulated pixel size (24.6 um).
    assert coarse.scales[0] == pytest.approx(1.219, abs=0.02)
    assert coarse.scales[1] == pytest.approx(1.219, abs=0.02)
    assert coarse.fit.reprojection_rms < 1.0


def test_coarse_mapping_detects_flip():
    slm, camera, model = _build_setup(fliplr=True)
    coarse = CoarseMapper(slm, camera, model).map_camera()
    assert coarse.is_mirrored
    assert coarse.fit.reprojection_rms < 1.0


def test_coarse_mapping_detects_rot90():
    slm, camera, model = _build_setup(rot="90")
    coarse = CoarseMapper(slm, camera, model).map_camera()
    assert not coarse.is_mirrored
    assert abs(coarse.rotation_degrees) == pytest.approx(90.0, abs=0.3)
    assert coarse.fit.reprojection_rms < 1.0


def test_map_camera_suggests_camera_orientation():
    """find_camera_orientation records the mounting whose residual affine aligns with
    the model plane, without modifying the camera.

    The mounting, not the correction: a camera mirrored by its own fliplr is aligned by
    mounting it unflipped, which is what set_orientation takes.
    """
    slm, camera, model = _build_setup(fliplr=True)  # mirrored camera
    shape_before, transform_before = camera.shape, camera.transform
    coarse = CoarseMapper(slm, camera, model).map_camera(
        find_camera_orientation=True
    )
    assert coarse.orientation.suggested == CameraOrientation()
    residual = _mapping_with_transform(
        np.asarray(coarse.orientation.residual_transform)[:, :2]
    )
    assert not residual.is_mirrored
    assert residual.rotation_degrees == pytest.approx(0.0, abs=0.5)
    # Suggest-only: the camera orientation is untouched.
    assert camera.shape == shape_before
    assert camera.transform is transform_before


def test_the_suggested_orientation_can_be_adopted():
    """The point of suggesting one: the camera takes it, and a second mapping of the
    reoriented camera then has nothing left to suggest."""
    slm, camera, model = _build_setup(fliplr=True)
    first = CoarseMapper(slm, camera, model).map_camera(find_camera_orientation=True)

    camera.set_orientation(first.orientation.suggested)
    assert camera.orientation == first.orientation.suggested

    second = CoarseMapper(slm, camera, model).map_camera(find_camera_orientation=True)
    # Already aligned, so the nearest orientation is the one it is already in.
    assert second.orientation.suggested == CameraOrientation()
    assert not second.is_mirrored


def test_map_camera_suggests_rot90_orientation():
    slm, camera, model = _build_setup(rot="90")
    coarse = CoarseMapper(slm, camera, model).map_camera(
        find_camera_orientation=True
    )
    # A quarter-turned camera over an aligned model is aligned by unturning it.
    assert coarse.orientation.suggested == CameraOrientation()
    residual = _mapping_with_transform(
        np.asarray(coarse.orientation.residual_transform)[:, :2]
    )
    assert residual.rotation_degrees == pytest.approx(0.0, abs=0.5)


def test_map_camera_orientation_off_by_default():
    slm, camera, model = _build_setup(camera_angle=10.0, camera_shift=(20, -10))
    coarse = CoarseMapper(slm, camera, model).map_camera()
    assert coarse.orientation is None


def test_camera_mapping_records_camera_data_and_pickles(tmp_path):
    slm, camera, model = _build_setup()
    coarse = CoarseMapper(slm, camera, model).map_camera()
    assert coarse.camera_data is not None
    assert np.asarray(coarse.camera_data.orientation).shape == (2, 3)
    # The mapping (with the CameraData snapshot) still round-trips through pickle.
    path = str(tmp_path / "coarse_mapping.asdf")
    coarse.save(path)
    loaded = CameraMapping.load(path)
    assert loaded.camera_data.resolution == tuple(camera.shape)


def test_coarse_mapping_accepts_initial_tilt():
    # Centred sensor: the zeroth order is on it, so tilt (0, 0) is a valid seed
    # and the spiral search is skipped.
    slm, camera, model = _build_setup()
    coarse = CoarseMapper(slm, camera, model).map_camera(initial_tilt=(0.0, 0.0))
    assert coarse.fit.reprojection_rms < 1.0


def test_coarse_mapping_initial_tilt_without_spot_raises():
    # Zeroth order off the sensor: tilt (0, 0) lands no spot on the sensor, so
    # the supplied seed is rejected.
    slm, camera, model = _build_setup(
        camera_angle=10.0, camera_shift=(60, 100), camera_resolution=(120, 160)
    )
    with pytest.raises(ValueError):
        CoarseMapper(slm, camera, model).map_camera(initial_tilt=(0.0, 0.0))


def test_coarse_mapping_with_zeroth_order_off_sensor():
    """The camera only sees a region away from the zeroth order: the spiral
    search must find the sensor, and the zeroth-order position is extrapolated
    (legitimately off the sensor)."""
    slm, camera, model = _build_setup(
        camera_angle=10.0, camera_shift=(60, 100), camera_resolution=(120, 160)
    )
    coarse = CoarseMapper(slm, camera, model).map_camera()

    assert coarse.rotation_degrees == pytest.approx(-10.0, abs=0.5)
    assert coarse.fit.reprojection_rms < 1.0
    zeroth_y, zeroth_x = coarse.zeroth_order_position
    on_sensor = 0 <= zeroth_x < 160 and 0 <= zeroth_y < 120
    assert not on_sensor

    # The fine mapper, seeded with the coarse mapping, places the array on the
    # actual sensor and never probes the (unreachable) zeroth order.
    mapper = SpotArrayMapper(slm, camera, model)
    mapping = mapper.map_camera(
        number_of_spots=8, seed=1, coarse_mapping=coarse
    )
    assert len(mapping.detected_points) == 8
    assert np.asarray(mapping.fit.reprojection_rms) < 1.0


def test_coarse_mapping_survives_pointing_instability():
    """Real beam-pointing drift jitters the focal spot frame-to-frame. The
    probe search (and _is_static_background in particular) must still find and
    accept the spots, and the recovered transform must stay correct: the
    deterministic tilt step dwarfs a 1 um (~sub-px) jitter."""
    slm, camera, model = _build_setup(
        camera_angle=10.0, camera_shift=(20, -10), pointing_focal_shift_std=1e-6
    )
    coarse = CoarseMapper(slm, camera, model).map_camera()

    assert not coarse.is_mirrored
    assert coarse.rotation_degrees == pytest.approx(-10.0, abs=1.0)
    assert coarse.scales[0] == pytest.approx(1.219, abs=0.05)
    assert coarse.fit.reprojection_rms < 2.0

    # The seeded fine mapper still matches almost all its spots (a common
    # per-frame tilt is absorbed by the affine translation; the odd edge spot may
    # be jittered out now that the array fills the sensor).
    mapper = SpotArrayMapper(slm, camera, model)
    mapping = mapper.map_camera(
        number_of_spots=8, seed=1, coarse_mapping=coarse
    )
    assert len(mapping.detected_points) >= 6
    assert np.asarray(mapping.fit.reprojection_rms) < 1.0


def test_coarse_mapping_survives_background_scatter():
    """A static laser-speckle background (stray light added before the ND filter)
    raises the camera floor, but the coarse mapper still recovers the transform."""
    slm, camera, model = _build_setup(
        camera_angle=10.0, camera_shift=(20, -10), background_scatter_power=2e-8
    )
    # The speckle raises the camera floor above the no-scatter case (median 0).
    assert np.median(np.asarray(camera.get_image())) > 0

    coarse = CoarseMapper(slm, camera, model).map_camera()
    assert not coarse.is_mirrored
    assert coarse.rotation_degrees == pytest.approx(-10.0, abs=0.5)
    assert coarse.fit.reprojection_rms < 1.0


# --- exposure calibration (_calibrate_exposure) ---------------------------------


def _calibration_args(slm, resolution):
    """(focal_length, half_extent, search_step, spot_radius) for
    _calibrate_exposure, mirroring how map_camera derives them."""
    slm = as_slm(slm)  # the mapper works on the native (adapter) interface
    focal_length = 0.25
    beam_diameter = min(
        slm.resolution[i] * slm.pixel_size[i] for i in range(2)
    )
    spot_radius = get_focal_spot_radius(
        beam_radius=0.5 * beam_diameter,
        wavelength=slm.wavelength,
        focal_length=focal_length,
    )
    half_extent = (
        slm.wavelength * focal_length / (2.0 * slm.pixel_size[1]),
        slm.wavelength * focal_length / (2.0 * slm.pixel_size[0]),
    )
    field_of_view = (resolution[1] * 30e-6, resolution[0] * 30e-6)
    search_step = (
        _PROBE_SPACING_FRACTION * min(field_of_view)
        - 2.0 * _WINDOW_SPOT_RADII * spot_radius
    )
    return focal_length, half_extent, search_step, spot_radius


def test_calibrate_exposure_uses_visible_array():
    """With the zeroth order off the sensor, the randomised-phase probe array is
    visible, so _calibrate_exposure returns a fixed exposure (not the per-probe-
    ladder sentinel None)."""
    slm, camera, model = _build_setup(
        camera_angle=10.0, camera_shift=(60, 100), camera_resolution=(120, 160)
    )
    mapper = CoarseMapper(slm, camera, model)
    exposure = mapper._calibrate_exposure(*_calibration_args(slm, (120, 160)))
    assert exposure is not None
    assert exposure >= 0.0


def test_calibrate_exposure_none_when_zeroth_order_on_sensor():
    """When the zeroth order lands on the sensor no array calibration is needed;
    the helper returns None so the search keeps its normal per-probe path."""
    slm, camera, model = _build_setup()  # centred sensor: DC on it
    mapper = CoarseMapper(slm, camera, model)
    assert mapper._calibrate_exposure(*_calibration_args(slm, (240, 320))) is None


def test_calibrate_exposure_rejects_speckle_as_zeroth_order():
    """A bright static speckle background must not be mistaken for the zeroth
    order. The zero-tilt probe latches onto a speckle grain, but the 0/pi-grating
    confirmation rejects it (it does not dim), so the array path is taken."""
    slm, camera, model = _build_setup(
        camera_angle=10.0, camera_shift=(60, 100), camera_resolution=(120, 160),
        background_scatter_power=2e-8, background_scatter_grain_radius=60e-6,
    )
    mapper = CoarseMapper(slm, camera, model)
    focal_length, _, _, spot_radius = _calibration_args(slm, (120, 160))
    # Precondition: the raw zero-tilt probe does find a spurious (speckle) spot.
    assert (
        mapper._spot_on_sensor((0.0, 0.0), focal_length, None, spot_radius)
        is not None
    )
    # But the 0/pi confirmation rejects it, so a fixed array exposure is returned.
    assert mapper._calibrate_exposure(*_calibration_args(slm, (120, 160))) is not None


def test_calibrate_exposure_warns_and_clamps_below_hardware_bound():
    """A per-probe exposure below the camera's minimum hardware exposure warns
    the user and clamps to the bound."""
    slm, camera, model = _build_setup(
        camera_angle=10.0, camera_shift=(60, 100), camera_resolution=(120, 160)
    )
    camera._exposure_bounds = (5e-3, 1.0)  # force a large hardware minimum
    mapper = CoarseMapper(slm, camera, model)
    with pytest.warns(UserWarning, match="below the camera's minimum"):
        exposure = mapper._calibrate_exposure(*_calibration_args(slm, (120, 160)))
    assert exposure == pytest.approx(5e-3)


def test_calibrate_exposure_falls_back_on_autoexposure_rail(monkeypatch):
    """A genuinely too-dim array rails the autoexposure; the helper returns None
    so the per-probe ladder takes over."""
    slm, camera, model = _build_setup(
        camera_angle=10.0, camera_shift=(60, 100), camera_resolution=(120, 160)
    )
    mapper = CoarseMapper(slm, camera, model)

    def rail(*args, **kwargs):
        raise RuntimeError("autoexposure has railed")

    monkeypatch.setattr(camera, "autoexpose", rail)
    assert mapper._calibrate_exposure(*_calibration_args(slm, (120, 160))) is None


def test_map_camera_falls_back_to_ladder_when_calibration_returns_none(monkeypatch):
    """When calibration returns None (dim array / rail), map_camera still maps
    correctly via the per-probe ladder."""
    slm, camera, model = _build_setup(
        camera_angle=10.0, camera_shift=(60, 100), camera_resolution=(120, 160)
    )
    mapper = CoarseMapper(slm, camera, model)
    monkeypatch.setattr(mapper, "_calibrate_exposure", lambda *a, **k: None)
    coarse = mapper.map_camera()
    assert coarse.rotation_degrees == pytest.approx(-10.0, abs=0.5)
    assert coarse.fit.reprojection_rms < 1.0


# --- CoarseMapperVisualizer -----------------------------------------------------


def test_coarse_visualization_data_populated_and_visualizer_renders():
    """A zeroth-order-off-sensor mapping records every stage capture as its
    visualization_data and the CoarseMapperVisualizer renders all four panels."""
    from matplotlib.figure import Figure

    slm, camera, model = _build_setup(
        camera_angle=10.0, camera_shift=(60, 100), camera_resolution=(120, 160)
    )
    mapping = CoarseMapper(slm, camera, model).map_camera()

    data = mapping.visualization_data
    assert data is not None
    assert data.array_image is not None       # array path was taken
    assert data.walk_image is not None         # centre-search walked
    assert np.asarray(data.probe_image).ndim == 2
    assert np.asarray(data.array_spot_positions).shape[1] == 2
    assert len(data.array_spot_positions) > 0
    assert np.asarray(data.sensor_rectangle).shape == (4, 2)
    assert data.output_resolution == (512, 512)

    figure = CoarseMapperVisualizer(data).render()
    assert isinstance(figure, Figure)
    import matplotlib.pyplot as plt

    plt.close(figure)


def test_coarse_visualizer_renders_with_zeroth_order_on_sensor():
    """When the zeroth order is on the sensor no array is displayed
    (array_image is None); the visualizer still renders (placeholder panel)."""
    from matplotlib.figure import Figure

    slm, camera, model = _build_setup()  # centred sensor: DC on it
    mapping = CoarseMapper(slm, camera, model).map_camera()
    assert mapping.visualization_data.array_image is None

    figure = CoarseMapperVisualizer(mapping.visualization_data).render()
    assert isinstance(figure, Figure)
    import matplotlib.pyplot as plt

    plt.close(figure)


def test_coarse_visualization_data_follows_visualization_data_pattern():
    """CoarseVisualizationData is a VisualizationData and CameraMapping exposes
    the standard optional visualization_data field."""
    import dataclasses

    from hologradpy.visualizer import VisualizationData

    assert issubclass(CoarseVisualizationData, VisualizationData)
    fields = {f.name: f for f in dataclasses.fields(CameraMapping)}
    assert "visualization_data" in fields
    assert fields["visualization_data"].default is None
