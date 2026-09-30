"""Tests for CoarseMapper and the CameraMapping orientation properties.

The coarse mapper bootstraps the camera position / rotation / flip from
sequential probe spots; it must work when the camera is rotated, flipped, or
positioned so the zeroth order misses the sensor entirely.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from hologradpy.hardware import (
    CameraOrientation,
    SimulatedCameraTorch,
    SimulatedSLMTorch,
)
from hologradpy.hardware import as_slm
from hologradpy.optics.complex_amplitude import (
    ComplexAmplitude,
    FieldGeometry,
)
from hologradpy.optics.systems import SLMFFT, SLMCZT
from hologradpy.optics.modules.pixel_crosstalk import (
    PixelCrosstalk,
    SuperGaussianCrosstalk,
)
from hologradpy.optics.modules.slm_fields import PixelwiseSLMField
from hologradpy.optics.modules.virtual_slms import VirtualSLM
from hologradpy.profiles.phase import (
    binary_phase_grating,
)
from hologradpy.fourier_optics import get_focal_spot_radius
from hologradpy.profiles.amplitude import gaussian_beam_intensity
from hologradpy.roi import ROI
from hologradpy.calibration.camera_mapping import (
    CameraMapping,
    FocalSpotFit,
    CoarseMapper,
    CoarseMapperVisualizer,
    CoarseVisualizationData,
    SpotArrayMapper,
)
from hologradpy.calibration.camera_mapping.coarse_mapping.coarse_mapper import (
    _PROBE_SPACING_FRACTION,
)
from hologradpy.calibration.spot_detection import (
    _WINDOW_SPOT_RADII,
    background_noise,
    detect_spot,
    get_diffraction_spot_position,
    has_prominent_peak,
    peak_prominence,
)
from tests.native_camera_fakes import CroppingCamera

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
    pixel_crosstalk: PixelCrosstalk | None = None,
):
    torch.manual_seed(0)
    geometry = FieldGeometry(
        resolution=(256, 320),
        pixel_size=torch.tensor([12.5e-6, 12.5e-6], device=DEVICE),
        wavelength=torch.tensor(0.630e-6, device=DEVICE),
    )
    slm = SimulatedSLMTorch(
        input_geometry=geometry, bitdepth=8, pixel_crosstalk=pixel_crosstalk
    )
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
        # The tests place the sensor in pixels, which reads more directly for a
        # sensor; the systems take the shift in focal-plane metres.
        camera_shift=tuple(
            s * p for s, p in zip(camera_shift, (30e-6, 30e-6))
        ),
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
        virtual_slm=VirtualSLM(full_scale_cycles=1.0),
        slm_field=PixelwiseSLMField(beam),
        focal_length=0.25,
        padded_resolution=(512, 512),
    )
    return slm, camera, model


def test_the_zeroth_order_is_located_through_an_overexposed_camera():
    """The probe that finds the spot at zero tilt stops as soon as it is detectable,
    which a saturated frame already is, so it leaves the camera far overexposed.

    An SLM that leaves light undiffracted then reads the same overexposed peak under the
    suppressing grating as without it. Pixel crosstalk leaves about a tenth of it, which
    is plenty to overexpose the sensor. The zeroth order is therefore located on a frame
    of its own below full scale.
    """
    slm, camera, model = _build_setup(
        pixel_crosstalk=SuperGaussianCrosstalk(upscale_factor=3, order=2.0, width=1.0)
    )
    mapper = CoarseMapper(slm, camera, model)
    spot_radius = 2.0 * float(min(camera.pixel_size))

    # The exposure handed over by the probe. It overexposes the zeroth order by a wide
    # margin, with and without the grating.
    flat = np.zeros(slm.resolution, dtype=np.float32)
    slm.set_phase(flat)
    camera.set_exposure(1e-3)
    assert np.asarray(camera.get_image()).max() >= camera.max_pixel_value
    slm.set_phase(binary_phase_grating(slm.resolution))
    assert np.asarray(camera.get_image()).max() >= camera.max_pixel_value
    slm.set_phase(flat)

    located = mapper._locate_zeroth_order(focal_length=0.25, spot_radius=spot_radius)

    assert located == pytest.approx(ALIGNED_ZEROTH_ORDER, abs=1.0)


# The zeroth order of the aligned bench, (row, column), at the centre of the sensor.
ALIGNED_ZEROTH_ORDER = (120.0, 160.0)

# The share of the beam the SLM leaves unmodulated, which makes the zeroth order one and
# a half times as bright as a probe.
UNMODULATED_FRACTION = 0.6


def _add_unmodulated_light(slm: SimulatedSLMTorch, fraction: float) -> None:
    """Leave ``fraction`` of the beam unmodulated by the simulated SLM.

    The field leaving the SLM is ``sqrt(1 - fraction) exp(i phase) + sqrt(fraction)``,
    so the zeroth order keeps ``fraction`` of the light whatever the SLM shows.
    """
    virtual_slm = slm.virtual_slm

    def forward(complex_amplitude: ComplexAmplitude) -> ComplexAmplitude:
        phase = virtual_slm.apply_phase_transforms(virtual_slm.get_phase())
        phase = virtual_slm.align_phase(phase, complex_amplitude.ndim)
        modulation = np.sqrt(1.0 - fraction) * torch.exp(1j * phase) + np.sqrt(fraction)
        output = complex_amplitude * modulation
        return output.with_geometry(
            wavelength=output.wavelength, pixel_size=virtual_slm.pixel_size_out
        )

    virtual_slm.forward = forward


@pytest.mark.parametrize(
    ("bench", "mirrored", "rotation_degrees"),
    [
        ({}, False, 0.0),
        ({"camera_angle": 10.0, "camera_shift": (20, -10)}, False, -10.0),
        ({"rot": "90"}, False, 90.0),
        ({"fliplr": True}, True, None),
    ],
    ids=["aligned", "rotated-and-shifted", "quarter-turn", "mirrored"],
)
def test_a_zeroth_order_brighter_than_the_probes_is_masked(
    bench: dict, mirrored: bool, rotation_degrees: float | None
) -> None:
    """The SLM leaves more light in the zeroth order than in a probe. The zeroth order
    is located and masked, so every probe is fitted at its own position.
    """
    slm, camera, model = _build_setup(**bench)
    _add_unmodulated_light(slm, UNMODULATED_FRACTION)

    coarse = CoarseMapper(slm, camera, model).map_camera()

    assert coarse.is_mirrored == mirrored
    if rotation_degrees is not None:
        assert abs(coarse.rotation_degrees) == pytest.approx(
            abs(rotation_degrees), abs=0.5
        )
    assert coarse.fit.reprojection_rms < 1.0


def test_a_located_zeroth_order_seeds_the_centre_search(monkeypatch):
    """The zeroth order is a spot on the sensor at zero tilt, so the mapping starts
    there. It needs neither the spiral search nor the exposure of a spot array, and a
    given initial tilt is not used, even one that misses the sensor.
    """
    slm, camera, model = _build_setup()
    mapper = CoarseMapper(slm, camera, model)

    def not_needed(*arguments, **options):
        raise AssertionError("Not needed with the zeroth order on the sensor.")

    monkeypatch.setattr(mapper, "_search_spot", not_needed)
    monkeypatch.setattr(mapper, "_calibrate_exposure", not_needed)

    coarse = mapper.map_camera(initial_tilt=(5e-3, 5e-3))

    assert coarse.zeroth_order_position == pytest.approx(ALIGNED_ZEROTH_ORDER, abs=1.0)
    assert coarse.fit.reprojection_rms < 1.0


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


def test_background_noise_is_estimated_from_the_masked_pixels_only():
    """One half of the frame holds saturated pixels, and the mask leaves that half
    out. The estimate is the sigma of the other half, and a mask that keeps nothing is
    refused.
    """
    rng = np.random.default_rng(1)
    sample = rng.normal(10.0, 3.0, size=(400, 400))
    sample[:, 200:] = 1023.0
    kept = np.zeros(sample.shape, dtype=bool)
    kept[:, :200] = True

    assert background_noise(sample, kept) == pytest.approx(3.0, rel=0.03)
    assert peak_prominence(sample, kept) == pytest.approx(
        sample[:, :200].max() - np.median(sample[:, :200])
    )
    with pytest.raises(ValueError, match="keeps no pixel"):
        background_noise(sample, np.zeros(sample.shape, dtype=bool))


# --- detect_spot ----------------------------------------------------------------

PIXEL_UM = 3.45
SPOT_RADIUS = 10e-6           # 1/e^2 radius -> ~2.9 px at 3.45 um pitch
BITRESOLUTION = 1024


class _PitchCamera(CroppingCamera):
    """A native camera that gives detect_spot its pitch and its full scale.

    detect_spot reads a frame passed to it, so this camera never captures one.
    """

    def __init__(self, pixel_um: float, bitresolution: int) -> None:
        super().__init__((80, 120), max_pixel_value=bitresolution - 1)
        self._pitch_m = pixel_um * 1e-6

    @property
    def pixel_size(self) -> np.ndarray:
        return np.array([self._pitch_m, self._pitch_m])

    def render_sensor_frame(self) -> np.ndarray:
        raise AssertionError("detect_spot reads the frame it is given.")


def _fake_camera(pixel_um=PIXEL_UM, bitresolution=BITRESOLUTION):
    # detect_spot reads native .pixel_size (y, x) metres and .max_pixel_value.
    return _PitchCamera(pixel_um, bitresolution)


def _gaussian_frame(shape, center, amplitude, sigma_px, *, background=5.0,
                    noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    rows, columns = np.indices(shape)
    column0, row0 = center
    frame = background + amplitude * np.exp(
        -((columns - column0) ** 2 + (rows - row0) ** 2) / (2 * sigma_px**2)
    )
    return frame + rng.normal(0.0, noise, shape)


def test_a_weak_peak_stays_weak_beside_a_large_mask():
    """A bump of 4 noise sigma is no prominent peak at the default 8 sigma. This holds
    with or without a mask over two thirds of the frame, since the masked pixels are
    left out of the noise estimate. Setting them to the frame median shrinks the
    median absolute deviation to zero, and the same bump then passes.
    """
    sigma_px = (SPOT_RADIUS / (PIXEL_UM * 1e-6)) / 2
    frame = _gaussian_frame(
        (80, 120), (30, 40), amplitude=160.0, sigma_px=sigma_px, noise=40.0
    )
    kept = np.zeros(frame.shape, dtype=bool)
    kept[:, :40] = True
    filled = np.where(kept, frame, np.median(frame))

    assert not has_prominent_peak(frame, _fake_camera())
    assert not has_prominent_peak(frame, _fake_camera(), mask=kept)
    assert has_prominent_peak(filled, _fake_camera())


def test_a_bright_masked_spot_is_no_prominent_peak():
    sigma_px = (SPOT_RADIUS / (PIXEL_UM * 1e-6)) / 2
    frame = _gaussian_frame((80, 120), (90, 40), amplitude=800.0, sigma_px=sigma_px)
    kept = np.ones(frame.shape, dtype=bool)
    kept[:, 60:] = False

    assert has_prominent_peak(frame, _fake_camera())
    assert not has_prominent_peak(frame, _fake_camera(), mask=kept)


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
    floor of 1 must not reject it.
    """
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
    order sitting off the sensor) is rejected by the enclosure gate.
    """
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
    reoriented camera then has nothing left to suggest.
    """
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


def test_camera_mapping_records_the_camera_and_the_output_plane(tmp_path):
    """The mapping records the camera it was measured with and the output plane of its
    model, and both survive a save and a load.
    """
    slm, camera, model = _build_setup()
    coarse = CoarseMapper(slm, camera, model).map_camera()
    output = model[-1]
    assert coarse.camera_data is not None
    assert np.asarray(coarse.camera_data.orientation).shape == (2, 3)
    assert coarse.output_pixel_size == pytest.approx(
        tuple(output.pixel_size_out.tolist()[0])
    )
    assert coarse.output_resolution == tuple(output.resolution_out)

    path = str(tmp_path / "coarse_mapping.asdf")
    coarse.save(path)
    loaded = CameraMapping.load(path)
    assert loaded.camera_data.resolution == tuple(camera.shape)
    assert loaded.output_pixel_size == coarse.output_pixel_size
    assert loaded.output_resolution == coarse.output_resolution


# --- checking a mapping against the camera it is used with ----------------------


@pytest.fixture(scope="module")
def measured_camera():
    """A camera and the coarse mapping measured with it, shared by the checks."""
    slm, camera, model = _build_setup()
    return camera, CoarseMapper(slm, camera, model).map_camera()


def test_a_mapping_accepts_the_camera_it_was_measured_with(measured_camera, tmp_path):
    """The region of interest is not compared, and the check holds after a save and a
    load.
    """
    camera, mapping = measured_camera
    mapping.check_camera(camera)

    path = tmp_path / "coarse_mapping.asdf"
    mapping.save(path)
    loaded = CameraMapping.load(path)
    camera.set_roi(ROI(10, 20, 50, 60))
    try:
        loaded.check_camera(camera)
    finally:
        camera.set_roi(None)


def test_a_mapping_refuses_a_camera_turned_since_it_was_measured(measured_camera):
    camera, mapping = measured_camera
    camera.set_orientation(CameraOrientation(rot="180"))
    try:
        with pytest.raises(ValueError, match="mounted"):
            mapping.check_camera(camera)
    finally:
        camera.set_orientation(CameraOrientation())


def test_a_mapping_refuses_a_sensor_of_another_resolution(measured_camera):
    camera, mapping = measured_camera
    other = replace(
        mapping,
        camera_data=replace(mapping.camera_data, sensor_resolution=(120, 160)),
    )
    with pytest.raises(ValueError, match="120 x 160 sensor"):
        other.check_camera(camera)


def test_a_mapping_refuses_a_camera_of_another_pixel_pitch(measured_camera):
    camera, mapping = measured_camera
    pitch_y, pitch_x = mapping.camera_data.pixel_size
    other = replace(
        mapping,
        camera_data=replace(mapping.camera_data, pixel_size=(2 * pitch_y, 2 * pitch_x)),
    )
    with pytest.raises(ValueError, match="pitch"):
        other.check_camera(camera)


def test_a_mapping_without_camera_data_is_not_checked(measured_camera):
    camera, mapping = measured_camera
    replace(mapping, camera_data=None).check_camera(camera)


def test_coarse_mapping_accepts_initial_tilt():
    # Centered sensor: the zeroth order is on it, so tilt (0, 0) is a valid seed
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


def test_a_diffraction_spot_that_misses_the_sensor_raises_after_autoexposure():
    """A camera that sees only read noise rails at its longest exposure. The rail is
    not an error on its own, but the final frame holds no prominent peak.
    """

    class _DarkCamera(CroppingCamera):
        def __init__(self) -> None:
            super().__init__((60, 80), exposure_bounds=(1e-4, 1.0))
            self._noise = np.random.default_rng(0)

        def render_sensor_frame(self) -> np.ndarray:
            counts = self._noise.normal(10.0, 2.0, self.sensor_resolution)
            return np.clip(np.rint(counts), 0, 255).astype(np.uint16)

    slm, _, _ = _build_setup()
    camera = _DarkCamera()

    with pytest.warns(UserWarning, match="railed"):
        with pytest.raises(RuntimeError, match="No spot on the sensor"):
            get_diffraction_spot_position(
                slm, camera, (0.0, 0.0), focal_length=0.25, verbose=False
            )


def test_map_camera_leaves_the_camera_as_it_found_it():
    """The whole sensor is mapped for any window held by the camera, and the window and
    the exposure are put back afterwards.
    """
    slm, camera, model = _build_setup(
        camera_angle=10.0, camera_shift=(60, 100), camera_resolution=(120, 160)
    )
    window = ROI(10, 20, 50, 60)
    camera.set_roi(window)
    camera.set_exposure(2e-3)

    coarse = CoarseMapper(slm, camera, model).map_camera()

    assert camera.roi == window
    assert camera.get_exposure() == 2e-3
    assert coarse.camera_data.roi == ROI(0, 0, 120, 160)
    assert coarse.rotation_degrees == pytest.approx(-10.0, abs=0.5)
    assert coarse.fit.reprojection_rms < 1.0


def test_no_exposure_request_leaves_the_camera_bounds(monkeypatch):
    """Every exposure requested by the mapping lies inside the bounds stated by the
    camera. This includes the cuts in the search for a main order and a spot.
    """
    slm, camera, model = _build_setup(
        camera_angle=10.0, camera_shift=(60, 100), camera_resolution=(120, 160)
    )
    camera._exposure_bounds = (1e-8, 1.0)
    requested = []
    set_exposure = camera.set_exposure

    def record_and_set(exposure_s):
        requested.append(float(exposure_s))
        set_exposure(exposure_s)

    monkeypatch.setattr(camera, "set_exposure", record_and_set)

    CoarseMapper(slm, camera, model).map_camera()

    assert requested
    assert min(requested) >= 1e-8
    assert max(requested) <= 1.0


def test_coarse_mapping_with_zeroth_order_off_sensor():
    """The camera only sees a region away from the zeroth order: the spiral
    search must find the sensor, and the zeroth-order position is extrapolated
    (legitimately off the sensor).
    """
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
    deterministic tilt step dwarfs a 1 um (~sub-px) jitter.
    """
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
    raises the camera floor, but the coarse mapper still recovers the transform.
    """
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
    _calibrate_exposure, mirroring how map_camera derives them.
    """
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
    ladder sentinel None).
    """
    slm, camera, model = _build_setup(
        camera_angle=10.0, camera_shift=(60, 100), camera_resolution=(120, 160)
    )
    mapper = CoarseMapper(slm, camera, model)
    exposure = mapper._calibrate_exposure(*_calibration_args(slm, (120, 160)))
    assert exposure is not None
    assert exposure >= 0.0


def test_a_speckle_grain_is_not_taken_for_the_zeroth_order():
    """A bright static speckle background must not be mistaken for the zeroth
    order. The zero-tilt probe latches onto a speckle grain, but the 0/pi grating
    leaves it as bright as it was, so no zeroth order is located.
    """
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

    assert mapper._locate_zeroth_order(focal_length, spot_radius) is None


def test_calibrate_exposure_warns_and_clamps_below_hardware_bound():
    """A per-probe exposure below the camera's minimum hardware exposure warns
    the user and clamps to the bound.
    """
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
    so the per-probe ladder takes over.
    """
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
    correctly via the per-probe ladder.
    """
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
    visualization_data and the CoarseMapperVisualizer renders all four panels.
    """
    from matplotlib.figure import Figure

    slm, camera, model = _build_setup(
        camera_angle=10.0, camera_shift=(60, 100), camera_resolution=(120, 160)
    )
    mapping = CoarseMapper(slm, camera, model).map_camera()

    data = mapping.visualization_data
    assert data is not None
    assert data.array_image is not None       # array path was taken
    assert data.walk_image is not None         # center-search walked
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
    (array_image is None); the visualizer still renders (placeholder panel).
    """
    from matplotlib.figure import Figure

    slm, camera, model = _build_setup()  # centered sensor: DC on it
    mapping = CoarseMapper(slm, camera, model).map_camera()
    assert mapping.visualization_data.array_image is None

    figure = CoarseMapperVisualizer(mapping.visualization_data).render()
    assert isinstance(figure, Figure)
    import matplotlib.pyplot as plt

    plt.close(figure)


def test_coarse_visualization_data_follows_visualization_data_pattern():
    """CoarseVisualizationData is a VisualizationData and CameraMapping exposes
    the standard optional visualization_data field.
    """
    import dataclasses

    from hologradpy.visualizer import VisualizationData

    assert issubclass(CoarseVisualizationData, VisualizationData)
    fields = {f.name: f for f in dataclasses.fields(CameraMapping)}
    assert "visualization_data" in fields
    assert fields["visualization_data"].default is None
