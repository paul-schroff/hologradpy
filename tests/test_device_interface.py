"""Tests for the HoloGradPy-native device interface (protocol.py / adapter.py).

Covers the ROI value object, the slmsuite<->HoloGradPy conversion helpers, the native
properties the simulated devices expose, and that the real-hardware adapter reports
identical native values for the same underlying device. Uses a non-square camera so
an axis swap in any conversion would show.
"""

from __future__ import annotations

import copy
import gc
import time
import weakref

import numpy as np
import pytest
import torch

from hologradpy.hardware import SimulatedSLMTorch, SimulatedCameraTorch
from hologradpy.hardware import Camera, CameraOrientation, SLM
from hologradpy.hardware.camera import reorient_pixels
from hologradpy.hardware.slm import SLMData
from hologradpy.hardware.slmsuite.conversions import (
    pixel_size_from_pitch_um,
    pitch_um_from_pixel_size,
    wavelength_from_wav_um,
    wav_um_from_wavelength,
    roi_from_woi,
    roi_to_woi,
)
from hologradpy.phase_levels import (
    LinearResponse,
    LookupResponse,
    PhaseResponseModule,
)
from hologradpy.roi import ROI
from hologradpy.hardware import (
    SLMSuiteCameraAdapter,
    SLMSuiteSLMAdapter,
    as_camera,
    as_slm,
    open_camera,
    open_slm,
    register_camera_backend,
    register_slm_backend,
)
from slmsuite.hardware.cameras.camera import Camera as SLMSuiteCamera
from slmsuite.hardware.slms.slm import SLM as SLMSuiteSLM
from hologradpy.optics.complex_amplitude import (
    ComplexAmplitude,
    FieldGeometry,
)
from hologradpy.optics.systems import SLMCZT
from hologradpy.optics.modules.slm_fields import PixelwiseSLMField
from hologradpy.optics.modules.virtual_slms import VirtualSLM
from hologradpy.profiles.amplitude import (
    gaussian_beam_intensity,
)
from tests.native_camera_fakes import CroppingCamera

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

WAVELENGTH = 0.630e-6
SLM_PITCH = 12.5e-6
# Non-square camera: pixel_size (y, x) = (30, 20) um -> pitch_um (x, y) = (20, 30) um.
CAMERA_PIXEL_SIZE = (30e-6, 20e-6)


# --- ROI ------------------------------------------------------------------------


def test_roi_woi_round_trip():
    roi = ROI(top_row=5, left_column=8, height=20, width=30)
    # slmsuite WOI is (x0, width, y0, height).
    assert roi_to_woi(roi) == (8, 30, 5, 20)
    assert roi_from_woi(roi_to_woi(roi)) == roi


def test_roi_centered_and_slices():
    roi = ROI.centered((100, 200), (20, 40))
    assert (roi.top_row, roi.left_column, roi.height, roi.width) == (90, 180, 20, 40)
    assert roi.rows == slice(90, 110)
    assert roi.columns == slice(180, 220)


def test_roi_from_bounds_round_trip():
    roi = ROI.from_bounds(top=5, bottom=25, left=8, right=38)
    assert (roi.top_row, roi.left_column, roi.height, roi.width) == (5, 8, 20, 30)
    assert roi.to_bounds() == (5, 25, 8, 38)


def test_roi_crop_and_pad_round_trip():
    image = np.arange(48).reshape(6, 8).astype(float)
    roi = ROI.from_bounds(top=1, bottom=4, left=2, right=6)
    cropped = roi.crop(image)
    assert cropped.shape == (3, 4)
    np.testing.assert_array_equal(cropped, image[1:4, 2:6])
    padded = roi.pad(cropped, image.shape)
    assert padded.shape == image.shape
    np.testing.assert_array_equal(padded[1:4, 2:6], cropped)
    assert padded[0, 0] == 0


def test_roi_detect_bounds_bright_block():
    """The bounds are slicing bounds, so they reproduce the block exactly."""
    image = np.zeros((10, 12))
    image[3:6, 4:9] = 1.0
    roi = ROI.detect(image, threshold=0.5, pad=0)

    assert roi.to_bounds() == (3, 6, 4, 9)
    np.testing.assert_array_equal(roi.crop(image), image[3:6, 4:9])
    np.testing.assert_array_equal(roi.pad(roi.crop(image), image.shape), image)


def test_roi_detect_without_a_threshold_is_an_exact_lossless_box():
    """``threshold=0, pad=0`` bounds every nonzero pixel, so crop and pad round trip."""
    image = np.zeros((10, 12))
    image[3:6, 4:9] = 1.0
    image[2, 3] = 0.1                       # faint, but still inside the box

    roi = ROI.detect(image, threshold=0.0, pad=0)

    assert roi.to_bounds() == (2, 6, 3, 9)
    np.testing.assert_array_equal(roi.pad(roi.crop(image), image.shape), image)


# --- conversion helpers ---------------------------------------------------------


def test_pixel_size_pitch_um_round_trip():
    pitch_um = (20.0, 30.0)  # (x, y)
    pixel_size = pixel_size_from_pitch_um(pitch_um)  # (y, x) metres
    np.testing.assert_allclose(pixel_size, (30e-6, 20e-6))
    np.testing.assert_allclose(pitch_um_from_pixel_size(pixel_size), pitch_um)


def test_wavelength_round_trip():
    assert wavelength_from_wav_um(0.63) == pytest.approx(0.63e-6)
    assert wav_um_from_wavelength(0.63e-6) == pytest.approx(0.63)


# --- native device properties ---------------------------------------------------


def _build():
    torch.manual_seed(0)
    geometry = FieldGeometry(
        resolution=(256, 320),
        pixel_size=torch.tensor([SLM_PITCH, SLM_PITCH]),
        wavelength=torch.tensor(WAVELENGTH),
    )
    slm = SimulatedSLMTorch(input_geometry=geometry, bitdepth=8)
    intensity = gaussian_beam_intensity(*geometry.get_spatial_grid(), beam_radius=1e-3)
    beam = ComplexAmplitude(
        intensity.sqrt() + 0j,
        wavelength=geometry.wavelength,
        pixel_size=geometry.pixel_size,
    )
    model = SLMCZT(
        input_geometry=geometry,
        virtual_slm=slm.virtual_slm,
        camera_resolution=(240, 320),
        camera_pixel_size=CAMERA_PIXEL_SIZE,
        focal_length=0.25,
        slm_field=PixelwiseSLMField(beam),
    )
    camera = SimulatedCameraTorch(model, bitdepth=8)
    return slm, camera


def test_camera_native_properties():
    """The simulated camera implements the native Camera interface directly (no
    slmsuite base), so as_camera passes it through unchanged.
    """
    _, sim = _build()
    assert isinstance(sim, Camera)      # native on its own
    assert as_camera(sim) is sim    # passthrough, no adapter
    # pixel_size (y, x) metres matches the model's (y, x) camera pixel size.
    np.testing.assert_allclose(sim.pixel_size, CAMERA_PIXEL_SIZE, rtol=1e-6)
    assert sim.resolution == (240, 320)
    assert sim.adu_levels == 256  # 2 ** 8
    roi = ROI.centered((120, 160), (40, 60))
    sim.set_roi(roi)
    assert sim.roi == roi
    sim.set_roi(None)
    assert sim.roi == ROI(0, 0, 240, 320)


def test_slm_native_properties():
    sim, _ = _build()
    assert isinstance(sim, SLM)
    assert as_slm(sim) is sim
    np.testing.assert_allclose(sim.pixel_size, (SLM_PITCH, SLM_PITCH), rtol=1e-6)
    assert sim.resolution == (256, 320)
    assert sim.wavelength == pytest.approx(WAVELENGTH)


# --- Grayscale levels ---------------------------------------------------


def _slm_at(bitdepth: int, wav_design_um: float | None = None, **options) -> SLM:
    geometry = FieldGeometry(
        resolution=(32, 48),
        pixel_size=torch.tensor([SLM_PITCH, SLM_PITCH]),
        wavelength=torch.tensor(WAVELENGTH),
    )
    return open_slm(
        SimulatedSLMTorch,
        input_geometry=geometry,
        bitdepth=bitdepth,
        wav_design_um=wav_design_um,
        **options,
    )


# Phases that exercise the wrap: inside the modulation range, below it, and several
# turns above it.
_PHASES = {
    "in range": lambda rng, shape: rng.random(shape) * 2 * np.pi,
    "negative": lambda rng, shape: rng.random(shape) * 2 * np.pi - 7.0,
    "many turns": lambda rng, shape: rng.random(shape) * 40,
}


@pytest.mark.parametrize("bitdepth", [8, 12])
@pytest.mark.parametrize("case", list(_PHASES))
def test_levels_replay_exactly_what_the_slm_displayed(bitdepth: int, case: str) -> None:
    """The pin the dataset format rests on.

    A stored pattern is the levels the SLM held, and putting them back on the model has
    to reproduce that display bit for bit. Getting the sign, the one-level shift or the
    wrap backwards would leave every fit quietly wrong rather than failing.
    """
    slm = _slm_at(bitdepth)
    phase = _PHASES[case](np.random.default_rng(0), slm.resolution)

    slm.set_phase(phase)
    displayed = slm.virtual_slm.get_phase().clone()

    levels = slm.phase_to_levels(phase)
    slm.virtual_slm.set_levels(levels, bitdepth)

    assert torch.equal(slm.virtual_slm.get_phase(), displayed)


@pytest.mark.parametrize("bitdepth", [8, 12])
def test_levels_are_what_the_device_displays(bitdepth: int) -> None:
    slm = _slm_at(bitdepth)
    phase = np.random.default_rng(1).random(slm.resolution) * 2 * np.pi

    slm.set_phase(phase)

    assert np.array_equal(slm.phase_to_levels(phase), slm.display)
    expected = np.uint8 if bitdepth == 8 else np.uint16
    assert slm.phase_to_levels(phase).dtype == expected


def test_quantizing_leaves_the_callers_pattern_alone() -> None:
    slm = _slm_at(8)
    phase = np.random.default_rng(2).random(slm.resolution) * 2 * np.pi
    original = phase.copy()

    slm.phase_to_levels(phase)

    assert np.array_equal(phase, original)


@pytest.mark.parametrize("bitdepth", [8, 12])
def test_integer_patterns_are_displayed_as_given(bitdepth: int) -> None:
    """Levels go to the display untouched, so a stored pattern can be shown again
    without a conversion that could round it somewhere else.
    """
    slm = _slm_at(bitdepth)
    levels = np.random.default_rng(4).integers(
        0, 2**bitdepth, slm.resolution, dtype=np.uint8 if bitdepth == 8 else np.uint16
    )

    slm.set_levels(levels)

    assert np.array_equal(slm.display, levels)
    # The phase those levels mean, to the precision the model runs in. Not bit equality:
    # the two sides reach it in a different order, and the state carries the field's
    # dtype rather than being computed wide and narrowed at the end.
    assert torch.allclose(
        slm.virtual_slm.get_phase(),
        slm.virtual_slm.levels_to_phase(levels, bitdepth).to(
            slm.virtual_slm.get_phase().dtype
        ),
        atol=1e-6,
    )
    # What must stay exact is the level itself, since that is what the panel holds.
    assert np.array_equal(
        slm.virtual_slm.phase_to_levels(slm.virtual_slm.get_phase().numpy(), bitdepth),
        levels,
    )


def test_a_pattern_survives_a_display_round_trip() -> None:
    """Display a phase, read the levels off, display those: the same pattern."""
    slm = _slm_at(8)
    phase = np.random.default_rng(5).random(slm.resolution) * 2 * np.pi

    slm.set_phase(phase)
    first = slm.display.copy()
    slm.set_levels(first)

    assert np.array_equal(slm.display, first)


def test_levels_replay_with_a_phase_scaling() -> None:
    """A target wavelength away from the design one takes the other branch of the
    conversion, where the wrap is folded into the scaling factor.
    """
    slm = _slm_at(8, wav_design_um=WAVELENGTH * 1e6 / 1.4)
    assert slm.phase_scaling != 1

    phase = np.random.default_rng(3).random(slm.resolution) * 2 * np.pi
    slm.set_phase(phase)
    displayed = slm.virtual_slm.get_phase().clone()

    slm.virtual_slm.set_levels(slm.phase_to_levels(phase), slm.bitdepth)

    assert torch.equal(slm.virtual_slm.get_phase(), displayed)


def _s_curve(bitdepth: int = 8, span: float = 1.9 * np.pi) -> LookupResponse:
    """The shape a real panel has: monotone, and not a straight line."""
    levels = np.arange(2**bitdepth)
    top = levels[-1]
    # No phase_scaling: the table's own span says how far the panel reaches.
    return LookupResponse(
        bitdepth=bitdepth,
        phases=-span * (0.5 - 0.5 * np.cos(np.pi * levels / top)),
    )


def test_a_nonlinear_response_is_what_a_level_means() -> None:
    """The point of the whole exercise: the phase a level imposes comes from the curve,
    not from an assumption that the panel is linear.
    """
    response = _s_curve()
    slm = _slm_at(8)
    slm.set_phase(np.zeros(slm.resolution))
    slm.virtual_slm.phase_response = PhaseResponseModule(response)

    slm.set_levels(np.full(slm.resolution, 100, dtype=np.uint8))

    assert float(slm.virtual_slm.get_phase()[0, 0]) == pytest.approx(
        response.phases[100]
    )


def test_a_desired_phase_lands_on_the_nearest_level_the_panel_has() -> None:
    response = _s_curve()
    slm = _slm_at(8)
    slm.set_phase(np.zeros(slm.resolution))
    slm.virtual_slm.phase_response = PhaseResponseModule(response)

    slm.set_phase(np.full(slm.resolution, -2.0))

    nearest = int(np.argmin(np.abs(response.phases + 2.0)))
    assert int(slm.display[0, 0]) == nearest
    assert float(slm.virtual_slm.get_phase()[0, 0]) == pytest.approx(
        response.phases[nearest]
    )


@pytest.mark.parametrize("bitdepth", [8, 12])
def test_a_nonlinear_response_round_trips_every_level(bitdepth: int) -> None:
    response = _s_curve(bitdepth)
    levels = np.arange(response.number_of_levels)

    assert np.array_equal(response.to_levels(response.to_phase(levels)), levels)


@pytest.mark.parametrize("bitdepth", [8, 12])
def test_the_panel_discretizes_the_same_way_for_a_pattern_and_a_gradient(
    bitdepth: int,
) -> None:
    """A pattern reaches the panel through display_levels and a gradient through
    quantize. They must land on the same level, or a simulated panel would be driven
    differently from the real one it stands for.
    """
    response = _s_curve(bitdepth)
    module = PhaseResponseModule(response)
    rng = np.random.default_rng(0)
    phase = rng.uniform(-30.0, 30.0, 5000)

    displayed = response.display_levels(phase)
    quantized = module.quantize(torch.as_tensor(response.to_levels(phase)))

    np.testing.assert_array_equal(quantized.numpy(), displayed.astype(float))


def test_quantize_leaves_the_gradient_alone() -> None:
    """Straight through: rounding has no useful derivative, so the estimator passes the
    incoming one on rather than killing the search.
    """
    module = PhaseResponseModule(_s_curve())
    levels = torch.tensor([12.3, 200.7, 4.5], requires_grad=True)

    module.quantize(levels).sum().backward()

    assert torch.equal(levels.grad, torch.ones(3))


def test_a_curve_that_cannot_reach_a_phase_clamps() -> None:
    """Under a full turn of modulation there are phases the panel simply cannot impose,
    and the nearest end is the honest answer rather than a wrap onto an unrelated
    level.
    """
    response = _s_curve(span=1.2 * np.pi)
    unreachable = np.array([-2.0 * np.pi])

    assert response.to_levels(unreachable)[0] == response.number_of_levels - 1


def test_the_response_travels_with_the_model() -> None:
    """A measured curve is part of the model's state, so a checkpoint carries it."""
    slm = _slm_at(8)
    slm.set_phase(np.zeros(slm.resolution))
    slm.virtual_slm.phase_response = PhaseResponseModule(_s_curve())

    assert "phase_response.table" in slm.virtual_slm.state_dict()


def test_levels_at_the_wrong_depth_are_refused() -> None:
    """A model reading levels at a depth they were not captured at would fit the wrong
    phase rather than fail, so it is caught.
    """
    slm = _slm_at(8)
    slm.set_phase(np.zeros(slm.resolution))

    with pytest.raises(ValueError, match="12-bit but this SLM's response is 8-bit"):
        slm.virtual_slm.set_levels(np.zeros(slm.resolution, dtype=np.uint16), 12)


class _BareSlm(SLM):
    """A native SLM with no bit depth, which converts phases itself."""

    pixel_size = np.array([SLM_PITCH, SLM_PITCH])
    resolution = (4, 4)
    wavelength = WAVELENGTH

    def set_phase(self, phase) -> None:
        pass


def test_a_device_without_a_bitdepth_says_so() -> None:
    device = _BareSlm()
    assert device.bitdepth is None
    with pytest.raises(ValueError, match="no bitdepth"):
        device.phase_to_levels(np.zeros((4, 4)))


# --- auto-wrap: as_camera / as_slm --------------------------------------


class _RawCamera(SLMSuiteCamera):
    """A minimal slmsuite camera with no native HoloGradPy properties.

    It stands in for real hardware, which is only reachable through the adapter, and
    it behaves as a driver does. The sensor holds fixed counts, modulo
    ``2 ** bitdepth`` in ``uint16``, and a frame is the part inside the window of
    interest. The exposure reads back as it was set. ``set_woi(None)`` opens the
    window to the whole sensor, and every window brings ``shape`` in line, as
    ``ThorCam.set_woi`` does.

    ``capture_calls`` counts the captures. The next ``missing_frames`` captures return
    None, as a driver does when a frame does not arrive in time. Both counters start
    at zero after slmsuite's own capture at construction. The driver holds no frames,
    so ``flush`` captures nothing.
    """

    def __init__(self, resolution, bitdepth, pitch_um, name="raw", **kwargs):
        width, height = resolution
        self._sensor = (
            np.arange(height * width).reshape(height, width) % 2**bitdepth
        ).astype(np.uint16)
        self._exposure_s = 1e-3
        self.capture_calls = 0
        self.missing_frames = 0
        super().__init__(
            resolution=resolution,
            bitdepth=bitdepth,
            pitch_um=pitch_um,
            name=name,
            **kwargs,
        )
        self.capture_calls = 0
        self.missing_frames = 0

    def _get_image_hw(self, timeout_s=None):
        self.capture_calls += 1
        if self.missing_frames > 0:
            self.missing_frames -= 1
            return None
        x0, width, y0, height = self.woi
        return self._sensor[y0 : y0 + height, x0 : x0 + width].copy()

    def flush(self, timeout_s=1):
        pass

    def _get_exposure_hw(self):
        return self._exposure_s

    def _set_exposure_hw(self, exposure_s):
        self._exposure_s = float(exposure_s)

    def set_woi(self, woi=None):
        if woi is None:
            height, width = self._sensor.shape
            woi = (0, width, 0, height)
        self.woi = tuple(int(value) for value in woi)
        self.shape = np.shape(self.transform(np.zeros((self.woi[3], self.woi[1]))))
        return self.woi

    def close(self):
        pass


class _RawSlm(SLMSuiteSLM):
    """A minimal slmsuite SLM with no native HoloGradPy properties.

    ``written`` holds the last array passed from slmsuite to the hardware, and it stays
    None until the first write. The settle time defaults to zero, so a write never
    sleeps unless a test asks for it.
    """

    def __init__(
        self, resolution, bitdepth, wav_um, pitch_um, name="raw", settle_time_s=0.0,
        **kwargs,
    ):
        super().__init__(
            resolution=resolution,
            bitdepth=bitdepth,
            wav_um=wav_um,
            pitch_um=pitch_um,
            name=name,
            settle_time_s=settle_time_s,
            **kwargs,
        )
        self.written = None

    def _set_phase_hw(self, display):
        self.written = display.copy()

    def close(self):
        pass


def _raw_slm(**options) -> _RawSlm:
    """A raw 8-bit slmsuite SLM of shape (4, 6), with ``options`` overriding any
    constructor argument.
    """
    arguments = dict(resolution=(6, 4), bitdepth=8, wav_um=0.5, pitch_um=(12.5, 12.5))
    arguments.update(options)
    return _RawSlm(**arguments)


def test_as_camera_wraps_slmsuite_and_is_idempotent():
    _, sim = _build()
    assert as_camera(sim) is sim                   # native sim -> passthrough

    raw = _RawCamera(resolution=(20, 30), bitdepth=8, pitch_um=(5.0, 7.0), name="raw")
    wrapped_raw = as_camera(raw)                   # raw hardware -> wrapped
    assert isinstance(wrapped_raw, SLMSuiteCameraAdapter)
    assert isinstance(wrapped_raw, Camera)
    assert as_camera(wrapped_raw) is wrapped_raw   # adapter -> idempotent
    # pitch_um (x, y) = (5, 7) um -> pixel_size (y, x) = (7, 5) um. Non-square, so an
    # axis swap would show.
    np.testing.assert_allclose(wrapped_raw.pixel_size, (7e-6, 5e-6))
    assert wrapped_raw.resolution == (30, 20)
    # The region crops each frame in software, so the driver keeps its whole-sensor
    # window. Every adapter of the camera reads the same region.
    whole_sensor_window = raw.woi
    roi = ROI(3, 5, 4, 8)
    wrapped_raw.set_roi(roi)
    assert raw.woi == whole_sensor_window
    assert wrapped_raw.roi == roi
    assert as_camera(raw).roi == roi
    assert wrapped_raw.get_image().shape == (4, 8)
    assert wrapped_raw.max_pixel_value == 2**raw.bitdepth - 1


def test_as_slm_wraps_slmsuite_and_is_idempotent():
    sim, _ = _build()
    assert as_slm(sim) is sim                      # native sim -> passthrough

    raw = _RawSlm(
        resolution=(20, 30), bitdepth=8, wav_um=0.5, pitch_um=(5.0, 7.0), name="raw"
    )
    wrapped_raw = as_slm(raw)
    assert isinstance(wrapped_raw, SLMSuiteSLMAdapter)
    assert isinstance(wrapped_raw, SLM)
    assert as_slm(wrapped_raw) is wrapped_raw
    np.testing.assert_allclose(wrapped_raw.pixel_size, (7e-6, 5e-6))
    assert wrapped_raw.resolution == (30, 20)
    assert wrapped_raw.wavelength == pytest.approx(0.5e-6)


def test_native_helpers_reject_non_devices():
    with pytest.raises(TypeError):
        as_camera(object())
    with pytest.raises(TypeError):
        as_slm(object())


# --- factory: open_camera / open_slm --------------------------------------------


def test_open_camera_builds_and_returns_native_from_class():
    camera = open_camera(
        _RawCamera, resolution=(20, 30), bitdepth=8, pitch_um=(5.0, 7.0)
    )
    assert isinstance(camera, SLMSuiteCameraAdapter)
    np.testing.assert_allclose(camera.pixel_size, (7e-6, 5e-6))


def test_open_slm_builds_and_returns_native_from_class():
    slm = open_slm(
        _RawSlm, resolution=(20, 30), bitdepth=8, wav_um=0.5, pitch_um=(5.0, 7.0)
    )
    assert isinstance(slm, SLMSuiteSLMAdapter)


def test_open_camera_accepts_registered_backend_name():
    register_camera_backend("raw_test_cam", _RawCamera)
    register_slm_backend("raw_test_slm", _RawSlm)
    camera = open_camera(
        "raw_test_cam", resolution=(20, 30), bitdepth=8, pitch_um=(5.0, 7.0)
    )
    slm = open_slm(
        "raw_test_slm", resolution=(20, 30), bitdepth=8, wav_um=0.5, pitch_um=(5.0, 7.0)
    )
    assert isinstance(camera, SLMSuiteCameraAdapter)
    assert isinstance(slm, SLMSuiteSLMAdapter)


def test_open_camera_unknown_backend_raises():
    with pytest.raises(KeyError, match="Unknown camera backend"):
        open_camera("nope", resolution=(20, 30), bitdepth=8, pitch_um=(5.0, 7.0))


# --- factory: lazy string-spec backends -----------------------------------------


def test_open_camera_resolves_lazy_string_spec():
    """A backend registered as a ``"module:Attr"`` string is imported on first open.

    Pointing at this module's own ``_RawCamera`` keeps the test free of any vendor
    SDK, while still exercising the full register -> import -> construct path.
    """
    register_camera_backend("lazy_raw_cam", f"{__name__}:_RawCamera")
    camera = open_camera(
        "lazy_raw_cam", resolution=(20, 30), bitdepth=8, pitch_um=(5.0, 7.0)
    )
    assert isinstance(camera, SLMSuiteCameraAdapter)
    np.testing.assert_allclose(camera.pixel_size, (7e-6, 5e-6))


def test_import_spec_colon_and_dotted_forms():
    from hologradpy.hardware.factory import _import_spec
    from hologradpy.roi import ROI as ExpectedROI

    assert _import_spec("hologradpy.roi:ROI", "x", "camera") is ExpectedROI
    assert _import_spec("hologradpy.roi.ROI", "x", "camera") is ExpectedROI


def test_import_spec_missing_module_raises_with_backend_name():
    from hologradpy.hardware.factory import _import_spec

    with pytest.raises(ImportError, match="badcam"):
        _import_spec("hologradpy._no_such_module:Thing", "badcam", "camera")


def test_import_spec_missing_attribute_raises_with_backend_name():
    from hologradpy.hardware.factory import _import_spec

    with pytest.raises(AttributeError, match="badslm"):
        _import_spec("hologradpy.roi:NoSuchClass", "badslm", "SLM")


# --- slmsuite backend table -----------------------------------------------------


def test_register_slmsuite_backends_registers_lazy_specs():
    """The opt-in registrar populates the factory registries with lazy specs, so no
    vendor SDK is imported until one of these backends is actually opened.
    """
    from hologradpy.hardware import register_slmsuite_backends
    from hologradpy.hardware.factory import _CAMERA_BACKENDS, _SLM_BACKENDS

    register_slmsuite_backends()
    assert _CAMERA_BACKENDS["thorlabs"] == "slmsuite.hardware.cameras.thorlabs:ThorCam"
    assert _SLM_BACKENDS["hamamatsu"] == "slmsuite.hardware.slms.hamamatsu:Hamamatsu"
    # Every slmsuite entry is a lazy "module:Attr" string, none an eager class.
    slmsuite_names = [*_CAMERA_BACKENDS, *_SLM_BACKENDS]
    slmsuite_entries = [
        entry
        for entry in (*_CAMERA_BACKENDS.values(), *_SLM_BACKENDS.values())
        if isinstance(entry, str) and entry.startswith("slmsuite.")
    ]
    assert len(slmsuite_entries) == 17  # 11 cameras + 6 SLMs
    assert all(":" in entry for entry in slmsuite_entries)
    assert "thorlabs" in slmsuite_names


def test_available_backends_list_registered_names():
    from hologradpy.hardware import (
        available_camera_backends,
        available_slm_backends,
        register_slmsuite_backends,
    )

    register_camera_backend("listed_cam", _RawCamera)
    register_slm_backend("listed_slm", _RawSlm)
    register_slmsuite_backends()
    cameras = available_camera_backends()
    slms = available_slm_backends()
    # Sorted, and covering both hand-registered and opt-in slmsuite names.
    assert cameras == sorted(cameras)
    assert slms == sorted(slms)
    assert "listed_cam" in cameras and "thorlabs" in cameras
    assert "listed_slm" in slms and "hamamatsu" in slms


# --- autoexpose: discrete exposure steps ----------------------------------------


class _QuantizedCamera(CroppingCamera):
    """A native camera whose exposure snaps to a coarse grid, to exercise the discrete
    step guard in ``autoexpose``. The response is linear (``peak = gain * exposure``),
    saturating at the top of the range. The grid is chosen so no achievable exposure
    lands the peak inside the 50 percent tolerance band, forcing an oscillation between
    two neighbouring steps that the guard must stop.
    """

    def __init__(self, step: float = 1e-3, gain: float = 55e3, adu_levels: int = 256):
        super().__init__(
            (4, 4),
            max_pixel_value=adu_levels - 1,
            exposure_bounds=(step, 100 * step),
            exposure_s=step,
        )
        self._step = step
        self._gain = gain
        self.set_exposure_calls = 0

    def set_exposure(self, exposure_s):
        self.set_exposure_calls += 1
        lo, hi = self.exposure_bounds
        snapped = round(exposure_s / self._step) * self._step
        self._exposure_s = float(min(max(snapped, lo), hi))

    def render_sensor_frame(self):
        peak = min(self.max_pixel_value, round(self._gain * self.get_exposure()))
        return np.full(self.sensor_resolution, peak, dtype=float)


def test_autoexpose_settles_on_best_discrete_step():
    """With a coarsely quantized exposure whose steps straddle the target, autoexpose
    settles on the closest achievable exposure instead of spending its whole budget.
    """
    camera = _QuantizedCamera()  # peaks: 55 (1x), 110 (2x), 165 (3x). Target 128
    with pytest.warns(UserWarning, match="no finer exposure step"):
        exposure = camera.autoexpose(
            set_fraction=0.5, tolerance=0.05, max_iterations=5
        )

    # 2 * step gives peak 110 (closest to 128). 3 * step overshoots to 165.
    assert exposure == pytest.approx(2e-3)
    assert camera.get_exposure() == pytest.approx(2e-3)
    # The guard stops after a couple of steps. Without it the loop would spend the whole
    # budget calling set_exposure on values the camera cannot distinguish.
    assert camera.set_exposure_calls <= 5


# --- autoexpose: hot pixels -----------------------------------------------------


class _HotPixelCamera(CroppingCamera):
    """A camera imaging a broad blob that peaks well below saturation, plus one stuck
    pixel pinned at the top of the range. The stuck pixel makes the raw peak read as
    permanent saturation, so autoexpose only reaches the real signal if it ignores it.
    """

    def __init__(self, adu_levels: int = 256):
        super().__init__(
            (32, 32),
            max_pixel_value=adu_levels - 1,
            exposure_bounds=(1e-4, 1.0),
            exposure_s=1e-3,
        )
        yy, xx = np.mgrid[0:32, 0:32] - 16
        self._blob = np.exp(-(xx**2 + yy**2) / (2 * 5.0**2))  # broad, peak 1 at center

    def render_sensor_frame(self):
        # Blob peaks at 0.4 of full scale at the initial 1e-3 s exposure.
        full_scale = self.max_pixel_value
        gain = 0.4 * full_scale / 1e-3
        frame = np.clip(
            np.round(self._blob * gain * self.get_exposure()), 0, full_scale
        )
        frame = frame.astype(float)
        frame[0, 0] = full_scale  # lone stuck pixel at saturation
        return frame


def test_autoexpose_excluded_pixels_targets_real_signal():
    """A stuck pixel reads as permanent saturation and rails the exposure. Excluding it
    via Camera.excluded_pixels lets autoexpose target the real blob near the target.
    """
    # Without excluding it, the stuck pixel forces the overexposed branch every step
    # until the exposure rails at the lower bound.
    railed = _HotPixelCamera()
    with pytest.raises(RuntimeError):
        railed.autoexpose(set_fraction=0.5, tolerance=0.05)

    camera = _HotPixelCamera()
    camera.excluded_pixels = [(0, 0)]  # the stuck pixel
    exposure = camera.autoexpose(set_fraction=0.5, tolerance=0.05)

    # Exposure rose from 1e-3 toward the target (blob was at 0.4, target 0.5).
    assert exposure > 1e-3
    # The real blob peak (stuck pixel excluded) sits near 50 percent of the range.
    frame = camera.get_image()
    frame[0, 0] = 0
    assert frame.max() == pytest.approx(0.5 * camera.adu_levels, rel=0.15)


def test_excluded_pixels_property_roundtrip():
    camera = _HotPixelCamera()
    assert camera.excluded_pixels == []  # empty by default
    camera.excluded_pixels = [(1, 2), (3, 4)]
    assert camera.excluded_pixels == [(1, 2), (3, 4)]  # stored as (row, col) tuples
    camera.excluded_pixels = None  # clears back to empty
    assert camera.excluded_pixels == []


def test_excluded_pixels_are_per_instance():
    """The class-level default is only a sentinel: setting on one camera must not leak
    to another (no shared mutable default).
    """
    first, second = _HotPixelCamera(), _HotPixelCamera()
    first.excluded_pixels = [(0, 0)]
    assert first.excluded_pixels == [(0, 0)]
    assert second.excluded_pixels == []


# Stuck pixels held at a fixed value across the sweep. Three lie in the illuminated
# disk with random values in [0, adu - 1] (detectable whatever their value), one is
# stuck high in the dark surround (stands out anywhere), and one is stuck low in the
# dark surround (indistinguishable from the unilluminated background).
_DISK_STUCK = {(13, 13), (16, 16), (18, 12)}
_DARK_STUCK_HIGH = (28, 28)
_DARK_STUCK_LOW = (2, 2)


class _SceneCamera(CroppingCamera):
    """A central illuminated disk on a dark surround, with additive read noise on every
    frame and several pixels stuck at fixed values (see the module-level constants).

    The working pixels vary from frame to frame, so this exercises the noise-tolerant
    detector rather than assuming exactly constant frames.
    """

    def __init__(self, adu_levels=256, saturating=False, seed=0):
        super().__init__(
            (32, 32),
            max_pixel_value=adu_levels - 1,
            exposure_bounds=(1e-4, 1.0),
            exposure_s=1e-3,
        )
        self._saturating = saturating
        self._noise = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:32, 0:32] - 16
        self._disk = (xx**2 + yy**2) <= 10**2
        values = np.random.default_rng(seed + 1)
        self._stuck = {
            pixel: int(values.integers(0, adu_levels)) for pixel in _DISK_STUCK
        }
        self._stuck[_DARK_STUCK_HIGH] = 200
        self._stuck[_DARK_STUCK_LOW] = 10  # below the noise floor, indistinguishable

    def render_sensor_frame(self):
        if self._saturating:
            signal = 2.0 * self.adu_levels  # above the ceiling, so it clamps to max
        else:
            # Unsaturated when dim, saturates when bright.
            signal = 2e5 * self.get_exposure()
        field = np.where(self._disk, signal, 0.0)
        frame = np.round(
            np.clip(
                field + self._noise.normal(0, 4.0, field.shape),
                0,
                self.max_pixel_value,
            )
        )
        for (row, col), stuck_value in self._stuck.items():
            frame[row, col] = float(stuck_value)  # stuck: exact value, no noise
        return frame


def test_find_stuck_pixels_flags_hot_dead_and_nonzero():
    """The stuck pixels inside the illuminated disk (fixed random values) and the pixel
    stuck high in the dark surround are all flagged despite read noise. The pixel stuck
    low in the dark surround is not, being indistinguishable from the background.
    """
    camera = _SceneCamera()
    found = set(camera.find_stuck_pixels())
    assert found == _DISK_STUCK | {_DARK_STUCK_HIGH}
    assert _DARK_STUCK_LOW not in found


def test_find_stuck_pixels_warns_on_overexposed_blob():
    """A disk saturated across the whole sweep is the camera overexposed, not hot
    pixels, so it warns and the saturated disk background is not excluded.
    """
    camera = _SceneCamera(saturating=True)
    with pytest.warns(UserWarning, match="overexposed"):
        found = set(camera.find_stuck_pixels())
    assert (10, 10) not in found  # a plain saturated disk pixel is overexposure


class _AutoDetectCamera(CroppingCamera):
    """A uniform field that saturates at the initial exposure, so autoexpose sweeps it
    down toward the target, capturing a wide exposure range, plus one dead pixel to find
    from those frames.
    """

    def __init__(self, adu_levels=256, seed=0):
        super().__init__(
            (16, 16),
            max_pixel_value=adu_levels - 1,
            exposure_bounds=(1e-5, 1.0),
            exposure_s=1e-3,  # starts saturated, autoexpose sweeps it down
        )
        self._noise = np.random.default_rng(seed)

    def render_sensor_frame(self):
        # Saturates at 1e-3, unsaturated when swept down.
        field = 5e5 * self.get_exposure()
        noise = self._noise.normal(0, 3.0, self.sensor_resolution)
        frame = np.round(np.clip(field + noise, 0, self.max_pixel_value))
        frame[8, 8] = 0.0  # dead pixel
        return frame


def test_autoexpose_detect_stuck_pixels_flag():
    """autoexpose(detect_stuck_pixels=True) runs the detection on the frames it captured
    while converging, populating excluded_pixels in the same call (no second sweep).
    """
    camera = _AutoDetectCamera()
    camera.autoexpose(set_fraction=0.5, tolerance=0.05, detect_stuck_pixels=True)
    assert camera.excluded_pixels == [(8, 8)]


def test_capture_exposure_sweep_drops_out_of_bounds_exposures():
    """Exposures above the upper bound are dropped, not clipped to it, so the sweep
    keeps its spacing and still detects the stuck pixels from the in-bounds ones.
    """
    camera = _SceneCamera()  # bounds (1e-4, 1.0)
    frames, exposures = camera._capture_exposure_sweep(
        exposures=[1e-4, 1e-1, 5.0, 10.0]
    )
    assert exposures == [1e-4, 1e-1]  # the two out-of-bounds values are dropped
    found = set(camera._detect_stuck_pixels(frames, exposures))
    assert found == _DISK_STUCK | {_DARK_STUCK_HIGH}


def test_find_stuck_pixels_needs_two_in_bounds_exposures():
    """Fewer than two exposures within the bounds cannot reveal a response, so it raises
    rather than guessing.
    """
    camera = _SceneCamera()
    with pytest.raises(ValueError, match="at least two exposures"):
        camera.find_stuck_pixels(exposures=[1e-3, 5.0])


def test_detect_stuck_pixels_ignores_repeated_exposures():
    """Two frames at one exposure have an exposure ratio of 1, so a stuck pixel keeps
    its count and passes as responding. The repeat is dropped, and the pair at 100 us
    and 100 ms finds every stuck pixel. Exposures closer than the response tolerance
    cannot be compared at all.
    """
    camera = _SceneCamera()
    exposures = [1e-4, 1e-4, 1e-1]
    frames = np.stack(
        [np.asarray(camera.get_image(exposure), dtype=float) for exposure in exposures]
    )

    found = set(camera._detect_stuck_pixels(frames, exposures))

    assert found == _DISK_STUCK | {_DARK_STUCK_HIGH}
    with pytest.raises(ValueError, match="two or more exposures"):
        camera._detect_stuck_pixels(frames[:2], [1e-3, 1.1e-3])


# --- CameraOrientation ----------------------------------------------------------


def test_the_eight_orientations_are_distinct_and_recoverable():
    """A mounting is read back from the transform a camera applies, so the two have to
    agree for all eight.
    """
    shape = (240, 320)
    matrices = set()
    for orientation in CameraOrientation.dihedral():
        matrix = orientation.matrix(shape)
        matrices.add(tuple(matrix.ravel()))
        assert CameraOrientation.from_matrix(matrix, shape) == orientation
    assert len(matrices) == 8


def test_a_transform_outside_the_eight_has_no_orientation():
    """A camera is free to apply any transform to its frames, and saying so beats
    naming the nearest of the eight.
    """
    stretch = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert CameraOrientation.from_matrix(stretch, (240, 320)) is None


def test_composing_two_mountings_gives_a_third():
    """Remounting an already-mounted camera is one of the eight, which is what lets a
    correction relative to the current mounting become the absolute one.
    """
    identity, flip = CameraOrientation(), CameraOrientation(fliplr=True)

    # self after other, matching GeometricTransform.compose.
    assert identity.compose(flip) == flip
    assert flip.compose(identity) == flip
    assert flip.compose(flip) == identity          # a flip undoes itself
    assert CameraOrientation("90").compose(CameraOrientation("270")) == identity
    assert CameraOrientation("90").compose(CameraOrientation("90")).rot == "180"

    # Closed, so composing never escapes the eight.
    every = CameraOrientation.dihedral()
    assert {a.compose(b) for a in every for b in every} == set(every)


def test_composition_matches_applying_both_transforms():
    """The algebra has to agree with the frames, since the frames are what a camera
    actually returns.
    """
    frame = np.arange(15).reshape(3, 5)
    for a in CameraOrientation.dihedral():
        for b in CameraOrientation.dihedral():
            both = a.transformation()(b.transformation()(frame))
            composed = a.compose(b).transformation()(frame)
            np.testing.assert_array_equal(composed, both)


def test_a_camera_reports_the_orientation_it_was_built_with():
    _, camera = _build()
    assert camera.orientation == CameraOrientation()

    _, rotated = _build()
    rotated.set_orientation(CameraOrientation("90", fliplr=True))
    assert rotated.orientation == CameraOrientation("90", fliplr=True)


def test_orientation_matrix_maps_raw_pixels_to_the_displayed_frame():
    """The matrix maps a pixel of the raw 240 x 320 sensor to its position in the
    displayed frame. This holds in all eight orientations, the quarter turns included.
    """
    _, camera = _build()
    raw_shape = (240, 320)
    raw_pixels = [(0, 0), (17, 250), (239, 319), (100, 3), (239, 0)]
    for orientation in CameraOrientation.dihedral():
        camera.set_orientation(orientation)
        matrix = camera.orientation_matrix()
        for row, col in raw_pixels:
            one_hot = np.zeros(raw_shape)
            one_hot[row, col] = 1.0
            ((shown_row, shown_col),) = np.argwhere(camera.transform(one_hot))
            np.testing.assert_allclose(
                matrix @ [col, row, 1.0], [shown_col, shown_row], atol=1e-9
            )
        assert camera.orientation == orientation


def test_orientation_records_probed_on_either_shape_decode_alike():
    """A matrix probed on the displayed shape and one probed on the raw shape name the
    same orientation, so snapshots probed either way decode alike.
    """
    for orientation in CameraOrientation.dihedral():
        for probed_shape in [(240, 320), (320, 240)]:
            matrix = orientation.matrix(probed_shape)
            for decoding_shape in [(240, 320), (320, 240)]:
                decoded = CameraOrientation.from_matrix(matrix, decoding_shape)
                assert decoded == orientation


def test_set_orientation_swaps_the_displayed_shape_for_a_quarter_turn():
    """What the constructor does for a rotated mount, done later: the frame comes back
    transposed and the geometry follows it.
    """
    _, camera = _build()
    camera.set_exposure(1e-3)
    assert camera.get_image().shape == (240, 320)

    camera.set_orientation(CameraOrientation("90"))
    assert camera.resolution == (320, 240)
    assert camera.shape == (320, 240)
    assert camera.sensor_resolution == (320, 240)
    assert camera.get_image().shape == (320, 240)

    # And back, which is the case a suggestion being adopted then undone would hit.
    camera.set_orientation(CameraOrientation())
    assert camera.resolution == (240, 320)
    assert camera.get_image().shape == (240, 320)


def test_a_snapshot_records_the_panel_and_derives_the_frame():
    """A snapshot used to store the frame shape beside the region that defines it, and
    not store the panel at all, so a saved record could not answer what the sensor
    was.
    """
    from hologradpy.hardware.camera.abstract import CameraData

    _, camera = _build()
    camera.set_roi(ROI(10, 20, 60, 80))
    recorded = CameraData.from_camera(camera)

    assert recorded.sensor_resolution == (240, 320)
    assert recorded.resolution == (60, 80)
    assert recorded.orientation_flags == CameraOrientation()

    rotated = CameraData.from_camera(_rotated_camera())
    assert rotated.orientation_flags == CameraOrientation("90")


def _rotated_camera() -> Camera:
    _, camera = _build()
    camera.set_orientation(CameraOrientation("90"))
    return camera


def test_set_orientation_resets_a_crop_from_the_old_frame():
    """A region of interest is expressed in the displayed frame, which the new mounting
    replaces, so keeping it would crop somewhere unintended.
    """
    _, camera = _build()
    camera.set_roi(ROI(10, 20, 30, 40))
    camera.set_orientation(CameraOrientation("90"))
    assert camera.roi == ROI(0, 0, 320, 240)


def test_the_simulator_moves_excluded_pixels_with_the_sensor():
    """An excluded pixel names a sensor pixel, so it is remapped to the position of
    that pixel under the new mounting. The slmsuite adapter does the same.
    """
    _, camera = _build()
    stuck = (17, 250)
    camera.excluded_pixels = [stuck]

    for orientation in [CameraOrientation("90"), CameraOrientation("180", True)]:
        camera.set_orientation(orientation)
        one_hot = np.zeros((240, 320))
        one_hot[stuck] = 1.0
        ((shown_row, shown_col),) = np.argwhere(orientation.transformation()(one_hot))
        assert camera.excluded_pixels == [(int(shown_row), int(shown_col))]
    camera.set_orientation(CameraOrientation())
    assert camera.excluded_pixels == [stuck]


def test_the_simulator_exchanges_its_pixel_pitch_under_a_quarter_turn():
    _, camera = _build()

    camera.set_orientation(CameraOrientation("90"))
    np.testing.assert_allclose(camera.pixel_size, CAMERA_PIXEL_SIZE[::-1], rtol=1e-6)
    x_grid, _ = camera.get_spatial_grid()
    assert tuple(x_grid.shape) == (320, 240)

    camera.set_orientation(CameraOrientation("180"))
    np.testing.assert_allclose(camera.pixel_size, CAMERA_PIXEL_SIZE, rtol=1e-6)


def test_the_simulated_camera_reports_overexposure_on_either_path():
    _, camera = _build()
    exposure = camera.autoexpose(raise_on_rail=False)

    camera.get_image_tensor()
    assert not camera.overexposed

    camera.set_exposure(10 * exposure)
    camera.get_image_tensor()
    assert camera.overexposed
    camera.get_image()
    assert camera.overexposed
    camera.get_image_tensor(mask=np.zeros(camera.resolution, dtype=bool))
    assert not camera.overexposed


def test_simulated_camera_clips_exposure_to_its_bounds():
    """Into the default bounds of (0, 1) s, with a warning, as slmsuite's cameras do."""
    _, camera = _build()

    with pytest.warns(UserWarning, match="outside the camera's bounds"):
        camera.set_exposure(2.0)
    assert camera.get_exposure() == 1.0

    with pytest.warns(UserWarning, match="outside the camera's bounds"):
        camera.set_exposure(-1.0)
    assert camera.get_exposure() == 0.0


def test_simulated_camera_rejects_an_off_frame_roi():
    _, camera = _build()
    kept = ROI(10, 20, 30, 40)
    camera.set_roi(kept)

    for region in [ROI(-1, 0, 4, 4), ROI(0, 0, 241, 320), ROI(0, 0, 0, 4)]:
        with pytest.raises(ValueError, match="does not lie inside"):
            camera.set_roi(region)
        assert camera.roi == kept


def test_a_camera_that_cannot_be_reoriented_says_so():
    class _Fixed(Camera):
        pixel_size = np.array([1.0, 1.0])
        resolution = (4, 4)
        sensor_resolution = (4, 4)
        max_pixel_value = 255
        exposure_bounds = None
        roi = ROI(0, 0, 4, 4)

        def set_roi(self, roi): ...
        def get_exposure(self): return 0.0
        def set_exposure(self, exposure_s): ...
        def _get_image(self, exposure=None, averaging=1):
            return np.zeros((4, 4))

    camera = _Fixed()
    # With no transform of its own it is axis-aligned, which it can still report.
    assert camera.orientation == CameraOrientation()
    with pytest.raises(NotImplementedError, match="reoriented"):
        camera.set_orientation(CameraOrientation("90"))


def test_phase_and_levels_are_told_apart_by_the_caller() -> None:
    """Not by dtype: an integer array of radians would otherwise become levels, and the
    call would look identical either way.
    """
    slm = _slm_at(8)

    with pytest.raises(TypeError, match="set_levels"):
        slm.set_phase(np.zeros(slm.resolution, dtype=np.uint8))

    # And the two agree where they overlap, since set_phase goes through set_levels.
    phase = np.full(slm.resolution, 1.0)
    slm.set_phase(phase)
    through_phase = slm.display.copy()

    slm.set_levels(slm.phase_to_levels(phase))
    assert np.array_equal(slm.display, through_phase)


# --- Corrections ----------------------------------------------------------------


def _aberrated_slm(rms: float = 1.5):
    """An SLM, and a measured field carrying a known aberration on this bench."""
    slm = _slm_at(8)
    grid_x, grid_y = slm.get_spatial_grid()
    aberration = rms * (
        (grid_x / grid_x.abs().max()) ** 2 - (grid_y / grid_y.abs().max()) ** 2
    )
    measured = ComplexAmplitude(
        torch.ones(slm.resolution) * torch.exp(1j * aberration),
        wavelength=torch.tensor(slm.wavelength),
        pixel_size=torch.as_tensor(slm.pixel_size),
    )
    return slm, aberration, measured


def _residual(displayed: torch.Tensor, aberration: torch.Tensor) -> float:
    """What is left once the bench adds its own aberration back on.

    Wrapped, because a phase means the same thing a turn away, and the panel returns it
    wrapped into its own range.
    """
    return float(torch.angle(torch.exp(1j * (displayed + aberration))).std())


def test_a_measured_wavefront_is_cancelled_not_doubled() -> None:
    """The whole point, and the one that catches the sign being backwards: a measurement
    says what aberration is present, so the correction is its negative.
    """
    slm, aberration, measured = _aberrated_slm()
    slm.load_measured_wavefront(measured)
    flat = np.zeros(slm.resolution)

    slm.set_phase(flat)
    uncorrected = _residual(slm.virtual_slm.get_phase(), aberration)
    slm.set_phase(flat, apply_phase_correction=True)
    corrected = _residual(slm.virtual_slm.get_phase(), aberration)

    assert corrected < uncorrected / 10


def test_the_correction_backwards_makes_it_worse() -> None:
    """What the negation on load buys. A bare array is taken as already a correction, so
    handing it the aberration itself is the mistake this guards.
    """
    slm, aberration, _ = _aberrated_slm()
    flat = np.zeros(slm.resolution)
    slm.set_phase(flat)
    uncorrected = _residual(slm.virtual_slm.get_phase(), aberration)

    slm.load_phase_correction(np.asarray(aberration))
    slm.set_phase(flat, apply_phase_correction=True)

    assert _residual(slm.virtual_slm.get_phase(), aberration) > uncorrected


def test_a_correction_is_off_unless_it_is_asked_for() -> None:
    """The default that keeps a wavefront calibration honest: measuring through an
    active correction recovers the wrong wavefront, and the error compounds.
    """
    slm, _, measured = _aberrated_slm()
    flat = np.zeros(slm.resolution)

    slm.set_phase(flat)
    before = slm.display.copy()
    slm.load_measured_wavefront(measured)
    slm.set_phase(flat)

    assert np.array_equal(slm.display, before)


def test_asking_for_a_correction_that_is_not_loaded_raises() -> None:
    """The difference between a correction switched off and one never loaded."""
    slm = _slm_at(8)
    with pytest.raises(ValueError, match="load_phase_correction"):
        slm.set_phase(np.zeros(slm.resolution), apply_phase_correction=True)
    with pytest.raises(ValueError, match="load_vendor_correction"):
        slm.set_phase(np.zeros(slm.resolution), apply_vendor_correction=True)


def test_a_correction_has_to_be_the_panels_shape() -> None:
    slm = _slm_at(8)
    with pytest.raises(ValueError, match="per pixel"):
        slm.load_phase_correction(np.zeros((4, 4)))


def test_only_the_phase_of_a_measurement_is_kept() -> None:
    """A phase-only panel cannot fix an amplitude, so the amplitude is dropped rather
    than quietly folded in.
    """
    slm, aberration, measured = _aberrated_slm()
    dim = ComplexAmplitude(
        0.01 * measured.as_tensor(),
        wavelength=measured.wavelength,
        pixel_size=measured.pixel_size,
    )
    slm.load_measured_wavefront(dim)

    np.testing.assert_allclose(
        slm.phase_correction, -np.asarray(aberration), atol=1e-5
    )


def test_a_vendor_correction_moves_the_phase_it_says() -> None:
    """A vendor correction is read on the nominal scale, where 256 levels make one
    cycle. It is added as a phase before the response converts the phase to a level.
    Under a measured curve, the corrected level therefore comes from the curve and
    differs from the uncorrected level plus the same number of levels.
    """
    response = _s_curve()
    slm = _slm_at(8)
    slm.load_phase_response(response)
    phase = np.full(slm.resolution, -2.0)

    slm.set_phase(phase)
    plain = int(slm.display[0, 0])
    slm.load_vendor_correction(np.full(slm.resolution, 7, dtype=np.uint8))
    slm.set_phase(phase, apply_vendor_correction=True)

    corrected = response.display_levels(np.array([-2.0 - 2 * np.pi * 7 / 256]))
    assert int(slm.display[0, 0]) == int(corrected[0])
    # On this curve, seven nominal levels of phase move the level by five.
    assert int(slm.display[0, 0]) != (plain + 7) % 256


def test_a_vendor_correction_wraps_rather_than_clipping() -> None:
    """Past the top of the range it comes back round, as the panel does."""
    slm = _slm_at(8)
    phase = np.full(slm.resolution, -2.0)
    slm.set_phase(phase)
    plain = int(slm.display[0, 0])

    slm.load_vendor_correction(np.full(slm.resolution, 250, dtype=np.uint16))
    slm.set_phase(phase, apply_vendor_correction=True)

    assert int(slm.display[0, 0]) == (plain + 250) % 256


def test_a_capture_carries_the_corrections_themselves() -> None:
    """Not a name for them. A dataset is reinterpreted long after the file a name points
    at has moved, and by then only the numbers are any use.
    """
    slm, aberration, measured = _aberrated_slm()
    assert SLMData.from_slm(slm).phase_correction is None

    slm.load_measured_wavefront(measured)
    vendor = np.full(slm.resolution, 3, dtype=np.uint8)
    slm.load_vendor_correction(vendor)
    recorded = SLMData.from_slm(slm)

    np.testing.assert_allclose(
        recorded.phase_correction, -np.asarray(aberration), atol=1e-5
    )
    np.testing.assert_array_equal(recorded.vendor_correction, vendor)


def test_a_vendor_correction_wraps_at_the_slms_full_cycle() -> None:
    """An SLM reaching one cycle at level 200 wraps a corrected phase there, not at
    256. Level 180 is 0.9 cycle, a correction of 100 nominal levels adds 0.39 cycle,
    and the sum wraps to 0.29 cycle, which is level 58.
    """
    raw = _raw_slm()
    slm = as_slm(raw)
    slm.load_phase_response(LinearResponse(bitdepth=8, phase_scaling=256 / 200))
    slm.load_vendor_correction(np.full(slm.resolution, 100, dtype=np.uint8))
    phase = np.full(slm.resolution, -2 * np.pi * 0.9)

    slm.set_phase(phase, apply_vendor_correction=False)
    assert np.all(raw.written == 180)
    slm.set_phase(phase)
    assert np.all(raw.written == 58)


def test_a_vendor_correction_adds_whole_levels_under_the_nominal_response() -> None:
    """Where 256 levels make one cycle, a correction of ``V`` levels moves every pixel
    by ``V`` levels, wrapped at 256.
    """
    raw = _raw_slm()
    slm = as_slm(raw)
    rng = np.random.default_rng(7)
    phase = rng.uniform(-20.0, 20.0, slm.resolution)
    vendor = rng.integers(0, 256, slm.resolution, dtype=np.uint8)

    slm.set_phase(phase)
    plain = raw.written.astype(np.int64)
    slm.load_vendor_correction(vendor)
    slm.set_phase(phase)

    np.testing.assert_array_equal(raw.written, (plain + vendor) % 256)


def test_a_loaded_vendor_correction_applies_by_default() -> None:
    """A vendor correction describes the SLM, so once loaded it reaches every phase
    unless a caller turns it off.
    """
    raw = _raw_slm()
    slm = as_slm(raw)
    phase = np.random.default_rng(8).uniform(-20.0, 20.0, slm.resolution)
    slm.set_phase(phase)
    plain = raw.written.astype(np.int64)

    slm.load_vendor_correction(np.full(slm.resolution, 7, dtype=np.uint8))
    slm.set_phase(phase)
    np.testing.assert_array_equal(raw.written, (plain + 7) % 256)

    slm.set_phase(phase, apply_vendor_correction=False)
    np.testing.assert_array_equal(raw.written, plain)


def test_a_phase_that_is_not_finite_or_real_is_refused() -> None:
    """A NaN or infinite phase has no level, and a complex array is a field, so each
    raises before anything reaches the SLM.
    """
    slm = _slm_at(8)
    slm.set_phase(np.full(slm.resolution, 1.0))
    before = slm.display.copy()

    for bad_value in (np.nan, np.inf):
        phase = np.zeros(slm.resolution)
        phase[3, 4] = bad_value
        with pytest.raises(ValueError, match="NaN or infinite at 1 of"):
            slm.set_phase(phase)
        np.testing.assert_array_equal(slm.display, before)

    field = np.exp(1j * np.ones(slm.resolution))
    with pytest.raises(TypeError, match="real phase"):
        slm.set_phase(field)
    np.testing.assert_array_equal(slm.display, before)


def test_a_complex_array_correction_keeps_only_its_phase() -> None:
    angle = np.random.default_rng(9).uniform(-3.0, 3.0, (32, 48))
    slm = _slm_at(8)

    slm.load_phase_correction(np.exp(1j * angle))
    assert np.isrealobj(slm.phase_correction)
    np.testing.assert_allclose(slm.phase_correction, angle)

    slm.load_measured_wavefront(np.exp(1j * angle))
    assert np.isrealobj(slm.phase_correction)
    np.testing.assert_allclose(slm.phase_correction, -angle)


def test_a_correction_that_is_not_finite_or_whole_is_refused() -> None:
    slm = _slm_at(8)
    correction = np.zeros(slm.resolution)
    correction[3, 4] = np.nan

    with pytest.raises(ValueError, match="NaN or infinite at 1 of"):
        slm.load_phase_correction(correction)
    with pytest.raises(TypeError, match="whole gray levels"):
        slm.load_vendor_correction(np.full(slm.resolution, 7.0))
    assert slm.phase_correction is None
    assert slm.vendor_correction is None


# --- The SLM contract: settle and display ---------------------------------------


def test_the_adapter_waits_settle_time_after_each_write(virtual_clock) -> None:
    """The wrapped SLM sleeps only for a write made with ``settle=True``. Its own
    ``settle`` flag is off, so the adapter passes ``settle=True`` on every write.
    """
    raw = _raw_slm()
    slm = as_slm(raw)
    slm.settle_time = 0.05
    assert raw.settle is False

    slm.set_phase(np.zeros(slm.resolution))
    assert virtual_clock.now == pytest.approx(0.05)
    slm.set_levels(np.zeros(slm.resolution, dtype=np.uint8))
    assert virtual_clock.now == pytest.approx(0.10)

    assert raw.settle_time_s == 0.05
    assert "settle_time" not in vars(slm)
    assert SLMData.from_slm(slm).settle_time == 0.05


def test_settle_time_comes_from_the_driver() -> None:
    slm = open_slm(
        _RawSlm,
        resolution=(6, 4),
        bitdepth=8,
        wav_um=0.5,
        pitch_um=(12.5, 12.5),
        settle_time_s=0.2,
    )
    assert slm.settle_time == 0.2
    assert SLMData.from_slm(slm).settle_time == 0.2


@pytest.mark.parametrize("seconds", [-0.1, float("nan"), float("inf")])
def test_a_settle_time_below_zero_is_refused(seconds: float) -> None:
    """A negative or non-finite wait is refused before it reaches slmsuite, which
    sleeps only after the SLM has already changed.
    """
    raw = _raw_slm(settle_time_s=0.2)
    slm = as_slm(raw)

    with pytest.raises(ValueError, match="finite number of zero or more"):
        slm.settle_time = seconds
    assert raw.settle_time_s == 0.2


def test_the_simulator_never_sleeps(monkeypatch: pytest.MonkeyPatch) -> None:
    """The simulated SLM shows a pattern at once and only records its settle time."""
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    sim = _slm_at(8, settle_time=0.3)

    sim.set_phase(np.zeros(sim.resolution))

    assert sleeps == []
    assert SLMData.from_slm(sim).settle_time == 0.3


def test_the_adapters_report_the_name_of_the_device() -> None:
    """The records of an SLM and a camera take the name the driver was opened with."""
    from hologradpy.hardware.camera.abstract import CameraData

    slm = as_slm(_raw_slm(name="bench slm"))
    camera = as_camera(_raw_camera(name="bench camera"))

    assert slm.name == "bench slm"
    assert SLMData.from_slm(slm).name == "bench slm"
    assert camera.name == "bench camera"
    assert CameraData.from_camera(camera).name == "bench camera"


def test_closing_the_adapter_closes_the_slm(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _raw_slm()
    closes: list[bool] = []
    monkeypatch.setattr(raw, "close", lambda: closes.append(True))

    as_slm(raw).close()

    assert closes == [True]


def test_the_adapter_reports_what_the_slm_shows() -> None:
    raw = _raw_slm()
    slm = as_slm(raw)
    levels = np.random.default_rng(10).integers(0, 256, slm.resolution, dtype=np.uint8)

    slm.set_levels(levels)

    assert slm.display is raw.display
    np.testing.assert_array_equal(slm.display, levels)


def test_a_native_slm_has_settle_and_display_defaults() -> None:
    device = _BareSlm()
    assert device.settle_time == 0.0
    assert device.display is None
    assert SLMData.from_slm(device).settle_time == 0.0


# --- The SLM contract: levels ---------------------------------------------------


_LEVEL_TYPES = {
    "int64": lambda values: values.astype(np.int64),
    "uint16": lambda values: values.astype(np.uint16),
    "torch int64": lambda values: torch.as_tensor(values, dtype=torch.int64),
    "cuda int64": lambda values: torch.as_tensor(
        values, dtype=torch.int64, device="cuda"
    ),
}


@pytest.mark.parametrize(
    "level_type",
    [
        pytest.param(
            name,
            marks=pytest.mark.skipif(
                name.startswith("cuda") and not torch.cuda.is_available(),
                reason="No CUDA device.",
            ),
        )
        for name in _LEVEL_TYPES
    ],
)
def test_the_adapter_hands_slmsuite_its_display_dtype(level_type: str) -> None:
    """The integer path of slmsuite takes only its own dtype, so whole levels of any
    integer type are cast to it.
    """
    raw = _raw_slm()
    slm = as_slm(raw)
    values = np.random.default_rng(11).integers(0, 256, slm.resolution)

    slm.set_levels(_LEVEL_TYPES[level_type](values))

    assert raw.written.dtype == np.uint8
    np.testing.assert_array_equal(raw.written, values)


_REFUSED_LEVELS = {
    "256 as uint16": (lambda shape: np.full(shape, 256, dtype=np.uint16), ValueError),
    "-1 as int64": (lambda shape: np.full(shape, -1, dtype=np.int64), ValueError),
    "floats": (lambda shape: np.full(shape, 10.0), TypeError),
    "larger": (
        lambda shape: np.zeros((shape[0] + 1, shape[1] + 1), dtype=np.uint8),
        ValueError,
    ),
    "smaller": (
        lambda shape: np.zeros((shape[0] - 1, shape[1]), dtype=np.uint8),
        ValueError,
    ),
}


@pytest.mark.parametrize("case", list(_REFUSED_LEVELS))
@pytest.mark.parametrize("device", ["adapter", "simulator"])
def test_levels_the_slm_cannot_show_are_refused(device: str, case: str) -> None:
    """The hardware and the simulator refuse the same levels, before anything is
    written.
    """
    raw = _raw_slm()
    slm = as_slm(raw) if device == "adapter" else _slm_at(8)
    slm.set_levels(np.full(slm.resolution, 5, dtype=np.uint8))
    before = slm.display.copy()
    raw.written = None

    make_levels, error = _REFUSED_LEVELS[case]
    with pytest.raises(error):
        slm.set_levels(make_levels(slm.resolution))

    np.testing.assert_array_equal(slm.display, before)
    assert raw.written is None


def test_the_adapter_and_the_simulator_show_the_same_levels() -> None:
    sim = _slm_at(8)
    height, width = sim.resolution
    slm = as_slm(
        _RawSlm(
            resolution=(width, height),
            bitdepth=8,
            wav_um=WAVELENGTH * 1e6,
            pitch_um=(SLM_PITCH * 1e6, SLM_PITCH * 1e6),
        )
    )
    phase = np.random.default_rng(12).uniform(-20.0, 20.0, sim.resolution)

    sim.set_phase(phase)
    slm.set_phase(phase)

    np.testing.assert_array_equal(slm.display, sim.display)


# --- The SLM contract: phase response -------------------------------------------


def test_the_adapter_reads_the_design_wavelength() -> None:
    """An SLM used below its design wavelength reaches more than one cycle. The number
    of cycles is the reciprocal of slmsuite's own ``phase_scaling``.
    """
    raw = _raw_slm(wav_um=0.5, wav_design_um=0.7)
    slm = as_slm(raw)

    assert slm.phase_response.bitdepth == 8
    assert slm.phase_scaling == pytest.approx(1.4)
    assert slm.phase_scaling == pytest.approx(1 / raw.phase_scaling)
    assert as_slm(_raw_slm(wav_um=0.5)).phase_response == LinearResponse(bitdepth=8)


def test_a_loaded_phase_response_drives_the_levels() -> None:
    """A response reaching one cycle at level 200 converts every later phase, and every
    record or model built afterwards carries it.
    """
    raw = _raw_slm()
    slm = as_slm(raw)
    response = LinearResponse(bitdepth=8, phase_scaling=256 / 200)

    slm.load_phase_response(response)
    slm.set_phase(np.full(slm.resolution, -2 * np.pi * 0.75))

    assert np.all(raw.written == 150)
    assert SLMData.from_slm(slm).phase_response == response
    assert VirtualSLM.from_slm(slm).phase_scaling == pytest.approx(1.28)

    slm.load_phase_response(None)
    assert slm.phase_response == LinearResponse(bitdepth=8)


def test_a_phase_response_the_slm_cannot_use_is_refused() -> None:
    for slm in (as_slm(_raw_slm()), _slm_at(8)):
        with pytest.raises(ValueError, match="12-bit"):
            slm.load_phase_response(LinearResponse(bitdepth=12))
        with pytest.raises(TypeError, match="PhaseResponseModule"):
            slm.load_phase_response(PhaseResponseModule(LinearResponse(bitdepth=8)))
        assert slm.phase_response == LinearResponse(bitdepth=8)

    with pytest.raises(ValueError, match="no bitdepth"):
        _BareSlm().load_phase_response(LinearResponse(bitdepth=8))


def test_loading_a_response_on_the_simulator_keeps_its_levels() -> None:
    """A hardware SLM keeps showing its levels when a new response is loaded, so the
    simulated SLM does too. The phase of those levels follows the new response.
    """
    sim = _slm_at(8)
    sim.set_levels(np.full(sim.resolution, 100, dtype=np.uint8))
    curve = _s_curve()

    sim.load_phase_response(curve)

    assert np.all(sim.display == 100)
    assert float(sim.virtual_slm.get_phase()[0, 0]) == pytest.approx(curve.phases[100])
    assert sim.phase_response is curve


# --- The SLM contract: state shared on the device -------------------------------


def test_every_adapter_of_one_slm_shares_what_was_loaded() -> None:
    """Each consumer coerces the SLM passed to it into an adapter of its own, so
    everything loaded is kept on the SLM itself.
    """
    raw = _raw_slm()
    vendor = np.full(raw.shape, 3, dtype=np.uint8)
    response = LinearResponse(bitdepth=8, phase_scaling=256 / 200)

    as_slm(raw).load_vendor_correction(vendor)
    as_slm(raw).load_phase_response(response)

    np.testing.assert_array_equal(as_slm(raw).vendor_correction, vendor)
    assert SLMSuiteSLMAdapter(raw).phase_response == response
    copied = copy.copy(as_slm(raw))
    assert copied.vendor_correction is as_slm(raw).vendor_correction


def test_dropping_the_slm_closes_it_at_once() -> None:
    """Nothing on the SLM refers back to an adapter, so dropping the last reference
    closes it without the garbage collector. A driver whose SDK holds one device at a
    time can then open it again straight away.
    """
    closed: list[bool] = []

    class _ClosingSlm(_RawSlm):
        def close(self):
            closed.append(True)

    gc.disable()
    try:
        raw = _ClosingSlm(
            resolution=(6, 4), bitdepth=8, wav_um=0.5, pitch_um=(12.5, 12.5)
        )
        as_slm(raw).load_vendor_correction(np.zeros(raw.shape, dtype=np.uint8))
        as_slm(raw).set_phase(np.zeros(raw.shape))
        reference = weakref.ref(raw)
        del raw

        assert reference() is None
        assert closed == [True]
    finally:
        gc.enable()


# --- set_resolution: an SLM smaller than the wrapped SLM ------------------------


def test_a_smaller_resolution_pads_each_pattern_at_the_top_left() -> None:
    """The SLM is the top left of the wrapped SLM, and the rows and columns past it
    show level zero.
    """
    raw = _raw_slm()
    slm = as_slm(raw)
    slm.set_resolution((3, 4))
    levels = np.random.default_rng(13).integers(1, 256, (3, 4), dtype=np.uint8)

    slm.set_levels(levels)

    assert slm.resolution == (3, 4)
    assert raw.written.shape == (4, 6)
    np.testing.assert_array_equal(raw.written[:3, :4], levels)
    assert not raw.written[3:, :].any()
    assert not raw.written[:, 4:].any()
    np.testing.assert_array_equal(slm.display, levels)

    slm.set_resolution(None)
    assert slm.resolution == (4, 6)
    assert slm.display is raw.display


def test_a_resolution_larger_than_the_frame_is_refused() -> None:
    slm = as_slm(_raw_slm())

    for resolution in [(5, 6), (4, 7)]:
        with pytest.raises(ValueError, match="does not fit"):
            slm.set_resolution(resolution)
    with pytest.raises(ValueError, match="two positive sizes"):
        slm.set_resolution((0, 6))
    assert slm.resolution == (4, 6)


def test_a_correction_at_another_shape_blocks_a_new_resolution() -> None:
    """A correction is per pixel, so a loaded correction holds the resolution at its
    own shape.
    """
    slm = as_slm(_raw_slm())
    slm.set_resolution((3, 4))
    slm.load_vendor_correction(np.zeros((3, 4), dtype=np.uint8))
    for resolution in [(2, 4), None]:
        with pytest.raises(ValueError, match="vendor correction"):
            slm.set_resolution(resolution)
    assert slm.resolution == (3, 4)

    whole = as_slm(_raw_slm())
    whole.load_phase_correction(np.zeros((4, 6)))
    with pytest.raises(ValueError, match="phase correction"):
        whole.set_resolution((3, 4))
    whole.set_resolution((4, 6))
    assert whole.resolution == (4, 6)


def test_a_phase_is_converted_at_the_set_resolution() -> None:
    """Everything that reads the SLM's shape reads the set resolution: the conversion,
    the grid, the record and a model built from the SLM.
    """
    raw = _raw_slm(resolution=(8, 5))
    slm = as_slm(raw)
    slm.set_resolution((4, 6))
    phase = np.random.default_rng(14).uniform(-20.0, 20.0, (4, 6))

    slm.set_phase(phase)
    np.testing.assert_array_equal(slm.display, slm.phase_to_levels(phase))
    with pytest.raises(ValueError, match="per pixel"):
        slm.set_phase(np.zeros(raw.shape))

    grid_x, grid_y = slm.get_spatial_grid()
    assert tuple(grid_x.shape) == tuple(grid_y.shape) == (4, 6)
    assert SLMData.from_slm(slm).resolution == (4, 6)

    virtual = VirtualSLM.from_slm(slm)
    virtual.initialize_for_slm_plane(
        FieldGeometry(
            resolution=slm.resolution,
            pixel_size=torch.tensor(slm.pixel_size.tolist()),
            wavelength=torch.tensor(slm.wavelength),
        )
    )
    virtual.set_levels(slm.display, slm.bitdepth)
    assert virtual.slm_resolution == (4, 6)
    np.testing.assert_array_equal(
        virtual.phase_to_levels(virtual.get_phase(), slm.bitdepth), slm.display
    )


def test_every_adapter_of_one_slm_shares_its_resolution() -> None:
    raw = _raw_slm()
    as_slm(raw).set_resolution((3, 4))

    assert as_slm(raw).resolution == (3, 4)
    assert SLMSuiteSLMAdapter(raw).resolution == (3, 4)
    as_slm(raw).set_levels(np.ones((3, 4), dtype=np.uint8))
    assert raw.written.shape == (4, 6)


# --- The slmsuite camera adapter ------------------------------------------------

# A non-square raw sensor of (height, width) = (5, 8) with a distinct count at every
# pixel, so a flip, a turn or a crop in the wrong place changes the frame. Its pitch is
# (x, y) = (5, 7) um, so a swap of the pixel size shows.
_RAW_SENSOR_RESOLUTION = (5, 8)


def _raw_camera(**options) -> _RawCamera:
    """A raw 8-bit slmsuite camera with a (5, 8) sensor, with ``options`` overriding
    any constructor argument.
    """
    arguments = dict(resolution=(8, 5), bitdepth=8, pitch_um=(5.0, 7.0))
    arguments.update(options)
    return _RawCamera(**arguments)


def _raw_orientation_id(orientation: CameraOrientation) -> str:
    return f"rot{orientation.rot}" + ("-fliplr" if orientation.fliplr else "")


def test_every_adapter_of_one_camera_shares_its_settings() -> None:
    """Each consumer coerces the camera passed to it into an adapter of its own, so
    the settings are kept on the camera itself.
    """
    raw = _raw_camera()
    assert as_camera(raw).frame_timeout_s == 1.0

    as_camera(raw).set_roi(ROI(1, 2, 3, 4))
    as_camera(raw).excluded_pixels = [(0, 1)]
    as_camera(raw).frame_timeout_s = 2.5

    assert as_camera(raw).roi == ROI(1, 2, 3, 4)
    assert SLMSuiteCameraAdapter(raw).excluded_pixels == [(0, 1)]
    assert as_camera(raw).frame_timeout_s == 2.5
    copied = copy.copy(as_camera(raw))
    assert copied.roi == ROI(1, 2, 3, 4)
    assert as_camera(_raw_camera()).roi == ROI(0, 0, *_RAW_SENSOR_RESOLUTION)


def test_dropping_the_camera_closes_it_at_once() -> None:
    """Nothing on the camera refers back to an adapter, so dropping the last reference
    closes it without the garbage collector, and the SDK can open it again straight
    away.
    """
    closed: list[bool] = []

    class _ClosingCamera(_RawCamera):
        def close(self):
            closed.append(True)

    gc.disable()
    try:
        raw = _ClosingCamera(resolution=(8, 5), bitdepth=8, pitch_um=(5.0, 7.0))
        as_camera(raw).set_roi(ROI(0, 0, 2, 2))
        as_camera(raw).excluded_pixels = [(1, 1)]
        as_camera(raw).get_image()
        reference = weakref.ref(raw)
        del raw

        assert reference() is None
        assert closed == [True]
    finally:
        gc.enable()


def test_max_pixel_value_ignores_slmsuite_averaging() -> None:
    """The ``bitresolution`` of slmsuite grows with its ``averaging`` setting, which
    the frames of the adapter do not use.
    """
    raw = _raw_camera(bitdepth=10)
    raw.averaging = 4

    assert as_camera(raw).max_pixel_value == 1023
    assert as_camera(raw).adu_levels == 1024


def test_get_image_takes_single_frames_whatever_slmsuite_defaults_say() -> None:
    raw = _raw_camera()
    raw.averaging = 4
    raw.hdr = 2

    frame = as_camera(raw).get_image()

    assert raw.capture_calls == 1
    assert frame.dtype == np.uint16
    np.testing.assert_array_equal(frame, raw._sensor)


def test_averaging_sums_frames_inside_a_region_of_interest() -> None:
    raw = _RawCamera(resolution=(20, 30), bitdepth=8, pitch_um=(5.0, 7.0))
    whole_sensor_window = raw.woi
    camera = as_camera(raw)
    region = ROI(3, 5, 4, 8)
    camera.set_roi(region)

    summed = camera.get_image(averaging=3)

    assert summed.shape == (4, 8)
    assert summed.dtype == np.float64
    np.testing.assert_array_equal(summed, 3.0 * region.crop(raw._sensor))
    assert raw.capture_calls == 3
    assert raw.woi == whole_sensor_window


@pytest.mark.parametrize(
    "orientation", CameraOrientation.dihedral(), ids=_raw_orientation_id
)
def test_every_orientation_reports_the_frame_it_returns(
    orientation: CameraOrientation,
) -> None:
    """The frame, the resolution, the sensor resolution and the pixel size all follow
    the orientation, and returning to the identity restores them.
    """
    raw = _raw_camera()
    camera = as_camera(raw)

    camera.set_orientation(orientation)

    shown = orientation.transformation()(raw._sensor)
    np.testing.assert_array_equal(camera.get_image(), shown)
    assert camera.resolution == shown.shape
    assert camera.sensor_resolution == shown.shape
    assert camera.get_image(averaging=2).shape == shown.shape
    assert camera.orientation == orientation
    expected_pixel_size = (5e-6, 7e-6) if orientation.swaps_axes() else (7e-6, 5e-6)
    np.testing.assert_allclose(camera.pixel_size, expected_pixel_size)

    camera.set_orientation(CameraOrientation())
    np.testing.assert_array_equal(camera.get_image(), raw._sensor)
    assert camera.resolution == _RAW_SENSOR_RESOLUTION
    np.testing.assert_allclose(camera.pixel_size, (7e-6, 5e-6))


def test_a_region_of_interest_crops_the_displayed_frame() -> None:
    raw = _raw_camera()
    camera = as_camera(raw)
    orientation = CameraOrientation("90", fliplr=True)
    camera.set_orientation(orientation)
    region = ROI(2, 1, 5, 4)

    camera.set_roi(region)

    np.testing.assert_array_equal(
        camera.get_image(), region.crop(orientation.transformation()(raw._sensor))
    )
    assert camera.resolution == (5, 4)


def test_a_region_off_the_frame_is_refused() -> None:
    camera = as_camera(_raw_camera())
    height, width = _RAW_SENSOR_RESOLUTION
    kept = ROI(1, 1, 2, 2)
    camera.set_roi(kept)

    for region in [
        ROI(-1, 0, 2, 2),
        ROI(0, -1, 2, 2),
        ROI(0, 0, height + 1, width),
        ROI(0, 1, height, width),
        ROI(0, 0, 0, 4),
    ]:
        with pytest.raises(ValueError, match="does not lie inside"):
            camera.set_roi(region)
        assert camera.roi == kept

    camera.set_roi(ROI(0, 0, height, width))
    assert camera.roi == ROI(0, 0, height, width)


def test_a_driver_without_a_readout_window_can_be_reoriented() -> None:
    """The base ``set_woi`` of slmsuite raises ``NotImplementedError``, and the
    adapter never calls it.
    """

    class _WindowlessCamera(_RawCamera):
        def set_woi(self, woi=None):
            raise NotImplementedError()

    raw = _WindowlessCamera(resolution=(8, 5), bitdepth=8, pitch_um=(5.0, 7.0))
    camera = as_camera(raw)

    camera.set_orientation(CameraOrientation("90"))

    assert tuple(raw.shape) == tuple(raw.default_shape) == (8, 5)
    assert camera.orientation == CameraOrientation("90")
    assert camera.resolution == (8, 5)
    assert camera.get_image().shape == (8, 5)


def test_excluded_pixels_follow_the_sensor_when_reoriented() -> None:
    """A stuck pixel belongs to the sensor, so it stays excluded at its position in the
    new frame. Autoexposure indexes the new frame without running off it.
    """
    raw = _raw_camera()
    camera = as_camera(raw)
    stuck = (4, 7)
    camera.excluded_pixels = [stuck]
    stuck_count = camera.get_image()[stuck]

    camera.set_orientation(CameraOrientation("90"))

    (moved,) = camera.excluded_pixels
    assert moved != stuck
    assert camera.get_image()[moved] == stuck_count
    camera.autoexpose(raise_on_rail=False, max_iterations=1)


def test_reorient_pixels_follows_every_sensor_pixel() -> None:
    """For every pair of the eight orientations, each pixel maps to the position of the
    same sensor pixel in the target frame. Mapping back returns the original pixel.
    """
    raw_shape = (3, 5)
    sensor = np.arange(15).reshape(raw_shape)
    every = CameraOrientation.dihedral()
    for source in every:
        shown_by_source = source.transformation()(sensor)
        pixels = [tuple(index) for index in np.ndindex(shown_by_source.shape)]
        for target in every:
            shown_by_target = target.transformation()(sensor)
            mapped = reorient_pixels(pixels, raw_shape, source, target)
            for pixel, mapped_pixel in zip(pixels, mapped):
                assert shown_by_target[mapped_pixel] == shown_by_source[pixel]
            assert reorient_pixels(mapped, raw_shape, target, source) == pixels
    assert reorient_pixels([], raw_shape, every[0], every[1]) == []


@pytest.mark.parametrize(
    "orientation",
    [CameraOrientation(), CameraOrientation("180")],
    ids=["rot0", "rot180"],
)
def test_a_missing_frame_raises_timeout(orientation: CameraOrientation) -> None:
    """A driver returns None for a frame that did not arrive. The adapter raises
    ``TimeoutError`` once every attempt has returned None, and it never passes None to
    the transform.
    """
    raw = _raw_camera()
    camera = as_camera(raw)
    camera.set_orientation(orientation)
    raw.capture_attempts = 3
    raw.missing_frames = 10

    with pytest.raises(TimeoutError, match="no frame in 3 attempt"):
        camera.get_image()
    assert raw.capture_calls == 3


def test_a_late_frame_is_captured_again_with_a_warning() -> None:
    raw = _raw_camera()
    camera = as_camera(raw)
    raw.missing_frames = 2

    with pytest.warns(UserWarning, match="returned no frame 2 time"):
        frame = camera.get_image()

    np.testing.assert_array_equal(frame, raw._sensor)
    assert raw.capture_calls == 3


def test_each_capture_flushes_the_driver_before_its_first_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The driver's own ``flush`` drops the held frames. Every frame returned or summed
    by a call was therefore exposed after the call.
    """
    raw = _raw_camera()
    captures_at_flush: list[int] = []
    monkeypatch.setattr(
        raw, "flush", lambda timeout_s=1: captures_at_flush.append(raw.capture_calls)
    )
    camera = as_camera(raw)

    camera.get_image()
    camera.get_image(averaging=3)

    assert captures_at_flush == [0, 1]
    assert raw.capture_calls == 4


def test_a_frame_of_another_shape_raises() -> None:
    """A readout window or a binning set directly on the driver changes the frame
    under the adapter. The adapter crops the whole frame.
    """
    raw = _raw_camera()
    camera = as_camera(raw)
    raw.set_woi((0, 4, 0, 3))
    with pytest.raises(RuntimeError, match="readout window"):
        camera.get_image()

    class _BinningCamera(_RawCamera):
        def _get_image_hw(self, timeout_s=None):
            return super()._get_image_hw(timeout_s)[::2, ::2]

    binned = as_camera(
        _BinningCamera(resolution=(8, 5), bitdepth=8, pitch_um=(5.0, 7.0))
    )
    with pytest.raises(RuntimeError, match="Binning"):
        binned.get_image()


def test_a_camera_without_a_pixel_pitch_says_so() -> None:
    with pytest.raises(ValueError, match="pixel pitch"):
        pixel_size_from_pitch_um(None)
    with pytest.raises(ValueError, match="pixel pitch"):
        as_camera(_raw_camera(pitch_um=None)).pixel_size


def test_closing_the_adapter_closes_the_camera(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _raw_camera()
    closes: list[dict[str, object]] = []
    monkeypatch.setattr(raw, "close", lambda *args, **kwargs: closes.append(kwargs))

    as_camera(raw).close(close_sdk=True)

    assert closes == [{"close_sdk": True}]
