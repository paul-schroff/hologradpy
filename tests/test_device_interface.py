"""Tests for the HoloGradPy-native device interface (protocol.py / adapter.py).

Covers the ROI value object, the slmsuite<->HoloGradPy conversion helpers, the native
properties the simulated devices expose, and that the real-hardware adapter reports
identical native values for the same underlying device. Uses a non-square camera so
an axis swap in any conversion would show.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hologradpy.hardware import SimulatedSLMTorch, SimulatedCameraTorch
from hologradpy.hardware import Camera, CameraOrientation, SLM
from hologradpy.hardware.slmsuite.conversions import (
    pixel_size_from_pitch_um,
    pitch_um_from_pixel_size,
    wavelength_from_wav_um,
    wav_um_from_wavelength,
    roi_from_woi,
    roi_to_woi,
)
from hologradpy.phase_levels import (
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
from hologradpy.profiles.amplitude import (
    gaussian_beam_intensity,
)

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


def _slm_at(bitdepth: int, wav_design_um: float | None = None) -> SLM:
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


def test_a_device_without_a_bitdepth_says_so() -> None:
    class Bare(SLM):
        pixel_size = np.array([SLM_PITCH, SLM_PITCH])
        resolution = (4, 4)
        wavelength = WAVELENGTH

        def set_phase(self, phase) -> None:
            pass

    device = Bare()
    assert device.bitdepth is None
    with pytest.raises(ValueError, match="no bitdepth"):
        device.phase_to_levels(np.zeros((4, 4)))


# --- auto-wrap: as_camera / as_slm --------------------------------------


class _RawCamera(SLMSuiteCamera):
    """A minimal slmsuite camera with no native HoloGradPy properties (stands in for
    real hardware, which is only reachable through the adapter).
    """

    def __init__(self, resolution, bitdepth, pitch_um, name="raw"):
        super().__init__(
            resolution=resolution, bitdepth=bitdepth, pitch_um=pitch_um, name=name
        )

    def _get_image_hw(self, timeout_s=None):
        return np.zeros(self.shape)

    def _get_exposure_hw(self):
        return 1.0

    def _set_exposure_hw(self, exposure_s):
        pass

    def set_woi(self, woi=None):
        self.woi = woi

    def close(self):
        pass


class _RawSlm(SLMSuiteSLM):
    """A minimal slmsuite SLM with no native HoloGradPy properties."""

    def __init__(self, resolution, bitdepth, wav_um, pitch_um, name="raw"):
        super().__init__(
            resolution=resolution,
            bitdepth=bitdepth,
            wav_um=wav_um,
            pitch_um=pitch_um,
            name=name,
        )

    def _set_phase_hw(self, phase):
        pass

    def close(self):
        pass


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
    # ROI round-trips through the slmsuite woi (corner) convention on real hardware.
    roi = ROI(3, 5, 4, 8)
    wrapped_raw.set_roi(roi)
    assert raw.woi == roi_to_woi(roi)
    assert wrapped_raw.roi == roi
    # Non-native attributes / methods delegate to the wrapped slmsuite device.
    assert wrapped_raw.bitdepth == raw.bitdepth


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


class _QuantizedCamera(Camera):
    """A native camera whose exposure snaps to a coarse grid, to exercise the discrete
    step guard in ``autoexpose``. The response is linear (``peak = gain * exposure``),
    saturating at the top of the range. The grid is chosen so no achievable exposure
    lands the peak inside the 50 percent tolerance band, forcing an oscillation between
    two neighbouring steps that the guard must stop.
    """

    def __init__(self, step: float = 1e-3, gain: float = 55e3, adu_levels: int = 256):
        self._step = step
        self._gain = gain
        self._adu = adu_levels
        self._exposure = step
        self._roi = ROI(0, 0, 4, 4)
        self.set_exposure_calls = 0

    @property
    def pixel_size(self):
        return np.array([1e-6, 1e-6])

    @property
    def resolution(self):
        return (4, 4)

    @property
    def max_pixel_value(self):
        return self._adu - 1

    @property
    def exposure_bounds(self):
        return (self._step, 100 * self._step)

    @property
    def roi(self):
        return self._roi

    def set_roi(self, roi):
        self._roi = ROI(0, 0, 4, 4) if roi is None else roi

    def get_exposure(self):
        return self._exposure

    def set_exposure(self, exposure_s):
        self.set_exposure_calls += 1
        lo, hi = self.exposure_bounds
        snapped = round(exposure_s / self._step) * self._step
        self._exposure = float(min(max(snapped, lo), hi))

    def get_image(self, exposure_s=None, averaging=1):
        if exposure_s is not None:
            self.set_exposure(exposure_s)
        peak = min(self._adu - 1, round(self._gain * self._exposure))
        return np.full(self.resolution, peak, dtype=float)


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


class _HotPixelCamera(Camera):
    """A camera imaging a broad blob that peaks well below saturation, plus one stuck
    pixel pinned at the top of the range. The stuck pixel makes the raw peak read as
    permanent saturation, so autoexpose only reaches the real signal if it ignores it.
    """

    def __init__(self, adu_levels: int = 256):
        self._adu = adu_levels
        self._exposure = 1e-3
        self._roi = ROI(0, 0, 32, 32)
        yy, xx = np.mgrid[0:32, 0:32] - 16
        self._blob = np.exp(-(xx**2 + yy**2) / (2 * 5.0**2))  # broad, peak 1 at center

    @property
    def pixel_size(self):
        return np.array([1e-6, 1e-6])

    @property
    def resolution(self):
        return (32, 32)

    @property
    def max_pixel_value(self):
        return self._adu - 1

    @property
    def exposure_bounds(self):
        return (1e-4, 1.0)

    @property
    def roi(self):
        return self._roi

    def set_roi(self, roi):
        self._roi = ROI(0, 0, 32, 32) if roi is None else roi

    def get_exposure(self):
        return self._exposure

    def set_exposure(self, exposure_s):
        lo, hi = self.exposure_bounds
        self._exposure = float(min(max(exposure_s, lo), hi))

    def get_image(self, exposure_s=None, averaging=1):
        if exposure_s is not None:
            self.set_exposure(exposure_s)
        # Blob peaks at 0.4 of full scale at the initial 1e-3 s exposure.
        gain = 0.4 * (self._adu - 1) / 1e-3
        frame = np.clip(np.round(self._blob * gain * self._exposure), 0, self._adu - 1)
        frame = frame.astype(float)
        frame[0, 0] = self._adu - 1  # lone stuck pixel at saturation
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


class _SceneCamera(Camera):
    """A central illuminated disk on a dark surround, with additive read noise on every
    frame and several pixels stuck at fixed values (see the module-level constants).

    The working pixels vary from frame to frame, so this exercises the noise-tolerant
    detector rather than assuming exactly constant frames.
    """

    def __init__(self, adu_levels=256, saturating=False, seed=0):
        self._adu = adu_levels
        self._exposure = 1e-3
        self._roi = ROI(0, 0, 32, 32)
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

    @property
    def pixel_size(self):
        return np.array([1e-6, 1e-6])

    @property
    def resolution(self):
        return (32, 32)

    @property
    def max_pixel_value(self):
        return self._adu - 1

    @property
    def exposure_bounds(self):
        return (1e-4, 1.0)

    @property
    def roi(self):
        return self._roi

    def set_roi(self, roi):
        self._roi = ROI(0, 0, 32, 32) if roi is None else roi

    def get_exposure(self):
        return self._exposure

    def set_exposure(self, exposure_s):
        lo, hi = self.exposure_bounds
        self._exposure = float(min(max(exposure_s, lo), hi))

    def get_image(self, exposure_s=None, averaging=1):
        if exposure_s is not None:
            self.set_exposure(exposure_s)
        if self._saturating:
            signal = 2.0 * self._adu  # well above the ceiling, so it clamps to max
        else:
            signal = 2e5 * self._exposure  # unsaturated when dim, saturates when bright
        field = np.where(self._disk, signal, 0.0)
        frame = np.round(
            np.clip(field + self._noise.normal(0, 4.0, field.shape), 0, self._adu - 1)
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


class _AutoDetectCamera(Camera):
    """A uniform field that saturates at the initial exposure, so autoexpose sweeps it
    down toward the target, capturing a wide exposure range, plus one dead pixel to find
    from those frames.
    """

    def __init__(self, adu_levels=256, seed=0):
        self._adu = adu_levels
        self._exposure = 1e-3  # starts saturated, autoexpose sweeps it down
        self._roi = ROI(0, 0, 16, 16)
        self._noise = np.random.default_rng(seed)

    @property
    def pixel_size(self):
        return np.array([1e-6, 1e-6])

    @property
    def resolution(self):
        return (16, 16)

    @property
    def max_pixel_value(self):
        return self._adu - 1

    @property
    def exposure_bounds(self):
        return (1e-5, 1.0)

    @property
    def roi(self):
        return self._roi

    def set_roi(self, roi):
        self._roi = ROI(0, 0, 16, 16) if roi is None else roi

    def get_exposure(self):
        return self._exposure

    def set_exposure(self, exposure_s):
        lo, hi = self.exposure_bounds
        self._exposure = float(min(max(exposure_s, lo), hi))

    def get_image(self, exposure_s=None, averaging=1):
        if exposure_s is not None:
            self.set_exposure(exposure_s)
        field = 5e5 * self._exposure  # saturates at 1e-3, unsaturated when swept down
        noise = self._noise.normal(0, 3.0, self.resolution)
        frame = np.round(np.clip(field + noise, 0, self._adu - 1))
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
    assert camera.default_shape == (320, 240)
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

    assert recorded.sensor_shape == (240, 320)
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


def test_a_camera_that_cannot_be_reoriented_says_so():
    class _Fixed(Camera):
        pixel_size = np.array([1.0, 1.0])
        resolution = (4, 4)
        max_pixel_value = 255
        exposure_bounds = None
        roi = ROI(0, 0, 4, 4)

        def set_roi(self, roi): ...
        def get_exposure(self): return 0.0
        def set_exposure(self, exposure_s): ...
        def get_image(self, exposure_s=None, averaging=1):
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


def test_a_vendor_correction_moves_the_level_it_says() -> None:
    """Vendors ship gray levels, calibrated against their own curve, so it is added
    after the conversion. Converting it through our response first would re-interpret
    their numbers, and under a measured curve would land somewhere else entirely.
    """
    slm = _slm_at(8)
    slm.virtual_slm.phase_response = PhaseResponseModule(_s_curve())
    phase = np.full(slm.resolution, -2.0)

    slm.set_phase(phase)
    plain = int(slm.display[0, 0])
    slm.load_vendor_correction(np.full(slm.resolution, 7, dtype=np.uint8))
    slm.set_phase(phase, apply_vendor_correction=True)

    assert int(slm.display[0, 0]) == (plain + 7) % 256


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
    from hologradpy.hardware.slm.abstract import SLMData

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
