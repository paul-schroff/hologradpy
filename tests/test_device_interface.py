"""Tests for the HoloGradPy-native device interface (protocol.py / adapter.py).

Covers the ROI value object, the slmsuite<->HoloGradPy conversion helpers, the native
properties the simulated devices expose, and that the real-hardware adapter reports
identical native values for the same underlying device. Uses a non-square camera so
an axis swap in any conversion would show.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from hologradpy.hardware import SimulatedSLMTorch, SimulatedCameraTorch  # noqa: E402
from hologradpy.hardware import Camera, SLM  # noqa: E402
from hologradpy.hardware.slmsuite.conversions import (  # noqa: E402
    pixel_size_from_pitch_um,
    pitch_um_from_pixel_size,
    wavelength_from_wav_um,
    wav_um_from_wavelength,
    roi_from_woi,
    roi_to_woi,
)
from hologradpy.roi import ROI  # noqa: E402
from hologradpy.hardware import (  # noqa: E402
    SLMSuiteCameraAdapter,
    SLMSuiteSLMAdapter,
    as_camera,
    as_slm,
    open_camera,
    open_slm,
    register_camera_backend,
    register_slm_backend,
)
from slmsuite.hardware.cameras.camera import Camera as SLMSuiteCamera  # noqa: E402
from slmsuite.hardware.slms.slm import SLM as SLMSuiteSLM  # noqa: E402
from hologradpy.propagation.complex_amplitude import (  # noqa: E402
    ComplexAmplitude,
    FieldGeometry,
)
from hologradpy.propagation.optical_systems import SLMCZT  # noqa: E402
from hologradpy.propagation.diagonal_elements import StaticSLMField  # noqa: E402
from hologradpy.propagation.amplitude_profiles import (  # noqa: E402
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
    image = np.zeros((10, 12))
    image[3:6, 4:9] = 1.0
    roi = ROI.detect(image, threshold=0.5, pad=0)
    # detect mirrors the old find_roi bounds (inclusive max index).
    assert roi.to_bounds() == (3, 5, 4, 8)


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
        static_slm_field=StaticSLMField(beam),
    )
    camera = SimulatedCameraTorch(model, bitdepth=8)
    return slm, camera


def test_camera_native_properties():
    """The simulated camera implements the native Camera interface directly (no
    slmsuite base), so as_camera passes it through unchanged."""
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


# --- auto-wrap: as_camera / as_slm --------------------------------------


class _RawCamera(SLMSuiteCamera):
    """A minimal slmsuite camera with no native HoloGradPy properties (stands in for
    real hardware, which is only reachable through the adapter)."""

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
    vendor SDK is imported until one of these backends is actually opened."""
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
    def adu_levels(self):
        return self._adu

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
    settles on the closest achievable exposure instead of oscillating to the timeout."""
    camera = _QuantizedCamera()  # peaks: 55 (1x), 110 (2x), 165 (3x). Target 128
    exposure = camera.autoexpose(set_fraction=0.5, tolerance=0.05, timeout=5.0)

    # 2 * step gives peak 110 (closest to 128). 3 * step overshoots to 165.
    assert exposure == pytest.approx(2e-3)
    assert camera.get_exposure() == pytest.approx(2e-3)
    # The guard stops after a couple of steps. Without it the loop would spin until the
    # timeout, calling set_exposure hundreds of times.
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
        self._blob = np.exp(-(xx**2 + yy**2) / (2 * 5.0**2))  # broad, peak 1 at centre

    @property
    def pixel_size(self):
        return np.array([1e-6, 1e-6])

    @property
    def resolution(self):
        return (32, 32)

    @property
    def adu_levels(self):
        return self._adu

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
    via Camera.excluded_pixels lets autoexpose target the real blob near the target."""
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
    to another (no shared mutable default)."""
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
    detector rather than assuming exactly constant frames."""

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
    def adu_levels(self):
        return self._adu

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
    low in the dark surround is not, being indistinguishable from the background."""
    camera = _SceneCamera()
    found = set(camera.find_stuck_pixels())
    assert found == _DISK_STUCK | {_DARK_STUCK_HIGH}
    assert _DARK_STUCK_LOW not in found


def test_find_stuck_pixels_warns_on_overexposed_blob():
    """A disk saturated across the whole sweep is the camera overexposed, not hot
    pixels, so it warns and the saturated disk background is not excluded."""
    camera = _SceneCamera(saturating=True)
    with pytest.warns(UserWarning, match="overexposed"):
        found = set(camera.find_stuck_pixels())
    assert (10, 10) not in found  # a plain saturated disk pixel is overexposure


class _AutoDetectCamera(Camera):
    """A uniform field that saturates at the initial exposure, so autoexpose sweeps it
    down toward the target, capturing a wide exposure range, plus one dead pixel to find
    from those frames."""

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
    def adu_levels(self):
        return self._adu

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
    while converging, populating excluded_pixels in the same call (no second sweep)."""
    camera = _AutoDetectCamera()
    camera.autoexpose(set_fraction=0.5, tolerance=0.05, detect_stuck_pixels=True)
    assert camera.excluded_pixels == [(8, 8)]


def test_capture_exposure_sweep_drops_out_of_bounds_exposures():
    """Exposures above the upper bound are dropped, not clipped to it, so the sweep
    keeps its spacing and still detects the stuck pixels from the in-bounds ones."""
    camera = _SceneCamera()  # bounds (1e-4, 1.0)
    frames, exposures = camera._capture_exposure_sweep(
        exposures=[1e-4, 1e-1, 5.0, 10.0]
    )
    assert exposures == [1e-4, 1e-1]  # the two out-of-bounds values are dropped
    found = set(camera._detect_stuck_pixels(frames, exposures))
    assert found == _DISK_STUCK | {_DARK_STUCK_HIGH}


def test_find_stuck_pixels_needs_two_in_bounds_exposures():
    """Fewer than two exposures within the bounds cannot reveal a response, so it raises
    rather than guessing."""
    camera = _SceneCamera()
    with pytest.raises(ValueError, match="at least two exposures"):
        camera.find_stuck_pixels(exposures=[1e-3, 5.0])
