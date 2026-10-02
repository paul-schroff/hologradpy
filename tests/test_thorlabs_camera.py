"""ThorlabsCamera against a fake pylablib driver, so it runs without a camera."""

from __future__ import annotations

import sys
import types
from collections import namedtuple

import numpy as np
import pytest

from hologradpy.hardware.camera import CameraData, CameraOrientation
from hologradpy.roi import ROI

SENSOR_HEIGHT = 480
SENSOR_WIDTH = 640
BIT_DEPTH = 10
MAX_COUNT = 2**BIT_DEPTH - 1
# Not square, so a quarter turn that forgets to exchange the pitch shows.
PIXEL_SIZE = (3.0e-6, 4.0e-6)

TDeviceInfo = namedtuple(
    "TDeviceInfo", ["model", "name", "serial_number", "firmware_version"]
)
TSensorInfo = namedtuple("TSensorInfo", ["sensor_type", "bit_depth"])


def fake_frame(exposure_s):
    """A ramp that brightens with exposure and saturates, as a sensor does.

    Responding to exposure is what lets autoexpose be exercised against the driver.
    """
    ramp = np.linspace(0.0, 1.0, SENSOR_HEIGHT * SENSOR_WIDTH).reshape(
        SENSOR_HEIGHT, SENSOR_WIDTH
    )
    return np.clip(ramp * exposure_s * 1e6, 0, MAX_COUNT).astype(np.uint16)


class FakeTLCamera:
    """The pylablib ThorlabsTLCamera calls ThorlabsCamera makes, on a sensor that
    exposes one frame per software trigger while armed.
    """

    def __init__(self, serial=None):
        self.serial = serial
        self.exposure = 1e-4
        self.trigger_mode = "ext"
        self.roi_calls = []
        self.start_options = None
        self.armed = False
        self.arms = 0
        self.disarms = 0
        self.closes = 0
        self.triggers = 0
        self.timeouts = []
        self.pending = []

    def get_device_info(self):
        return TDeviceInfo("CS165MU", "Zelux", self.serial or "00001", "1.0")

    def set_roi(self, *args, **kwargs):
        self.roi_calls.append((args, kwargs))

    def set_trigger_mode(self, mode):
        self.trigger_mode = mode

    def get_detector_size(self):
        return (SENSOR_WIDTH, SENSOR_HEIGHT)  # pylablib states (width, height).

    def get_sensor_info(self):
        return TSensorInfo("mono", BIT_DEPTH)

    def start_acquisition(self, frames_per_trigger="default", auto_start=True):
        self.start_options = {
            "frames_per_trigger": frames_per_trigger,
            "auto_start": auto_start,
        }
        self.armed = True
        self.arms += 1

    def stop_acquisition(self):
        if self.armed:
            self.armed = False
            self.disarms += 1

    def get_exposure(self):
        return self.exposure

    def set_exposure(self, exposure):
        self.exposure = exposure
        return exposure

    def send_software_trigger(self):
        if not self.armed or self.trigger_mode != "int":
            raise RuntimeError("A software trigger needs the armed camera in 'int'.")
        self.triggers += 1
        self.pending.append(fake_frame(self.exposure))

    def read_multiple_images(self):
        frames, self.pending = self.pending, []
        return frames

    def wait_for_frame(self, timeout=20.0):
        self.timeouts.append(timeout)
        if not self.pending:
            raise TimeoutError("No frame arrived.")

    def read_oldest_image(self):
        return self.pending.pop(0)

    def close(self):
        self.closes += 1


@pytest.fixture
def fake_pylablib(monkeypatch):
    """Install a fake ``pylablib.devices`` so the driver's import finds the fake."""
    built = {}

    def open_camera(serial=None):
        device = FakeTLCamera(serial)
        built["device"] = device
        return device

    devices = types.ModuleType("pylablib.devices")
    devices.Thorlabs = types.SimpleNamespace(ThorlabsTLCamera=open_camera)
    package = types.ModuleType("pylablib")
    package.devices = devices

    monkeypatch.setitem(sys.modules, "pylablib", package)
    monkeypatch.setitem(sys.modules, "pylablib.devices", devices)
    return built


@pytest.fixture
def camera(fake_pylablib):
    from hologradpy.hardware.camera.thorlabs import ThorlabsCamera

    device = ThorlabsCamera(PIXEL_SIZE, serial="23702", exposure_bounds=(40e-6, 1.0))
    yield device, fake_pylablib["device"]
    device.close()


def test_opens_the_camera_and_arms_it_for_software_triggers(camera):
    _, fake = camera
    assert fake.serial == "23702"
    assert fake.roi_calls == [((), {})]  # The whole sensor.
    assert fake.trigger_mode == "int"
    assert fake.start_options == {"frames_per_trigger": 1, "auto_start": False}
    assert fake.arms == 1
    # No frame is exposed before the first capture asks for one.
    assert fake.triggers == 0


def test_names_the_camera_by_model_and_serial(camera):
    device, _ = camera
    assert device.name == "CS165MU 23702"
    assert CameraData.from_camera(device).name == "CS165MU 23702"


def test_capture_triggers_once_and_keeps_the_camera_armed(camera):
    device, fake = camera
    device.set_exposure(100e-6)
    frames = [device.get_image() for _ in range(3)]

    assert fake.triggers == 3
    # Armed once and never disarmed, which is what makes a frame cheap.
    assert fake.arms == 1
    assert fake.disarms == 0
    for frame in frames:
        np.testing.assert_array_equal(frame, fake_frame(100e-6))


def test_frames_left_from_earlier_triggers_are_dropped(camera):
    """A frame still waiting from before is never handed out as the new one."""
    device, fake = camera
    device.set_exposure(100e-6)
    fake.pending.append(np.full((SENSOR_HEIGHT, SENSOR_WIDTH), MAX_COUNT, np.uint16))

    frame = device.get_image()

    np.testing.assert_array_equal(frame, fake_frame(100e-6))
    assert fake.pending == []


def test_the_frame_wait_allows_for_the_exposure(camera):
    from hologradpy.hardware.camera.thorlabs import FRAME_TIMEOUT_S

    device, fake = camera
    device.set_exposure(0.5)
    device.get_image()
    assert fake.timeouts[-1] == pytest.approx(0.5 + FRAME_TIMEOUT_S)


def test_averaging_sums_frames(camera):
    device, fake = camera
    device.set_exposure(100e-6)
    single = device.get_image()
    summed = device.get_image(averaging=3)

    assert fake.triggers == 4
    assert summed.dtype == np.float64
    np.testing.assert_array_equal(summed, single.astype(np.float64) * 3)


def test_capture_sets_the_exposure_first(camera):
    device, fake = camera
    device.get_image(exposure=1e-3)
    assert fake.exposure == pytest.approx(1e-3)


def test_exposure_is_set_without_stopping_the_acquisition(camera):
    device, fake = camera
    device.set_exposure(2.5e-3)
    assert device.get_exposure() == pytest.approx(2.5e-3)
    assert fake.disarms == 0


def test_an_exposure_outside_the_bounds_is_clipped_with_a_warning(camera):
    device, fake = camera

    with pytest.warns(UserWarning, match="outside the camera's bounds"):
        device.set_exposure(20.0)
    assert fake.exposure == 1.0

    with pytest.warns(UserWarning, match="outside the camera's bounds"):
        device.set_exposure(1e-6)
    assert fake.exposure == 40e-6


def test_without_bounds_none_are_stated_or_applied(fake_pylablib):
    from hologradpy.hardware.camera.thorlabs import ThorlabsCamera

    device = ThorlabsCamera(PIXEL_SIZE)
    try:
        assert device.exposure_bounds is None
        device.set_exposure(5.0)
        assert fake_pylablib["device"].exposure == 5.0
    finally:
        device.close()


def test_max_pixel_value_follows_the_bit_depth(camera):
    device, _ = camera
    assert device.max_pixel_value == MAX_COUNT


def test_the_region_starts_as_the_whole_sensor(camera):
    device, _ = camera
    assert device.roi == ROI(0, 0, SENSOR_HEIGHT, SENSOR_WIDTH)
    assert device.resolution == (SENSOR_HEIGHT, SENSOR_WIDTH)
    assert device.sensor_resolution == (SENSOR_HEIGHT, SENSOR_WIDTH)
    np.testing.assert_array_equal(device.pixel_size, PIXEL_SIZE)


def test_the_region_of_interest_crops_the_frame(camera):
    device, _ = camera
    device.set_exposure(100e-6)
    device.set_roi(ROI(top_row=100, left_column=200, height=64, width=128))

    frame = device.get_image()

    assert device.resolution == (64, 128)
    np.testing.assert_array_equal(frame, fake_frame(100e-6)[100:164, 200:328])


def test_a_region_off_the_frame_is_rejected(camera):
    device, _ = camera
    with pytest.raises(ValueError, match="does not lie inside"):
        device.set_roi(ROI(400, 600, 100, 100))
    assert device.roi == ROI(0, 0, SENSOR_HEIGHT, SENSOR_WIDTH)


def test_orientation_reorients_frames_and_exchanges_the_geometry(camera):
    device, _ = camera
    orientation = CameraOrientation(rot="270", fliplr=True, flipud=False)
    device.set_exposure(100e-6)

    device.set_orientation(orientation)
    frame = device.get_image()

    assert device.orientation == orientation
    assert device.sensor_resolution == (SENSOR_WIDTH, SENSOR_HEIGHT)
    assert device.roi == ROI(0, 0, SENSOR_WIDTH, SENSOR_HEIGHT)
    np.testing.assert_array_equal(device.pixel_size, PIXEL_SIZE[::-1])
    np.testing.assert_array_equal(
        frame, orientation.transformation()(fake_frame(100e-6))
    )


def test_orientation_returns_the_region_to_the_whole_frame(camera):
    device, _ = camera
    device.set_roi(ROI(10, 20, 64, 128))
    device.set_orientation(CameraOrientation(rot="180"))
    assert device.roi == ROI(0, 0, SENSOR_HEIGHT, SENSOR_WIDTH)


def test_excluded_pixels_follow_the_sensor(camera):
    """An excluded pixel names the same sensor pixel after a remount."""
    device, _ = camera
    raw = np.zeros((SENSOR_HEIGHT, SENSOR_WIDTH))
    raw[10, 20] = 1.0
    device.excluded_pixels = [(10, 20)]

    orientation = CameraOrientation(rot="90", fliplr=True)
    device.set_orientation(orientation)

    (row, col), = device.excluded_pixels
    assert orientation.transformation()(raw)[row, col] == 1.0


def test_close_tears_down_once(fake_pylablib):
    from hologradpy.hardware.camera.thorlabs import ThorlabsCamera

    device = ThorlabsCamera(PIXEL_SIZE)
    fake = fake_pylablib["device"]

    device.close()
    device.close()

    assert fake.disarms == 1
    assert fake.closes == 1


def test_context_manager_closes(fake_pylablib):
    from hologradpy.hardware.camera.thorlabs import ThorlabsCamera

    with ThorlabsCamera(PIXEL_SIZE):
        pass
    assert fake_pylablib["device"].closes == 1


def test_without_pylablib_the_error_says_how_to_install(monkeypatch):
    from hologradpy.hardware.camera.thorlabs import ThorlabsCamera

    monkeypatch.setitem(sys.modules, "pylablib.devices", None)
    with pytest.raises(ImportError, match="pip install pylablib"):
        ThorlabsCamera(PIXEL_SIZE)


def test_autoexpose_runs_through_the_base_class(camera):
    """The inherited template methods work against the driver."""
    device, _ = camera
    device.autoexpose(set_fraction=0.5)
    low, high = device.exposure_search_bounds
    assert low <= device.get_exposure() <= high
