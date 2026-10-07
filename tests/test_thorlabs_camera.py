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
# The smallest window the fake reads out, as a real sensor has one.
MIN_WINDOW_HEIGHT = 4
MIN_WINDOW_WIDTH = 8
BIT_DEPTH = 10
MAX_COUNT = 2**BIT_DEPTH - 1
# Every pixel holds its own index, so a crop or a turn that is off by a pixel shows.
NUMBERED = np.arange(SENSOR_HEIGHT * SENSOR_WIDTH, dtype=np.uint32).reshape(
    SENSOR_HEIGHT, SENSOR_WIDTH
)
# Not square, so a quarter turn that forgets to exchange the pitch shows.
PIXEL_SIZE = (3.0e-6, 4.0e-6)

TDeviceInfo = namedtuple(
    "TDeviceInfo", ["model", "name", "serial_number", "firmware_version"]
)
TSensorInfo = namedtuple("TSensorInfo", ["sensor_type", "bit_depth"])
# The gain range of the fake in dB, and its step, since the SDK sets the gain through an
# integer index.
GAIN_RANGE_DB = (0.0, 48.0)
GAIN_STEP_DB = 0.1


class FakeTLCameraError(RuntimeError):
    """What the pylablib driver raises for a request the camera refuses."""


def fake_frame(exposure_s):
    """A ramp that brightens with exposure and saturates, as a sensor does.

    Responding to exposure is what lets autoexpose be exercised against the driver. The
    fake camera passes its equivalent exposure, the exposure times the linear gain.
    """
    ramp = np.linspace(0.0, 1.0, SENSOR_HEIGHT * SENSOR_WIDTH).reshape(
        SENSOR_HEIGHT, SENSOR_WIDTH
    )
    return np.clip(ramp * exposure_s * 1e6, 0, MAX_COUNT).astype(np.uint16)


def snap(start, end, grid, maximum):
    """Round both edges of one axis of a window to the nearest multiple of ``grid``,
    as a Zelux does. This can cut off the first or the last row asked for.
    """
    end = maximum if end is None else end
    return tuple(grid * int(np.floor(edge / grid + 0.5)) for edge in (start, end))


def truncate(start, end, minimum, maximum):
    """The truncation pylablib applies to one axis of a window, for a position step of
    one: clip to the sensor, then grow to the minimum size.
    """
    start, end = max(0, start), min(maximum if end is None else end, maximum)
    if end - start < minimum:
        end = start + minimum
    if end > maximum:
        start, end = maximum - minimum, maximum
    return start, end


class FakeTLCamera:
    """The pylablib ThorlabsTLCamera calls ThorlabsCamera makes, on a sensor that
    exposes one frame per software trigger while armed and reads out its window.

    The gain is held as an index of ``GAIN_STEP_DB`` steps and starts at 3 dB, so
    opening at another gain shows. A ``gain_range`` of None makes a camera without
    gain, whose range request raises the driver's error.
    """

    Error = FakeTLCameraError
    gain_range = GAIN_RANGE_DB

    def __init__(self, serial=None):
        self.serial = serial
        self.exposure = 1e-4
        self.gain_index = 30
        self.gain_requests = []
        self.trigger_mode = "ext"
        self.roi_calls = []
        # (hstart, hend, vstart, vend), as pylablib states a window.
        self.window = (0, SENSOR_WIDTH, 0, SENSOR_HEIGHT)
        self.start_options = None
        self.armed = False
        self.arms = 0
        self.disarms = 0
        self.closes = 0
        self.triggers = 0
        self.timeouts = []
        self.pending = []
        self.pattern = None  # What the sensor sees, or None for the exposure ramp.
        self.grid = 1  # The window's edges are rounded to multiples of this.

    def get_device_info(self):
        return TDeviceInfo("CS165MU", "Zelux", self.serial or "00001", "1.0")

    def set_roi(self, hstart=0, hend=None, vstart=0, vend=None):
        """Takes the window, and as pylablib does, re-arms an armed camera with a bare
        start_acquisition, which free-runs.
        """
        self.roi_calls.append((hstart, hend, vstart, vend))
        was_armed = self.armed
        self.stop_acquisition()
        hstart, hend = snap(hstart, hend, self.grid, SENSOR_WIDTH)
        vstart, vend = snap(vstart, vend, self.grid, SENSOR_HEIGHT)
        hstart, hend = truncate(hstart, hend, MIN_WINDOW_WIDTH, SENSOR_WIDTH)
        vstart, vend = truncate(vstart, vend, MIN_WINDOW_HEIGHT, SENSOR_HEIGHT)
        self.window = (hstart, hend, vstart, vend)
        if was_armed:
            self.start_acquisition()
        return (*self.window, 1, 1)

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

    def get_gain_range(self):
        if self.gain_range is None:
            raise FakeTLCameraError("This camera has no gain.")
        return self.gain_range

    def get_gain(self):
        return self.gain_index * GAIN_STEP_DB

    def set_gain(self, gain, truncate=True):
        """Takes the gain in dB and applies the nearest index, as pylablib does."""
        self.gain_requests.append(gain)
        index = int(round(gain / GAIN_STEP_DB))
        if truncate:
            low, high = (int(round(limit / GAIN_STEP_DB)) for limit in self.gain_range)
            index = max(low, min(index, high))
        self.gain_index = index
        return self.get_gain()

    def send_software_trigger(self):
        if not self.armed or self.trigger_mode != "int":
            raise RuntimeError("A software trigger needs the armed camera in 'int'.")
        self.triggers += 1
        equivalent_exposure = self.exposure * 10.0 ** (self.get_gain() / 20.0)
        whole = (
            fake_frame(equivalent_exposure) if self.pattern is None else self.pattern
        )
        hstart, hend, vstart, vend = self.window
        self.pending.append(whole[vstart:vend, hstart:hend])

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
    assert fake.roi_calls == [(0, None, 0, None)]  # The whole sensor.
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
    # Armed once and never disarmed, so a frame is cheap.
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


def test_only_the_region_of_interest_is_read_out(camera):
    device, fake = camera
    fake.pattern = NUMBERED
    device.set_roi(ROI(top_row=100, left_column=200, height=64, width=128))

    frame = device.get_image()

    assert fake.window == (200, 328, 100, 164)  # (hstart, hend, vstart, vend)
    assert device.resolution == (64, 128)
    np.testing.assert_array_equal(frame, NUMBERED[100:164, 200:328])


def test_setting_a_region_rearms_for_one_frame_per_trigger(camera):
    """An armed camera is re-armed free-running by pylablib, so the driver disarms
    first.
    """
    device, fake = camera
    device.set_roi(ROI(10, 20, 64, 128))

    assert fake.disarms == 1
    assert fake.arms == 2
    assert fake.armed
    assert fake.start_options == {"frames_per_trigger": 1, "auto_start": False}


def test_an_unchanged_region_is_not_read_out_again(camera):
    device, fake = camera
    device.set_roi(ROI(10, 20, 64, 128))
    device.set_roi(ROI(10, 20, 64, 128))
    device.set_roi(None)
    device.set_roi(None)

    assert len(fake.roi_calls) == 3  # Opening, the region, and the whole sensor.
    assert fake.arms == 3


def test_a_window_grown_to_the_minimum_size_is_cropped_to_the_region(camera):
    device, fake = camera
    fake.pattern = NUMBERED
    device.set_roi(ROI(top_row=5, left_column=6, height=2, width=3))

    frame = device.get_image()

    assert fake.window == (6, 6 + MIN_WINDOW_WIDTH, 5, 5 + MIN_WINDOW_HEIGHT)
    np.testing.assert_array_equal(frame, NUMBERED[5:7, 6:9])


def test_a_window_grown_at_the_sensor_edge_still_holds_the_region(camera):
    """Growing past the edge moves the window back onto the sensor."""
    device, fake = camera
    fake.pattern = NUMBERED
    region = ROI(SENSOR_HEIGHT - 1, SENSOR_WIDTH - 2, 1, 2)
    device.set_roi(region)

    frame = device.get_image()

    assert fake.window == (
        SENSOR_WIDTH - MIN_WINDOW_WIDTH,
        SENSOR_WIDTH,
        SENSOR_HEIGHT - MIN_WINDOW_HEIGHT,
        SENSOR_HEIGHT,
    )
    np.testing.assert_array_equal(frame, region.crop(NUMBERED))


def test_a_window_moved_onto_the_camera_grid_is_widened_where_it_cuts_off(camera):
    """Asked for rows from 35, the camera starts the window at 36. The top is asked
    for again further out until the window holds row 35, and only the top.
    """
    device, fake = camera
    fake.pattern = NUMBERED
    fake.grid = 4
    region = ROI(top_row=35, left_column=33, height=23, width=25)
    device.set_roi(region)

    frame = device.get_image()

    assert fake.window == (32, 60, 32, 60)  # (hstart, hend, vstart, vend)
    np.testing.assert_array_equal(frame, region.crop(NUMBERED))


@pytest.mark.parametrize("orientation", CameraOrientation.dihedral(), ids=repr)
def test_a_window_moved_onto_the_camera_grid_in_every_orientation(camera, orientation):
    device, fake = camera
    fake.pattern = NUMBERED
    fake.grid = 4
    device.set_orientation(orientation)
    region = ROI(top_row=37, left_column=53, height=61, width=97)
    device.set_roi(region)

    np.testing.assert_array_equal(
        device.get_image(), region.crop(orientation.transformation()(NUMBERED))
    )


def test_a_window_that_does_not_hold_the_region_is_an_error(camera):
    device, fake = camera
    fake.set_roi = lambda *_: (0, 8, 0, 4, 1, 1)  # A camera that ignores the request.
    with pytest.raises(RuntimeError, match="does not hold"):
        device.set_roi(ROI(100, 200, 64, 128))
    # Armed again for triggers, whatever the window.
    assert fake.start_options == {"frames_per_trigger": 1, "auto_start": False}


@pytest.mark.parametrize("orientation", CameraOrientation.dihedral(), ids=repr)
def test_a_region_reads_out_only_its_sensor_pixels_in_every_orientation(
    camera, orientation
):
    """The frame is the region of the reoriented whole frame, and only the sensor
    pixels under it are read out.
    """
    device, fake = camera
    fake.pattern = NUMBERED
    device.set_orientation(orientation)
    region = ROI(top_row=37, left_column=53, height=61, width=97)
    device.set_roi(region)

    frame = device.get_image()

    np.testing.assert_array_equal(
        frame, region.crop(orientation.transformation()(NUMBERED))
    )
    hstart, hend, vstart, vend = fake.window
    assert (hend - hstart) * (vend - vstart) == region.height * region.width


@pytest.mark.parametrize("orientation", CameraOrientation.dihedral(), ids=repr)
def test_the_whole_frame_in_every_orientation(camera, orientation):
    device, fake = camera
    fake.pattern = NUMBERED
    device.set_orientation(orientation)

    np.testing.assert_array_equal(
        device.get_image(), orientation.transformation()(NUMBERED)
    )
    assert fake.window == (0, SENSOR_WIDTH, 0, SENSOR_HEIGHT)


def test_a_region_off_the_frame_is_rejected(camera):
    device, fake = camera
    with pytest.raises(ValueError, match="does not lie inside"):
        device.set_roi(ROI(400, 600, 100, 100))
    assert device.roi == ROI(0, 0, SENSOR_HEIGHT, SENSOR_WIDTH)
    assert fake.window == (0, SENSOR_WIDTH, 0, SENSOR_HEIGHT)


def test_orientation_reorients_frames_and_exchanges_the_geometry(camera):
    device, fake = camera
    fake.pattern = NUMBERED
    orientation = CameraOrientation(rot="270", fliplr=True, flipud=False)

    device.set_orientation(orientation)
    frame = device.get_image()

    assert device.orientation == orientation
    assert device.sensor_resolution == (SENSOR_WIDTH, SENSOR_HEIGHT)
    assert device.roi == ROI(0, 0, SENSOR_WIDTH, SENSOR_HEIGHT)
    np.testing.assert_array_equal(device.pixel_size, PIXEL_SIZE[::-1])
    np.testing.assert_array_equal(frame, orientation.transformation()(NUMBERED))


def test_orientation_returns_the_region_and_the_readout_to_the_whole_frame(camera):
    device, fake = camera
    device.set_roi(ROI(10, 20, 64, 128))
    device.set_orientation(CameraOrientation(rot="180"))
    assert device.roi == ROI(0, 0, SENSOR_HEIGHT, SENSOR_WIDTH)
    assert fake.window == (0, SENSOR_WIDTH, 0, SENSOR_HEIGHT)


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


def test_opens_at_zero_db_and_reads_the_gain_range(camera):
    device, fake = camera
    assert device.gain_bounds == GAIN_RANGE_DB
    assert fake.get_gain() == 0.0
    assert device.get_gain() == 0.0


def test_opens_at_the_gain_asked_for(fake_pylablib):
    from hologradpy.hardware.camera.thorlabs import ThorlabsCamera

    device = ThorlabsCamera(PIXEL_SIZE, gain=6.0)
    try:
        assert device.get_gain() == pytest.approx(6.0)
        assert CameraData.from_camera(device).gain == pytest.approx(6.0)
    finally:
        device.close()


def test_a_gain_is_applied_in_the_steps_of_the_camera(camera):
    device, fake = camera
    device.set_gain(6.04)
    assert fake.gain_requests[-1] == 6.04
    assert device.get_gain() == pytest.approx(6.0)
    assert fake.disarms == 0


def test_a_gain_outside_the_range_is_clipped_with_a_warning(camera):
    device, _ = camera

    with pytest.warns(UserWarning, match="outside the camera's gain range"):
        device.set_gain(60.0)
    assert device.get_gain() == pytest.approx(48.0)

    with pytest.warns(UserWarning, match="outside the camera's gain range"):
        device.set_gain(-3.0)
    assert device.get_gain() == 0.0


def test_a_camera_without_a_gain_range_keeps_its_gain(fake_pylablib, monkeypatch):
    from hologradpy.hardware.camera.thorlabs import ThorlabsCamera

    monkeypatch.setattr(FakeTLCamera, "gain_range", None)
    device = ThorlabsCamera(PIXEL_SIZE)
    try:
        assert device.gain_bounds is None
        assert device.get_gain() == 0.0
        with pytest.raises(NotImplementedError, match="no gain range"):
            device.set_gain(6.0)
        assert fake_pylablib["device"].gain_requests == []
    finally:
        device.close()


def test_autoexpose_raises_the_gain_through_the_driver(camera):
    """With the exposure held at 200 us, the scene needs a factor 2.56 of gain, about
    8.2 dB, to reach its target of 511 counts.
    """
    device, _ = camera

    exposure = device.autoexpose(set_fraction=0.5, exposure_bounds=(40e-6, 200e-6))

    assert exposure == pytest.approx(200e-6, rel=0.06)
    assert device.get_gain() == pytest.approx(20.0 * np.log10(2.56), abs=0.5)
    peak = float(device.get_image().max())
    assert abs(peak - 0.5 * MAX_COUNT) <= 0.05 * MAX_COUNT
