"""GenTLCamera against a fake GenTL producer, so every path runs without a camera."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from hologradpy.roi import ROI


SENSOR_HEIGHT = 480
SENSOR_WIDTH = 640


class FakeNode:
    """One GenICam feature node, with the bounds a real one carries."""

    def __init__(self, value, minimum=None, maximum=None, on_write=None):
        self._value = value
        self.min = minimum
        self.max = maximum
        self._on_write = on_write
        self.writes = []

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self.writes.append(new_value)
        if self._on_write is not None:
            new_value = self._on_write(new_value)
        self._value = new_value


class FakeCommand:
    """A GenICam command node, which is executed rather than written."""

    def __init__(self):
        self.executions = 0

    def execute(self):
        self.executions += 1


class FakeNodeMap:
    """The nodes GenTLCamera touches, and nothing else.

    Missing nodes are genuinely missing, so the driver's optional-feature handling is
    exercised the way a camera without them would.
    """

    def __init__(self):
        self.order = []

        self.ExposureAuto = FakeNode("On")
        self.GainAuto = FakeNode("On")
        self.BlackLevelAuto = FakeNode("On")
        # mvLowLight is deliberately absent: it is vendor-specific.

        self.PixelFormat = FakeNode("Mono8")
        self.PixelDynamicRangeMax = FakeNode(4095)
        self.Gain = FakeNode(0.0)

        self.TriggerSelector = FakeNode("AcquisitionStart")
        self.TriggerMode = FakeNode("Off")
        self.TriggerSource = FakeNode("Line0")
        self.TriggerSoftware = FakeCommand()

        self.AcquisitionMode = FakeNode("MultiFrame")
        self.AcquisitionFrameCount = FakeNode(1, minimum=1, maximum=65535)
        self.AcquisitionFrameRateEnable = FakeNode(True)
        self.AcquisitionFrameRate = FakeNode(12.0)

        # ExposureTime is in microseconds, as GenICam states it.
        self.ExposureTime = FakeNode(100.0, minimum=20.0, maximum=1e7)

        self.Width = FakeNode(
            SENSOR_WIDTH, maximum=SENSOR_WIDTH, on_write=self._record("Width")
        )
        self.Height = FakeNode(
            SENSOR_HEIGHT, maximum=SENSOR_HEIGHT, on_write=self._record("Height")
        )
        self.OffsetX = FakeNode(
            0, maximum=SENSOR_WIDTH, on_write=self._record("OffsetX")
        )
        self.OffsetY = FakeNode(
            0, maximum=SENSOR_HEIGHT, on_write=self._record("OffsetY")
        )
        self.WidthMax = FakeNode(SENSOR_WIDTH)
        self.HeightMax = FakeNode(SENSOR_HEIGHT)

    def _record(self, name):
        """Note the write, and enforce the bound a real camera enforces."""

        def hook(value):
            self.order.append((name, value))
            pending = {
                "Width": self.Width.value,
                "Height": self.Height.value,
                "OffsetX": self.OffsetX.value,
                "OffsetY": self.OffsetY.value,
            }
            pending[name] = value
            off_sensor = (
                pending["OffsetX"] + pending["Width"] > SENSOR_WIDTH
                or pending["OffsetY"] + pending["Height"] > SENSOR_HEIGHT
            )
            if off_sensor:
                raise ValueError(
                    f"{name}={value} would put the region off the sensor, which a "
                    "camera rejects."
                )
            return value

        return hook


MAX_COUNT = 4095


def fake_frame(height, width, exposure_us):
    """A ramp that brightens with exposure and saturates, as a sensor does.

    Responding to exposure is what lets autoexpose be exercised against the driver.
    """
    ramp = np.linspace(0.0, 1.0, height * width).reshape(height, width)
    counts = ramp * exposure_us
    return np.clip(counts, 0, MAX_COUNT).astype(np.uint16)


class FakeComponent:
    def __init__(self, height, width, exposure_us):
        self.height = height
        self.width = width
        self.data = fake_frame(height, width, exposure_us).reshape(-1)


class FakeBuffer:
    def __init__(self, height, width, exposure_us):
        self.payload = types.SimpleNamespace(
            components=[FakeComponent(height, width, exposure_us)]
        )
        self.returned = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.returned = True


class FakeStatistics:
    num_images = 7
    fps = 12.0


class FakeAcquirer:
    def __init__(self, node_map):
        self.remote_device = types.SimpleNamespace(node_map=node_map)
        self._node_map = node_map
        self.starts = 0
        self.stops = 0
        self.destroys = 0
        self.fetches = 0
        self.statistics = FakeStatistics()
        self.buffers = []

    def start(self):
        self.starts += 1

    def stop(self):
        self.stops += 1

    def destroy(self):
        self.destroys += 1

    def is_acquiring(self):
        return self.starts > self.stops

    def fetch(self):
        self.fetches += 1
        buffer = FakeBuffer(
            self._node_map.Height.value,
            self._node_map.Width.value,
            self._node_map.ExposureTime.value,
        )
        self.buffers.append(buffer)
        return buffer


class FakeHarvester:
    def __init__(self):
        self.node_map = FakeNodeMap()
        self.acquirer = FakeAcquirer(self.node_map)
        self.files = []
        self.updates = 0
        self.resets = 0
        self.device_info_list = [
            types.SimpleNamespace(serial_number="F0600075"),
            types.SimpleNamespace(serial_number="F0600086"),
        ]

    def add_file(self, path):
        self.files.append(path)

    def update(self):
        self.updates += 1

    def create(self, index):
        self.created = index
        return self.acquirer

    def reset(self):
        self.resets += 1


@pytest.fixture
def fake_harvesters(monkeypatch):
    """Install a fake ``harvesters.core`` so importing the driver's dependency works."""
    built = {}

    def make_harvester():
        harvester = FakeHarvester()
        built["harvester"] = harvester
        return harvester

    module = types.ModuleType("harvesters.core")
    module.Harvester = make_harvester
    package = types.ModuleType("harvesters")
    package.core = module

    monkeypatch.setitem(sys.modules, "harvesters", package)
    monkeypatch.setitem(sys.modules, "harvesters.core", module)
    return built


@pytest.fixture
def camera(fake_harvesters):
    from hologradpy.hardware.camera.gentl import GenTLCamera

    device = GenTLCamera(
        cti_path="fake.cti",
        pixel_size=(3.45e-6, 3.45e-6),
        pixel_format="Mono16",
    )
    yield device, fake_harvesters["harvester"]
    device.close()


def test_opens_the_producer_and_starts_streaming(camera):
    device, harvester = camera
    assert harvester.files == ["fake.cti"]
    assert harvester.updates == 1
    assert harvester.acquirer.starts == 1


def test_selects_the_camera_by_serial(fake_harvesters):
    from hologradpy.hardware.camera.gentl import GenTLCamera

    device = GenTLCamera("fake.cti", (3.45e-6, 3.45e-6), serial="F0600086")
    try:
        assert fake_harvesters["harvester"].created == 1
    finally:
        device.close()


def test_rejects_an_unknown_serial(fake_harvesters):
    from hologradpy.hardware.camera.gentl import GenTLCamera

    with pytest.raises(ValueError, match="No camera with serial"):
        GenTLCamera("fake.cti", (3.45e-6, 3.45e-6), serial="nope")


def test_opens_every_automatic_loop(camera):
    _, harvester = camera
    node_map = harvester.node_map
    assert node_map.ExposureAuto.value == "Off"
    assert node_map.GainAuto.value == "Off"
    assert node_map.BlackLevelAuto.value == "Off"


def test_configures_a_software_triggered_continuous_acquisition(camera):
    _, harvester = camera
    node_map = harvester.node_map
    assert node_map.AcquisitionMode.value == "Continuous"
    assert node_map.TriggerMode.value == "On"
    assert node_map.TriggerSource.value == "Software"
    assert node_map.TriggerSelector.value == "FrameStart"
    # A rate limit alongside a trigger throttles the trigger on some producers.
    assert node_map.AcquisitionFrameRateEnable.value is False
    # Raised to the ceiling, in case the producer stayed in MultiFrame.
    assert node_map.AcquisitionFrameCount.value == 65535


def test_exposure_round_trips_through_microseconds(camera):
    device, harvester = camera
    device.set_exposure(2.5e-3)
    assert harvester.node_map.ExposureTime.value == pytest.approx(2500.0)
    assert device.get_exposure() == pytest.approx(2.5e-3)


def test_exposure_bounds_are_seconds(camera):
    device, _ = camera
    low, high = device.exposure_bounds
    assert low == pytest.approx(20e-6)
    assert high == pytest.approx(10.0)


def test_max_pixel_value_follows_the_stated_dynamic_range(camera):
    device, harvester = camera
    # Mono16 on a 12-bit sensor still only fills 12 bits.
    assert harvester.node_map.PixelFormat.value == "Mono16"
    assert device.max_pixel_value == 4095


def test_max_pixel_value_falls_back_to_the_pixel_format(camera):
    device, harvester = camera
    del harvester.node_map.PixelDynamicRangeMax
    assert device.max_pixel_value == 65535


def test_region_of_interest_reads_back_the_geometry_nodes(camera):
    device, _ = camera
    assert device.roi == ROI(0, 0, SENSOR_HEIGHT, SENSOR_WIDTH)
    assert device.resolution == (SENSOR_HEIGHT, SENSOR_WIDTH)
    assert device.sensor_shape == (SENSOR_HEIGHT, SENSOR_WIDTH)


def test_setting_a_region_writes_offsets_last(camera):
    device, harvester = camera
    harvester.node_map.order.clear()

    device.set_roi(ROI(top_row=100, left_column=200, height=64, width=128))

    assert device.roi == ROI(100, 200, 64, 128)
    # Zeroed, then sized, then moved: an order the camera always accepts.
    assert harvester.node_map.order == [
        ("OffsetX", 0),
        ("OffsetY", 0),
        ("Width", 128),
        ("Height", 64),
        ("OffsetX", 200),
        ("OffsetY", 100),
    ]


def test_growing_a_region_from_a_far_corner_is_accepted(camera):
    """The order matters: sizing up before moving back would be rejected."""
    device, _ = camera
    device.set_roi(ROI(top_row=400, left_column=500, height=64, width=128))
    device.set_roi(
        ROI(top_row=0, left_column=0, height=SENSOR_HEIGHT, width=SENSOR_WIDTH)
    )
    assert device.roi == ROI(0, 0, SENSOR_HEIGHT, SENSOR_WIDTH)


def test_resetting_the_region_restores_the_whole_sensor(camera):
    device, _ = camera
    device.set_roi(ROI(10, 20, 64, 128))
    device.set_roi(None)
    assert device.roi == ROI(0, 0, SENSOR_HEIGHT, SENSOR_WIDTH)


def test_setting_a_region_cycles_acquisition(camera):
    device, harvester = camera
    before = harvester.acquirer.starts
    device.set_roi(ROI(10, 20, 64, 128))
    # The sizes are not writable while streaming.
    assert harvester.acquirer.stops == 1
    assert harvester.acquirer.starts == before + 1


def test_capture_triggers_once_and_returns_the_frame(camera):
    device, harvester = camera
    frame = device.get_image()

    assert harvester.node_map.TriggerSoftware.executions == 1
    assert frame.shape == (SENSOR_HEIGHT, SENSOR_WIDTH)
    # The buffer went back to the transport, so the frame has to be a copy.
    assert harvester.acquirer.buffers[0].returned


def test_capture_sets_the_exposure_first(camera):
    device, harvester = camera
    device.get_image(exposure_s=1e-3)
    assert harvester.node_map.ExposureTime.value == pytest.approx(1000.0)


def test_averaging_sums_frames(camera):
    device, harvester = camera
    device.set_exposure(100e-6)
    single = device.get_image()
    summed = device.get_image(averaging=3)

    assert harvester.node_map.TriggerSoftware.executions == 4
    np.testing.assert_array_equal(summed, single.astype(np.int64) * 3)


def test_no_restart_when_no_budget_is_set(camera):
    device, harvester = camera
    for _ in range(5):
        device.get_image()
    assert harvester.acquirer.stops == 0


def test_frame_budget_cycles_acquisition_without_rebuilding(fake_harvesters):
    from hologradpy.hardware.camera.gentl import GenTLCamera

    device = GenTLCamera(
        "fake.cti", (3.45e-6, 3.45e-6), frames_before_restart=2
    )
    harvester = fake_harvesters["harvester"]
    try:
        for _ in range(5):
            device.get_image()

        assert harvester.acquirer.stops == 2
        # The producer and the acquirer stay, which is what makes the restart cheap.
        assert harvester.acquirer.destroys == 0
        assert harvester.resets == 0
    finally:
        device.close()


def test_diagnostics_reads_the_nodes_back(camera):
    device, _ = camera
    device.get_image()
    report = device.diagnostics()

    assert report["AcquisitionMode"] == "Continuous"
    assert report["TriggerSource"] == "Software"
    assert report["is_acquiring"] is True
    assert report["frames_since_start"] == 1
    assert report["num_images"] == 7
    # Absent on this camera, and reported as such rather than raising.
    assert report["AcquisitionFrameRate"] == 12.0


def test_close_tears_down_once(fake_harvesters):
    from hologradpy.hardware.camera.gentl import GenTLCamera

    device = GenTLCamera("fake.cti", (3.45e-6, 3.45e-6))
    harvester = fake_harvesters["harvester"]

    device.close()
    device.close()

    assert harvester.acquirer.stops == 1
    assert harvester.acquirer.destroys == 1
    assert harvester.resets == 1


def test_context_manager_closes(fake_harvesters):
    from hologradpy.hardware.camera.gentl import GenTLCamera

    with GenTLCamera("fake.cti", (3.45e-6, 3.45e-6)):
        pass
    assert fake_harvesters["harvester"].acquirer.destroys == 1


def test_autoexpose_runs_through_the_base_class(camera):
    """The inherited template methods work against a real driver."""
    device, _ = camera
    device.autoexpose(set_fraction=0.5)
    low, high = device.exposure_limits
    assert low <= device.get_exposure() <= high
