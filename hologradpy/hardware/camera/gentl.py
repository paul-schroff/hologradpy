"""A GenTL / GenICam camera, driven through ``harvesters``.

Pointed at a different producer ``.cti`` it drives any compliant camera. Developed 
for a MATRIX VISION BlueFOX3.
"""

from __future__ import annotations

import re
import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ...roi import ROI
from .abstract import Camera

# GenICam states ExposureTime in microseconds. This package states it in seconds.
MICROSECONDS_PER_SECOND = 1e6

# What a pixel format implies about the count a pixel can report, when the camera does
# not state PixelDynamicRangeMax.
PIXEL_FORMAT_BITS = re.compile(r"(\d+)")


class GenTLCamera(Camera):
    """A camera reached over GenTL, through a vendor's producer library.

    Every frame is taken with a software trigger, so a capture is one exposure the
    caller asked for rather than whatever the free-running sensor last produced.

    Geometry, dynamic range and exposure bounds are read from the node map, so nothing
    here is specific to one sensor. The pixel pitch is the exception, since GenICam has
    no standard node for it.
    """

    def __init__(
        self,
        cti_path: str,
        pixel_size: tuple[float, float],
        serial: str | None = None,
        pixel_format: str = "Mono16",
        roi: ROI | None = None,
        gain: float = 0.0,
        frames_before_restart: int | None = None,
    ) -> None:
        """Open the camera and start acquisition.

        Args:
            cti_path: The vendor's GenTL producer, the ``.cti`` that speaks to the
                transport.
            pixel_size: Pixel pitch ``(y, x)`` in metres, from the datasheet.
            serial: Which camera to open, when several are attached. Defaults to the
                first one the producer reports.
            pixel_format: The format to stream in, for example ``"Mono16"``.
            roi: The region to read out. Defaults to the whole sensor.
            gain: Analog gain, in the units the camera states it in.
            frames_before_restart: Cycle acquisition after this many frames. A
                workaround for a producer that stops streaming after a fixed count.
                Defaults to None, which never restarts.

        Raises:
            ImportError: If ``harvesters`` is not installed.
            ValueError: If no camera matches ``serial``.
        """
        try:
            from harvesters.core import Harvester
        except ImportError as error:
            raise ImportError(
                "GenTLCamera needs the 'harvesters' package, which is not installed. "
                "Install it with 'pip install harvesters'."
            ) from error

        self._pixel_size = np.asarray(pixel_size, dtype=np.float64)
        self.frames_before_restart: int | None = frames_before_restart
        self.frames_since_start: int = 0
        # The region the caller asked for, and the window on the camera's increment
        # grid that is read out to hold it. None is the whole sensor.
        self._requested_roi: ROI | None = None
        self._window: ROI | None = None

        self._harvester = Harvester()
        self._harvester.add_file(str(cti_path))
        self._harvester.update()

        self._acquirer = self._harvester.create(self._device_index(serial))
        self._node_map = self._acquirer.remote_device.node_map

        self._configure(pixel_format=pixel_format, gain=gain)
        # Before the first start, since the sizes are not writable while streaming.
        self._write_roi(roi)

        self._sensor_resolution: tuple[int, int] = (
            int(self._read("HeightMax", self._read("Height", 0))),
            int(self._read("WidthMax", self._read("Width", 0))),
        )

        self._acquirer.start()

    def _device_index(self, serial: str | None) -> int:
        """Which of the attached cameras to open."""
        devices = self._harvester.device_info_list
        if not devices:
            raise ValueError(
                "The producer reported no cameras. Check that the camera is connected "
                "and that the .cti path is the right one for this transport."
            )
        if serial is None:
            return 0

        serials = [str(info.serial_number) for info in devices]
        if serial not in serials:
            raise ValueError(
                f"No camera with serial {serial!r}. The producer reports {serials}."
            )
        return serials.index(serial)

    def _node(self, name: str):
        """One node of the map, or None when this camera does not carry it.

        Cameras differ in which optional features they expose, so a driver meant for
        more than one has to ask rather than assume.
        """
        return getattr(self._node_map, name, None)

    def _read(self, name: str, default: Any = None) -> Any:
        node = self._node(name)
        if node is None:
            return default
        try:
            return node.value
        except Exception:
            return default

    def _write(self, name: str, value: Any) -> bool:
        """Set a node, reporting whether this camera has it.

        Returns:
            bool: True when the node exists and took the value.
        """
        node = self._node(name)
        if node is None:
            return False
        node.value = value
        return True

    def _configure(self, pixel_format: str, gain: float) -> None:
        """Everything that is set once and holds for the session."""
        # Anything automatic would move under the fit, so every loop is opened.
        for name in ("ExposureAuto", "GainAuto", "BlackLevelAuto", "mvLowLight"):
            self._write(name, "Off")

        self._write("PixelFormat", pixel_format)
        self._write("Gain", float(gain))

        # One frame per software trigger, which is what makes a capture correspond to
        # the exposure asked for.
        self._write("TriggerSelector", "FrameStart")
        self._write("TriggerMode", "On")
        self._write("TriggerSource", "Software")

        # A frame rate limit alongside a trigger throttles the trigger response on some
        # producers, and under triggering it buys nothing.
        self._write("AcquisitionFrameRateEnable", False)

        self._write("AcquisitionMode", "Continuous")

        # Continuous ignores the frame count, but a producer that quietly stays in
        # MultiFrame does not, so the count is raised to its ceiling either way.
        count = self._node("AcquisitionFrameCount")
        if count is not None:
            count.value = int(count.max)

    @property
    def pixel_size(self) -> NDArray[np.float64]:
        return self._pixel_size

    @property
    def sensor_resolution(self) -> tuple[int, int]:
        """The whole sensor's ``(height, width)``, whatever the region of interest."""
        return self._sensor_resolution

    @property
    def max_pixel_value(self) -> int:
        """The largest count a pixel can report.

        Read from ``PixelDynamicRangeMax`` when the camera states it, since a 12-bit
        sensor streaming ``Mono16`` still only fills 12 bits.
        """
        stated = self._read("PixelDynamicRangeMax")
        if stated is not None:
            return int(stated)

        match = PIXEL_FORMAT_BITS.search(str(self._read("PixelFormat", "Mono8")))
        bits = int(match.group(1)) if match else 8
        return 2**bits - 1

    @property
    def roi(self) -> ROI:
        """The region each frame covers, as set, or the readout window of the
        geometry nodes when no region was set.
        """
        if self._requested_roi is not None:
            return self._requested_roi
        return ROI(
            top_row=int(self._read("OffsetY", 0)),
            left_column=int(self._read("OffsetX", 0)),
            height=int(self._read("Height", self._sensor_resolution[0])),
            width=int(self._read("Width", self._sensor_resolution[1])),
        )

    def set_roi(self, roi: ROI | None) -> None:
        """Set the region of interest, restarting acquisition around the change.

        ``Width`` and ``Height`` are not writable while the camera streams. The camera
        reads out the smallest window on its increment grid that holds the region, and
        every frame is cropped to the region.
        """
        self._acquirer.stop()
        try:
            self._write_roi(roi)
        finally:
            self._acquirer.start()
            self.frames_since_start = 0

    def _write_roi(self, roi: ROI | None) -> None:
        """Write the four geometry nodes in an order the camera always accepts.

        GenICam holds ``OffsetX + Width <= SensorWidth`` at every write, so growing the
        width before moving the offset back can be rejected. Zeroing both offsets first
        makes the full sensor available, and the offsets then always fit.
        """
        self._write("OffsetX", 0)
        self._write("OffsetY", 0)
        self._requested_roi = None
        self._window = None

        if roi is None:
            width = self._node("Width")
            height = self._node("Height")
            if width is not None:
                width.value = width.max
            if height is not None:
                height.value = height.max
            return

        window = self._window_holding(roi)
        self._write("Width", window.width)
        self._write("Height", window.height)
        self._write("OffsetX", window.left_column)
        self._write("OffsetY", window.top_row)
        self._requested_roi = roi
        self._window = window

    def _window_holding(self, roi: ROI) -> ROI:
        """The smallest readout window on the camera's increment grid that holds
        ``roi``.

        The offsets are rounded down to their increments and the sizes up to theirs, so
        the window holds the whole region.
        """
        top = _round_down(int(roi.top_row), self._increment("OffsetY"))
        left = _round_down(int(roi.left_column), self._increment("OffsetX"))
        bottom = int(roi.top_row) + int(roi.height)
        right = int(roi.left_column) + int(roi.width)
        return ROI(
            top_row=top,
            left_column=left,
            height=_round_up(bottom - top, self._increment("Height")),
            width=_round_up(right - left, self._increment("Width")),
        )

    def _increment(self, name: str) -> int:
        """The step between the values a geometry node accepts, one when the camera
        does not state it.
        """
        increment = getattr(self._node(name), "inc", None)
        return int(increment) if increment else 1

    @property
    def exposure_bounds(self) -> tuple[float, float] | None:
        node = self._node("ExposureTime")
        if node is None:
            return None
        return (
            float(node.min) / MICROSECONDS_PER_SECOND,
            float(node.max) / MICROSECONDS_PER_SECOND,
        )

    def get_exposure(self) -> float:
        return float(self._read("ExposureTime", 0.0)) / MICROSECONDS_PER_SECOND

    def set_exposure(self, exposure_s: float) -> None:
        """Set the exposure time in seconds.

        An exposure outside :attr:`exposure_bounds` is clipped into them with a
        warning, since the camera rejects a value outside its node's range. The value
        is clipped in the node's microseconds, so a bound converted to seconds and back
        still lands on the node's limit.

        Args:
            exposure_s: The exposure in seconds.
        """
        exposure = float(exposure_s)
        exposure_us = exposure * MICROSECONDS_PER_SECOND
        node = self._node("ExposureTime")
        if node is not None:
            low_us, high_us = float(node.min), float(node.max)
            bounds = (
                low_us / MICROSECONDS_PER_SECOND,
                high_us / MICROSECONDS_PER_SECOND,
            )
            exposure_us = min(max(exposure_us, low_us), high_us)
            if not bounds[0] <= exposure <= bounds[1]:
                warnings.warn(
                    f"An exposure of {exposure} s is outside the camera's bounds "
                    f"{bounds} s, so {exposure_us / MICROSECONDS_PER_SECOND} s is "
                    "applied.",
                    stacklevel=2,
                )
        self._write("ExposureTime", exposure_us)

    def _get_image(
        self, exposure: float | None = None, averaging: int = 1
    ) -> NDArray:
        """Capture a frame as a ``(height, width)`` array of digital counts.

        Each frame is triggered on its own, and several frames are summed.

        Args:
            exposure: The exposure in seconds, set before the capture when given.
            averaging: The number of frames to sum.

        Returns:
            NDArray: One frame in the streamed pixel format, or the float64 sum of
            ``averaging`` frames.
        """
        if exposure is not None:
            self.set_exposure(exposure)

        frames = [self._fetch_frame() for _ in range(max(int(averaging), 1))]
        if len(frames) == 1:
            return frames[0]
        return np.sum(np.stack(frames).astype(np.float64), axis=0)

    def _fetch_frame(self) -> NDArray:
        """Trigger one exposure, copy the frame out of the transport buffer, and crop
        it to the region of interest.
        """
        self._restart_if_due()

        self._execute("TriggerSoftware")

        with self._acquirer.fetch() as buffer:
            component = buffer.payload.components[0]
            # A copy, since the buffer returns to the transport at the end of the with.
            frame = np.array(
                component.data.reshape(int(component.height), int(component.width))
            )

        self.frames_since_start += 1
        roi, window = self._requested_roi, self._window
        if roi is None or window is None:
            return frame
        top = int(roi.top_row) - window.top_row
        left = int(roi.left_column) - window.left_column
        return frame[top : top + int(roi.height), left : left + int(roi.width)]

    def _execute(self, name: str) -> None:
        """Fire a GenICam command node."""
        node = self._node(name)
        if node is None:
            raise RuntimeError(
                f"This camera has no {name} command, so a frame cannot be triggered. "
                "Check that TriggerSource is set to Software."
            )
        node.execute()

    def _restart_if_due(self) -> None:
        """Cycle acquisition once the frame budget is spent."""
        if self.frames_before_restart is None:
            return
        if self.frames_since_start < self.frames_before_restart:
            return
        self.restart_acquisition()

    def restart_acquisition(self) -> None:
        """Stop and start streaming, keeping the camera open.

        Far cheaper than rebuilding the producer and the acquirer, which is what makes
        ``frames_before_restart`` bearable as a workaround.
        """
        self._acquirer.stop()
        self._acquirer.start()
        self.frames_since_start = 0

    # Diagnostics and teardown ---------------------------------------------------

    def diagnostics(self) -> dict:
        """What the camera says about itself right now, as plain values.

        A GenICam write can be rejected, clamped, or ignored while streaming without
        raising, so the nodes are read back rather than assumed. Reach for this when
        acquisition stops and it is not obvious why.
        """
        names = (
            "AcquisitionMode",
            "AcquisitionFrameCount",
            "AcquisitionFrameRateEnable",
            "AcquisitionFrameRate",
            "TriggerMode",
            "TriggerSource",
            "TriggerSelector",
            "ExposureTime",
            "PixelFormat",
            "PixelDynamicRangeMax",
            "Gain",
            "Width",
            "Height",
            "OffsetX",
            "OffsetY",
        )
        report = {name: self._read(name) for name in names}
        report["is_acquiring"] = self._is_acquiring()
        report["frames_since_start"] = self.frames_since_start

        statistics = getattr(self._acquirer, "statistics", None)
        if statistics is not None:
            report["num_images"] = getattr(statistics, "num_images", None)
            report["fps"] = getattr(statistics, "fps", None)
        return report

    def _is_acquiring(self) -> bool | None:
        acquiring = getattr(self._acquirer, "is_acquiring", None)
        if acquiring is None:
            return None
        return bool(acquiring() if callable(acquiring) else acquiring)

    def close(self) -> None:
        """Stop streaming and hand the camera back. Safe to call more than once."""
        acquirer, self._acquirer = getattr(self, "_acquirer", None), None
        if acquirer is not None:
            try:
                acquirer.stop()
            finally:
                acquirer.destroy()

        harvester, self._harvester = getattr(self, "_harvester", None), None
        if harvester is not None:
            harvester.reset()

    def __enter__(self) -> GenTLCamera:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _round_down(value: int, step: int) -> int:
    """``value`` rounded down to a multiple of ``step``."""
    return (value // step) * step


def _round_up(value: int, step: int) -> int:
    """``value`` rounded up to a multiple of ``step``."""
    return -(-value // step) * step
