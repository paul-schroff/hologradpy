"""A Thorlabs scientific camera, such as a Zelux, driven through ``pylablib``.

Developed for a Zelux CS165MU. pylablib's driver speaks to the Thorlabs TSI SDK, which
also runs the Kiralux and Quantalux cameras.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ...roi import ROI
from .abstract import Camera, CameraOrientation, reorient_pixels

# How long to wait for a triggered frame beyond its exposure, in seconds.
FRAME_TIMEOUT = 2.0


class ThorlabsCamera(Camera):
    """A Thorlabs scientific camera, armed once and triggered in software.

    The camera is armed for one frame per software trigger, and every frame is taken
    with a trigger. Only the part of the sensor under the region of interest is read
    out, which shortens the transfer of every frame. The pixel pitch and the exposure
    bounds are passed in, since the driver does not report them. The gain range is read
    from the camera.
    """

    def __init__(
        self,
        pixel_size: tuple[float, float],
        serial: str | None = None,
        exposure_bounds: tuple[float, float] | None = None,
        gain: float = 0.0,
    ) -> None:
        """Open the camera and arm it for software triggers.

        Args:
            pixel_size: Pixel pitch ``(y, x)`` of the sensor in metres, from the
                datasheet.
            serial: Which camera to open, when several are attached. Defaults to the
                first one found.
            exposure_bounds: The ``(min, max)`` exposure in seconds to keep to, from the
                datasheet. Defaults to None, which states no bounds.
            gain: The gain in dB to open the camera at. Defaults to 0 dB. A camera
                that states no gain range keeps its own gain.

        Raises:
            ImportError: If ``pylablib`` is not installed.
        """
        try:
            from pylablib.devices import Thorlabs
        except ImportError as error:
            raise ImportError(
                "ThorlabsCamera needs the 'pylablib' package, which is not installed. "
                "Install it with 'pip install pylablib'."
            ) from error

        self._device = Thorlabs.ThorlabsTLCamera(serial)
        info = self._device.get_device_info()
        self.name = f"{info.model} {info.serial_number}"

        width, height = self._device.get_detector_size()
        self._raw_shape: tuple[int, int] = (int(height), int(width))
        self._device.set_trigger_mode("int")  # Software triggers.
        self._sensor_region = ROI(0, 0, *self._raw_shape)
        self._window = _window(self._device.set_roi())
        self._raw_pixel_size = np.asarray(pixel_size, dtype=np.float64)
        self._max_pixel_value = 2 ** int(self._device.get_sensor_info().bit_depth) - 1
        self._exposure_bounds: tuple[float, float] | None = (
            None
            if exposure_bounds is None
            else (float(exposure_bounds[0]), float(exposure_bounds[1]))
        )
        self._gain_bounds = self._read_gain_range()
        if self._gain_bounds is not None:
            self.set_gain(gain)

        self._sensor_resolution: tuple[int, int] = self._raw_shape
        self._pixel_size = self._raw_pixel_size
        self._roi = ROI(0, 0, *self._raw_shape)

        # One frame per trigger, and none until the first trigger.
        self._device.start_acquisition(frames_per_trigger=1, auto_start=False)

    @property
    def pixel_size(self) -> NDArray[np.float64]:
        """Pixel pitch ``(y, x)`` of the displayed frame in metres."""
        return self._pixel_size

    @property
    def sensor_resolution(self) -> tuple[int, int]:
        """The whole sensor's ``(height, width)`` in the displayed frame, whatever the
        region of interest.
        """
        return self._sensor_resolution

    @property
    def max_pixel_value(self) -> int:
        """The largest count a pixel can report, from the sensor's bit depth."""
        return self._max_pixel_value

    @property
    def roi(self) -> ROI:
        """The current region of interest, in displayed ``(row, col)`` coordinates."""
        return self._roi

    def _set_roi(self, roi: ROI | None) -> None:
        """Read out the part of the sensor under ``roi``, or the whole sensor for None.

        Changing the region re-arms the camera, which takes a moment.
        """
        if roi is None:
            roi = ROI(0, 0, *self._sensor_resolution)
        self._read_out(self._sensor_region_of(roi))
        self._roi = roi

    def _sensor_region_of(self, roi: ROI) -> ROI:
        """The region of the raw sensor that ``roi`` shows in displayed coordinates.

        A rotation or a flip takes a rectangle to a rectangle, so its two opposite
        corners are enough.
        """
        corners = reorient_pixels(
            [
                (roi.top_row, roi.left_column),
                (roi.top_row + roi.height - 1, roi.left_column + roi.width - 1),
            ],
            self._raw_shape,
            self.orientation,
            CameraOrientation(),
        )
        rows, columns = zip(*corners)
        return ROI(
            min(rows),
            min(columns),
            max(rows) - min(rows) + 1,
            max(columns) - min(columns) + 1,
        )

    def _read_out(self, region: ROI) -> None:
        """Read out the window the camera allows around ``region`` of the raw sensor.

        The camera moves the edges of a window onto a grid of its own, which can cut off
        the edge rows or columns of the region. The camera is disarmed for the change
        and armed again for one frame per trigger.

        Args:
            region: The raw sensor region to read out, in raw ``(row, col)``
                coordinates.

        Raises:
            RuntimeError: No window the camera takes holds ``region``.
        """
        if region == self._sensor_region:
            return
        margins = (0, 0, 0, 0)  # (top, left, bottom, right)
        asked = None
        self._device.stop_acquisition()
        try:
            while (ask := _widened(region, margins, self._raw_shape)) != asked:
                asked = ask
                window = _window(
                    self._device.set_roi(
                        ask.left_column,
                        ask.left_column + ask.width,
                        ask.top_row,
                        ask.top_row + ask.height,
                    )
                )
                cut_off = _cut_off(window, region)
                if not any(cut_off):
                    break
                margins = tuple(
                    max(2 * margin, 1) if cut else margin
                    for margin, cut in zip(margins, cut_off)
                )
        finally:
            self._device.start_acquisition(frames_per_trigger=1, auto_start=False)
        if any(cut_off):
            raise RuntimeError(
                f"The camera reads out {window}, which does not hold the sensor region "
                f"{region} asked for."
            )
        self._window = window
        self._sensor_region = region

    def _set_orientation(
        self, orientation: CameraOrientation, raw_shape: tuple[int, int]
    ) -> None:
        """Remount the sensor, exchanging the displayed height and width and the pixel
        pitch under a quarter turn.
        """
        if orientation.swaps_axes():
            self._sensor_resolution = (raw_shape[1], raw_shape[0])
            self._pixel_size = self._raw_pixel_size[::-1].copy()
        else:
            self._sensor_resolution = (raw_shape[0], raw_shape[1])
            self._pixel_size = self._raw_pixel_size
        self.transform = orientation.transformation()

    @property
    def exposure_bounds(self) -> tuple[float, float] | None:
        return self._exposure_bounds

    def get_exposure(self) -> float:
        return float(self._device.get_exposure())

    def _set_exposure(self, exposure: float) -> None:
        """Set the exposure time in seconds, from the next frame on. The acquisition
        keeps running.
        """
        self._device.set_exposure(exposure)

    def _read_gain_range(self) -> tuple[float, float] | None:
        """The ``(min, max)`` gain in dB the camera states, or None when it states no
        range or a range of a single gain.
        """
        try:
            low, high = (float(gain) for gain in self._device.get_gain_range())
        except self._device.Error:
            return None
        return (low, high) if high > low else None

    @property
    def gain_bounds(self) -> tuple[float, float] | None:
        """The ``(min, max)`` gain in dB, read from the camera when it was opened, or
        None for a camera whose gain cannot be set.
        """
        return self._gain_bounds

    def get_gain(self) -> float:
        """The current gain in dB, or 0 dB for a camera whose gain cannot be set."""
        if self._gain_bounds is None:
            return 0.0
        return float(self._device.get_gain())

    def _set_gain(self, gain: float) -> None:
        """Set the gain in dB, from the next frame on. The acquisition keeps running.

        The camera sets the gain in steps of its own. :meth:`get_gain` reads back the
        step it applied.
        """
        self._device.set_gain(gain)

    def _get_image(self) -> NDArray:
        """Trigger one exposure, and return its frame reoriented and cropped to the
        region of interest.

        Frames left from earlier triggers are dropped first, so the frame returned is
        the one this trigger exposed.
        """
        self._device.read_multiple_images()
        self._device.send_software_trigger()
        self._device.wait_for_frame(timeout=self.get_exposure() + FRAME_TIMEOUT)
        frame = self._device.read_oldest_image()
        # The window can be larger than the region, when the camera has a minimum size.
        part = self._sensor_region.relative_to(self._window).crop(frame)
        # A copy, since the driver reuses its frame buffer.
        return np.array(self.transform(part))

    def close(self) -> None:
        """Disarm the camera and hand it back. Safe to call more than once."""
        device, self._device = getattr(self, "_device", None), None
        if device is not None:
            try:
                device.stop_acquisition()
            finally:
                device.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _window(roi: tuple) -> ROI:
    """A window as pylablib states it, ``(hstart, hend, vstart, vend, hbin, vbin)``,
    as an ROI.
    """
    left, right, top, bottom = roi[:4]
    return ROI.from_bounds(top, bottom, left, right)


def _widened(
    region: ROI, margins: tuple[int, int, int, int], shape: tuple[int, int]
) -> ROI:
    """``region`` widened by ``(top, left, bottom, right)`` margins, kept on a sensor
    of ``shape``.
    """
    top, bottom, left, right = region.to_bounds()
    return ROI.from_bounds(
        top - margins[0], bottom + margins[2], left - margins[1], right + margins[3]
    ).trimmed_to(shape)


def _cut_off(window: ROI, region: ROI) -> tuple[bool, bool, bool, bool]:
    """Which sides of ``region`` ``window`` cuts off, ``(top, left, bottom, right)``."""
    return (
        window.top_row > region.top_row,
        window.left_column > region.left_column,
        window.top_row + window.height < region.top_row + region.height,
        window.left_column + window.width < region.left_column + region.width,
    )
