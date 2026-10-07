"""A Thorlabs scientific camera, such as a Zelux, driven through ``pylablib``.

Developed for a Zelux CS165MU. pylablib's driver speaks to the Thorlabs TSI SDK, which
also runs the Kiralux and Quantalux cameras.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

from ...roi import ROI
from .abstract import Camera, CameraOrientation, reorient_pixels

# How long to wait for a triggered frame beyond its exposure, in seconds.
FRAME_TIMEOUT_S = 2.0


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
        self.transform = CameraOrientation().transformation()
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

    def set_roi(self, roi: ROI | None) -> None:
        """Set the region of interest, or return to the whole frame with None.

        Only the part of the sensor under the region is read out. Changing it re-arms
        the camera, which takes a moment.

        Args:
            roi: The region in displayed ``(row, col)`` coordinates, or None.

        Raises:
            ValueError: ``roi`` is empty or reaches off the displayed frame. The region
                is then left as it was.
        """
        height, width = self._sensor_resolution
        if roi is None:
            roi = ROI(0, 0, height, width)
        elif not roi.lies_inside((height, width)):
            raise ValueError(f"{roi} does not lie inside the {height} x {width} frame.")
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

    def set_orientation(self, orientation: CameraOrientation) -> None:
        """Remount the sensor, reorienting every frame from here on.

        The region of interest returns to the whole frame, since a crop is given in the
        coordinates of the previous frame. The excluded pixels follow the sensor into
        the new frame (:func:`~hologradpy.hardware.camera.reorient_pixels`), and the
        pixel pitch is exchanged under a quarter turn.

        Args:
            orientation: The orientation to mount the sensor in.
        """
        excluded = self.excluded_pixels
        if excluded:
            excluded = reorient_pixels(
                excluded, self._raw_shape, self.orientation, orientation
            )
        if orientation.swaps_axes():
            self._sensor_resolution = (self._raw_shape[1], self._raw_shape[0])
            self._pixel_size = self._raw_pixel_size[::-1].copy()
        else:
            self._sensor_resolution = self._raw_shape
            self._pixel_size = self._raw_pixel_size
        self.transform = orientation.transformation()
        self.set_roi(None)
        self.excluded_pixels = excluded

    @property
    def exposure_bounds(self) -> tuple[float, float] | None:
        return self._exposure_bounds

    def get_exposure(self) -> float:
        return float(self._device.get_exposure())

    def set_exposure(self, exposure_s: float) -> None:
        """Set the exposure time in seconds, from the next frame on.

        The acquisition keeps running. An exposure outside :attr:`exposure_bounds` is
        clipped into them with a warning.

        Args:
            exposure_s: The exposure in seconds.
        """
        exposure = float(exposure_s)
        bounds = self._exposure_bounds
        if bounds is not None and not bounds[0] <= exposure <= bounds[1]:
            clipped = min(max(exposure, bounds[0]), bounds[1])
            warnings.warn(
                f"An exposure of {exposure} s is outside the camera's bounds {bounds} "
                f"s, so {clipped} s is applied.",
                stacklevel=2,
            )
            exposure = clipped
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

    def set_gain(self, gain: float) -> None:
        """Set the gain in dB, from the next frame on.

        The acquisition keeps running. A gain outside :attr:`gain_bounds` is clipped
        into them with a warning. The camera sets the gain in steps of its own, and
        :meth:`get_gain` reads back the step it applied.

        Args:
            gain: The gain in dB.

        Raises:
            NotImplementedError: The camera states no gain range.
        """
        bounds = self._gain_bounds
        if bounds is None:
            raise NotImplementedError(
                f"The {self.name} states no gain range, so its gain cannot be set."
            )
        gain = float(gain)
        if not bounds[0] <= gain <= bounds[1]:
            clipped = min(max(gain, bounds[0]), bounds[1])
            warnings.warn(
                f"A gain of {gain} dB is outside the camera's gain range {bounds} dB, "
                f"so {clipped} dB is applied.",
                stacklevel=2,
            )
            gain = clipped
        self._device.set_gain(gain)

    def _get_image(
        self, exposure: float | None = None, averaging: int = 1
    ) -> NDArray:
        """Capture a frame as a ``(height, width)`` array of digital counts.

        Each frame is triggered on its own, and several frames are summed.

        Args:
            exposure: The exposure in seconds, set before the capture when given.
            averaging: The number of frames to sum.

        Returns:
            NDArray: One frame in the sensor's dtype, or the float64 sum of
            ``averaging`` frames.
        """
        if exposure is not None:
            self.set_exposure(exposure)

        frames = [self._capture_frame() for _ in range(max(int(averaging), 1))]
        if len(frames) == 1:
            return frames[0]
        return np.sum(np.stack(frames).astype(np.float64), axis=0)

    def _capture_frame(self) -> NDArray:
        """Trigger one exposure, and return its frame reoriented and cropped to the
        region of interest.

        Frames left from earlier triggers are dropped first, so the frame returned is
        the one this trigger exposed.
        """
        self._device.read_multiple_images()
        self._device.send_software_trigger()
        self._device.wait_for_frame(timeout=self.get_exposure() + FRAME_TIMEOUT_S)
        frame = self._device.read_oldest_image()
        # The window can be larger than the region, when the camera has a minimum size.
        region, window = self._sensor_region, self._window
        top = region.top_row - window.top_row
        left = region.left_column - window.left_column
        part = frame[top : top + region.height, left : left + region.width]
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

    def __enter__(self) -> ThorlabsCamera:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _window(roi: tuple) -> ROI:
    """A window as pylablib states it, ``(hstart, hend, vstart, vend, hbin, vbin)``,
    as an ROI.
    """
    left, right, top, bottom = (int(value) for value in roi[:4])
    return ROI(top, left, bottom - top, right - left)


def _widened(
    region: ROI, margins: tuple[int, int, int, int], shape: tuple[int, int]
) -> ROI:
    """``region`` widened by ``(top, left, bottom, right)`` margins, kept on a sensor
    of ``shape``.
    """
    top = max(region.top_row - margins[0], 0)
    left = max(region.left_column - margins[1], 0)
    bottom = min(region.top_row + region.height + margins[2], shape[0])
    right = min(region.left_column + region.width + margins[3], shape[1])
    return ROI(top, left, bottom - top, right - left)


def _cut_off(window: ROI, region: ROI) -> tuple[bool, bool, bool, bool]:
    """Which sides of ``region`` ``window`` cuts off, ``(top, left, bottom, right)``."""
    return (
        window.top_row > region.top_row,
        window.left_column > region.left_column,
        window.top_row + window.height < region.top_row + region.height,
        window.left_column + window.width < region.left_column + region.width,
    )
