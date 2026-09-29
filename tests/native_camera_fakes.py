"""Native cameras for the tests, built on the HoloGradPy camera templates.

:class:`CroppingCamera` is the base for a fake that renders whole sensor frames. It
crops each frame to its region of interest and keeps the exposure inside its bounds, as
a real camera does. :class:`StrictSimulatedCamera` holds the simulated camera to the
camera contract. It refuses an exposure outside its limits and a region off its frame,
and it returns integer counts.
"""

from __future__ import annotations

from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray

from hologradpy.hardware.camera import Camera, SimulatedCameraTorch
from hologradpy.roi import ROI
from hologradpy.utils import gpu_to_numpy


def _check_region_on_frame(roi: ROI, frame_shape: tuple[int, int]) -> None:
    """Raise ``ValueError`` unless ``roi`` is a nonempty region inside the frame."""
    if not roi.lies_inside(frame_shape):
        raise ValueError(f"{roi} is not a region of the {frame_shape} frame.")


class CroppingCamera(Camera):
    """A native camera that renders whole sensor frames and crops them to its region
    of interest.

    A subclass implements :meth:`render_sensor_frame`. :meth:`get_image` renders one
    sensor frame per averaged frame and crops each one, so the region and the averaging
    behave as on a real camera. A single frame keeps the rendered dtype, and a sum of
    several frames is float64.

    Args:
        sensor_resolution: The ``(height, width)`` of the sensor.
        max_pixel_value: The largest count a pixel reports.
        exposure_bounds: The ``(min, max)`` exposure in seconds, or None for a camera
            that states no bounds.
        exposure_s: The starting exposure in seconds.
    """

    def __init__(
        self,
        sensor_resolution: tuple[int, int],
        *,
        max_pixel_value: int = 255,
        exposure_bounds: tuple[float, float] | None = (1e-4, 1.0),
        exposure_s: float = 1e-3,
    ) -> None:
        self._sensor_resolution = (int(sensor_resolution[0]), int(sensor_resolution[1]))
        self._max_pixel_value = int(max_pixel_value)
        self._exposure_bounds = (
            None
            if exposure_bounds is None
            else (float(exposure_bounds[0]), float(exposure_bounds[1]))
        )
        self._exposure_s = float(exposure_s)
        self._roi = ROI(0, 0, *self._sensor_resolution)
        self.captured_frames = 0

    @abstractmethod
    def render_sensor_frame(self) -> NDArray:
        """One frame of the whole sensor at the current exposure."""

    @property
    def pixel_size(self) -> NDArray[np.float64]:
        """Pixel pitch ``(y, x)`` in metres."""
        return np.array([1e-6, 1e-6])

    @property
    def max_pixel_value(self) -> int:
        """The largest count a pixel reports."""
        return self._max_pixel_value

    @property
    def exposure_bounds(self) -> tuple[float, float] | None:
        """The ``(min, max)`` exposure in seconds, or None if the camera states none."""
        return self._exposure_bounds

    @property
    def sensor_resolution(self) -> tuple[int, int]:
        """The ``(height, width)`` of the sensor."""
        return self._sensor_resolution

    @property
    def roi(self) -> ROI:
        """The current region of interest."""
        return self._roi

    def set_roi(self, roi: ROI | None) -> None:
        """Set the region of interest, None for the whole sensor.

        Raises:
            ValueError: The region is empty or reaches off the sensor.
        """
        if roi is None:
            self._roi = ROI(0, 0, *self._sensor_resolution)
            return
        _check_region_on_frame(roi, self._sensor_resolution)
        self._roi = roi

    def get_exposure(self) -> float:
        """The current exposure in seconds."""
        return self._exposure_s

    def set_exposure(self, exposure_s: float) -> None:
        """Set the exposure in seconds, clamped into the bounds when there are any."""
        if self._exposure_bounds is None:
            self._exposure_s = float(exposure_s)
            return
        low, high = self._exposure_bounds
        self._exposure_s = float(min(max(exposure_s, low), high))

    def _get_image(self, exposure: float | None = None, averaging: int = 1) -> NDArray:
        """Capture ``averaging`` fresh frames cropped to the region, and sum them.

        Args:
            exposure: The exposure to set first, if any.
            averaging: The number of frames to sum.

        Returns:
            NDArray: The cropped frame in the rendered dtype for one frame, and the
            float64 sum for several.

        Raises:
            ValueError: A rendered frame does not have the sensor resolution.
        """
        if exposure is not None:
            self.set_exposure(exposure)
        crops = []
        for _ in range(max(1, int(averaging))):
            frame = self.render_sensor_frame()
            self.captured_frames += 1
            if frame.shape != self._sensor_resolution:
                raise ValueError(
                    f"A rendered frame has shape {frame.shape}, and the sensor is "
                    f"{self._sensor_resolution}."
                )
            crops.append(self._roi.crop(frame))
        if len(crops) == 1:
            return crops[0]
        return np.sum(np.stack(crops).astype(np.float64), axis=0)


class StrictSimulatedCamera(SimulatedCameraTorch):
    """The simulated camera, held to the camera contract.

    It adds three rules to :class:`~hologradpy.hardware.camera.SimulatedCameraTorch`.

    - An exposure outside :attr:`exposure_search_bounds` raises ``ValueError``.
    - A region that is empty or reaches off the displayed frame raises ``ValueError``.
    - :meth:`get_image` rounds and clips each frame to integer counts in ``uint16``,
      and sums several frames into float64.

    The tensor path, :meth:`get_image_tensor`, is inherited unchanged.

    Args:
        *args: Passed on to :class:`SimulatedCameraTorch`.
        **kwargs: Passed on to :class:`SimulatedCameraTorch`.

    Raises:
        ValueError: The sensor counts past what ``uint16`` holds.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.max_pixel_value > np.iinfo(np.uint16).max:
            raise ValueError(
                f"A {self.bitdepth}-bit sensor counts past what uint16 holds."
            )

    def set_exposure(self, exposure_s: float) -> None:
        """Set the exposure in seconds.

        Raises:
            ValueError: The exposure is outside :attr:`exposure_search_bounds`.
        """
        low, high = self.exposure_search_bounds
        if not low <= exposure_s <= high:
            raise ValueError(
                f"An exposure of {exposure_s} s is outside the camera's limits "
                f"({low}, {high}) s."
            )
        super().set_exposure(exposure_s)

    def set_roi(self, roi: ROI | None) -> None:
        """Set the region of interest, None for the whole displayed frame.

        Raises:
            ValueError: The region is empty or reaches off the displayed frame.
        """
        if roi is not None:
            _check_region_on_frame(roi, self.sensor_resolution)
        super().set_roi(roi)

    def _get_image(
        self, exposure: float | None = None, averaging: int = 1
    ) -> NDArray:
        """Capture a frame of integer counts, reoriented and cropped to the region.

        Args:
            exposure: The exposure to set first, if any.
            averaging: The number of frames to sum.

        Returns:
            NDArray: A ``uint16`` frame for one frame, and the float64 sum for
            several.
        """
        if exposure is not None:
            self.set_exposure(exposure)
        frames = [self._capture_counts() for _ in range(max(1, int(averaging)))]
        image = (
            frames[0]
            if len(frames) == 1
            else np.sum(np.stack(frames).astype(np.float64), axis=0)
        )
        return self._roi.crop(self.transform(image))

    def _capture_counts(self) -> NDArray[np.uint16]:
        """Capture one raw frame, rounded and clipped to integer counts."""
        counts = gpu_to_numpy(self._capture_frame())
        return np.clip(np.rint(counts), 0, self.max_pixel_value).astype(np.uint16)
