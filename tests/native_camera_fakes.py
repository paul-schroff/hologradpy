"""Native cameras for the tests, built on the HoloGradPy camera templates."""

from __future__ import annotations

from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray

from hologradpy.hardware.camera import Camera, SimulatedCameraTorch
from hologradpy.roi import ROI
from hologradpy.utils import gpu_to_numpy


class CroppingCamera(Camera):
    """A native camera that renders whole sensor frames and crops them to its region
    of interest.

    A subclass implements :meth:`render_sensor_frame`. :meth:`get_image` renders and
    crops one sensor frame per averaged frame, as on a real camera.

    A camera built with ``gain_bounds`` sets its gain. Every gain applied is recorded
    in :attr:`requested_gains`. A subclass renders its counts from
    :meth:`get_equivalent_exposure`, so the gain scales them.

    Args:
        sensor_resolution: The ``(height, width)`` of the sensor.
        max_pixel_value: The largest count a pixel reports.
        exposure_bounds: The ``(min, max)`` exposure in seconds, or None for a camera
            that states no bounds.
        exposure_s: The starting exposure in seconds.
        gain_bounds: The ``(min, max)`` gain in dB, or None for a camera whose gain
            cannot be set.
        gain: The starting gain in dB.
        gain_step: The step in dB that every gain is rounded to, or None to apply each
            gain as asked.
    """

    def __init__(
        self,
        sensor_resolution: tuple[int, int],
        *,
        max_pixel_value: int = 255,
        exposure_bounds: tuple[float, float] | None = (1e-4, 1.0),
        exposure_s: float = 1e-3,
        gain_bounds: tuple[float, float] | None = None,
        gain: float = 0.0,
        gain_step: float | None = None,
    ) -> None:
        self._sensor_resolution = (int(sensor_resolution[0]), int(sensor_resolution[1]))
        self._max_pixel_value = int(max_pixel_value)
        self._exposure_bounds = (
            None
            if exposure_bounds is None
            else (float(exposure_bounds[0]), float(exposure_bounds[1]))
        )
        self._exposure_s = float(exposure_s)
        self._gain_bounds = (
            None
            if gain_bounds is None
            else (float(gain_bounds[0]), float(gain_bounds[1]))
        )
        self._current_gain = float(gain)
        self._gain_step = gain_step
        self.requested_gains: list[float] = []
        self._roi = ROI(0, 0, *self._sensor_resolution)
        self.captured_frames = 0

    @abstractmethod
    def render_sensor_frame(self) -> NDArray:
        """One frame of the whole sensor at the current exposure and gain."""

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

    def _set_roi(self, roi: ROI | None) -> None:
        """Crop every frame to ``roi``, or to the whole sensor for None."""
        self._roi = ROI(0, 0, *self._sensor_resolution) if roi is None else roi

    def get_exposure(self) -> float:
        """The current exposure in seconds."""
        return self._exposure_s

    def _set_exposure(self, exposure: float) -> None:
        """Apply the exposure in seconds, which the camera base keeps in the bounds."""
        self._exposure_s = float(exposure)

    @property
    def gain_bounds(self) -> tuple[float, float] | None:
        """The ``(min, max)`` gain in dB, or None for a camera whose gain cannot be
        set.
        """
        return self._gain_bounds

    def get_gain(self) -> float:
        """The current gain in dB."""
        return self._current_gain

    def _set_gain(self, gain: float) -> None:
        """Apply the gain in dB, which the camera base keeps in the bounds.

        The applied gain is rounded to the gain step.
        """
        self.requested_gains.append(float(gain))
        applied = float(gain)
        if self._gain_step is not None:
            applied = round(applied / self._gain_step) * self._gain_step
        self._current_gain = float(applied)

    def _get_image(self) -> NDArray:
        """Render one fresh sensor frame and crop it to the region.

        Returns:
            NDArray: The cropped frame in the rendered dtype.

        Raises:
            ValueError: A rendered frame does not have the sensor resolution.
        """
        frame = self.render_sensor_frame()
        self.captured_frames += 1
        if frame.shape != self._sensor_resolution:
            raise ValueError(
                f"A rendered frame has shape {frame.shape}, and the sensor is "
                f"{self._sensor_resolution}."
            )
        return self._roi.crop(frame)


class StrictSimulatedCamera(SimulatedCameraTorch):
    """The simulated camera, held to the camera contract.

    It adds two rules to :class:`~hologradpy.hardware.camera.SimulatedCameraTorch`.

    - An exposure outside :attr:`exposure_search_bounds` raises ``ValueError``.
    - :meth:`get_image` rounds and clips each frame to integer counts in ``uint16``.

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

    def set_exposure(self, exposure: float) -> None:
        """Set the exposure in seconds.

        Raises:
            ValueError: The exposure is outside :attr:`exposure_search_bounds`.
        """
        low, high = self.exposure_search_bounds
        if not low <= exposure <= high:
            raise ValueError(
                f"An exposure of {exposure} s is outside the camera's limits "
                f"({low}, {high}) s."
            )
        super().set_exposure(exposure)

    def _get_image(self) -> NDArray:
        """Capture one frame of integer counts, reoriented and cropped to the region.

        Returns:
            NDArray: The ``uint16`` frame.
        """
        return self._roi.crop(self.transform(self._capture_counts()))

    def _capture_counts(self) -> NDArray[np.uint16]:
        """Capture one raw frame, rounded and clipped to integer counts."""
        counts = gpu_to_numpy(self._capture_frame())
        return np.clip(np.rint(counts), 0, self.max_pixel_value).astype(np.uint16)
