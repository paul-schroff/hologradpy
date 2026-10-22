"""The native camera template.

A device subclasses the ``Camera`` base class. The module also defines the
``CameraOrientation`` and ``CameraData`` records, the ``reorient_pixels`` helper, and
the conversion of a gain in dB to a linear factor and back.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
import warnings

import numpy as np
import torch
from numpy.typing import NDArray

from ...geometry import dihedral_affine_matrix, dihedral_array_transform
from ...grids import get_spatial_grid as _spatial_grid
from ...roi import ROI
from ...serialization import SaveableRecord, record_type
from .stuck_pixels import STUCK_PIXEL_RESPONSE_TOLERANCE, stuck_pixel_mask


# The longest exposure in seconds that an exposure search reaches unless its caller
# passes longer bounds.
DEFAULT_MAX_EXPOSURE = 1.0

# The exposure in seconds from which an exposure search raises the gain, on a camera
# whose gain can be set (Camera.gain_onset_exposure).
DEFAULT_GAIN_ONSET_EXPOSURE = 0.1

# A pixel counts as overexposed from this fraction of the largest count. Some sensors
# clip a count or two short of it, such as a 10-bit Thorlabs Zelux at 1022 of 1023.
SATURATION_FRACTION = 0.99


def linear_gain(gain: float) -> float:
    """The factor ``10 ** (gain / 20)`` by which a gain of ``gain`` dB multiplies the
    counts above the black level.
    """
    return 10.0 ** (float(gain) / 20.0)


def gain_in_db(factor: float) -> float:
    """The gain ``20 * log10(factor)`` in dB, the inverse of :func:`linear_gain`."""
    return 20.0 * float(np.log10(factor))


@dataclass(frozen=True)
class _ExposureMeasurement:
    """A frame measured by the exposure search of :meth:`Camera.autoexpose`.

    Attributes:
        peak: The peak of the measured region in counts.
        exposure: The exposure in seconds that the camera read back.
        gain: The gain in dB that the camera read back.
        frame: The whole frame, kept for stuck-pixel detection, or None.
    """

    peak: float
    exposure: float
    gain: float
    frame: NDArray | None = None


@dataclass
class _ExposureSearch:
    """The target and the measured frames of the exposure search of
    :meth:`Camera.autoexpose`.

    Every exposure here is an equivalent exposure in seconds
    (:meth:`Camera.get_equivalent_exposure`). The measurements are keyed by it, in the
    order of the first frame at each exposure.

    Attributes:
        set_fraction: The target peak as a fraction of full scale.
        set_value: The target peak in counts.
        saturation: The count from which a pixel is overexposed.
        measurements: The latest measurement at each exposure.
    """

    set_fraction: float
    set_value: float
    saturation: float
    measurements: dict[float, _ExposureMeasurement] = field(default_factory=dict)

    def record(
        self,
        equivalent_exposure: float,
        peak: float,
        exposure: float,
        gain: float,
        frame: NDArray | None,
    ) -> None:
        """Record a frame at ``equivalent_exposure``.

        A repeated exposure takes the new peak, exposure and gain, and keeps the frame
        of its first measurement.
        """
        first = self.measurements.get(equivalent_exposure)
        if first is not None:
            frame = first.frame
        self.measurements[equivalent_exposure] = _ExposureMeasurement(
            peak, exposure, gain, frame
        )

    @property
    def overexposed_exposures(self) -> list[float]:
        """The positive exposures with a peak at :attr:`saturation` or above."""
        return [
            exposure
            for exposure, measurement in self.measurements.items()
            if exposure > 0.0 and measurement.peak >= self.saturation
        ]

    @property
    def exposures_below_saturation(self) -> list[float]:
        """The positive exposures with a peak below :attr:`saturation`."""
        return [
            exposure
            for exposure, measurement in self.measurements.items()
            if exposure > 0.0 and measurement.peak < self.saturation
        ]

    @property
    def every_frame_overexposed(self) -> bool:
        """Whether every frame at a positive exposure is overexposed."""
        return all(
            measurement.peak >= self.saturation
            for exposure, measurement in self.measurements.items()
            if exposure > 0.0
        )

    @property
    def recorded_frames(self) -> tuple[list[NDArray], list[float]]:
        """The frames kept for stuck-pixel detection and their exposures."""
        kept = [
            (measurement.frame, exposure)
            for exposure, measurement in self.measurements.items()
            if measurement.frame is not None
        ]
        return [frame for frame, _ in kept], [exposure for _, exposure in kept]


def _next_exposure(
    search: _ExposureSearch, exposure: float, peak: float, overexposed_factor: float
) -> float:
    """The exposure the search asks for after a frame at ``exposure`` peaked at
    ``peak``.

    The steps are described in :meth:`Camera.autoexpose`. An overexposed frame gives no
    proportional step, since its true peak is unknown.

    Args:
        search: The state of the search.
        exposure: The equivalent exposure of the last frame in seconds.
        peak: The peak of the last frame in counts.
        overexposed_factor: The factor that multiplies the exposure after an
            overexposed peak.

    Returns:
        float: The equivalent exposure in seconds, before the bounds of the search.
    """
    overexposed_at = search.overexposed_exposures
    below_saturation_at = search.exposures_below_saturation
    if peak >= search.saturation:
        if (
            overexposed_at
            and below_saturation_at
            and max(below_saturation_at) < min(overexposed_at)
        ):
            return float(np.sqrt(max(below_saturation_at) * min(overexposed_at)))
        return exposure * overexposed_factor
    desired = exposure * search.set_value / max(peak, 1.0)
    # The counts grow in proportion to the exposure, so the target lies at or below
    # set_fraction times any exposure that overexposed the region. A step from a peak
    # of a few counts is coarse. This bound keeps such a step below those exposures.
    if overexposed_at:
        desired = min(desired, search.set_fraction * min(overexposed_at))
    return desired


def _settled_exposure(search: _ExposureSearch) -> float:
    """The measured exposure a search settles on when it does not converge
    (:meth:`Camera.autoexpose`).

    Raises:
        RuntimeError: The search measured no frame at a positive exposure.
    """
    positive_peaks = {
        exposure: measurement.peak
        for exposure, measurement in search.measurements.items()
        if exposure > 0.0
    }
    if not positive_peaks:
        raise RuntimeError(
            "autoexposure took no frame at a positive exposure, so it has no exposure "
            "to settle on."
        )
    peaks_below_saturation = {
        exposure: peak
        for exposure, peak in positive_peaks.items()
        if peak < search.saturation
    }
    if peaks_below_saturation:
        return min(
            peaks_below_saturation,
            key=lambda candidate: abs(
                peaks_below_saturation[candidate] - search.set_value
            ),
        )
    return min(positive_peaks)


class Camera(ABC):
    """A HoloGradPy-native camera with SI units, ``(y, x)`` geometry and a
    ``(row, col)`` ROI.

    A device implements the abstract geometry and exposure members below and three
    hooks. :meth:`_set_roi` applies a region of interest, :meth:`_set_exposure` applies
    an exposure, and :meth:`_get_image` captures one frame. The public methods around
    them check the region, clip the exposure into the bounds, and sum and check the
    frames, so every device shares that behaviour. A device that can set its gain also
    overrides :attr:`gain_bounds`, :meth:`get_gain` and :meth:`_set_gain`, which state
    the gain in dB. A device that can be reoriented implements :meth:`_set_orientation`.
    A device that holds a driver session overrides :meth:`close`. The
    ``get_spatial_grid`` and ``autoexpose`` template methods are provided here as well.
    Third-party devices subclass this base or register a wrapper with
    :func:`hologradpy.hardware.as_native.as_camera`.
    """

    transform: Callable[[NDArray], NDArray] = staticmethod(dihedral_array_transform())
    """The function that reorients a raw sensor frame into the displayed frame. The base
    camera displays the frame as the sensor reads it out."""
    _excluded_pixels: list[tuple[int, int]] | None = None
    _overexposed: bool = False
    _gain_onset_exposure: float | None = DEFAULT_GAIN_ONSET_EXPOSURE

    @property
    @abstractmethod
    def pixel_size(self) -> NDArray[np.float64]:
        """Pixel pitch ``(y, x)`` in metres."""

    @property
    def resolution(self) -> tuple[int, int]:
        """Resolution ``(height, width)`` of the frame :meth:`get_image` returns.
        Describes the region of interest, not the whole sensor.
        """
        roi = self.roi
        return (int(roi.height), int(roi.width))

    @property
    def adu_levels(self) -> int:
        """How many digital levels a pixel can take, one more than
        :attr:`max_pixel_value`.
        """
        return self.max_pixel_value + 1

    @property
    def exposure_search_bounds(self) -> tuple[float, float]:
        """The ``(min, max)`` exposure in seconds that an exposure search uses.

        It is :attr:`exposure_bounds` with the maximum held at ``DEFAULT_MAX_EXPOSURE``,
        or ``(0, DEFAULT_MAX_EXPOSURE)`` for a camera that states no bounds.
        """
        bounds = self.exposure_bounds
        if bounds is None:
            return (0.0, DEFAULT_MAX_EXPOSURE)
        low, high = float(bounds[0]), float(bounds[1])
        return (low, min(high, max(low, DEFAULT_MAX_EXPOSURE)))

    @property
    @abstractmethod
    def sensor_resolution(self) -> tuple[int, int]:
        """The whole sensor's ``(height, width)`` in the displayed frame, whatever the
        region of interest.
        """

    @property
    @abstractmethod
    def max_pixel_value(self) -> int:
        """The largest count a pixel can report (``2 ** bitdepth - 1``)."""

    @property
    def saturation_level(self) -> float:
        """The count from which a pixel counts as overexposed, ``SATURATION_FRACTION``
        of :attr:`max_pixel_value`.
        """
        return SATURATION_FRACTION * self.max_pixel_value

    @property
    @abstractmethod
    def exposure_bounds(self) -> tuple[float, float] | None:
        """The ``(min, max)`` exposure time in seconds that the device accepts, or None
        when the device does not state them.

        An exposure search uses :attr:`exposure_search_bounds`, which holds the maximum
        of this range at ``DEFAULT_MAX_EXPOSURE``.
        """

    @property
    @abstractmethod
    def roi(self) -> ROI:
        """The current region of interest."""

    def set_roi(self, roi: ROI | None) -> None:
        """Set the region of interest, or return to the whole frame with None.

        Args:
            roi: The region in displayed ``(row, col)`` coordinates, or None.

        Raises:
            ValueError: ``roi`` is empty or reaches off the displayed frame. The region
                is then left as it was.
        """
        if roi is not None:
            height, width = self.sensor_resolution
            if not roi.lies_inside((height, width)):
                raise ValueError(
                    f"{roi} does not lie inside the {height} x {width} frame."
                )
        self._set_roi(roi)

    @abstractmethod
    def _set_roi(self, roi: ROI | None) -> None:
        """Apply a region of interest for :meth:`set_roi`.

        Args:
            roi: A nonempty region inside the displayed frame, or None for the whole
                frame.
        """

    @abstractmethod
    def get_exposure(self) -> float:
        """The current exposure time in seconds."""

    def set_exposure(self, exposure: float) -> None:
        """Set the exposure time in seconds.

        An exposure outside :attr:`exposure_bounds` is clipped into them with a warning.

        Args:
            exposure: The exposure in seconds.
        """
        exposure = float(exposure)
        bounds = self.exposure_bounds
        if bounds is not None and not bounds[0] <= exposure <= bounds[1]:
            clipped = min(max(exposure, bounds[0]), bounds[1])
            warnings.warn(
                f"An exposure of {exposure} s is outside the camera's bounds {bounds} "
                f"s, so {clipped} s is applied.",
                stacklevel=2,
            )
            exposure = clipped
        self._set_exposure(exposure)

    @abstractmethod
    def _set_exposure(self, exposure: float) -> None:
        """Apply an exposure for :meth:`set_exposure`.

        Args:
            exposure: The exposure in seconds, inside :attr:`exposure_bounds` when the
                camera states them.
        """

    @property
    def gain_bounds(self) -> tuple[float, float] | None:
        """The ``(min, max)`` gain in dB that the device accepts, or None when its gain
        cannot be set.
        """
        return None

    def get_gain(self) -> float:
        """The current gain in dB. The base camera states 0 dB."""
        return 0.0

    def set_gain(self, gain: float) -> None:
        """Set the gain in dB.

        A gain outside :attr:`gain_bounds` is clipped into them with a warning.

        Args:
            gain: The gain in dB.

        Raises:
            NotImplementedError: The camera states no gain range, so its gain cannot
                be set.
        """
        bounds = self.gain_bounds
        if bounds is None:
            raise NotImplementedError(
                f"{type(self).__name__} states no gain range (its gain_bounds are "
                "None), so its gain cannot be set."
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
        self._set_gain(gain)

    def _set_gain(self, gain: float) -> None:
        """Apply a gain for :meth:`set_gain`.

        Args:
            gain: The gain in dB, inside :attr:`gain_bounds`.

        Raises:
            NotImplementedError: The device states :attr:`gain_bounds` and does not
                override this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} states a gain range, and it implements no "
            "_set_gain to apply a gain."
        )

    @property
    def gain_onset_exposure(self) -> float | None:
        """The exposure in seconds from which :meth:`set_equivalent_exposure` and
        :meth:`autoexpose` raise the gain, or None to leave the gain as it is.

        Up to this exposure the gain stays at 0 dB, or at the lowest gain of the camera
        when that lies above 0 dB. The onset is ``DEFAULT_GAIN_ONSET_EXPOSURE`` until it
        is set, and it acts only on a camera whose gain can be set
        (:attr:`gain_bounds`). Setting a value that is not a positive finite number
        raises ``ValueError`` and leaves the onset as it was.
        """
        return self._gain_onset_exposure

    @gain_onset_exposure.setter
    def gain_onset_exposure(self, exposure: float | None) -> None:
        if exposure is not None:
            exposure = float(exposure)
            if not (np.isfinite(exposure) and exposure > 0.0):
                raise ValueError(
                    "gain_onset_exposure is an exposure in seconds, so it has to be a "
                    f"positive finite number, or None. Got {exposure}."
                )
        self._gain_onset_exposure = exposure

    def get_equivalent_exposure(self) -> float:
        """The equivalent exposure in seconds, the exposure at 0 dB that gives the
        counts of the current exposure and gain.

        It is ``get_exposure() * 10 ** (get_gain() / 20)``. The counts above the black
        level grow in proportion to it, so the brightness of frames taken at different
        gains is compared through it.
        """
        return float(self.get_exposure()) * linear_gain(self.get_gain())

    def set_equivalent_exposure(
        self,
        equivalent_exposure: float,
        exposure_bounds: tuple[float, float] | None = None,
    ) -> None:
        """Set the exposure and the gain that give the counts of an exposure of
        ``equivalent_exposure`` seconds at 0 dB (:meth:`get_equivalent_exposure`).

        The gain stays at its floor up to an exposure of :attr:`gain_onset_exposure`.
        The floor is 0 dB, or the lowest gain of the camera when that lies above 0 dB.
        Beyond the onset, the exposure holds there and the gain rises up to the largest
        gain of the camera. Beyond the largest gain, the exposure rises again up to the
        upper bound. The onset is clipped into the bounds. A camera whose gain cannot be
        set, or whose onset is None, keeps its gain, and its exposure alone is set.

        The gain is set first and read back, and the exposure is calculated from the
        applied gain, so the exposure takes up the step size of the gain. The exposure
        is clipped into the bounds. Each setting is written only when it differs from
        the value the camera reads back.

        Args:
            equivalent_exposure: The exposure in seconds at 0 dB to reach.
            exposure_bounds: The ``(min, max)`` exposure in seconds to stay within,
                narrowed to the camera's :attr:`exposure_bounds`, or None for
                :attr:`exposure_search_bounds`.

        Raises:
            ValueError: The bounds leave no positive exposure.
        """
        low, high = self._narrow_exposure_bounds(exposure_bounds)
        gain_range = self._gain_floor_and_ceiling
        if gain_range is not None:
            floor, ceiling = gain_range
            onset = min(max(float(self.gain_onset_exposure), low), high)
            gain = floor
            if equivalent_exposure > onset * linear_gain(floor):
                gain = min(gain_in_db(equivalent_exposure / onset), ceiling)
            if gain != float(self.get_gain()):
                self.set_gain(gain)
        exposure = equivalent_exposure / linear_gain(self.get_gain())
        exposure = min(max(exposure, low), high)
        if exposure != float(self.get_exposure()):
            self.set_exposure(exposure)

    @property
    def _gain_floor_and_ceiling(self) -> tuple[float, float] | None:
        """The lowest and the highest gain in dB that :meth:`set_equivalent_exposure`
        sets, or None when it leaves the gain as it is.

        The floor is 0 dB, raised to the lowest gain of the camera when that lies above
        0 dB. The ceiling is the largest gain of the camera. The gain is left as it is
        on a camera whose gain cannot be set, and while :attr:`gain_onset_exposure` is
        None.
        """
        bounds = self.gain_bounds
        if bounds is None or self.gain_onset_exposure is None:
            return None
        lowest, highest = float(bounds[0]), float(bounds[1])
        return min(max(0.0, lowest), highest), highest

    def _narrow_exposure_bounds(
        self, exposure_bounds: tuple[float, float] | None
    ) -> tuple[float, float]:
        """``exposure_bounds`` narrowed to the camera's own :attr:`exposure_bounds`, or
        :attr:`exposure_search_bounds` for None.

        Raises:
            ValueError: The narrowed bounds leave no positive exposure.
        """
        requested_bounds = (
            self.exposure_search_bounds if exposure_bounds is None else exposure_bounds
        )
        low, high = float(requested_bounds[0]), float(requested_bounds[1])
        stated_bounds = self.exposure_bounds
        if stated_bounds is not None:
            low = max(low, float(stated_bounds[0]))
            high = min(high, float(stated_bounds[1]))
        if not (0.0 <= low <= high and high > 0.0):
            raise ValueError(
                f"The exposure bounds {tuple(requested_bounds)} s, narrowed to the "
                f"camera's bounds {stated_bounds} s, leave no positive exposure."
            )
        return low, high

    def _set_exposure_and_gain(self, exposure: float, gain: float) -> None:
        """Set ``gain`` in dB and then ``exposure`` in seconds, each only when it
        differs from the value the camera reads back.

        The exposure is set even when setting the gain fails.
        """
        try:
            if float(self.get_gain()) != gain:
                self.set_gain(gain)
        finally:
            if float(self.get_exposure()) != exposure:
                self.set_exposure(exposure)

    def get_image(
        self,
        exposure: float | None = None,
        averaging: int = 1,
        mask: NDArray[np.bool_] | None = None,
    ) -> NDArray:
        """Capture a frame as a ``(height, width)`` array of digital counts.

        The device prepares the capture (:meth:`_prepare_capture`) after a given
        exposure is set. Whether the result is overexposed is recorded in
        :attr:`overexposed`.

        Args:
            exposure: The exposure in seconds, set before the capture when given.
            averaging: The number of fresh frames to sum. The sum is not divided, and
                :meth:`get_averaged_image` returns the mean.
            mask: True at the pixels to check for overexposure, in the shape of the
                frame, or None to check every pixel. The frame is returned whole either
                way.

        Returns:
            NDArray: One frame in the device's dtype (integer counts on hardware), or
                the float64 sum of ``averaging`` frames.

        Raises:
            ValueError: ``mask`` does not have the shape of the frame.
        """
        if exposure is not None:
            self.set_exposure(exposure)
        self._prepare_capture()
        frame_count = max(1, int(averaging))
        frame = self._get_image()
        if frame_count > 1:
            # A copy, so the sum never writes into a buffer of the device.
            frame = np.asarray(frame).astype(np.float64)
            for _ in range(frame_count - 1):
                frame += self._get_image()
        self._overexposed = self._is_overexposed(frame, frame_count, mask)
        return frame

    def _prepare_capture(self) -> None:
        """Prepare the device for the frames of one :meth:`get_image` call.

        The base camera has nothing to prepare.
        """

    @abstractmethod
    def _get_image(self) -> NDArray:
        """Capture one frame for :meth:`get_image`.

        Returns:
            NDArray: The frame in the device's dtype, reoriented by :attr:`transform`
                and cropped to :attr:`roi`.
        """

    def get_spatial_grid(
        self, device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The sensor-plane ``(x, y)`` coordinate meshgrid, in metres.

        Args:
            device: The device to build the grid on, the CPU for None.
        """
        return _spatial_grid(self.resolution, self.pixel_size, device=device)

    @contextmanager
    def preserve_roi(self, full_sensor: bool = False) -> Generator[Camera, None, None]:
        """Restore the region of interest when the block ends, whether the block
        finishes or raises.

        The region is written back only when the block changed it, since a device such
        as a GenTL camera restarts its acquisition to change the region.

        Args:
            full_sensor: Read out the whole sensor inside the block, so every frame,
                mask and region there is in whole-sensor pixels. The region is reset
                only when it does not already cover the sensor.

        Yields:
            Camera: This camera.
        """
        stored_roi = self.roi
        try:
            if full_sensor and stored_roi != ROI(0, 0, *self.sensor_resolution):
                self.set_roi(None)
            yield self
        finally:
            if self.roi != stored_roi:
                self.set_roi(stored_roi)

    @contextmanager
    def preserve_exposure_gain_and_roi(
        self, full_sensor: bool = False
    ) -> Generator[Camera, None, None]:
        """Restore the exposure, the gain and the region of interest when the block
        ends, whether the block finishes or raises.

        A setting is written back only when the block changed it. The region is
        restored first (:meth:`preserve_roi`), since a device can bound the exposure by
        its readout window. The gain and then the exposure are restored even when
        restoring the region fails, and the exposure even when restoring the gain fails.

        Args:
            full_sensor: Read out the whole sensor inside the block, as in
                :meth:`preserve_roi`.

        Yields:
            Camera: This camera.
        """
        stored_exposure = float(self.get_exposure())
        stored_gain = float(self.get_gain())
        try:
            with self.preserve_roi(full_sensor=full_sensor):
                yield self
        finally:
            self._set_exposure_and_gain(stored_exposure, stored_gain)

    def orientation_matrix(self) -> NDArray:
        """The ``(2, 3)`` pixel-space affine of this camera's frame transform.

        It maps the ``(column, row)`` of a pixel in the raw sensor frame to the
        ``(column, row)`` of the same pixel in the displayed frame. The matrix of the
        base camera is the identity.
        """
        raw_shape = np.shape(
            self.transform(np.zeros(self.sensor_resolution, dtype=np.uint8))
        )
        return dihedral_affine_matrix(self.transform, raw_shape)

    @property
    def orientation(self) -> CameraOrientation | None:
        """How the sensor is mounted, or None when the frame transform is not one of the
        eight rotate and flip orientations.
        """
        return CameraOrientation.from_matrix(
            self.orientation_matrix(), self.sensor_resolution
        )

    def set_orientation(self, orientation: CameraOrientation) -> None:
        """Mount the sensor in ``orientation``, reorienting every frame from here on.

        The shape of the raw sensor frame is worked out from the current orientation,
        and the device remounts the sensor in :meth:`_set_orientation`. The region of
        interest then returns to the whole frame, since a region is given in the
        coordinates of the previous frame. The :attr:`excluded_pixels` are converted
        into the new frame (:func:`reorient_pixels`). Everything is worked out before
        the device changes, so a failure leaves the camera as it was.
        :meth:`~hologradpy.calibration.camera_mapping.CoarseMapper.map_camera` can
        suggest an orientation.

        Args:
            orientation: The orientation to mount the sensor in.

        Raises:
            NotImplementedError: The current frame transform is not one of the eight
                orientations, or the camera cannot be reoriented.
        """
        current = self.orientation
        if current is None:
            raise NotImplementedError(
                f"{type(self).__name__} applies a frame transform that is not one of "
                "the eight orientations, so its raw sensor frame cannot be worked out. "
                "Set its transform to one of the eight first."
            )
        height, width = self.sensor_resolution
        raw_shape = (width, height) if current.swaps_axes() else (height, width)
        excluded = reorient_pixels(
            self.excluded_pixels, raw_shape, current, orientation
        )
        self._set_orientation(orientation, raw_shape)
        self.set_roi(None)
        self.excluded_pixels = excluded

    def _set_orientation(
        self, orientation: CameraOrientation, raw_shape: tuple[int, int]
    ) -> None:
        """Remount the sensor for :meth:`set_orientation`.

        A device that can be reoriented sets :attr:`transform` to
        ``orientation.transformation()``, and its :attr:`sensor_resolution` and
        :attr:`pixel_size` to those of the reoriented frame.

        Args:
            orientation: The orientation to mount the sensor in.
            raw_shape: The ``(height, width)`` of the raw sensor frame.

        Raises:
            NotImplementedError: The camera cannot be reoriented.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support being reoriented."
        )

    @property
    def excluded_pixels(self) -> list[tuple[int, int]]:
        """The ``(row, column)`` indices of the pixels of the whole frame to leave out
        of measurements, such as hot or dead pixels, as a list of pairs (empty when
        there are none).

        :meth:`autoexpose` leaves these out of its peak measurement, so a stuck pixel
        cannot rail the exposure. It is set from pairs or from an ``(N, 2)`` integer
        array, such as ``np.argwhere(dark_frame > threshold)`` for a characterization
        from a dark frame. None or an empty input clears it. Anything else raises
        ``ValueError``. The list returned is a copy, so changing it leaves the camera
        as it was.
        """
        pixels = self._excluded_pixels
        return list(pixels) if pixels is not None else []

    @excluded_pixels.setter
    def excluded_pixels(
        self, pixels: Sequence[tuple[int, int]] | NDArray[np.integer] | None
    ) -> None:
        if pixels is None:
            self._excluded_pixels = None
            return
        rows_and_columns = np.asarray(pixels, dtype=np.int64)
        if rows_and_columns.size == 0:
            self._excluded_pixels = None
            return
        if rows_and_columns.ndim != 2 or rows_and_columns.shape[1] != 2:
            raise ValueError(
                "excluded_pixels takes (row, column) index pairs or an (N, 2) integer "
                f"array, got an array of shape {rows_and_columns.shape}."
            )
        self._excluded_pixels = [
            (int(row), int(col)) for row, col in rows_and_columns.tolist()
        ]

    @property
    def overexposed(self) -> bool:
        """Whether the last frame from :meth:`get_image` was overexposed.

        A pixel is overexposed when it reads :attr:`saturation_level` or more. Only the
        pixels where the ``mask`` of that capture is True are checked, and the
        :attr:`excluded_pixels` are left out. Set by every capture. False before the
        first capture.
        """
        return bool(self._overexposed)

    def _is_overexposed(
        self,
        frame: NDArray,
        averaging: int,
        mask: NDArray[np.bool_] | None = None,
    ) -> bool:
        """Whether ``frame`` holds an overexposed pixel.

        ``frame`` is the sum of ``averaging`` frames. A pixel is overexposed when its
        mean over them reaches :attr:`saturation_level`. Only the pixels where ``mask``
        is True are checked, and the excluded pixels are left out.

        Raises:
            ValueError: ``mask`` does not have the shape of ``frame``.
        """
        threshold = averaging * self.saturation_level
        at_full_scale = np.asarray(frame) >= threshold
        if mask is not None:
            if tuple(np.shape(mask)) != tuple(at_full_scale.shape):
                raise ValueError(
                    f"The mask has shape {tuple(np.shape(mask))}, and the frame is "
                    f"{tuple(at_full_scale.shape)}."
                )
            at_full_scale &= np.asarray(mask, dtype=bool)
        # The frame is cropped to the region of interest.
        at_full_scale &= ~self.excluded_mask(self.roi)
        return bool(at_full_scale.any())

    def excluded_mask(self, region: ROI) -> NDArray[np.bool_]:
        """True at the :attr:`excluded_pixels` inside ``region`` of the whole frame, in
        the shape of the region.

        Args:
            region: The region of the whole frame that the mask covers.
        """
        excluded = np.asarray(self.excluded_pixels, dtype=np.int64).reshape(-1, 2)
        rows = excluded[:, 0] - region.top_row
        columns = excluded[:, 1] - region.left_column
        inside = (
            (rows >= 0)
            & (rows < region.height)
            & (columns >= 0)
            & (columns < region.width)
        )
        mask = np.zeros((region.height, region.width), dtype=bool)
        mask[rows[inside], columns[inside]] = True
        return mask

    def find_stuck_pixels(
        self,
        *,
        exposures: list[float] | None = None,
        steps: int = 7,
        lower_threshold: float = 0.1,
        tolerance: float = STUCK_PIXEL_RESPONSE_TOLERANCE,
        blob_min_size: int = 4,
    ) -> list[tuple[int, int]]:
        """Capture an exposure sweep and flag the sensor's hot and dead pixels, storing
        them in :attr:`excluded_pixels`.

        The sweep is :meth:`_capture_exposure_sweep`, and the analysis is
        :func:`~hologradpy.hardware.camera.stuck_pixels.stuck_pixel_mask`.
        :meth:`autoexpose` can run the same analysis on the frames of its own search,
        through its ``detect_stuck_pixels`` flag.

        Args:
            exposures: The exposures in seconds to capture at, or None for ``steps``
                exposures a half decade apart.
            steps: The number of exposures in the default sweep.
            lower_threshold: The fraction of full scale above which a pixel carries
                signal.
            tolerance: The relative tolerance of the response test.
            blob_min_size: A region saturated in every frame counts as overexposure
                when it holds at least this many pixels.

        Returns:
            list[tuple[int, int]]: The ``(row, col)`` stuck pixels of the whole frame.

        Raises:
            ValueError: Fewer than two exposures lie within
                :attr:`exposure_search_bounds`, or fewer than two of them lie far
                enough apart to compare.
        """
        frames, exposures = self._capture_exposure_sweep(exposures, steps=steps)
        stuck = stuck_pixel_mask(
            frames,
            exposures,
            self.max_pixel_value,
            self.saturation_level,
            lower_threshold=lower_threshold,
            tolerance=tolerance,
            blob_min_size=blob_min_size,
        )
        self.excluded_pixels = np.argwhere(stuck)
        return self.excluded_pixels

    def _capture_exposure_sweep(
        self, exposures: list[float] | None = None, *, steps: int = 7
    ) -> tuple[NDArray, list[float]]:
        """Capture whole frames across an exposure sweep from short to long, at the
        current gain.

        The whole sensor is read out for the sweep, and the region of interest and the
        exposure are put back afterwards (:meth:`preserve_exposure_gain_and_roi`).

        Args:
            exposures: The exposures in seconds to capture at. None gives ``steps``
                exposures a half decade apart, from the lower search bound, or from
                100 us when that bound is zero. An exposure outside
                :attr:`exposure_search_bounds` is dropped, so the sweep keeps its
                spacing.
            steps: The number of exposures in the default sweep.

        Returns:
            tuple[NDArray, list[float]]: The frames stacked as ``(n, height, width)``,
            and the read-back exposure of each frame.

        Raises:
            ValueError: Fewer than two exposures lie within
                :attr:`exposure_search_bounds`.
        """
        low, high = self.exposure_search_bounds
        if exposures is None:
            base = low if low > 0 else 100e-6
            exposures = [base * 10.0 ** (step / 2) for step in range(steps)]

        exposures = [
            float(exposure) for exposure in exposures if low <= exposure <= high
        ]
        if len(exposures) < 2:
            raise ValueError(
                "the exposure sweep needs at least two exposures within the bounds "
                f"{(low, high)}. Got {exposures}."
            )

        frames = []
        applied_exposures = []
        with self.preserve_exposure_gain_and_roi(full_sensor=True):
            for exposure in exposures:
                self.set_exposure(exposure)
                applied_exposures.append(float(self.get_exposure()))
                frames.append(np.asarray(self.get_image(), dtype=float))
        return np.stack(frames), applied_exposures

    def autoexpose(
        self,
        *,
        set_fraction: float = 0.5,
        tolerance: float = 0.05,
        roi: ROI | None = None,
        mask: NDArray[np.bool_] | None = None,
        exposure_bounds: tuple[float, float] | None = None,
        overexposed_factor: float = 0.01,
        raise_on_rail: bool = True,
        max_iterations: int = 5,
        detect_stuck_pixels: bool = False,
        verbose: bool = False,
    ) -> float:
        """Set the exposure and the gain so the peak of the measured region sits at
        ``set_fraction`` of full scale, and return the exposure.

        The search runs on the equivalent exposure, the exposure at 0 dB that gives the
        same counts (:meth:`get_equivalent_exposure`). Each step is set through
        :meth:`set_equivalent_exposure`, so the gain stays at 0 dB up to an exposure of
        :attr:`gain_onset_exposure` and rises beyond it. A camera whose gain cannot be
        set keeps its gain, and the search runs on its exposure. Every exposure in the
        rest of this description is an equivalent exposure.

        The region is the whole frame, cropped to ``roi`` and reduced to the pixels
        where ``mask`` is True, and the camera's :attr:`excluded_pixels` are always left
        out. The whole sensor is read out for the search, and the region of interest is
        put back afterwards (:meth:`preserve_roi`). An exception from the search also
        puts the exposure and the gain back where the search started.

        The camera's exposure stays within ``exposure_bounds``, narrowed to its own
        :attr:`exposure_bounds`. The search therefore reaches from the lower bound at
        the gain floor to the upper bound at the largest gain. It starts from the
        current exposure clipped into this range, or from the lower end of the range
        when the camera sits at zero, or from the upper end when the lower end is zero
        too. Each step sets an exposure, reads back the applied exposure, and measures
        one frame, so the search measures at most ``1 + max_iterations`` frames. A peak
        at :attr:`saturation_level` or above is overexposed, and cuts the exposure by
        ``overexposed_factor`` until a frame peaks below it. From then on, an
        overexposed peak steps to the geometric mean of the longest exposure that was
        not overexposed and the shortest overexposed one, when the first is the
        shorter. A peak below saturation scales the exposure to the target, up to
        ``set_fraction`` times the shortest exposure that overexposed the region. The
        target lies at or below this cap for a linear sensor.

        A step cut short by the bounds lands on the bound and measures it. The search
        has railed when its next step asks to pass an already measured bound, when the
        camera reads back zero, or when the camera repeats an exposure while every frame
        so far has been overexposed. A hidden limit is one that the camera does not
        state. The search rails at a hidden floor, and at a hidden ceiling reached
        through a request at a bound. A hidden ceiling reached through a request inside
        the bounds ends the search with no finer exposure step to take.

        The search settles on a measured exposure when it does not converge. It picks
        the exposure whose peak lies below full scale and closest to the target, or the
        shortest exposure when every frame was overexposed. The exposure and the gain of
        that frame are set again and read back, and one more frame is measured when the
        camera lands on an unmeasured exposure. A warning then reports the final
        exposure, the gain and the peak of its frame.

        The frames of the search are analysed for stuck pixels
        (:meth:`find_stuck_pixels`) when ``detect_stuck_pixels`` is set and the search
        did not rail. The analysis compares the frames by their equivalent exposures,
        and fills :attr:`excluded_pixels` without a second sweep. The detection is
        skipped with a warning when the frames span fewer than two exposures far enough
        apart to compare.

        Args:
            set_fraction: The target peak as a fraction of :attr:`max_pixel_value`.
            tolerance: The largest accepted distance between the peak and the target,
                as a fraction of full scale.
            roi: The region of the whole frame to measure, or None for the whole frame.
                Without a mask, a region reaching off the frame is trimmed to it.
            mask: True at the pixels of the region to measure, in the region's shape.
                For example, a mask can select everything outside a zeroth order. None
                measures every pixel.
            exposure_bounds: The ``(min, max)`` exposure in seconds that the camera is
                set within, or None for :attr:`exposure_search_bounds`. A maximum above
                ``DEFAULT_MAX_EXPOSURE`` is reached only through these bounds. The gain
                onset is clipped into them.
            overexposed_factor: The factor that multiplies the exposure after an
                overexposed peak, until a frame peaks below full scale.
            raise_on_rail: Whether a rail raises ``RuntimeError``. Otherwise the search
                settles and warns.
            max_iterations: The number of exposure steps after the first frame.
            detect_stuck_pixels: Whether to find stuck pixels in the frames of the
                search.
            verbose: Whether to print the exposure and the peak of every frame, and
                the gain when the search sets it.

        Returns:
            float: The final exposure in seconds, as read back from the camera. The
            gain it was reached with is :meth:`get_gain`.

        Raises:
            ValueError: The bounds leave no positive exposure, ``set_fraction`` is not
                below ``SATURATION_FRACTION``, or the region, the mask or an excluded
                pixel does not fit the frame. Nothing is captured then.
            RuntimeError: The search railed and ``raise_on_rail`` is set.
        """
        low, high = self._narrow_exposure_bounds(exposure_bounds)
        if not 0.0 < set_fraction < SATURATION_FRACTION:
            raise ValueError(
                f"The target set_fraction {set_fraction} has to lie between 0 and "
                f"the saturation fraction {SATURATION_FRACTION}."
            )

        full_scale = self.max_pixel_value
        search = _ExposureSearch(
            set_fraction=set_fraction,
            set_value=set_fraction * full_scale,
            saturation=self.saturation_level,
        )
        stored_exposure = float(self.get_exposure())
        stored_gain = float(self.get_gain())
        # The gain moves between its floor and its ceiling, or stays where it is.
        gain_range = self._gain_floor_and_ceiling
        if gain_range is None:
            floor_factor = ceiling_factor = linear_gain(stored_gain)
        else:
            floor_factor, ceiling_factor = (linear_gain(gain) for gain in gain_range)
        # Every exposure of the search is an equivalent exposure, the exposure at 0 dB
        # that gives the same counts. The search reaches from the lower bound at the
        # gain floor to the upper bound at the gain ceiling.
        lowest, highest = low * floor_factor, high * ceiling_factor
        finished = False

        try:
            with self.preserve_roi(full_sensor=True):
                region, keep = self._select_measured_region(roi, mask)

                def measure(requested: float | None) -> tuple[float, float, bool]:
                    """Set ``requested`` when given, then measure one fresh frame.

                    Returns the exposure the camera applied, the peak of the region,
                    and whether that exposure was measured before.
                    """
                    if requested is not None:
                        self.set_equivalent_exposure(requested, (low, high))
                    applied_exposure = float(self.get_exposure())
                    applied_gain = float(self.get_gain())
                    applied = applied_exposure * linear_gain(applied_gain)
                    image = self.get_image()
                    repeated = applied in search.measurements
                    frame = None
                    if detect_stuck_pixels and applied > 0.0 and not repeated:
                        frame = np.asarray(image, dtype=float)
                    pixels = np.where(keep, region.crop(image), 0)
                    peak = float(np.amax(pixels))
                    search.record(applied, peak, applied_exposure, applied_gain, frame)
                    if verbose:
                        gain_text = ""
                        if gain_range is not None:
                            gain_text = f"gain = {applied_gain:.2f} dB, "
                        print(
                            f"Autoexposure: exposure = {applied_exposure:<.3e} s, "
                            f"{gain_text}peak = {peak:.0f}/{full_scale}."
                        )
                    return applied, peak, repeated

                stored_equivalent = stored_exposure * linear_gain(stored_gain)
                start = float(np.clip(stored_equivalent, lowest, highest))
                if start <= 0.0:
                    # A zero start after the clip means the lower bound is zero.
                    start = highest
                # The first frame is taken at the exposure and the gain set for the
                # start, which writes only the settings that differ.
                exposure, peak, _ = measure(start)
                # The bound of the last request, or None for a request inside the
                # bounds. A step asking past this bound again has railed.
                measured_bound = start if start in (lowest, highest) else None

                steps = 0
                while True:
                    if (
                        peak < search.saturation
                        and abs(peak - search.set_value) <= tolerance * full_scale
                    ):
                        outcome = "converged"
                        break
                    desired = _next_exposure(search, exposure, peak, overexposed_factor)
                    requested = float(np.clip(desired, lowest, highest))
                    if requested != desired and requested == measured_bound:
                        outcome = "rail"
                        break
                    if steps >= max_iterations:
                        outcome = "budget"
                        break
                    steps += 1
                    exposure, peak, repeated = measure(requested)
                    measured_bound = (
                        requested if requested in (lowest, highest) else None
                    )
                    if exposure <= 0.0 or repeated:
                        railed = (
                            exposure <= 0.0
                            or requested != desired
                            or search.every_frame_overexposed
                        )
                        outcome = "rail" if railed else "no finer step"
                        break

                if outcome == "rail" and raise_on_rail:
                    raise RuntimeError(
                        "autoexposure has railed at "
                        f"{self._describe_exposure_and_gain(gain_range)}: the region "
                        f"peaks at {peak:.0f} of {full_scale} there, and the target "
                        "lies beyond "
                        f"{self._describe_search_bounds(low, high, gain_range)}."
                    )

                if outcome != "converged":
                    # Settle on the measured exposure closest to the target, and report
                    # a frame taken at the applied exposure.
                    settled = _settled_exposure(search)
                    if settled != exposure:
                        measured = search.measurements[settled]
                        self._set_exposure_and_gain(measured.exposure, measured.gain)
                        exposure = self.get_equivalent_exposure()
                        if exposure in search.measurements:
                            peak = search.measurements[exposure].peak
                        else:
                            exposure, peak, _ = measure(None)
            finished = True
        finally:
            if not finished:
                self._set_exposure_and_gain(stored_exposure, stored_gain)

        if outcome != "converged":
            reason = {
                "rail": "the search railed against "
                + self._describe_search_bounds(low, high, gain_range),
                "budget": f"the budget of {max_iterations} exposure steps ran out",
                "no finer step": "the camera has no finer exposure step to take",
            }[outcome]
            warnings.warn(
                f"Autoexposure did not reach its target: {reason}. The region peaks at "
                f"{peak:.0f} of {full_scale} ({peak / full_scale:.1%}) against a "
                f"target of {set_fraction:.0%}, at "
                f"{self._describe_exposure_and_gain(gain_range)}. The frames that "
                "follow are exposed as reported here, not as asked for.",
                stacklevel=2,
            )

        if detect_stuck_pixels and outcome != "rail":
            frames, frame_exposures = search.recorded_frames
            try:
                stuck = stuck_pixel_mask(
                    frames, frame_exposures, self.max_pixel_value, self.saturation_level
                )
            except ValueError:
                warnings.warn(
                    "Stuck pixels were not detected: autoexposure took frames at fewer "
                    "than two exposures far enough apart to compare. find_stuck_pixels "
                    "captures a sweep of its own.",
                    stacklevel=2,
                )
            else:
                self.excluded_pixels = np.argwhere(stuck)

        return float(self.get_exposure())

    def _describe_exposure_and_gain(
        self, gain_range: tuple[float, float] | None
    ) -> str:
        """The current exposure for a message, with the gain when the search sets it."""
        exposure = f"an exposure of {float(self.get_exposure()):.3e} s"
        if gain_range is None:
            return exposure
        return f"{exposure} and a gain of {float(self.get_gain()):.2f} dB"

    @staticmethod
    def _describe_search_bounds(
        low: float, high: float, gain_range: tuple[float, float] | None
    ) -> str:
        """The exposure bounds of a search for a message, with the gain range when the
        search sets the gain.
        """
        bounds = f"the exposure bounds {(low, high)} s"
        if gain_range is None:
            return bounds
        return f"{bounds} and the gain range {gain_range} dB"

    def _select_measured_region(
        self, roi: ROI | None, mask: NDArray[np.bool_] | None
    ) -> tuple[ROI, NDArray[np.bool_]]:
        """The measured region of the whole frame for :meth:`autoexpose`, and the
        pixels to keep.

        It runs with the whole sensor read out. A ``roi`` without a mask is trimmed to
        the frame (:meth:`~hologradpy.roi.ROI.trimmed_to`), so a window centred near an
        edge measures the part of it on the frame. A ``roi`` with a mask has to lie on
        the frame, since the mask is given in the shape of the whole region.

        Args:
            roi: The region of the whole frame to measure, or None for the whole frame.
            mask: True at the pixels of the region to measure, in the region's shape,
                or None to measure every pixel.

        Returns:
            tuple[ROI, NDArray[np.bool_]]: The region, and the pixels of it to measure,
            which leave out :attr:`excluded_pixels`.

        Raises:
            ValueError: No part of ``roi`` lies on the frame, ``roi`` reaches off the
                frame while a mask is given, the mask is not the region's shape, an
                excluded pixel lies off the frame, or no pixel is left to measure.
        """
        height, width = self.resolution
        if roi is None:
            region = ROI(0, 0, height, width)
        else:
            if mask is not None and not roi.lies_inside((height, width)):
                raise ValueError(
                    f"{roi} reaches off the {height} x {width} frame, and a mask is "
                    "given in its shape, so the region cannot be trimmed to the frame."
                )
            region = roi.trimmed_to((height, width))

        if mask is not None and np.shape(mask) != (region.height, region.width):
            raise ValueError(
                f"The mask has shape {np.shape(mask)}, and the region it selects from "
                f"is {(region.height, region.width)}."
            )

        excluded = np.asarray(self.excluded_pixels, dtype=np.int64).reshape(-1, 2)
        if len(excluded) > 0:
            off_frame = (
                (excluded < 0).any(axis=1)
                | (excluded[:, 0] >= height)
                | (excluded[:, 1] >= width)
            )
            if off_frame.any():
                listed = [
                    (int(row), int(column))
                    for row, column in excluded[off_frame][:5].tolist()
                ]
                raise ValueError(
                    f"The excluded pixels {listed} lie off the {height} x {width} "
                    "frame. excluded_pixels names (row, col) pixels of the whole frame "
                    "the camera returns."
                )
        keep = ~self.excluded_mask(region)
        if mask is not None:
            keep &= np.asarray(mask, dtype=bool)
        if not keep.any():
            raise ValueError(
                f"The mask and the excluded pixels leave no pixel of {region} to "
                "measure."
            )
        return region, keep

    def get_averaged_image(
        self, exposure: float | None = None, averaging: int = 1
    ) -> NDArray:
        """The mean of ``averaging`` fresh frames as a float array.

        :meth:`get_image` returns the sum of the frames, and this method divides it by
        ``averaging``. The mean cuts the noise variance by a factor ``averaging`` for a
        camera that draws fresh noise for every frame.

        Args:
            exposure: The exposure in seconds, set before the capture when given.
            averaging: The number of frames to average.

        Returns:
            NDArray: The mean frame in float64.
        """
        frames = max(1, int(averaging))
        summed = np.asarray(self.get_image(exposure, averaging=frames), dtype=float)
        return summed / frames

    def close(self) -> None:
        """Release the device, which a ``with`` block does when it ends.

        The base camera holds nothing to release. An override is safe to call more than
        once.
        """

    def __enter__(self) -> Camera:
        return self

    def __exit__(self, *_) -> None:
        self.close()


@record_type("camera_orientation")
@dataclass(frozen=True)
class CameraOrientation:
    """How a sensor is mounted, as the rotate and flip flags a device takes.

    The flags carry the raw frame to the displayed frame, with the rotation before the
    flips. Each quarter turn of ``rot`` is one :func:`numpy.rot90`, which turns the
    displayed image counterclockwise. A camera that shows the focal plane unturned in
    its raw frame therefore reads a :attr:`~.CameraMapping.camera_angle` of 90 once it
    is mounted with ``rot="90"``. ``fliplr`` reverses the columns and ``flipud`` the
    rows.

    A turn or a flip reverses the pixel array about its middle, ``((width - 1) / 2,
    (height - 1) / 2)``. The centre pixel, :func:`~hologradpy.grids.plane_center`, lies
    half a pixel past the middle along an axis of even length. The pixel at the centre
    of the raw frame therefore lands one pixel from the centre pixel of the displayed
    frame along each reversed axis of even length, and the
    :attr:`~.CameraMapping.camera_shift` of a mapping moves by one pitch along it.
    """

    rot: str = "0"
    fliplr: bool = False
    flipud: bool = False

    def transformation(self) -> Callable[[NDArray], NDArray]:
        """The transform a camera in this orientation applies to its raw frames."""
        return dihedral_array_transform(self.rot, self.fliplr, self.flipud)

    def matrix(self, shape: tuple[int, int]) -> NDArray:
        """The ``(2, 3)`` pixel-space affine of :meth:`transformation`.

        Args:
            shape: The ``(height, width)`` of the raw frame, which sets the
                translation.

        Returns:
            NDArray: The matrix mapping the ``(column, row, 1)`` of a raw pixel to its
            ``(column, row)`` in the transformed frame.
        """
        return dihedral_affine_matrix(self.transformation(), shape)

    def swaps_axes(self) -> bool:
        """True when the rotation exchanges height and width."""
        return self.rot in ("90", "270", 1, 3)

    def compose(self, other: CameraOrientation) -> CameraOrientation:
        """The orientation that applies ``other`` first and then this one.

        Args:
            other: The orientation applied first.

        Returns:
            CameraOrientation: The combined orientation, one of the eight.
        """

        def combined(image: NDArray) -> NDArray:
            return self.transformation()(other.transformation()(image))

        # Non-square, so no two of the eight probe to the same matrix.
        shape = (3, 5)
        return CameraOrientation.from_matrix(
            dihedral_affine_matrix(combined, shape), shape
        )

    @classmethod
    def dihedral(cls) -> list[CameraOrientation]:
        """The eight orientations a sensor can be mounted in."""
        return [
            cls(rot, fliplr, False)
            for rot in ("0", "90", "180", "270")
            for fliplr in (False, True)
        ]

    @classmethod
    def from_matrix(
        cls, matrix: NDArray, shape: tuple[int, int]
    ) -> CameraOrientation | None:
        """The orientation whose :meth:`matrix` has the linear part of ``matrix``.

        Args:
            matrix: A ``(2, 3)`` pixel-space affine.
            shape: The ``(height, width)`` for probing the orientations, at least 2 in
                each dimension.

        Returns:
            CameraOrientation | None: The orientation, or None when the linear part is
                not one of the eight.
        """
        target = np.asarray(matrix, dtype=np.float64)
        for orientation in cls.dihedral():
            if np.allclose(orientation.matrix(shape)[:, :2], target[:, :2]):
                return orientation
        return None


def reorient_pixels(
    pixels: Sequence[tuple[int, int]] | NDArray,
    raw_shape: tuple[int, int],
    source: CameraOrientation,
    target: CameraOrientation,
) -> list[tuple[int, int]]:
    """Convert the positions of sensor pixels in the displayed frame from one camera
    orientation to another.

    A camera names a sensor pixel by its position in the displayed frame, and that
    position depends on the orientation. A pixel list such as
    :attr:`Camera.excluded_pixels` is therefore converted whenever the orientation
    changes. Each position is first mapped back to the raw sensor frame with the inverse
    of the ``source`` affine, :meth:`CameraOrientation.matrix` for ``raw_shape``. It is
    then mapped forward into the displayed frame with the ``target`` affine.

    Args:
        pixels: The ``(row, col)`` positions in the frame displayed in the ``source``
            orientation, as pairs or as an ``(N, 2)`` array.
        raw_shape: The ``(height, width)`` of the raw sensor frame, before any rotation
            or flip.
        source: The orientation in which the positions are given.
        target: The orientation to convert them to.

    Returns:
        list[tuple[int, int]]: The ``(row, col)`` positions in the frame displayed in
        the ``target`` orientation, in the same order as ``pixels``.
    """
    points = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    if points.shape[0] == 0:
        return []
    source_matrix = source.matrix(raw_shape)
    target_matrix = target.matrix(raw_shape)
    # The matrices act on (column, row) positions.
    shown_xy = points[:, ::-1]
    sensor_xy = np.linalg.solve(
        source_matrix[:, :2], (shown_xy - source_matrix[:, 2]).T
    ).T
    target_xy = sensor_xy @ target_matrix[:, :2].T + target_matrix[:, 2]
    target_rows_cols = np.rint(target_xy[:, ::-1]).astype(np.int64)
    return [(int(row), int(col)) for row, col in target_rows_cols]


@record_type("camera_data")
@dataclass(frozen=True, unsafe_hash=True)
class CameraData(SaveableRecord):
    """A native snapshot of a camera's geometry, exposure and gain.

    The gain is in dB.
    """

    name: str
    sensor_resolution: tuple[int, int]
    pixel_size: tuple[float, float]
    adu_levels: int
    exposure: float
    exposure_bounds: tuple[float, float] | None
    roi: ROI
    orientation: NDArray = field(compare=False, hash=False)
    gain: float = 0.0

    @property
    def resolution(self) -> tuple[int, int]:
        """The shape of a frame this camera returns, which is the region of interest."""
        return (self.roi.height, self.roi.width)

    @property
    def equivalent_exposure(self) -> float:
        """The exposure in seconds at 0 dB that gives the counts of the recorded
        exposure and gain (:meth:`Camera.get_equivalent_exposure`).
        """
        return float(self.exposure) * linear_gain(self.gain)

    @property
    def orientation_flags(self) -> CameraOrientation | None:
        """:attr:`orientation` as the rotate and flip flags a device takes, or None."""
        return CameraOrientation.from_matrix(self.orientation, self.sensor_resolution)

    @classmethod
    def from_camera(cls, camera: Camera) -> CameraData:
        """Record a camera as it stands.

        Args:
            camera: The camera, as :func:`~hologradpy.hardware.factory.open_camera` or
                :func:`~hologradpy.hardware.as_native.as_camera` returns it.
        """
        return cls(
            name=getattr(camera, "name", ""),
            sensor_resolution=camera.sensor_resolution,
            pixel_size=tuple(float(v) for v in camera.pixel_size),
            adu_levels=camera.adu_levels,
            exposure=camera.get_exposure(),
            exposure_bounds=camera.exposure_bounds,
            roi=camera.roi,
            orientation=camera.orientation_matrix(),
            gain=float(camera.get_gain()),
        )
