"""The native camera template.

A device subclasses the ``Camera`` base class. The module also defines the
``CameraOrientation`` and ``CameraData`` records and the ``reorient_pixels`` helper.
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
from scipy.ndimage import binary_erosion, label

from ...geometry import dihedral_affine_matrix, dihedral_array_transform
from ...grids import get_spatial_grid as _spatial_grid
from ...roi import ROI
from ...serialization import SaveableRecord, record_type


# The longest exposure in seconds that an exposure search reaches unless its caller
# passes longer bounds.
DEFAULT_MAX_EXPOSURE = 1.0

# A pixel counts as overexposed from this fraction of the largest count. Some sensors
# clip a count or two short of it, such as a 10-bit Thorlabs Zelux at 1022 of 1023.
SATURATION_FRACTION = 0.99

# The relative tolerance of the response test in stuck-pixel detection. A working
# pixel rises by the ratio of two exposures within this tolerance.
STUCK_PIXEL_RESPONSE_TOLERANCE = 0.2


def _select_distinct_exposures(
    exposures: NDArray[np.float64], tolerance: float
) -> NDArray[np.intp]:
    """The indices of the frames that stuck-pixel detection can compare, in increasing
    order of exposure.

    The shortest exposure is kept first. Each further exposure is kept when it exceeds
    the last kept exposure by more than a factor ``1 / (1 - tolerance)``. The response
    test expects a working pixel to rise by the ratio of two exposures, within
    ``tolerance``. A pixel that does not respond at all also passes this test on a pair
    of exposures closer than that factor.

    Args:
        exposures: The exposure of each frame in seconds.
        tolerance: The relative tolerance of the response test.

    Returns:
        NDArray[np.intp]: Indices into ``exposures``, ordered by increasing exposure.
    """
    exposures = np.asarray(exposures, dtype=np.float64)
    kept: list[int] = []
    for index in np.argsort(exposures, kind="stable"):
        if not kept or exposures[index] * (1.0 - tolerance) > exposures[kept[-1]]:
            kept.append(int(index))
    return np.asarray(kept, dtype=np.intp)


class Camera(ABC):
    """A HoloGradPy-native camera: SI units, ``(y, x)`` geometry, ``(row, col)`` ROI.

    A device implements the geometry / exposure / capture abstract members below. The
    ``get_spatial_grid`` and ``autoexpose`` template methods are provided here, so every
    camera shares them. A device captures a frame in :meth:`_get_image`.
    :meth:`get_image` calls it and checks the frame for overexposure. Third-party
    devices subclass this base or register a wrapper with
    :func:`hologradpy.hardware.as_native.as_camera`.
    """

    _excluded_pixels: list[tuple[int, int]] | None = None
    _overexposed: bool = False

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

    @abstractmethod
    def set_roi(self, roi: ROI | None) -> None:
        """Set the region of interest (``None`` resets to the full sensor)."""

    @abstractmethod
    def get_exposure(self) -> float:
        """The current exposure time in seconds."""

    @abstractmethod
    def set_exposure(self, exposure_s: float) -> None:
        """Set the exposure time in seconds."""

    def get_image(
        self,
        exposure: float | None = None,
        averaging: int = 1,
        mask: NDArray[np.bool_] | None = None,
    ) -> NDArray:
        """Capture a frame as a ``(height, width)`` array of digital counts.

        The device captures the frame in :meth:`_get_image`. The frame is then checked
        for overexposure at the pixels where ``mask`` is True, and the result is
        recorded in :attr:`overexposed`.

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
        frame = self._get_image(exposure, averaging)
        self._overexposed = self._is_overexposed(frame, averaging, mask)
        return frame

    @abstractmethod
    def _get_image(
        self, exposure: float | None = None, averaging: int = 1
    ) -> NDArray:
        """Capture a frame for :meth:`get_image`.

        Args:
            exposure: The exposure in seconds, set before the capture when given.
            averaging: The number of fresh frames to sum.

        Returns:
            NDArray: One frame in the device's dtype, or the float64 sum of
                ``averaging`` frames, cropped to :attr:`roi`.
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
    def preserve_exposure_and_roi(
        self, full_sensor: bool = False
    ) -> Generator[Camera, None, None]:
        """Restore the exposure and the region of interest when the block ends, whether
        the block finishes or raises.

        A setting is written back only when the block changed it. The region is
        restored first (:meth:`preserve_roi`), since a device can bound the exposure by
        its readout window. The exposure is restored even when restoring the region
        fails.

        Args:
            full_sensor: Read out the whole sensor inside the block, as in
                :meth:`preserve_roi`.

        Yields:
            Camera: This camera.
        """
        stored_exposure = float(self.get_exposure())
        try:
            with self.preserve_roi(full_sensor=full_sensor):
                yield self
        finally:
            if float(self.get_exposure()) != stored_exposure:
                self.set_exposure(stored_exposure)

    def orientation_matrix(self) -> NDArray:
        """The ``(2, 3)`` pixel-space affine of this camera's frame transform.

        It maps the ``(x, y)`` of a pixel in the raw sensor frame to the ``(x, y)`` of
        the same pixel in the displayed frame. A camera without a ``transform`` is
        axis-aligned, so its matrix is the identity.
        """
        transform = getattr(self, "transform", None)
        if transform is None:
            return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        raw_shape = np.shape(
            transform(np.zeros(self.sensor_resolution, dtype=np.uint8))
        )
        return dihedral_affine_matrix(transform, raw_shape)

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

        The region of interest then returns to the whole frame, and
        :attr:`excluded_pixels` follow the sensor into the new frame
        (:func:`reorient_pixels`). A device that can be reoriented overrides this
        method. :meth:`~hologradpy.calibration.camera_mapping.CoarseMapper.map_camera`
        can suggest an orientation, and this method applies it.

        Args:
            orientation: The orientation to mount the sensor in.

        Raises:
            NotImplementedError: The camera cannot be reoriented.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support being reoriented."
        )

    @property
    def excluded_pixels(self) -> list[tuple[int, int]]:
        """The ``(row, col)`` pixels of the whole frame to leave out of measurements,
        such as hot or dead pixels, as a list of pairs (empty when there are none).

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
                "excluded_pixels takes (row, col) pairs or an (N, 2) integer array, "
                f"got an array of shape {rows_and_columns.shape}."
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
        frame: NDArray | torch.Tensor,
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
        threshold = max(1, int(averaging)) * self.saturation_level
        if isinstance(frame, torch.Tensor):
            at_full_scale = frame.detach() >= threshold
        else:
            at_full_scale = np.asarray(frame) >= threshold
        if mask is not None:
            if tuple(np.shape(mask)) != tuple(at_full_scale.shape):
                raise ValueError(
                    f"The mask has shape {tuple(np.shape(mask))}, and the frame is "
                    f"{tuple(at_full_scale.shape)}."
                )
            kept = (
                torch.as_tensor(mask, dtype=torch.bool, device=at_full_scale.device)
                if isinstance(at_full_scale, torch.Tensor)
                else np.asarray(mask, dtype=bool)
            )
            at_full_scale &= kept
        if self._excluded_pixels:
            # Excluded pixels name whole-frame pixels, and the frame is cropped to the
            # region of interest.
            top, left = self.roi.top_row, self.roi.left_column
            height, width = at_full_scale.shape[-2:]
            for row, col in self._excluded_pixels:
                if 0 <= row - top < height and 0 <= col - left < width:
                    at_full_scale[..., row - top, col - left] = False
        return bool(at_full_scale.any())

    def find_stuck_pixels(
        self,
        *,
        exposures: list[float] | None = None,
        steps: int = 7,
        lower_threshold: float = 0.1,
        tolerance: float = STUCK_PIXEL_RESPONSE_TOLERANCE,
        blob_min_size: int = 4,
        verbose: bool = False,
    ) -> list[tuple[int, int]]:
        """Capture an exposure sweep and flag the sensor's hot and dead pixels, storing
        them in :attr:`excluded_pixels`.

        This characterizes the sensor in one call, independent of :meth:`autoexpose`.
        :meth:`autoexpose` can run the same analysis on the frames of its own search,
        through its ``detect_stuck_pixels`` flag. The sweep is
        :meth:`_capture_exposure_sweep`, and the analysis is
        :meth:`_detect_stuck_pixels`.

        Args:
            exposures: The exposures in seconds to capture at, or None for ``steps``
                exposures a half decade apart.
            steps: The number of exposures in the default sweep.
            lower_threshold: The fraction of full scale above which a pixel carries
                signal.
            tolerance: The relative tolerance of the response test.
            blob_min_size: A region saturated in every frame counts as overexposure
                when it holds at least this many pixels.
            verbose: Whether to print how many stuck pixels were found and where.

        Returns:
            list[tuple[int, int]]: The ``(row, col)`` stuck pixels of the whole frame.

        Raises:
            ValueError: Fewer than two exposures lie within
                :attr:`exposure_search_bounds`.
        """
        frames, exposures = self._capture_exposure_sweep(exposures, steps=steps)
        return self._detect_stuck_pixels(
            frames,
            exposures,
            lower_threshold=lower_threshold,
            tolerance=tolerance,
            blob_min_size=blob_min_size,
            verbose=verbose,
        )

    def _capture_exposure_sweep(
        self, exposures: list[float] | None = None, *, steps: int = 7
    ) -> tuple[NDArray, list[float]]:
        """Capture whole frames across an exposure sweep from short to long.

        The whole sensor is read out for the sweep, and the region of interest and the
        exposure are put back afterwards (:meth:`preserve_exposure_and_roi`).

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
        with self.preserve_exposure_and_roi(full_sensor=True):
            for exposure in exposures:
                self.set_exposure(exposure)
                applied_exposures.append(float(self.get_exposure()))
                frames.append(np.asarray(self.get_image(), dtype=float))
        return np.stack(frames), applied_exposures

    def _detect_stuck_pixels(
        self,
        frames: NDArray,
        exposures: list[float],
        *,
        lower_threshold: float = 0.1,
        tolerance: float = STUCK_PIXEL_RESPONSE_TOLERANCE,
        blob_min_size: int = 4,
        verbose: bool = False,
    ) -> list[tuple[int, int]]:
        """Find hot and dead pixels from frames captured at different exposures, and
        store them in :attr:`excluded_pixels`.

        A working pixel scales with the exposure. Each frame is compared with the frame
        at the next longer exposure. A pixel is tested when it carries signal (above
        ``lower_threshold`` of full scale) and stays below full scale at the longer
        exposure. A working pixel then rises by the exposure ratio within
        ``tolerance``. A pixel that climbs to full scale from below also responds, and
        a stuck pixel does neither. A pixel stuck above ``lower_threshold`` of full
        scale is flagged anywhere. A pixel stuck lower (dead) is flagged only where its
        neighbours respond, since elsewhere it cannot be told from an unilluminated
        pixel.

        The comparison uses only exposures far enough apart to tell a working pixel
        from a stuck one (:func:`_select_distinct_exposures`), so a repeated exposure
        is dropped. A connected region saturated in every frame is taken as
        overexposure when it holds at least ``blob_min_size`` pixels. Such a region is
        not excluded, and a ``UserWarning`` reports that the camera is overexposed.

        Args:
            frames: A stack ``(n, height, width)`` of whole frames, from
                :meth:`_capture_exposure_sweep` or from the search of
                :meth:`autoexpose` with ``detect_stuck_pixels=True``.
            exposures: The ``n`` exposures in seconds.
            lower_threshold: The fraction of full scale above which a pixel carries
                signal.
            tolerance: The relative tolerance of the response test.
            blob_min_size: A region saturated in every frame counts as overexposure
                when it holds at least this many pixels.
            verbose: Whether to print how many stuck pixels were found and where.

        Returns:
            list[tuple[int, int]]: The flagged ``(row, col)`` pixels, which replace
            :attr:`excluded_pixels`.

        Raises:
            ValueError: The frames are not a stack of at least two, the exposures do not
                match them, or fewer than two exposures lie far enough apart.
        """
        frames = np.asarray(frames, dtype=float)
        exposures = np.asarray(exposures, dtype=float)
        if frames.ndim != 3 or frames.shape[0] < 2:
            raise ValueError(
                "frames must be a stack of at least two (height, width) images."
            )
        if exposures.shape != (frames.shape[0],):
            raise ValueError("exposures must give one exposure time for each frame.")

        # Exposure increases along the stack, and neighbouring frames are compared.
        distinct = _select_distinct_exposures(exposures, tolerance)
        if len(distinct) < 2:
            raise ValueError(
                "Stuck-pixel detection needs frames at two or more exposures, each "
                f"longer than the one before by more than a factor 1 / (1 - "
                f"{tolerance}). Got the exposures {exposures.tolist()} s."
            )
        exposures = exposures[distinct]
        frames = frames[distinct]

        full_scale = self.max_pixel_value
        saturation = self.saturation_level
        frames_min = frames.min(axis=0)
        frames_max = frames.max(axis=0)

        # A working pixel scales with the exposure. Each frame is compared with the
        # frame at the next longer exposure. A pixel is tested when it reads above
        # lower_threshold of full scale (signal, not noise) and stays below saturation
        # at the longer exposure. A working pixel then rises by the exposure ratio
        # within tolerance. A pixel that climbs to saturation from below also responds,
        # and a stuck pixel does neither. Neighbouring exposures are compared on
        # readings below saturation, so the comparison holds over a wide sweep.
        responding = (frames_max >= saturation) & (frames_min < saturation)
        for shorter, longer, exposure_short, exposure_long in zip(
            frames[:-1], frames[1:], exposures[:-1], exposures[1:]
        ):
            ratio = exposure_long / exposure_short
            testable = (shorter > lower_threshold * full_scale) & (
                shorter < saturation / ratio
            )
            expected = shorter * ratio
            rose_as_expected = np.abs(longer - expected) <= tolerance * expected
            responding |= testable & rose_as_expected
        non_responding = ~responding

        # A large connected region saturated even at the lowest exposure means the
        # camera is overexposed, not a field of hot pixels.
        saturated = frames_min >= saturation

        # The connected components of the saturated pixels, and their sizes in pixels.
        components, count = label(saturated)
        sizes = np.bincount(components.ravel())

        # A component of blob_min_size pixels or more is overexposure, and a smaller one
        # is a group of hot pixels.
        overexposed_blob = np.zeros_like(saturated)
        for component in range(1, count + 1):
            if sizes[component] >= blob_min_size:
                overexposed_blob |= components == component
        if overexposed_blob.any():
            warnings.warn(
                f"Camera is overexposed: a blob of >= {blob_min_size} pixels is "
                "saturated across the whole exposure sweep.",
                stacklevel=2,
            )

        # A pixel stuck above the noise floor stands out from the dark background, so it
        # is flagged anywhere. This covers a dark corner, since a hot pixel there still
        # rails the exposure. A pixel stuck low (dead) can be told from an unilluminated
        # one only where its neighbours respond. Only pixels with signal count as
        # responding, so noise in a dark background never marks the neighbours of a
        # dark pixel as responding.
        neighbours = np.ones((3, 3), dtype=bool)
        neighbours[1, 1] = False
        in_illuminated_region = binary_erosion(
            responding, structure=neighbours, border_value=False
        )
        stuck = (
            non_responding
            & ~overexposed_blob
            & ((frames_min > lower_threshold * full_scale) | in_illuminated_region)
        )

        rows, columns = np.nonzero(stuck)
        self.excluded_pixels = list(zip(rows.tolist(), columns.tolist()))

        if verbose:
            stuck_pixels = self.excluded_pixels
            if not stuck_pixels:
                print("detect_stuck_pixels: no stuck pixels found.")
            else:
                shown = stuck_pixels[:20]
                more = len(stuck_pixels) - len(shown)
                suffix = f" and {more} more" if more else ""
                print(
                    f"detect_stuck_pixels: found {len(stuck_pixels)} stuck pixel(s) at "
                    f"(row, col) {shown}{suffix}."
                )

        return self.excluded_pixels

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
        """Set the exposure so the peak of the measured region sits at ``set_fraction``
        of full scale, and return it.

        The region is the whole frame, cropped to ``roi`` and reduced to the pixels
        where ``mask`` is True, and the camera's :attr:`excluded_pixels` are always left
        out. The whole sensor is read out for the search, and the region of interest is
        put back afterwards (:meth:`preserve_roi`). An exception from the search also
        puts the exposure back where the search started.

        The search works within ``exposure_bounds``, narrowed to the camera's own
        :attr:`exposure_bounds`. It starts from the current exposure clipped into those
        bounds, or from the lower bound when the camera sits at zero, or from the upper
        bound when the lower one is zero too. Each step sets an exposure, reads back the
        applied exposure, and measures one frame, so the search measures at most
        ``1 + max_iterations`` frames. A peak at :attr:`saturation_level` or above is
        overexposed, and cuts the exposure by ``overexposed_factor`` until a frame
        peaks below it. From then on, an overexposed peak steps to the geometric mean of
        the longest exposure that was not overexposed and the shortest overexposed one,
        when the first is the shorter. A peak below saturation scales the exposure to
        the target, up to ``set_fraction`` times the shortest exposure that overexposed
        the region. The target lies at or below this cap for a linear sensor.

        A step cut short by the bounds lands on the bound and measures it. The search
        has railed when its next step asks to pass an already measured bound, when the
        camera reads back zero, or when the camera repeats an exposure while every frame
        so far has been overexposed. A hidden limit is one that the camera does not
        state. The search rails at a hidden floor, and at a hidden ceiling reached
        through a request at a bound. A hidden ceiling reached through a request inside
        the bounds ends the search with no finer exposure step to take.

        The search settles on a measured exposure when it does not converge. It picks
        the exposure whose peak lies below full scale and closest to the target, or the
        shortest exposure when every frame was overexposed. The applied exposure is
        read back, and one more frame is measured when the camera lands on an
        unmeasured exposure. A warning then reports the final exposure and the peak of
        its frame.

        The frames of the search are analysed for stuck pixels
        (:meth:`find_stuck_pixels`) when ``detect_stuck_pixels`` is set and the search
        did not rail. The analysis fills :attr:`excluded_pixels` without a second
        sweep. The detection is skipped with a warning when the frames span fewer than
        two exposures far enough apart to compare.

        Args:
            set_fraction: The target peak as a fraction of :attr:`max_pixel_value`.
            tolerance: The largest accepted distance between the peak and the target,
                as a fraction of full scale.
            roi: The region of the whole frame to measure, or None for the whole frame.
                Without a mask, a region reaching off the frame is trimmed to it.
            mask: True at the pixels of the region to measure, in the region's shape.
                For example, a mask can select everything outside a zeroth order. None
                measures every pixel.
            exposure_bounds: The ``(min, max)`` exposure in seconds to search within,
                or None for :attr:`exposure_search_bounds`. A maximum above
                ``DEFAULT_MAX_EXPOSURE`` is reached only through these bounds.
            overexposed_factor: The factor that multiplies the exposure after an
                overexposed peak, until a frame peaks below full scale.
            raise_on_rail: Whether a rail raises ``RuntimeError``. Otherwise the search
                settles and warns.
            max_iterations: The number of exposure steps after the first frame.
            detect_stuck_pixels: Whether to find stuck pixels in the frames of the
                search.
            verbose: Whether to print the exposure and the peak of every frame.

        Returns:
            float: The final exposure in seconds, as read back from the camera.

        Raises:
            ValueError: The bounds leave no positive exposure, ``set_fraction`` is not
                below ``SATURATION_FRACTION``, or the region, the mask or an excluded
                pixel does not fit the frame. Nothing is captured then.
            RuntimeError: The search railed and ``raise_on_rail`` is set.
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
                f"camera's bounds {stated_bounds} s, leave no positive exposure to "
                "search."
            )
        if not 0.0 < set_fraction < SATURATION_FRACTION:
            raise ValueError(
                f"The target set_fraction {set_fraction} has to lie between 0 and "
                f"the saturation fraction {SATURATION_FRACTION}."
            )

        full_scale = self.max_pixel_value
        saturation = self.saturation_level
        set_value = set_fraction * full_scale
        stored_exposure = float(self.get_exposure())
        # The peak of the measured region at every applied exposure.
        peak_by_exposure: dict[float, float] = {}
        recorded_frames: list[NDArray] = []
        recorded_exposures: list[float] = []
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
                        self.set_exposure(requested)
                    applied = float(self.get_exposure())
                    image = self.get_image()
                    repeated = applied in peak_by_exposure
                    if detect_stuck_pixels and applied > 0.0 and not repeated:
                        recorded_frames.append(np.asarray(image, dtype=float))
                        recorded_exposures.append(applied)
                    pixels = region.crop(image)
                    if keep is not None:
                        pixels = np.where(keep, pixels, 0)
                    peak = float(np.amax(pixels))
                    peak_by_exposure[applied] = peak
                    if verbose:
                        print(
                            f"Autoexposure: exposure = {applied:<.3e} s, "
                            f"peak = {peak:.0f}/{full_scale}."
                        )
                    return applied, peak, repeated

                start = float(np.clip(stored_exposure, low, high))
                if start <= 0.0:
                    # A zero start after the clip means the lower bound is zero.
                    start = high
                exposure, peak, _ = measure(
                    None if start == stored_exposure else start
                )
                # The bound of the last request, or None for a request inside the
                # bounds. A step asking past this bound again has railed.
                measured_bound = start if start in (low, high) else None

                steps = 0
                while True:
                    if (
                        peak < saturation
                        and abs(peak - set_value) <= tolerance * full_scale
                    ):
                        outcome = "converged"
                        break
                    overexposed_at = [
                        measured_exposure
                        for measured_exposure, measured_peak in peak_by_exposure.items()
                        if measured_exposure > 0.0 and measured_peak >= saturation
                    ]
                    below_saturation_at = [
                        measured_exposure
                        for measured_exposure, measured_peak in peak_by_exposure.items()
                        if measured_exposure > 0.0 and measured_peak < saturation
                    ]
                    if peak >= saturation:
                        # An overexposed peak hides the true one, so no proportional
                        # step can be computed. Once a shorter exposure has kept the
                        # peak below full scale, the step goes to the geometric mean of
                        # the longest exposure below full scale and the shortest
                        # overexposed one.
                        if (
                            overexposed_at
                            and below_saturation_at
                            and max(below_saturation_at) < min(overexposed_at)
                        ):
                            desired = float(
                                np.sqrt(max(below_saturation_at) * min(overexposed_at))
                            )
                        else:
                            desired = exposure * overexposed_factor
                    else:
                        desired = exposure * set_value / max(peak, 1.0)
                        # The counts grow in proportion to the exposure, so the target
                        # lies at or below set_fraction times any exposure that
                        # overexposed the region. A step from a peak of a few counts
                        # is coarse, and this bound keeps it from returning to an
                        # exposure that overexposes the region.
                        if overexposed_at:
                            desired = min(
                                desired, set_fraction * min(overexposed_at)
                            )
                    requested = float(np.clip(desired, low, high))
                    if requested != desired and requested == measured_bound:
                        outcome = "rail"
                        break
                    if steps >= max_iterations:
                        outcome = "budget"
                        break
                    steps += 1
                    exposure, peak, repeated = measure(requested)
                    measured_bound = requested if requested in (low, high) else None
                    if exposure <= 0.0 or repeated:
                        every_frame_overexposed = all(
                            peak_by_exposure[measured_exposure] >= saturation
                            for measured_exposure in peak_by_exposure
                            if measured_exposure > 0.0
                        )
                        railed = (
                            exposure <= 0.0
                            or requested != desired
                            or every_frame_overexposed
                        )
                        outcome = "rail" if railed else "no finer step"
                        break

                if outcome == "rail" and raise_on_rail:
                    raise RuntimeError(
                        f"autoexposure has railed at {exposure:.3e} s: the region "
                        f"peaks at {peak:.0f} of {full_scale} there, and the target "
                        f"lies beyond the exposure bounds {(low, high)} s."
                    )

                if outcome != "converged":
                    # Settle on the measured exposure closest to the target, and report
                    # a frame taken at the applied exposure.
                    positive_peaks = {
                        measured_exposure: measured_peak
                        for measured_exposure, measured_peak in peak_by_exposure.items()
                        if measured_exposure > 0.0
                    }
                    if not positive_peaks:
                        raise RuntimeError(
                            "autoexposure took no frame at a positive exposure, so it "
                            "has no exposure to settle on."
                        )
                    peaks_below_saturation = {
                        measured_exposure: measured_peak
                        for measured_exposure, measured_peak in positive_peaks.items()
                        if measured_peak < saturation
                    }
                    if peaks_below_saturation:
                        settled = min(
                            peaks_below_saturation,
                            key=lambda candidate: abs(
                                peaks_below_saturation[candidate] - set_value
                            ),
                        )
                    else:
                        settled = min(positive_peaks)
                    if settled != exposure:
                        self.set_exposure(settled)
                        exposure = float(self.get_exposure())
                        if exposure in peak_by_exposure:
                            peak = peak_by_exposure[exposure]
                        else:
                            exposure, peak, _ = measure(None)
            finished = True
        finally:
            if not finished and float(self.get_exposure()) != stored_exposure:
                self.set_exposure(stored_exposure)

        if outcome != "converged":
            reason = {
                "rail": f"the exposure railed against its bounds {(low, high)} s",
                "budget": f"the budget of {max_iterations} exposure steps ran out",
                "no finer step": "the camera has no finer exposure step to take",
            }[outcome]
            warnings.warn(
                f"Autoexposure did not reach its target: {reason}. The region peaks at "
                f"{peak:.0f} of {full_scale} ({peak / full_scale:.1%}) against a "
                f"target of {set_fraction:.0%}, at an exposure of {exposure:.3e} s. "
                "The frames that follow are exposed as reported here, not as asked "
                "for.",
                stacklevel=2,
            )

        if detect_stuck_pixels and outcome != "rail":
            distinct = _select_distinct_exposures(
                np.asarray(recorded_exposures, dtype=np.float64),
                STUCK_PIXEL_RESPONSE_TOLERANCE,
            )
            if len(distinct) >= 2:
                self._detect_stuck_pixels(
                    np.stack(recorded_frames), recorded_exposures, verbose=verbose
                )
            else:
                warnings.warn(
                    "Stuck pixels were not detected: autoexposure took frames at fewer "
                    "than two exposures far enough apart to compare. find_stuck_pixels "
                    "captures a sweep of its own.",
                    stacklevel=2,
                )

        return exposure

    def _select_measured_region(
        self, roi: ROI | None, mask: NDArray[np.bool_] | None
    ) -> tuple[ROI, NDArray[np.bool_] | None]:
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
            tuple[ROI, NDArray[np.bool_] | None]: The region, and the pixels of it to
            measure, which leave out :attr:`excluded_pixels`. The pixels are None when
            there is neither a mask nor an excluded pixel.

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

        keep = None
        if mask is not None:
            keep = np.asarray(mask, dtype=bool)
            if keep.shape != (region.height, region.width):
                raise ValueError(
                    f"The mask has shape {keep.shape}, and the region it selects from "
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
            excluded_in_frame = np.zeros((height, width), dtype=bool)
            excluded_in_frame[excluded[:, 0], excluded[:, 1]] = True
            kept_by_exclusion = ~region.crop(excluded_in_frame)
            keep = kept_by_exclusion if keep is None else keep & kept_by_exclusion

        if keep is not None and not keep.any():
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


@record_type("camera_orientation")
@dataclass(frozen=True)
class CameraOrientation:
    """How a sensor is mounted, as the rotate and flip flags a device takes."""

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
            NDArray: The matrix mapping the ``(x, y, 1)`` of a raw pixel to its
            ``(x, y)`` in the transformed frame.
        """
        return dihedral_affine_matrix(self.transformation(), shape)

    def swaps_axes(self) -> bool:
        """True when the rotation exchanges height and width."""
        return self.rot in ("90", "270", 1, 3)

    def compose(self, other: CameraOrientation) -> CameraOrientation:
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
    # The matrices act on (x, y) = (col, row) positions.
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
    """A native snapshot of a camera's geometry and exposure state."""

    name: str
    sensor_resolution: tuple[int, int]
    pixel_size: tuple[float, float]
    adu_levels: int
    exposure: float
    exposure_bounds: tuple[float, float] | None
    roi: ROI
    orientation: NDArray = field(compare=False, hash=False)

    @property
    def resolution(self) -> tuple[int, int]:
        """The shape of a frame this camera returns, which is the region of interest."""
        return (self.roi.height, self.roi.width)

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
        # transform is a device detail, which the slmsuite adapter exposes. Without a
        # transform the sensor is axis-aligned.
        return cls(
            name=getattr(camera, "name", ""),
            sensor_resolution=camera.sensor_resolution,
            pixel_size=tuple(float(v) for v in camera.pixel_size),
            adu_levels=camera.adu_levels,
            exposure=camera.get_exposure(),
            exposure_bounds=camera.exposure_bounds,
            roi=camera.roi,
            orientation=camera.orientation_matrix(),
        )
