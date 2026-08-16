"""The native camera template: the ``Camera`` base class a device subclasses, together
with the ``CameraData`` snapshot record and the ``probe_orientation`` helper.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import reduce
from typing import Callable
import warnings

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.ndimage import binary_erosion, label

from array_api_compat import array_namespace

from ...grids import get_spatial_grid as _spatial_grid
from ...roi import ROI
from ...serialization import SaveableRecord, record_type


class Camera(ABC):
    """A HoloGradPy-native camera: SI units, ``(y, x)`` geometry, ``(row, col)`` ROI.

    A device implements the geometry / exposure / capture abstract members below. The
    ``get_spatial_grid``, ``flush`` and ``autoexpose`` template methods are provided
    here, so every camera shares them. Third-party devices subclass this base or
    register a wrapper with :func:`hologradpy.hardware.as_native.as_camera`.
    """

    _excluded_pixels: list[tuple[int, int]] | None = None

    @property
    @abstractmethod
    def pixel_size(self) -> NDArray[np.float64]:
        """Pixel pitch ``(y, x)`` in metres."""

    @property
    @abstractmethod
    def resolution(self) -> tuple[int, int]:
        """Sensor resolution ``(height, width)`` in pixels."""

    @property
    @abstractmethod
    def adu_levels(self) -> int:
        """Number of digital levels (``2 ** bitdepth``). The max pixel value is one
        less."""

    @property
    @abstractmethod
    def exposure_bounds(self) -> tuple[float, float] | None:
        """The ``(min, max)`` exposure time in seconds, or ``None`` if unbounded."""

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

    @abstractmethod
    def get_image(
        self, exposure_s: float | None = None, averaging: int = 1
    ) -> NDArray:
        """Capture a frame as a ``(height, width)`` array of digital counts.

        ``exposure_s`` sets the exposure first when given. ``averaging`` sums that many
        frames (integer sum, not mean), matching the slmsuite convention.
        :meth:`get_averaged_image` returns the float mean instead.
        """

    def get_spatial_grid(
        self, device: torch.device = torch.device("cpu")
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The sensor-plane ``(x, y)`` coordinate meshgrid, in metres."""
        return _spatial_grid(self.resolution, self.pixel_size, device=device)

    def flush(self) -> None:
        """Discard two frames so the next :meth:`get_image` is fresh (matches the
        slmsuite two-frame buffer flush)."""
        self.get_image()
        self.get_image()

    @property
    def orientation(self) -> CameraOrientation | None:
        """How the sensor is mounted, or None when the frame transform is not one of the
        eight rotate and flip orientations.
        """
        transform = getattr(self, "transform", None)
        if transform is None:
            return CameraOrientation()
        shape = getattr(self, "default_shape", self.resolution)
        return CameraOrientation.from_matrix(probe_orientation(transform, shape), shape)

    def set_orientation(self, orientation: CameraOrientation) -> None:
        """Mount the sensor in ``orientation``, reorienting every frame from here on.

        What :meth:`~hologradpy.calibration.camera_mapping.CoarseMapper.map_camera`
        suggests is applied through this, so a device that can be reoriented overrides
        it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support being reoriented."
        )

    @property
    def excluded_pixels(self) -> list[tuple[int, int]]:
        """The ``(row, col)`` pixels to leave out of measurements, such as hot or dead
        pixels (empty if none).

        :meth:`autoexpose` drops these from its peak and saturation, so a stuck pixel
        cannot pose as the peak and rail the exposure. Set it from a sensor
        characterization (for example the pixels above a threshold in a dark frame).
        """
        return self._excluded_pixels if self._excluded_pixels is not None else []

    @excluded_pixels.setter
    def excluded_pixels(self, pixels: list[tuple[int, int]] | None) -> None:
        self._excluded_pixels = (
            None if not pixels else [(int(row), int(col)) for row, col in pixels]
        )

    def find_stuck_pixels(
        self,
        *,
        exposures: list[float] | None = None,
        steps: int = 4,
        lower_threshold: float = 0.1,
        tolerance: float = 0.2,
        blob_min_size: int = 4,
        verbose: bool = False,
    ) -> list[tuple[int, int]]:
        """Capture an exposure sweep and flag the sensor's hot / dead pixels, storing
        them in :attr:`excluded_pixels`.

        The public one-call way to characterize the sensor independently of
        :meth:`autoexpose`: it captures a low-to-high exposure sweep and analyses it
        (see :meth:`_detect_stuck_pixels` for the arguments and the method).
        ``autoexpose`` can instead reuse the frames it already captured, via its
        ``detect_stuck_pixels`` flag.
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
        self, exposures: list[float] | None = None, *, steps: int = 4
    ) -> tuple[NDArray, list[float]]:
        """Capture full-frame images across a low-to-high exposure sweep.

        Returns the stacked frames ``(n, height, width)`` and the exposures used.
        ``exposures`` defaults to ``steps`` values increasing by factors of ten from the
        minimum exposure, and any value outside the exposure bounds is dropped rather
        than clipped. The roi and exposure are reset for the capture and restored after.
        """
        low, high = (
            self.exposure_bounds if self.exposure_bounds is not None else (0.0, np.inf)
        )
        if exposures is None:
            # Lower bound if non-zero, otherwise 100 us, then a decade per step.
            base = low if low > 0 else 100e-6
            exposures = [base * 10.0**step for step in range(steps)]

        # Drop exposures outside the bounds rather than clip them.
        exposures = [
            float(exposure) for exposure in exposures if low <= exposure <= high
        ]
        if len(exposures) < 2:
            raise ValueError(
                "the exposure sweep needs at least two exposures within the bounds "
                f"{(low, high)}. Got {exposures}."
            )

        stored_roi = self.roi
        stored_exposure = self.get_exposure()
        self.set_roi(None)
        try:
            frames = []
            for exposure in exposures:
                self.set_exposure(exposure)
                self.flush()
                frames.append(np.asarray(self.get_image(), dtype=float))
        finally:
            self.set_roi(stored_roi)
            self.set_exposure(stored_exposure)
        return np.stack(frames), exposures

    def _detect_stuck_pixels(
        self,
        frames: NDArray,
        exposures: list[float],
        *,
        lower_threshold: float = 0.1,
        tolerance: float = 0.2,
        blob_min_size: int = 4,
        verbose: bool = False,
    ) -> list[tuple[int, int]]:
        """Find hot / dead pixels from frames captured at different exposures and store
        them in :attr:`excluded_pixels`.

        ``frames`` is a stack ``(n, height, width)`` and ``exposures`` the ``n``
        exposure times, from :meth:`_capture_exposure_sweep` or the frames captured by
        :meth:`autoexpose` (``detect_stuck_pixels=True``).

        A working pixel scales with exposure (linear response). Between a frame and the
        next longer exposure, a pixel bright enough to carry signal (above
        ``lower_threshold`` of full scale) that would not clip at the longer exposure
        should rise by the exposure ratio within ``tolerance``. A pixel that instead
        climbs to the ceiling (saturating from below) also responds. A stuck pixel does
        neither. A pixel stuck in the upper half of the range is flagged anywhere, a
        pixel stuck low (dead) only where its neighbours respond (elsewhere it looks
        unilluminated).

        A connected region of at least ``blob_min_size`` pixels saturated across every
        frame is likely caused by overexposure rather than hot pixels, so it is *not*
        excluded and a ``UserWarning`` reports that the camera is overexposed. The
        flagged ``(row, col)`` pixels replace :attr:`excluded_pixels` and are returned.
        ``verbose`` prints how many stuck pixels were found and where.
        """
        frames = np.asarray(frames, dtype=float)
        exposures = np.asarray(exposures, dtype=float)
        if frames.ndim != 3 or frames.shape[0] < 2:
            raise ValueError(
                "frames must be a stack of at least two (height, width) images."
            )
        if exposures.shape != (frames.shape[0],):
            raise ValueError("exposures must give one exposure time for each frame.")

        # Sort so exposure increases along the stack, then compare neighbouring frames.
        order = np.argsort(exposures)
        exposures = exposures[order]
        frames = frames[order]

        full_scale = self.adu_levels - 1
        frames_min = frames.min(axis=0)
        frames_max = frames.max(axis=0)

        # A working pixel scales linearly with exposure. Between a frame and the next
        # longer exposure, a pixel above lower_threshold of full scale (signal, not
        # noise) that would not clip at the longer exposure should rise by the exposure
        # ratio within tolerance. A pixel that instead climbs to the ceiling (saturating
        # from below) also responds. A stuck pixel does neither. Comparing neighbouring
        # exposures on unsaturated readings avoids the clipping that a single
        # dimmest-to-brightest ratio hits over a wide sweep.
        responding = (frames_max >= full_scale) & (frames_min < full_scale)
        for shorter, longer, exposure_short, exposure_long in zip(
            frames[:-1], frames[1:], exposures[:-1], exposures[1:]
        ):
            ratio = exposure_long / exposure_short
            testable = (shorter > lower_threshold * full_scale) & (
                shorter < full_scale / ratio
            )
            expected = shorter * ratio
            rose_as_expected = np.abs(longer - expected) <= tolerance * expected
            responding |= testable & rose_as_expected
        non_responding = ~responding

        # A large connected region saturated even at the lowest exposure means the
        # camera is overexposed, not a field of hot pixels.
        saturated = frames_min >= full_scale

        # Labelling connected components and counting their sizes in pixels
        components, count = label(saturated)
        sizes = np.bincount(components.ravel())

        # Keep only the connected components that are less than or equal to 
        # blob_min_size pixels in size. Larger blobs are deemed to be overexposed.
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
        # is flagged anywhere (a hot pixel in a dark corner still rails the exposure). A
        # pixel stuck low (dead) is only distinguishable from an unilluminated one where
        # its neighbours respond. Only pixels with real signal count as responding, so a
        # noisy dark background stays uniformly non-responding rather than posing as
        # illumination around a dark pixel.
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
        of the dynamic range, and return it.

        The region is the full sensor frame, optionally cropped to ``roi`` and then
        reduced to the pixels where ``mask`` is ``True`` (a boolean array of the region
        shape). ``mask`` drops bright pixels that must not drive the exposure, such as a
        zeroth order. The camera's :attr:`excluded_pixels` (hot or dead pixels) are
        always dropped as well. ``roi`` selects a window of the *full* sensor, so the
        current :attr:`roi` is reset for the measurement and restored afterwards.

        The loop targets ``set_fraction`` directly. Each step scales the exposure to hit
        the target. When the peak is clipped the true peak is hidden and no such step
        can be computed, so it cuts by ``overexposed_factor`` and looks again. It stops
        at ``tolerance`` or after ``max_iterations`` frames, whichever comes first, and
        warns if it stopped without reaching the target. If the exposure rails against
        ``exposure_bounds`` (falling back to :attr:`exposure_bounds`, then unbounded),
        it raises ``RuntimeError`` when ``raise_on_rail`` is set, otherwise it settles
        at the bound and returns.

        The exposure is read back after each step, so the loop works from the value the
        camera actually applied rather than the one requested. A real camera tunes in
        discrete steps, so when it lands on an exposure already tried (a request below
        its resolution, or an oscillation between two adjacent steps that both miss
        ``tolerance``) the loop stops and settles on the closest exposure it reached.

        With ``detect_stuck_pixels`` the full frames captured along the way (spanning
        the exposures the loop visited) are analysed for stuck pixels at the end,
        populating :attr:`excluded_pixels` in the same call and avoiding a second sweep
        (see :meth:`find_stuck_pixels`). This runs only when the loop converged or
        settled, not when it railed.
        """
        if exposure_bounds is None:
            exposure_bounds = (
                self.exposure_bounds
                if self.exposure_bounds is not None
                else (0.0, np.inf)
            )

        set_value = set_fraction * self.adu_levels
        clipped_value = self.adu_levels - 1
        exposure = self.get_exposure()
        stored_roi = self.roi
        self.set_roi(None)

        # Pixels to keep: the caller's mask minus the sensor's excluded (hot/dead)
        # pixels, both cropped to roi so a stuck pixel cannot pose as the peak.
        keep = mask
        if self.excluded_pixels:
            rows, columns = zip(*self.excluded_pixels)
            excluded = np.zeros(self.resolution, dtype=bool)
            excluded[list(rows), list(columns)] = True
            excluded = excluded if roi is None else roi.crop(excluded)
            keep = ~excluded if keep is None else (keep & ~excluded)

        recorded_frames: list[NDArray] = []
        recorded_exposures: list[float] = []

        def measure() -> tuple[float, float]:
            self.flush()
            image = self.get_image()
            if detect_stuck_pixels:
                recorded_frames.append(np.asarray(image, dtype=float))
                recorded_exposures.append(self.get_exposure())
            region = image if roi is None else roi.crop(image)
            if keep is not None:
                region = np.where(keep, region, 0)
            peak = float(np.amax(region))
            valid = int(keep.sum()) if keep is not None else region.size
            saturated = int(np.count_nonzero(region >= clipped_value))
            return peak, saturated / valid if valid else 0.0

        try:
            image_max, sat_fraction = measure()
            error = np.abs(image_max - set_value) / self.adu_levels

            clipped = image_max >= clipped_value
            best_exposure = exposure
            best_error = np.inf if clipped else error
            tried = {exposure}
            unconverged_reason: str | None = None
            iterations = 0

            while (error > tolerance or clipped) and iterations < max_iterations:
                iterations += 1
                if image_max >= clipped_value:
                    # Overexposed
                    desired = exposure * overexposed_factor
                else:
                    desired = exposure * set_value / max(image_max, 1.0)

                requested = float(
                    np.clip(desired, exposure_bounds[0], exposure_bounds[1])
                )
                if desired != requested:  # railed against a bound
                    if raise_on_rail:
                        raise RuntimeError(
                            f"autoexposure has railed (exposure: {desired}, "
                            f"bounds: {exposure_bounds})."
                        )
                    self.set_exposure(requested)
                    exposure = self.get_exposure()
                    unconverged_reason = (
                        f"the exposure railed against its bounds {exposure_bounds}"
                    )
                    break

                self.set_exposure(requested)
                # Read back the applied exposure: a real camera tunes in discrete steps
                # and may snap the request to a nearby value. Working from the actual
                # exposure keeps the next proportional step honest.
                exposure = self.get_exposure()
                image_max, sat_fraction = measure()
                error = np.abs(image_max - set_value) / self.adu_levels
                clipped = image_max >= clipped_value
                if not clipped and error < best_error:
                    best_exposure, best_error = exposure, error

                if verbose:
                    print(
                        f"Autoexposure: exposure = {exposure:<.2e} s, "
                        f"image_max = {image_max}/{clipped_value},"
                    )

                # If the camera lands on an exposure already tried (a request below its
                # step resolution, or an oscillation between two adjacent steps that
                # both miss the target), the discrete steps cannot get closer, so settle
                # on the best exposure seen instead of spending the rest of the budget.
                if exposure in tried:
                    if best_exposure != exposure:
                        self.set_exposure(best_exposure)
                    exposure = best_exposure
                    if best_error > tolerance:
                        unconverged_reason = (
                            "the camera has no finer exposure step to take"
                        )
                    break
                tried.add(exposure)
            else:
                if error > tolerance or clipped:
                    unconverged_reason = (
                        f"the budget of {max_iterations} frames ran out"
                    )
        finally:
            self.set_roi(stored_roi)

        if unconverged_reason is not None:
            warnings.warn(
                f"Autoexposure did not reach its target: {unconverged_reason}. The "
                f"region peaks at {image_max:.0f} of {self.adu_levels} "
                f"({image_max / self.adu_levels:.1%}) against a target of "
                f"{set_fraction:.0%}, at an exposure of {exposure:.3e} s. The frames "
                "that follow are exposed as reported here, not as asked for.",
                stacklevel=2,
            )

        # Reuse the frames the loop captured to find stuck pixels, avoiding a second
        # sweep. Only reached when the loop converged or settled, not when it railed.
        if detect_stuck_pixels and len(recorded_frames) >= 2:
            self._detect_stuck_pixels(
                np.stack(recorded_frames), recorded_exposures, verbose=verbose
            )

        return exposure

    def get_averaged_image(
        self, exposure_s: float | None = None, averaging: int = 1
    ) -> NDArray:
        """Mean of ``averaging`` frames as a float array, the lower-noise counterpart
        of :meth:`get_image` (which returns the integer sum).

        A real (and the simulated) camera draws fresh read and shot noise per frame, so
        the mean has ``averaging`` times lower noise variance.
        """
        frames = max(1, int(averaging))
        summed = np.asarray(self.get_image(exposure_s, averaging=frames), dtype=float)
        return summed / frames


def get_orientation_transformation(
    rot: str | int = "0", fliplr: bool = False, flipud: bool = False
) -> Callable[[NDArray], NDArray]:
    """Compile a discrete image transform (a rot90/flip composition) from simple
    rotate and flip flags.

    ``rot`` rotates by the given degrees in ``["90", "180", "270"]`` or the
    :func:`numpy.rot90` code in ``[1, 2, 3]`` (no rotation otherwise). ``fliplr`` and
    ``flipud`` mirror left-right and up-down. The flips are applied before the
    rotation. Returns a function mapping an array to the reoriented array, matching the
    ``transform`` a camera applies to its raw frames. Backend-agnostic: a numpy frame
    comes back as numpy, a torch frame as torch on the same device.
    """
    transforms = []

    if fliplr:
        transforms.append(lambda img: array_namespace(img).fliplr(img))
    if flipud:
        transforms.append(lambda img: array_namespace(img).flipud(img))

    if rot == "90" or rot == 1:
        transforms.append(lambda img: array_namespace(img).rot90(img, 1))
    elif rot == "180" or rot == 2:
        transforms.append(lambda img: array_namespace(img).rot90(img, 2))
    elif rot == "270" or rot == 3:
        transforms.append(lambda img: array_namespace(img).rot90(img, 3))

    return reduce(lambda f, g: lambda x: f(g(x)), transforms, lambda x: x)


def probe_orientation(
    transform_fn: Callable[[NDArray], NDArray], shape: tuple[int, int]
) -> NDArray:
    """Pixel-space affine ``(x, y)_out = M @ [x, y, 1]`` of a discrete image transform
    (a rot90/flip composition, e.g. ``Camera.transform``).

    Found by applying ``transform_fn`` to row/column index arrays of ``shape`` (height,
    width) and reading where three input corners land. Robust for any of the 8
    dihedral orientations. Returns a ``(2, 3)`` matrix.
    """
    height, width = int(shape[0]), int(shape[1])
    rows = np.broadcast_to(np.arange(height)[:, None], (height, width))
    columns = np.broadcast_to(np.arange(width)[None, :], (height, width))
    source_rows = np.asarray(transform_fn(rows))
    source_columns = np.asarray(transform_fn(columns))
    out_h, out_w = source_rows.shape

    # Output corner (i, j) came from input (source_rows[i, j], source_columns[i, j]).
    # fit input (x=col, y=row) -> output (x'=j, y'=i) at three corners.
    corners = [(0, 0), (0, out_w - 1), (out_h - 1, 0)]
    source = np.array(
        [[source_columns[i, j], source_rows[i, j], 1.0] for i, j in corners],
        dtype=np.float64,
    )
    destination = np.array([[j, i] for i, j in corners], dtype=np.float64)
    return np.linalg.solve(source, destination).T


@record_type("camera_orientation")
@dataclass(frozen=True)
class CameraOrientation:
    """How a sensor is mounted, as the rotate and flip flags a device takes."""

    rot: str = "0"
    fliplr: bool = False
    flipud: bool = False

    def transformation(self) -> Callable[[NDArray], NDArray]:
        """The transform a camera in this orientation applies to its raw frames."""
        return get_orientation_transformation(self.rot, self.fliplr, self.flipud)

    def matrix(self, shape: tuple[int, int]) -> NDArray:
        """The ``(2, 3)`` pixel-space affine of :meth:`transformation` on ``shape``."""
        return probe_orientation(self.transformation(), shape)

    def swaps_axes(self) -> bool:
        """True when the rotation exchanges height and width."""
        return self.rot in ("90", "270", 1, 3)

    def compose(self, other: CameraOrientation) -> CameraOrientation:
        def combined(image: NDArray) -> NDArray:
            return self.transformation()(other.transformation()(image))

        # Non-square, so no two of the eight probe to the same matrix.
        shape = (3, 5)
        return CameraOrientation.from_matrix(
            probe_orientation(combined, shape), shape
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
        """The orientation whose :meth:`matrix` on ``shape`` is ``matrix``."""
        target = np.asarray(matrix, dtype=np.float64)
        for orientation in cls.dihedral():
            if np.allclose(orientation.matrix(shape), target):
                return orientation
        return None


@record_type("camera_data")
@dataclass(frozen=True, unsafe_hash=True)
class CameraData(SaveableRecord):
    """A native snapshot of a camera's geometry and exposure state."""

    name: str
    resolution: tuple[int, int]
    pixel_size: tuple[float, float]
    adu_levels: int
    exposure: float
    exposure_bounds: tuple[float, float] | None
    roi: ROI
    orientation: NDArray = field(compare=False, hash=False)

    @classmethod
    def from_camera(cls, camera: Camera) -> CameraData:
        # transform / default_shape are device details (a real slmsuite camera or the
        # adapter exposes them). Without a transform the sensor is axis-aligned.
        transform = getattr(camera, "transform", None)
        if transform is None:
            orientation = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        else:
            default_shape = getattr(camera, "default_shape", camera.resolution)
            orientation = probe_orientation(transform, default_shape)
        return cls(
            name=getattr(camera, "name", ""),
            resolution=camera.resolution,
            pixel_size=tuple(float(v) for v in camera.pixel_size),
            adu_levels=camera.adu_levels,
            exposure=camera.get_exposure(),
            exposure_bounds=camera.exposure_bounds,
            roi=camera.roi,
            orientation=orientation,
        )

    # save / load come from SaveableRecord.
