"""The template methods of the native ``Camera`` base, on fakes that crop each frame.

The fakes build on ``CroppingCamera`` (``tests/native_camera_fakes.py``), so a region of
interest crops each frame and an exposure stays inside the bounds, as on a real camera.
The tests cover the ``exposure_search_bounds`` ceiling and ``sensor_resolution``, then
``preserve_roi`` and ``preserve_exposure_and_roi``. The exposure search of
``autoexpose`` follows, with its rails, its final exposure and the final state of the
camera. The last tests cover ``get_averaged_image``, ``excluded_pixels`` and
``find_stuck_pixels``.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from hologradpy.hardware.camera.abstract import DEFAULT_MAX_EXPOSURE, Camera
from hologradpy.roi import ROI
from tests.native_camera_fakes import CroppingCamera

SENSOR_RESOLUTION = (16, 24)
# A region in the top-left corner of the sensor, set before a routine runs.
PRESET_REGION = ROI(0, 0, 8, 8)
# Counts per second that saturate an 8-bit pixel at every exposure in the bounds.
SATURATING_RATE = 1e9
# The default target peak of autoexposure, half of the 8-bit full scale, and the
# tolerance on the final peak.
DEFAULT_TARGET_COUNTS = 0.5 * 255
DEFAULT_TOLERANCE_COUNTS = 0.05 * 255


class _LinearSensorCamera(CroppingCamera):
    """A sensor whose counts grow in proportion to the exposure and clip at full scale.

    Args:
        rate: The counts per second at each pixel, a ``(height, width)`` array.
        read_noise: The standard deviation in counts of the Gaussian noise on every
            pixel of every frame.
        stuck_pixels: ``(row, col)`` pixels, each mapped to its constant count.
        seed: The seed of the read noise.
        **camera_options: Passed on to :class:`CroppingCamera`.
    """

    def __init__(
        self,
        rate: NDArray,
        *,
        read_noise: float = 0.0,
        stuck_pixels: dict[tuple[int, int], int] | None = None,
        seed: int = 0,
        **camera_options,
    ) -> None:
        super().__init__(np.shape(rate), **camera_options)
        self._rate = np.asarray(rate, dtype=np.float64)
        self._read_noise = read_noise
        self._stuck_pixels = dict(stuck_pixels or {})
        self._noise = np.random.default_rng(seed)
        self.rendered_frames: list[NDArray[np.uint16]] = []

    def render_sensor_frame(self) -> NDArray[np.uint16]:
        counts = self._rate * self.get_exposure()
        if self._read_noise > 0:
            counts = counts + self._noise.normal(0.0, self._read_noise, counts.shape)
        counts = np.clip(np.rint(counts), 0, self.max_pixel_value)
        for (row, col), count in self._stuck_pixels.items():
            counts[row, col] = count
        frame = counts.astype(np.uint16)
        self.rendered_frames.append(frame)
        return frame


class _ResponsiveCamera(CroppingCamera):
    """An 8-bit camera for the exposure search, whose counts follow a response of the
    applied exposure.

    The count of a pixel is ``response(exposure) * scene``, rounded, clipped at full
    scale and stored in ``uint16``. The camera records every requested exposure, the
    exposure of every rendered frame and every region written to it.

    Args:
        response: The counts of a pixel of scene value 1 at an applied exposure in
            seconds.
        scene: The relative brightness of each pixel, a ``(height, width)`` array, or
            None for a uniform scene of 1.
        exposure_bounds: The ``(min, max)`` exposure stated by the camera, or None.
        exposure_s: The starting exposure in seconds.
        hidden_limits: Unstated ``(min, max)`` limits on the applied exposure, or None.
        whole_microseconds: Truncate each exposure to whole microseconds, as the
            slmsuite ThorCam driver does.
        failing_frame: The frame, counted from 1, whose capture raises
            ``TimeoutError``, or None.
        **camera_options: Passed on to :class:`CroppingCamera`.
    """

    def __init__(
        self,
        response: Callable[[float], float],
        *,
        scene: NDArray | None = None,
        exposure_bounds: tuple[float, float] | None = (1e-4, 1.0),
        exposure_s: float = 1e-3,
        hidden_limits: tuple[float, float] | None = None,
        whole_microseconds: bool = False,
        failing_frame: int | None = None,
        **camera_options,
    ) -> None:
        shape = SENSOR_RESOLUTION if scene is None else np.shape(scene)
        super().__init__(
            shape,
            exposure_bounds=exposure_bounds,
            exposure_s=exposure_s,
            **camera_options,
        )
        self._response = response
        self._scene = np.ones(shape) if scene is None else np.asarray(scene, float)
        self._hidden_limits = hidden_limits
        self._whole_microseconds = whole_microseconds
        self._failing_frame = failing_frame
        if whole_microseconds:
            self._exposure_s = round(exposure_s * 1e6) / 1e6
        self.requested_exposures: list[float] = []
        self.frame_exposures: list[float] = []
        self.written_regions: list[ROI | None] = []

    def set_exposure(self, exposure_s: float) -> None:
        self.requested_exposures.append(float(exposure_s))
        applied = float(exposure_s)
        for limits in (self._exposure_bounds, self._hidden_limits):
            if limits is not None:
                applied = min(max(applied, limits[0]), limits[1])
        if self._whole_microseconds:
            applied = int(applied * 1e6) / 1e6
        self._exposure_s = applied

    def set_roi(self, roi: ROI | None) -> None:
        self.written_regions.append(roi)
        super().set_roi(roi)

    def render_sensor_frame(self) -> NDArray[np.uint16]:
        self.frame_exposures.append(self.get_exposure())
        if len(self.frame_exposures) == self._failing_frame:
            raise TimeoutError("The fake camera returned no frame.")
        counts = self._response(self.get_exposure()) * self._scene
        return np.clip(np.rint(counts), 0, self.max_pixel_value).astype(np.uint16)


def _saturated(exposure_s: float) -> float:
    """A response that overexposes the sensor at every exposure."""
    return SATURATING_RATE


# --- the exposure ceiling and the sensor resolution -----------------------------------


def test_the_exposure_search_bounds_fall_back_when_a_camera_states_no_bounds() -> None:
    camera = _LinearSensorCamera(np.full(SENSOR_RESOLUTION, 1e4), exposure_bounds=None)

    assert camera.exposure_bounds is None
    assert camera.exposure_search_bounds == (0.0, DEFAULT_MAX_EXPOSURE)


def test_exposure_search_bounds_hold_the_ceiling() -> None:
    """A stated maximum above 1 s is held at 1 s, one below it is kept, and a stated
    minimum above 1 s leaves the minimum as the only exposure.
    """
    camera = _ResponsiveCamera(_saturated, exposure_bounds=(2e-5, 10.0))
    assert camera.exposure_search_bounds == (2e-5, 1.0)
    assert camera.exposure_bounds == (2e-5, 10.0)

    camera = _ResponsiveCamera(_saturated, exposure_bounds=(2e-5, 0.5))
    assert camera.exposure_search_bounds == (2e-5, 0.5)

    camera = _ResponsiveCamera(_saturated, exposure_bounds=(2.0, 10.0), exposure_s=3.0)
    assert camera.exposure_search_bounds == (2.0, 2.0)


def test_the_sensor_resolution_is_the_whole_sensor_while_a_region_is_set() -> None:
    camera = _ResponsiveCamera(_saturated)
    camera.set_roi(ROI(2, 3, 5, 6))
    assert camera.sensor_resolution == SENSOR_RESOLUTION


def test_a_camera_states_its_sensor_resolution() -> None:
    """The whole sensor is part of the camera contract, so a camera that leaves it out
    cannot be built.
    """
    assert "sensor_resolution" in Camera.__abstractmethods__


# --- restoring the region and the exposure -------------------------------------------


def test_preserve_exposure_and_roi_puts_both_back() -> None:
    camera = _ResponsiveCamera(_saturated)
    camera.set_roi(ROI(2, 3, 10, 8))
    camera.set_exposure(1e-3)

    with camera.preserve_exposure_and_roi():
        camera.set_roi(None)
        camera.set_exposure(5e-3)

    assert camera.roi == ROI(2, 3, 10, 8)
    assert camera.get_exposure() == 1e-3


def test_preserve_exposure_and_roi_puts_both_back_after_an_error() -> None:
    camera = _ResponsiveCamera(_saturated)
    camera.set_roi(ROI(2, 3, 10, 8))
    camera.set_exposure(1e-3)

    with pytest.raises(RuntimeError, match="inside the block"):
        with camera.preserve_exposure_and_roi(full_sensor=True):
            camera.set_exposure(5e-3)
            raise RuntimeError("An error inside the block.")

    assert camera.roi == ROI(2, 3, 10, 8)
    assert camera.get_exposure() == 1e-3


def test_the_exposure_is_restored_when_restoring_the_roi_fails() -> None:
    """A device can refuse a region write, for example through a Thorlabs window
    assert or a GenTL restart. The exposure is still put back.
    """

    class _RefusingCamera(_ResponsiveCamera):
        refuse_regions = False

        def set_roi(self, roi: ROI | None) -> None:
            if self.refuse_regions:
                raise RuntimeError("The camera refused the region.")
            super().set_roi(roi)

    camera = _RefusingCamera(_saturated)
    camera.set_roi(ROI(2, 3, 10, 8))
    camera.set_exposure(1e-3)

    with pytest.raises(RuntimeError, match="refused the region"):
        with camera.preserve_exposure_and_roi(full_sensor=True):
            camera.set_exposure(5e-3)
            camera.refuse_regions = True

    assert camera.get_exposure() == 1e-3


def test_full_sensor_reads_out_the_whole_sensor_inside_the_block() -> None:
    camera = _ResponsiveCamera(_saturated)
    camera.set_roi(ROI(2, 3, 10, 8))

    with camera.preserve_roi(full_sensor=True):
        assert camera.resolution == SENSOR_RESOLUTION
        assert camera.get_image().shape == SENSOR_RESOLUTION

    assert camera.roi == ROI(2, 3, 10, 8)
    assert camera.get_image().shape == (10, 8)


def test_unchanged_settings_are_not_written_again() -> None:
    camera = _ResponsiveCamera(_saturated)
    camera.written_regions.clear()

    with camera.preserve_exposure_and_roi(full_sensor=True):
        camera.get_image()

    assert camera.written_regions == []
    assert camera.requested_exposures == []


# --- autoexpose: bounds and rails ----------------------------------------------------


def test_autoexpose_puts_back_the_region_when_it_raises() -> None:
    camera = _LinearSensorCamera(np.full(SENSOR_RESOLUTION, SATURATING_RATE))
    camera.set_roi(PRESET_REGION)

    with pytest.raises(RuntimeError):
        camera.autoexpose(set_fraction=0.5)

    assert camera.roi == PRESET_REGION


def test_autoexpose_settles_on_the_bound_when_told_not_to_raise() -> None:
    camera = _LinearSensorCamera(np.full(SENSOR_RESOLUTION, SATURATING_RATE))
    lowest_exposure, _ = camera.exposure_bounds

    with pytest.warns(UserWarning, match="railed"):
        exposure = camera.autoexpose(set_fraction=0.5, raise_on_rail=False)

    assert exposure == lowest_exposure
    assert camera.get_exposure() == lowest_exposure


def test_autoexpose_measures_the_whole_sensor_under_a_preset_region() -> None:
    """The preset region holds only a dim background, and the one bright pixel lies
    outside it. The exposure is metered on the whole sensor, so the bright pixel ends
    at the target.
    """
    rate = np.full(SENSOR_RESOLUTION, 1e3)
    rate[12, 20] = 5e4
    camera = _LinearSensorCamera(rate)
    camera.set_roi(PRESET_REGION)

    camera.autoexpose(set_fraction=0.5, tolerance=0.05)

    assert camera.roi == PRESET_REGION
    camera.set_roi(None)
    peak = float(camera.get_image().max())
    full_scale = camera.max_pixel_value
    assert abs(peak - 0.5 * full_scale) <= 0.05 * full_scale


def test_autoexpose_measures_the_bound_before_railing() -> None:
    """The first step cuts past the lower bound. The frame at the bound is on target,
    so the search converges there without a rail.
    """
    camera = _ResponsiveCamera(
        lambda exposure: 128 * exposure / 5e-5,
        exposure_bounds=(5e-5, 1.0),
        exposure_s=2e-3,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        exposure = camera.autoexpose()

    assert exposure == 5e-5
    assert camera.frame_exposures == [2e-3, 5e-5]


def test_autoexpose_rails_once_the_bound_is_measured() -> None:
    """With a scene ten times brighter than in the previous test, the frame at the
    bound is still overexposed. The search therefore rails and puts the exposure and
    the region back.
    """
    camera = _ResponsiveCamera(
        lambda exposure: 1280 * exposure / 5e-5,
        exposure_bounds=(5e-5, 1.0),
        exposure_s=2e-3,
    )
    camera.set_roi(PRESET_REGION)

    with pytest.raises(RuntimeError, match="railed"):
        camera.autoexpose()

    assert 5e-5 in camera.requested_exposures
    assert camera.frame_exposures == [2e-3, 5e-5]
    assert camera.get_exposure() == 2e-3
    assert camera.roi == PRESET_REGION


def test_autoexpose_rails_without_repeating_a_bound_it_reached() -> None:
    """The cut from 10 ms lands exactly on the lower bound. That frame counts as a
    measurement of the bound, so the bound is captured once.
    """
    camera = _ResponsiveCamera(_saturated, exposure_bounds=(1e-4, 1.0), exposure_s=1e-2)

    with pytest.raises(RuntimeError, match="railed"):
        camera.autoexpose()

    assert camera.frame_exposures == [1e-2, 1e-4]


def test_autoexpose_searches_within_the_camera_bounds() -> None:
    """The caller's bounds are narrowed to the camera's, so no request goes below the
    stated minimum.
    """
    camera = _ResponsiveCamera(_saturated, exposure_bounds=(1e-4, 1.0), exposure_s=3e-2)

    with pytest.raises(RuntimeError, match="railed"):
        camera.autoexpose(exposure_bounds=(0.0, 1.0))

    assert min(camera.requested_exposures) >= 1e-4


def test_autoexpose_reaches_past_one_second_through_explicit_bounds() -> None:
    """The camera states a maximum of 20 s, and its exposure search bounds hold the
    search at 1 s. Bounds passed by the caller reach the 5.1 s needed by the dim scene.
    """

    def dim(exposure: float) -> float:
        return 25.0 * exposure

    camera = _ResponsiveCamera(dim, exposure_bounds=(1e-4, 20.0), exposure_s=1.0)
    with pytest.raises(RuntimeError, match="railed"):
        camera.autoexpose()
    assert camera.frame_exposures == [1.0]

    camera = _ResponsiveCamera(dim, exposure_bounds=(1e-4, 20.0), exposure_s=1.0)
    exposure = camera.autoexpose(exposure_bounds=(1e-3, 10.0))
    assert exposure == pytest.approx(5.1)


def test_autoexpose_treats_a_zero_readback_as_a_rail() -> None:
    """A camera that truncates to whole microseconds reads back zero for a request
    below 1 us, which rails the search.
    """

    def build() -> _ResponsiveCamera:
        return _ResponsiveCamera(
            lambda exposure: 3e7 * exposure,
            exposure_bounds=None,
            whole_microseconds=True,
            exposure_s=1e-3,
        )

    camera = build()
    with pytest.warns(UserWarning, match="railed"):
        exposure = camera.autoexpose(exposure_bounds=(0.0, 1.0), raise_on_rail=False)

    assert 0.0 in camera.frame_exposures
    assert exposure > 0.0
    assert exposure == camera.get_exposure()
    assert exposure == min(value for value in camera.frame_exposures if value > 0.0)

    camera = build()
    with pytest.raises(RuntimeError, match="railed"):
        camera.autoexpose(exposure_bounds=(0.0, 1.0))
    assert camera.get_exposure() == 1e-3


# --- autoexpose: where the search settles --------------------------------------------


def test_autoexpose_reports_the_frame_it_settles_on() -> None:
    """The step from 0.64 s lands on the upper bound, where the peak is 100 counts.
    The next step asks past that bound, and the warning reports the frame at 1 s.
    """
    camera = _ResponsiveCamera(
        lambda exposure: 100.0 * np.sqrt(exposure), exposure_s=0.64
    )

    with pytest.warns(UserWarning, match="peaks at 100 of 255"):
        exposure = camera.autoexpose(raise_on_rail=False)

    assert exposure == 1.0
    assert camera.frame_exposures == [0.64, 1.0]


def test_autoexpose_never_returns_to_an_overexposed_start() -> None:
    """An unstated floor at 50 us holds every request below it. The search rails there
    and never goes back to its overexposed start at 10 ms.
    """

    def build() -> _ResponsiveCamera:
        return _ResponsiveCamera(
            _saturated,
            exposure_bounds=None,
            hidden_limits=(5e-5, np.inf),
            exposure_s=1e-2,
        )

    camera = build()
    with pytest.raises(RuntimeError, match="railed"):
        camera.autoexpose(exposure_bounds=(0.0, 1.0))
    assert camera.get_exposure() == 1e-2

    camera = build()
    with pytest.warns(UserWarning, match="railed"):
        exposure = camera.autoexpose(exposure_bounds=(0.0, 1.0), raise_on_rail=False)
    assert exposure == 5e-5
    assert camera.get_exposure() == 5e-5


def test_autoexpose_keeps_the_closest_exposure_when_the_budget_runs_out() -> None:
    """A cubic response overshoots each proportional step. The frames peak at 100, 207
    and 48 counts, and the first is the closest to the target of 127.5.
    """
    camera = _ResponsiveCamera(lambda exposure: 1e8 * exposure**3, exposure_s=1e-2)

    with pytest.warns(UserWarning, match="budget of 2 exposure steps"):
        exposure = camera.autoexpose(max_iterations=2)

    assert exposure == 1e-2
    assert camera.get_exposure() == 1e-2
    assert camera.frame_exposures == pytest.approx(
        [1e-2, 1.275e-2, 7.853e-3], rel=1e-3
    )


def test_autoexpose_steps_up_below_an_exposure_that_overexposed() -> None:
    """The cuts from 1 ms stay overexposed down to 100 ns and land on 2 counts at 1 ns,
    a floored 2.9. The step to the target from there asks for 102 ns, which is past
    the overexposed 100 ns. The request is therefore held at 0.8 of 100 ns, and the
    search converges in its budget.
    """
    camera = _ResponsiveCamera(
        lambda exposure: np.floor(2.9e9 * exposure),
        exposure_bounds=(1e-10, 1.0),
        exposure_s=1e-3,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        camera.autoexpose(set_fraction=0.8)

    assert camera.frame_exposures[:5] == pytest.approx(
        [1e-3, 1e-5, 1e-7, 1e-9, 0.8e-7]
    )
    assert abs(camera.get_image().max() - 0.8 * 255) <= 0.05 * 255


def test_autoexpose_starts_from_a_bound_when_the_exposure_is_zero() -> None:
    """From the lower bound when it is positive, and from the upper bound when the
    lower one is zero.
    """
    camera = _ResponsiveCamera(
        lambda exposure: 1e5 * exposure, exposure_bounds=(1e-5, 1.0), exposure_s=0.0
    )
    camera.autoexpose()
    assert camera.requested_exposures[0] == 1e-5
    assert abs(camera.get_image().max() - DEFAULT_TARGET_COUNTS) <= (
        DEFAULT_TOLERANCE_COUNTS
    )

    camera = _ResponsiveCamera(
        lambda exposure: 1e3 * exposure, exposure_bounds=None, exposure_s=0.0
    )
    camera.autoexpose()
    assert camera.requested_exposures[0] == DEFAULT_MAX_EXPOSURE
    assert abs(camera.get_image().max() - DEFAULT_TARGET_COUNTS) <= (
        DEFAULT_TOLERANCE_COUNTS
    )


def test_autoexpose_reports_the_exposure_it_lands_on_when_settling() -> None:
    """The search settles on 249 us. The camera truncates this exposure to 248 us when
    it is set again, and the reported frame is one taken at 248 us.
    """
    camera = _ResponsiveCamera(
        lambda exposure: 100.0 * (exposure / 249e-6) ** 3,
        whole_microseconds=True,
        exposure_s=249e-6,
    )

    with pytest.warns(UserWarning, match="peaks at 99 of 255"):
        exposure = camera.autoexpose(max_iterations=1)

    assert exposure == camera.get_exposure() == 248e-6
    assert camera.frame_exposures == [249e-6, 317e-6, 248e-6]


# --- autoexpose: restoring the camera and measuring a region -------------------------


def test_autoexpose_restores_the_camera_when_a_capture_fails() -> None:
    camera = _ResponsiveCamera(
        lambda exposure: 1e5 * exposure, exposure_s=1.0, failing_frame=3
    )
    camera.set_roi(PRESET_REGION)

    with pytest.raises(TimeoutError, match="no frame"):
        camera.autoexpose()

    assert camera.roi == PRESET_REGION
    assert camera.get_exposure() == 1.0


@pytest.mark.parametrize(
    ("excluded_pixels", "options"),
    [
        ([(16, 3)], {}),
        ([(-1, 3)], {}),
        ([], {"roi": PRESET_REGION, "mask": np.ones((3, 3), dtype=bool)}),
        ([], {"roi": PRESET_REGION, "mask": np.ones((1, 8), dtype=bool)}),
        ([], {"roi": ROI(12, 20, 8, 8), "mask": np.ones((8, 8), dtype=bool)}),
        ([], {"roi": ROI(20, 30, 4, 4)}),
        ([], {"roi": PRESET_REGION, "mask": np.zeros((8, 8), dtype=bool)}),
    ],
    ids=[
        "excluded-pixel-off-the-frame",
        "negative-excluded-pixel",
        "mask-of-another-shape",
        "mask-of-one-row",
        "mask-with-an-overhanging-region",
        "region-off-the-frame",
        "mask-that-keeps-nothing",
    ],
)
def test_autoexpose_leaves_the_camera_as_it_was_when_the_setup_fails(
    excluded_pixels: list[tuple[int, int]], options: dict[str, object]
) -> None:
    camera = _ResponsiveCamera(_saturated, exposure_s=2e-3)
    camera.set_roi(ROI(2, 3, 5, 6))
    camera.excluded_pixels = excluded_pixels

    with pytest.raises(ValueError):
        camera.autoexpose(**options)

    assert camera.captured_frames == 0
    assert camera.roi == ROI(2, 3, 5, 6)
    assert camera.get_exposure() == 2e-3


def test_autoexpose_meters_the_part_of_a_roi_on_the_frame() -> None:
    """A region overhanging the far corner meters its pixels on the frame. A region
    with a negative corner meters the covered part of the frame. A plain slice with a
    negative start wraps round to the far corner, and the metering leaves that corner
    out.
    """
    scene = np.full(SENSOR_RESOLUTION, 0.05)
    scene[14, 22] = 1.0
    scene[8, 12] = 3.0
    camera = _ResponsiveCamera(lambda exposure: 1e5 * exposure, scene=scene)

    camera.autoexpose(roi=ROI(12, 20, 8, 8))
    frame = camera.get_image()
    assert abs(frame[14, 22] - DEFAULT_TARGET_COUNTS) <= DEFAULT_TOLERANCE_COUNTS
    assert frame[8, 12] == camera.max_pixel_value

    # A slice from row -4 and column -4 wraps round to rows 12 to 15 and columns 20 to
    # 23, which hold the pixel at (14, 22) and not the brighter one at (8, 12).
    camera.autoexpose(roi=ROI(-4, -4, 24, 30))
    frame = camera.get_image()
    assert abs(frame[8, 12] - DEFAULT_TARGET_COUNTS) <= DEFAULT_TOLERANCE_COUNTS


# --- autoexpose: stuck pixels --------------------------------------------------------


def test_autoexpose_skips_stuck_pixel_detection_when_it_rails() -> None:
    camera = _ResponsiveCamera(_saturated, exposure_s=1e-2)
    camera.excluded_pixels = [(0, 1)]

    with pytest.warns(UserWarning, match="railed"):
        camera.autoexpose(raise_on_rail=False, detect_stuck_pixels=True)

    assert camera.excluded_pixels == [(0, 1)]


def test_autoexpose_warns_when_one_exposure_is_too_few_for_detection() -> None:
    camera = _ResponsiveCamera(
        lambda exposure: DEFAULT_TARGET_COUNTS * exposure / 1e-3, exposure_s=1e-3
    )
    camera.excluded_pixels = [(0, 1)]

    with pytest.warns(UserWarning, match="Stuck pixels were not detected"):
        camera.autoexpose(detect_stuck_pixels=True)

    assert camera.frame_exposures == [1e-3]
    assert camera.excluded_pixels == [(0, 1)]


# --- averaged frames, excluded pixels and stuck pixels -------------------------------


def test_an_averaged_image_is_the_mean_of_fresh_frames() -> None:
    camera = _LinearSensorCamera(np.full(SENSOR_RESOLUTION, 1e5), read_noise=5.0)

    averaged = camera.get_averaged_image(averaging=8)

    frames = camera.rendered_frames
    assert len(frames) == 8
    np.testing.assert_allclose(
        averaged, np.mean(np.stack(frames).astype(np.float64), axis=0)
    )
    # Each frame carries its own read noise, so no frame repeats the one before.
    assert not any(
        np.array_equal(earlier, later) for earlier, later in zip(frames, frames[1:])
    )


def test_excluded_pixels_accepts_an_array_of_pairs() -> None:
    """An ``(N, 2)`` array from ``np.argwhere`` sets the pixels, and the list read back
    is a copy.
    """
    camera = _ResponsiveCamera(_saturated)
    dark_frame = np.zeros(SENSOR_RESOLUTION)
    dark_frame[3, 5] = 40.0
    dark_frame[10, 2] = 40.0

    camera.excluded_pixels = np.argwhere(dark_frame > 20.0)
    assert camera.excluded_pixels == [(3, 5), (10, 2)]
    assert all(
        type(row) is int and type(col) is int for row, col in camera.excluded_pixels
    )

    camera.excluded_pixels = np.array([[7, 1]])
    assert camera.excluded_pixels == [(7, 1)]

    camera.excluded_pixels = []
    assert camera.excluded_pixels == []
    camera.excluded_pixels = [(7, 1)]
    camera.excluded_pixels = np.empty((0, 2), dtype=np.int64)
    assert camera.excluded_pixels == []

    with pytest.raises(ValueError, match="pairs"):
        camera.excluded_pixels = np.array([1, 2, 3])

    camera.excluded_pixels = [(7, 1)]
    returned = camera.excluded_pixels
    returned.append((0, 0))
    assert camera.excluded_pixels == [(7, 1)]


def test_find_stuck_pixels_puts_back_the_region_and_exposure() -> None:
    """The stuck pixel lies outside the preset region, and it is found at its sensor
    coordinates.
    """
    stuck_pixel = (3, 12)
    camera = _LinearSensorCamera(
        np.full(SENSOR_RESOLUTION, 1e5), stuck_pixels={stuck_pixel: 200}
    )
    camera.set_roi(PRESET_REGION)
    camera.set_exposure(2e-3)

    found = camera.find_stuck_pixels(exposures=[1e-4, 3e-4, 1e-3, 3e-3])

    assert found == [stuck_pixel]
    assert camera.roi == PRESET_REGION
    assert camera.get_exposure() == 2e-3


def test_default_sweep_keeps_a_dim_working_pixel() -> None:
    """A working pixel at a fifth of its neighbours' brightness rises with the
    exposure between two half-decade steps. With decade steps, no pair of consecutive
    readings has the first above the signal floor and the second below full scale.
    The same pixel therefore looks stuck.
    """
    rate = np.full(SENSOR_RESOLUTION, 1e4)
    dim_pixel = (5, 7)
    rate[dim_pixel] = 2e3
    camera = _LinearSensorCamera(rate)

    assert camera.find_stuck_pixels() == []

    frames, exposures = camera._capture_exposure_sweep()
    decades = [0, 2, 4, 6]
    flagged = camera._detect_stuck_pixels(
        frames[decades], [exposures[index] for index in decades]
    )
    assert flagged == [dim_pixel]


def test_capture_exposure_sweep_defaults_to_half_decades() -> None:
    """Seven exposures a half decade apart from the lower bound. The camera truncates
    each exposure to whole microseconds, and the sweep reports the value read back.
    """
    camera = _ResponsiveCamera(
        lambda exposure: 1e3 * exposure, whole_microseconds=True, exposure_s=2e-3
    )

    frames, exposures = camera._capture_exposure_sweep()

    requested = [1e-4 * 10.0 ** (step / 2) for step in range(7)]
    assert frames.shape == (7, *SENSOR_RESOLUTION)
    assert camera.requested_exposures[:7] == pytest.approx(requested)
    assert exposures == camera.frame_exposures
    assert exposures[1] == 316e-6
    assert camera.get_exposure() == 2e-3
