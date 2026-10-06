"""The backgrounds subtracted from camera frames by a caller, and how a camera reports
overexposure.

``background_at`` evaluates a background for a frame. The background is given as one
level, a whole-sensor frame or a function of the exposure. ``Camera.overexposed`` tells
whether the last frame held a pixel at full scale. The ``preserve_roi`` and
``preserve_exposure_and_roi`` blocks are tested in ``tests/test_camera_base.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from hologradpy.hardware import as_camera
from hologradpy.hardware.camera import background_at
from hologradpy.roi import ROI
from tests.native_camera_fakes import CroppingCamera
from tests.test_device_interface import _raw_camera

SENSOR_RESOLUTION = (16, 24)


class _FrameCamera(CroppingCamera):
    """An 8-bit camera that returns one fixed frame."""

    def __init__(self, frame: NDArray) -> None:
        super().__init__(np.shape(frame))
        self._frame = np.asarray(frame)

    def render_sensor_frame(self) -> NDArray:
        return self._frame.copy()


# --- Backgrounds -------------------------------------------------------------------


def test_a_background_level_applies_to_every_pixel() -> None:
    assert background_at(7.5, None, SENSOR_RESOLUTION) == 7.5
    assert type(background_at(np.float32(7.5), 1e-3, SENSOR_RESOLUTION)) is float


def test_a_background_frame_has_the_shape_of_the_frames() -> None:
    frame = np.arange(np.prod(SENSOR_RESOLUTION), dtype=np.uint16).reshape(
        SENSOR_RESOLUTION
    )

    counts = background_at(frame, None, SENSOR_RESOLUTION)

    assert counts.dtype == np.float64
    np.testing.assert_array_equal(counts, frame)
    with pytest.raises(ValueError, match="shape"):
        background_at(frame.T, None, SENSOR_RESOLUTION)


def test_a_background_function_is_evaluated_at_the_exposure() -> None:
    """A level or a frame that grows with the exposure, as dark counts do."""
    frame = np.full(SENSOR_RESOLUTION, 3.0)

    def growing_level(exposure: float) -> float:
        return 2.0 + 1e3 * exposure

    def growing_frame(exposure: float) -> NDArray:
        return frame * exposure / 1e-3

    assert background_at(growing_level, 1e-3, SENSOR_RESOLUTION) == pytest.approx(3.0)
    np.testing.assert_allclose(
        background_at(growing_frame, 2e-3, SENSOR_RESOLUTION), 2.0 * frame
    )
    with pytest.raises(ValueError, match="no exposure"):
        background_at(growing_level, None, SENSOR_RESOLUTION)


def test_a_background_holds_finite_counts() -> None:
    frame = np.zeros(SENSOR_RESOLUTION)
    frame[1, 2] = np.inf

    with pytest.raises(ValueError, match="finite"):
        background_at(float("nan"), None, SENSOR_RESOLUTION)
    with pytest.raises(ValueError, match="finite"):
        background_at(frame, None, SENSOR_RESOLUTION)


# --- Overexposure --------------------------------------------------------------------


def test_overexposed_reports_a_pixel_of_the_last_frame_at_full_scale() -> None:
    """Every capture sets it, and an excluded pixel is left out. This also holds inside
    a region of interest.
    """
    frame = np.full(SENSOR_RESOLUTION, 100, dtype=np.uint16)
    camera = _FrameCamera(frame)
    assert not camera.overexposed

    frame[3, 4] = 255
    camera.get_image()
    assert camera.overexposed

    camera.set_roi(ROI(2, 3, 4, 4))
    camera.excluded_pixels = [(3, 4)]
    camera.get_image()
    assert not camera.overexposed

    camera.set_roi(ROI(8, 8, 4, 4))
    camera.excluded_pixels = None
    camera.get_image()
    assert not camera.overexposed


def test_a_mask_limits_the_pixels_whose_overexposure_counts() -> None:
    """The frame comes back whole, and only the pixels kept by the mask are checked
    for overexposure.
    """
    frame = np.full(SENSOR_RESOLUTION, 100, dtype=np.uint16)
    frame[3, 4] = 255
    camera = _FrameCamera(frame)
    mask = np.ones(SENSOR_RESOLUTION, dtype=bool)
    mask[3, 4] = False

    image = camera.get_image(mask=mask)

    assert not camera.overexposed
    np.testing.assert_array_equal(image, frame)
    camera.get_image(mask=~mask)
    assert camera.overexposed
    with pytest.raises(ValueError, match="mask"):
        camera.get_image(mask=np.ones((4, 4), dtype=bool))


def test_an_averaged_capture_is_overexposed_where_every_frame_was() -> None:
    frame = np.full(SENSOR_RESOLUTION, 100, dtype=np.uint16)
    frame[3, 4] = 255
    camera = _FrameCamera(frame)

    camera.get_image(averaging=3)
    assert camera.overexposed

    frame[3, 4] = 252
    camera.get_image(averaging=3)
    assert not camera.overexposed


def test_a_pixel_clipped_just_below_full_scale_is_overexposed() -> None:
    """From 99 % of full scale, as a sensor can clip a count or two short of it: 253 of
    255 here.
    """
    frame = np.full(SENSOR_RESOLUTION, 100, dtype=np.uint16)
    frame[3, 4] = 253
    camera = _FrameCamera(frame)

    camera.get_image()
    assert camera.overexposed

    frame[3, 4] = 252
    camera.get_image()
    assert not camera.overexposed


def test_every_adapter_of_one_camera_shares_whether_it_is_overexposed() -> None:
    """A consumer coerces the camera passed to it into an adapter of its own, so the
    flag is kept on the camera itself.
    """
    raw = _raw_camera()
    raw._sensor[2, 3] = 255

    as_camera(raw).get_image()

    assert as_camera(raw).overexposed
