"""Contract checks on a real camera and SLM, selected with the ``hardware`` marker.

Each device is named in the environment and opened through the factory. A test skips
when its device is not named, so an ordinary run skips the whole file.

``HOLOGRADPY_CAMERA`` and ``HOLOGRADPY_SLM`` hold a backend name registered by
:func:`~hologradpy.hardware.register_slmsuite_backends`, such as ``thorlabs``, or a
``module:Attr`` import spec, such as ``devices:ZeluxCamera`` with that module on the
path. The constructor arguments go in ``HOLOGRADPY_CAMERA_OPTIONS`` and
``HOLOGRADPY_SLM_OPTIONS`` as a JSON object. For example::

    HOLOGRADPY_CAMERA=devices:ZeluxCamera
    HOLOGRADPY_CAMERA_OPTIONS={"serial": "12345"}
    pytest -m hardware tests/test_hardware_contract.py
"""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest

from hologradpy.hardware import (
    Camera,
    SLM,
    open_camera,
    open_slm,
    register_slmsuite_backends,
)
from hologradpy.hardware.camera import CameraData
from hologradpy.hardware.slm import SLMData
from hologradpy.phase_levels import level_dtype
from hologradpy.roi import ROI

pytestmark = pytest.mark.hardware


def _driver_named_by(variable: str) -> tuple[Callable[..., Any] | str, dict]:
    """The driver named in the environment variable ``variable``, and its options.

    The test skips when the variable is unset.
    """
    name = os.environ.get(variable)
    if not name:
        pytest.skip(f"{variable} is not set, so there is no device to test.")
    options = json.loads(os.environ.get(f"{variable}_OPTIONS", "{}"))
    if ":" in name:
        module_path, _, attribute = name.partition(":")
        return getattr(importlib.import_module(module_path), attribute), options
    register_slmsuite_backends()
    return name, options


def _close(device: Camera | SLM) -> None:
    close = getattr(device, "close", None)
    if close is not None:
        close()


@pytest.fixture(scope="module")
def camera() -> Iterator[Camera]:
    driver, options = _driver_named_by("HOLOGRADPY_CAMERA")
    device = open_camera(driver, **options)
    try:
        yield device
    finally:
        _close(device)


@pytest.fixture(scope="module")
def slm() -> Iterator[SLM]:
    driver, options = _driver_named_by("HOLOGRADPY_SLM")
    device = open_slm(driver, **options)
    try:
        yield device
    finally:
        _close(device)


def _mean_frame(camera: Camera, number_of_frames: int) -> np.ndarray:
    """The mean of ``number_of_frames`` single frames."""
    return np.mean(
        [
            np.asarray(camera.get_image(), dtype=np.float64)
            for _ in range(number_of_frames)
        ],
        axis=0,
    )


# --- Camera ------------------------------------------------------------------------


def test_a_frame_matches_the_reported_geometry(camera: Camera) -> None:
    with camera.preserve_roi(full_sensor=True):
        frame = np.asarray(camera.get_image())

    assert frame.shape == tuple(camera.sensor_resolution)
    assert frame.max() <= camera.max_pixel_value


def test_an_exposure_inside_the_bounds_reads_back(camera: Camera) -> None:
    low, high = camera.exposure_search_bounds
    exposures = np.geomspace(max(low, 1e-5), min(high, 0.1), 5)

    with camera.preserve_exposure_gain_and_roi():
        for exposure in exposures:
            camera.set_exposure(float(exposure))
            assert camera.get_exposure() == pytest.approx(
                exposure, rel=1e-2, abs=1e-6
            )


def test_a_gain_inside_the_bounds_reads_back(camera: Camera) -> None:
    """A gain inside the camera's range reads back inside it, and a higher request
    never reads back lower. A gain read back is a step of the camera, so setting it
    again reads it back unchanged. A camera whose gain cannot be set states 0 dB.
    """
    bounds = camera.gain_bounds
    if bounds is None:
        assert camera.get_gain() == 0.0
        return
    low, high = bounds

    applied = []
    with camera.preserve_exposure_gain_and_roi():
        for gain in np.linspace(low, high, 5):
            camera.set_gain(float(gain))
            applied.append(camera.get_gain())
            camera.set_gain(applied[-1])
            assert camera.get_gain() == applied[-1]

    assert all(low <= gain <= high for gain in applied)
    assert applied == sorted(applied)


def test_a_region_round_trips(camera: Camera) -> None:
    height, width = camera.sensor_resolution
    region = ROI(height // 4, width // 4, height // 2, width // 2)

    with camera.preserve_roi():
        camera.set_roi(region)
        assert camera.roi == region
        assert np.shape(camera.get_image()) == (region.height, region.width)


def test_a_region_shows_the_same_pixels_as_the_whole_frame(camera: Camera) -> None:
    """The region reads the part of the sensor that the whole frame shows at its
    position. The region sits off the centre, so a mirrored or transposed region
    shows other pixels.
    """
    height, width = camera.sensor_resolution
    region = ROI(height // 8, width // 3, height // 4, width // 4)

    with camera.preserve_roi(full_sensor=True):
        whole = _mean_frame(camera, 8)
        camera.set_roi(region)
        cropped = _mean_frame(camera, 8)

    expected = region.crop(whole)
    if np.std(expected) < 1.0:
        pytest.skip("The scene shows too little structure to compare the regions.")
    assert np.corrcoef(cropped.ravel(), expected.ravel())[0, 1] > 0.9


def test_a_camera_snapshot_describes_the_camera(camera: Camera) -> None:
    record = CameraData.from_camera(camera)

    assert record.sensor_resolution == tuple(camera.sensor_resolution)
    assert record.resolution == tuple(camera.resolution)
    assert record.pixel_size == pytest.approx(tuple(camera.pixel_size))
    assert record.exposure == pytest.approx(camera.get_exposure())
    assert record.gain == pytest.approx(camera.get_gain())


# --- SLM ---------------------------------------------------------------------------


def test_levels_read_back_from_the_slm(slm: SLM) -> None:
    if slm.bitdepth is None:
        pytest.skip("This SLM states no bit depth.")
    ramp = np.indices(tuple(slm.resolution)).sum(axis=0) % 2**slm.bitdepth
    levels = ramp.astype(level_dtype(slm.bitdepth))

    slm.set_levels(levels)

    if slm.display is None:
        pytest.skip("This SLM does not report the levels it shows.")
    assert np.array_equal(np.asarray(slm.display), levels)


def test_an_slm_snapshot_describes_the_slm(slm: SLM) -> None:
    record = SLMData.from_slm(slm)

    assert record.resolution == tuple(slm.resolution)
    assert record.pixel_size == pytest.approx(tuple(slm.pixel_size))
    assert record.wavelength == pytest.approx(slm.wavelength)
    assert record.bitdepth == slm.bitdepth
