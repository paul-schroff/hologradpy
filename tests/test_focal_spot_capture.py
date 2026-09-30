"""Capturing the focal spot that seeds a PSF kernel.

The failure these guard against is quiet: with the zeroth order away from the sensor
center the crop used to be taken wherever the zeroth order landed, so near an edge it
held a corner of the spot, and off the sensor it held no spot at all. The kernel still
had the right shape and the fit still ran, it just started from noise.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
import torch

from hologradpy.calibration.camera_mapping import (
    CameraMapping,
    FocalSpotFit,
)
from hologradpy.calibration.spot_detection import (
    capture_focal_spot,
    tilt_to_sensor_center,
)
from hologradpy.geometry import PartialAffineTransform
from hologradpy.grids import plane_center
from hologradpy.hardware import SimulatedCameraTorch, SimulatedSLMTorch
from hologradpy.optics.complex_amplitude import ComplexAmplitude, FieldGeometry
from hologradpy.optics.modules.slm_fields import PixelwiseSLMField
from hologradpy.optics.systems import SLMFFTAffine
from hologradpy.profiles.amplitude import gaussian_beam_intensity
from hologradpy.roi import ROI
from tests.native_camera_fakes import CroppingCamera

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

DEVICE = torch.device("cpu")
SLM_RESOLUTION = (64, 64)
CAMERA_RESOLUTION = (96, 96)
CAMERA_PIXEL_SIZE = (30e-6, 30e-6)
FOCAL_LENGTH = 0.1


def _mapping(
    zeroth_order_position,
    scale: float = 1.0,
    angle: float = 0.0,
    output_pixel_size: tuple[float, float] | None = CAMERA_PIXEL_SIZE,
):
    """A camera mapping whose similarity sends the zeroth order at
    ``zeroth_order_position``, ``(row, column)``, to the centre of an output plane of
    the camera's size, at ``output_pixel_size``.
    """
    center = np.asarray(plane_center(CAMERA_RESOLUTION), dtype=float)  # (x, y)
    zeroth_xy = np.array([zeroth_order_position[1], zeroth_order_position[0]])
    turned = PartialAffineTransform.from_components(scale=scale, angle_deg=angle)
    transform = PartialAffineTransform.from_components(
        scale=scale,
        angle_deg=angle,
        shift=tuple(center - turned.transform_points([zeroth_xy])[0]),
    )
    detected = np.random.default_rng(0).uniform(0.0, 96.0, size=(8, 2))
    return CameraMapping(
        timestamp=datetime.now(),
        name="synthetic",
        transform=transform.as_matrix(homogeneous=False),
        detected_points=detected.tolist(),
        calculated_points=transform.transform_points(detected).tolist(),
        zeroth_order_position=zeroth_order_position,
        spot_fit=FocalSpotFit(waist=CAMERA_PIXEL_SIZE[0] * 2),
        output_pixel_size=output_pixel_size,
        output_resolution=CAMERA_RESOLUTION,
    )


class _FakeCamera(CroppingCamera):
    """A native camera for the tilt arithmetic. The arithmetic reads only the geometry,
    so the camera never captures a frame.
    """

    def __init__(self) -> None:
        super().__init__(CAMERA_RESOLUTION)

    @property
    def pixel_size(self) -> np.ndarray:
        return np.asarray(CAMERA_PIXEL_SIZE)

    def render_sensor_frame(self) -> np.ndarray:
        raise AssertionError("The tilt is worked out from the geometry alone.")


def test_a_centered_zeroth_order_needs_no_steering() -> None:
    """The tilt is a correction, so a beam already in the middle gets none. Anything
    else would move a working setup off center.
    """
    center = (CAMERA_RESOLUTION[0] / 2, CAMERA_RESOLUTION[1] / 2)

    tilt = tilt_to_sensor_center(_FakeCamera(), _mapping(center))

    assert tilt == pytest.approx((0.0, 0.0), abs=1e-15)


def test_the_tilt_carries_the_beam_from_the_zeroth_order_to_the_middle() -> None:
    """With an identity mapping the tilt is the camera-plane displacement in metres,
    (x, y), from a zeroth order stored (row, column).
    """
    zeroth = (20.0, 70.0)  # (row, column)

    tilt = tilt_to_sensor_center(_FakeCamera(), _mapping(zeroth))

    expected_x = (CAMERA_RESOLUTION[1] / 2 - zeroth[1]) * CAMERA_PIXEL_SIZE[1]
    expected_y = (CAMERA_RESOLUTION[0] / 2 - zeroth[0]) * CAMERA_PIXEL_SIZE[0]
    assert tilt == pytest.approx((expected_x, expected_y))


def test_the_tilt_is_taken_through_the_mapping_not_along_the_sensor() -> None:
    """A tilt steers along the *model* axes, which are rotated with respect to the
    sensor, so the camera-plane displacement cannot be used directly. Rotating the
    mapping by 90 degrees must swap the two components.
    """
    zeroth = (20.0, 70.0)
    camera = _FakeCamera()

    straight = tilt_to_sensor_center(camera, _mapping(zeroth))
    rotated = tilt_to_sensor_center(camera, _mapping(zeroth, angle=90.0))

    # A quarter turn sends (x, y) to (-y, x). The partial affine is fitted from the
    # point pairs, which holds it to a relative 1e-8.
    assert rotated == pytest.approx((-straight[1], straight[0]), rel=1e-6)


def test_a_model_plane_sampled_finer_than_the_sensor_gives_the_same_tilt() -> None:
    """A model plane at half the camera pitch holds twice the pixels over the same
    distance, so the tilt in metres is unchanged.
    """
    zeroth = (20.0, 70.0)
    camera = _FakeCamera()
    half_pitch = (CAMERA_PIXEL_SIZE[0] / 2, CAMERA_PIXEL_SIZE[1] / 2)

    unit = tilt_to_sensor_center(camera, _mapping(zeroth))
    finer = tilt_to_sensor_center(
        camera, _mapping(zeroth, scale=2.0, output_pixel_size=half_pitch)
    )

    assert finer == pytest.approx(unit)


def test_a_demagnified_focal_plane_doubles_the_tilt() -> None:
    """At the camera pitch, a mapping of scale two means the focal plane is imaged at
    half size, so the spot has twice as far to go in the focal plane.
    """
    zeroth = (20.0, 70.0)
    camera = _FakeCamera()

    unit = tilt_to_sensor_center(camera, _mapping(zeroth))
    doubled = tilt_to_sensor_center(camera, _mapping(zeroth, scale=2.0))

    assert doubled == pytest.approx((2 * unit[0], 2 * unit[1]))


def test_a_mapping_without_an_output_plane_is_refused() -> None:
    with pytest.raises(ValueError, match="no output plane"):
        tilt_to_sensor_center(
            _FakeCamera(), _mapping((20.0, 70.0), output_pixel_size=None)
        )


def _build_hardware(camera_shift):
    """A small simulated bench whose zeroth order sits wherever ``camera_shift`` puts
    it.
    """
    geometry = FieldGeometry(
        resolution=SLM_RESOLUTION,
        pixel_size=torch.tensor([12.5e-6, 12.5e-6], device=DEVICE),
        wavelength=torch.tensor(0.63e-6, device=DEVICE),
    )
    slm = SimulatedSLMTorch(input_geometry=geometry, bitdepth=8)

    grid_x, grid_y = geometry.get_spatial_grid()
    amplitude = gaussian_beam_intensity(grid_x, grid_y, beam_radius=3e-4).sqrt()
    beam = ComplexAmplitude(
        amplitude.to(torch.complex64),
        wavelength=geometry.wavelength,
        pixel_size=geometry.pixel_size,
        power=1e-3,
    )
    hardware = SLMFFTAffine(
        input_geometry=geometry,
        virtual_slm=slm.virtual_slm,
        camera_resolution=CAMERA_RESOLUTION,
        camera_pixel_size=CAMERA_PIXEL_SIZE,
        focal_length=FOCAL_LENGTH,
        slm_field=PixelwiseSLMField(beam),
        padded_resolution=(256, 256),
        camera_angle=0.0,
        camera_shift=tuple(
            s * p for s, p in zip(camera_shift, CAMERA_PIXEL_SIZE)
        ),
    )
    camera = SimulatedCameraTorch(hardware, add_noise=False)
    # Roughly mid-scale for this bench. A milliwatt onto a 96 x 96 sensor with no
    # attenuation saturates by eight orders of magnitude. An overexposed frame hides the
    # peak, so autoexposure from there only steps down geometrically and uses up its
    # whole budget.
    camera.set_exposure(2.5e-11)
    camera.get_image()
    return slm, camera


def _zeroth_order_position(slm, camera) -> tuple[float, float]:
    """Where a flat pattern actually puts the spot, in camera pixels ``(row, col)``.

    Metered first and taken as a centroid rather than an argmax. At the bench's default
    exposure the spot saturates over hundreds of pixels, and the argmax of a flat top is
    whichever pixel the scan reaches first, which is not the spot.
    """
    slm.set_phase(np.zeros(tuple(slm.resolution)))
    camera.autoexpose(set_fraction=0.6)
    image = np.asarray(camera.get_image(), dtype=float)

    bright = image > 0.5 * image.max()
    rows, columns = np.nonzero(bright)
    weights = image[bright]
    return (
        float((rows * weights).sum() / weights.sum()),
        float((columns * weights).sum() / weights.sum()),
    )


@pytest.mark.parametrize(
    "camera_shift", [(0.0, 0.0), (18.0, -12.0)], ids=["centered", "off center"]
)
def test_the_captured_spot_lands_in_the_middle_of_the_kernel(camera_shift) -> None:
    """The kernel is applied about its own center, so a spot sitting off center in the
    crop is an unwanted tilt baked into the seed. Steering puts the spot in the middle
    whatever the geometry, and the same crop then holds it.
    """
    slm, camera = _build_hardware(camera_shift)
    zeroth = _zeroth_order_position(slm, camera)
    kernel_size = 21

    kernel = capture_focal_spot(
        slm, camera, _mapping(zeroth), FOCAL_LENGTH, kernel_size
    )

    assert kernel.shape == (kernel_size, kernel_size)
    peak = np.unravel_index(int(np.argmax(kernel)), kernel.shape)
    # Exact: abs=1 here is the width of the very off-by-one this would catch.
    assert peak == (kernel_size // 2, kernel_size // 2)
    # And it is a spot, not the noise floor a mis-aimed crop would return.
    edge = np.concatenate(
        [kernel[0], kernel[-1], kernel[:, 0], kernel[:, -1]]
    ).mean()
    assert kernel.max() > 10 * max(edge, 1e-12)


def test_capture_ignores_a_window_left_on_the_camera() -> None:
    """The spot is steered to the middle of the whole sensor and captured there, so a
    window left in a corner changes nothing. The window and the exposure are put back
    afterwards.
    """
    slm, camera = _build_hardware((18.0, -12.0))
    zeroth = _zeroth_order_position(slm, camera)
    exposure = camera.get_exposure()
    expected = capture_focal_spot(slm, camera, _mapping(zeroth), FOCAL_LENGTH, 21)
    assert camera.get_exposure() == exposure

    window = ROI(0, 0, 30, 40)
    camera.set_roi(window)
    kernel = capture_focal_spot(slm, camera, _mapping(zeroth), FOCAL_LENGTH, 21)

    np.testing.assert_array_equal(kernel, expected)
    assert camera.roi == window
    assert camera.get_exposure() == exposure


def test_a_spot_that_never_arrives_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty frame holds no point spread function, so the capture raises an error
    and returns no kernel.

    Autoexposure starts from a bound when the camera sits at zero. The test holds
    autoexposure off, so every frame stays empty.
    """
    slm, camera = _build_hardware((0.0, 0.0))
    zeroth = _zeroth_order_position(slm, camera)
    camera.set_exposure(0.0)
    monkeypatch.setattr(camera, "autoexpose", lambda **_: camera.get_exposure())

    # A mapping that claims the beam is somewhere it is not sends the steering the
    # wrong way, which is exactly the case that must not pass silently.
    with pytest.raises(RuntimeError, match="No focal spot found"):
        capture_focal_spot(
            slm, camera, _mapping((zeroth[0], zeroth[1])), FOCAL_LENGTH, 21
        )
