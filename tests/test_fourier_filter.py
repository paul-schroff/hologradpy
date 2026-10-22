"""The band of frequencies that light from the SLM can put on the camera, and the
Fourier filter to it.

The model is a stand-in carrying only the SLM-plane geometry and the focal length,
since the band is computed from these alone.
"""

from __future__ import annotations

import math
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hologradpy.analysis.fourier_filter import (
    filtered_noise_rms,
    fourier_filter,
    intensity_band_mask,
    white_noise_rms,
)
from hologradpy.calibration.camera_mapping import CameraMapping, FocalSpotFit
from hologradpy.grids import pixel_frame_angle, pixel_frame_vector
from hologradpy.optics.complex_amplitude import FieldGeometry

SLM_RESOLUTION = (64, 64)
SLM_PIXEL_SIZE = 12.5e-6
CAMERA_RESOLUTION = (64, 64)
CAMERA_PIXEL_SIZE = 20e-6
WAVELENGTH = 0.670e-6
FOCAL_LENGTH = 0.1
# The autocorrelation of the SLM aperture reaches L / (wavelength * focal_length).
BAND = SLM_RESOLUTION[1] * SLM_PIXEL_SIZE / (WAVELENGTH * FOCAL_LENGTH)
# An SLM twice as wide as it is tall. Its band reaches BAND along x and half of it
# along y.
WIDE_SLM_RESOLUTION = (32, 64)


def _model(resolution: tuple[int, int] = SLM_RESOLUTION) -> SimpleNamespace:
    geometry = FieldGeometry(
        resolution=resolution,
        pixel_size=torch.tensor([SLM_PIXEL_SIZE, SLM_PIXEL_SIZE]),
        wavelength=torch.tensor(WAVELENGTH),
    )
    return SimpleNamespace(input_geometry=geometry, focal_length=FOCAL_LENGTH)


def _mapping(
    angle_degrees: float = 0.0, camera_pixel_size: float = CAMERA_PIXEL_SIZE
) -> CameraMapping:
    """A mapping turned by ``angle_degrees`` about the centre of the sensor, from
    +column towards +row.
    """
    center = np.array([CAMERA_RESOLUTION[1] / 2, CAMERA_RESOLUTION[0] / 2])
    detected = np.array([(10.0, 12.0), (40.0, 15.0), (20.0, 45.0), (50.0, 50.0)])
    angle = math.radians(angle_degrees)
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    calculated = (detected - center) @ rotation.T + center
    return CameraMapping(
        timestamp=datetime(2026, 1, 1),
        name="turned",
        transform=np.column_stack([rotation, center - rotation @ center]),
        detected_points=[tuple(point) for point in detected],
        calculated_points=[tuple(point) for point in calculated],
        spot_fit=FocalSpotFit(waist=2.0),
        output_pixel_size=(camera_pixel_size, camera_pixel_size),
        output_resolution=CAMERA_RESOLUTION,
    )


def _image_plane_cosine(
    mapping: CameraMapping, frequency: float, direction_degrees: float = 0.0
) -> torch.Tensor:
    """A cosine in the image plane, with its wave vector at ``direction_degrees`` from
    +x towards +y. It is drawn on the sensor and tapered by a Hann window so its
    spectrum stays near its own frequency.
    """
    rows, columns = np.indices(CAMERA_RESOLUTION)
    pixels = np.stack([columns.ravel(), rows.ravel()], axis=1).astype(float)
    x, y = mapping.sensor_to_image_plane(pixels).T.reshape(2, *CAMERA_RESOLUTION)
    direction = math.radians(direction_degrees)
    along_wave_vector = math.cos(direction) * x + math.sin(direction) * y
    window = np.outer(
        np.hanning(CAMERA_RESOLUTION[0]), np.hanning(CAMERA_RESOLUTION[1])
    )
    return torch.as_tensor(np.cos(2 * np.pi * frequency * along_wave_vector) * window)


def _retained(image: torch.Tensor, mask: torch.Tensor) -> float:
    """The share of the image's power left after the Fourier filter."""
    filtered = fourier_filter(image, mask)
    return float((filtered**2).sum() / (image**2).sum())


def test_the_band_reaches_the_autocorrelation_of_the_aperture() -> None:
    mask = intensity_band_mask(_model(), _mapping(), CAMERA_RESOLUTION)

    inside = _image_plane_cosine(_mapping(), 0.6 * BAND)
    outside = _image_plane_cosine(_mapping(), 1.5 * BAND)

    assert _retained(inside, mask) > 0.9
    assert _retained(outside, mask) < 0.1


def test_the_band_turns_with_the_camera() -> None:
    """On a camera turned by 30 degrees, the same image-plane cosines keep and lose
    their power. The linear part of the mapping is the turned one.
    """
    turned = _mapping(30.0)
    mask = intensity_band_mask(_model(), turned, CAMERA_RESOLUTION)

    assert _retained(_image_plane_cosine(turned, 0.6 * BAND), mask) > 0.9
    assert _retained(_image_plane_cosine(turned, 1.5 * BAND), mask) < 0.1

    relative = turned.sensor_pixels_per_metre @ np.linalg.inv(
        _mapping().sensor_pixels_per_metre
    )
    # The partial affine is fitted from the point pairs, to about 1e-8.
    assert relative @ relative.T == pytest.approx(np.eye(2), abs=1e-6)
    assert abs(math.degrees(math.atan2(relative[1, 0], relative[0, 0]))) == (
        pytest.approx(30.0, abs=1e-4)
    )


@pytest.mark.parametrize("angle_degrees", [20.0, -35.0])
def test_the_band_of_a_wide_slm_turns_with_the_camera(angle_degrees: float) -> None:
    """An SLM twice as wide as tall holds a band twice as wide along x as along y. On a
    turned camera the band is that rectangle turned by the angle of the mapping.

    A bin of the sensor's FFT grid is a frequency in cycles per pixel along the
    columns and the rows. Divided by the pitch, it is a frequency along the camera's x
    and y, which the angle of the mapping turns into the image plane.
    """
    mapping = _mapping(angle_degrees)
    mask = intensity_band_mask(_model(WIDE_SLM_RESOLUTION), mapping, CAMERA_RESOLUTION)

    height, width = CAMERA_RESOLUTION
    rows, columns = np.indices(CAMERA_RESOLUTION)
    camera_frequency_x, camera_frequency_y = pixel_frame_vector(
        (
            (columns - width // 2) / (width * CAMERA_PIXEL_SIZE),
            (rows - height // 2) / (height * CAMERA_PIXEL_SIZE),
        )
    )
    angle = math.radians(pixel_frame_angle(mapping.affine.rotation_degrees))
    frequency_x = (
        math.cos(angle) * camera_frequency_x - math.sin(angle) * camera_frequency_y
    )
    frequency_y = (
        math.sin(angle) * camera_frequency_x + math.cos(angle) * camera_frequency_y
    )
    band_x, band_y = BAND, BAND / 2
    # Bins within a relative 1e-6 of an edge are left out, so rounding decides none.
    inside = (np.abs(frequency_x) < (1 - 1e-6) * band_x) & (
        np.abs(frequency_y) < (1 - 1e-6) * band_y
    )
    outside = (np.abs(frequency_x) > (1 + 1e-6) * band_x) | (
        np.abs(frequency_y) > (1 + 1e-6) * band_y
    )

    assert mask.numpy()[inside].all()
    assert not mask.numpy()[outside].any()
    # A cosine between the two edges keeps its power along x and loses it along y.
    assert _retained(_image_plane_cosine(mapping, 0.75 * BAND), mask) > 0.9
    assert _retained(_image_plane_cosine(mapping, 0.75 * BAND, 90.0), mask) < 0.1


def test_the_filter_is_idempotent_and_keeps_the_band() -> None:
    mask = intensity_band_mask(_model(), _mapping(), CAMERA_RESOLUTION)
    generator = torch.Generator().manual_seed(0)
    noise = torch.randn(CAMERA_RESOLUTION, generator=generator, dtype=torch.float64)

    in_band = fourier_filter(noise, mask)

    assert torch.allclose(fourier_filter(in_band, mask), in_band, atol=1e-12)


def test_white_noise_keeps_the_share_of_the_band() -> None:
    mask = intensity_band_mask(_model(), _mapping(), CAMERA_RESOLUTION)
    share = float(mask.double().mean())
    generator = torch.Generator().manual_seed(1)
    noise = torch.randn(CAMERA_RESOLUTION, generator=generator, dtype=torch.float64)

    filtered = fourier_filter(noise, mask)

    assert float(filtered.std()) == pytest.approx(math.sqrt(share), rel=0.05)


def test_the_noise_floor_matches_injected_white_noise() -> None:
    mask = intensity_band_mask(_model(), _mapping(), CAMERA_RESOLUTION)
    share = float(mask.double().mean())
    generator = torch.Generator().manual_seed(2)
    noise = torch.randn(CAMERA_RESOLUTION, generator=generator, dtype=torch.float64)
    noise = 0.3 * noise
    region = torch.ones(CAMERA_RESOLUTION, dtype=torch.bool)

    floor = filtered_noise_rms(noise, region, mask)

    assert floor == pytest.approx(0.3 * math.sqrt(share), rel=0.05)


def test_no_noise_floor_when_the_band_fills_the_spectrum() -> None:
    """A camera coarser than the band has no bins that hold noise alone."""
    coarse = _mapping(camera_pixel_size=60e-6)
    mask = intensity_band_mask(_model(), coarse, CAMERA_RESOLUTION)

    assert bool(mask.all())
    region = torch.ones(CAMERA_RESOLUTION, dtype=torch.bool)
    assert filtered_noise_rms(torch.rand(CAMERA_RESOLUTION), region, mask) is None
    assert white_noise_rms(torch.rand(CAMERA_RESOLUTION), region, mask) is None


def test_the_noise_level_of_several_frames_matches_injected_white_noise() -> None:
    """The standard deviation comes from the bins outside the band of every frame,
    with the frames set to zero outside the region.
    """
    mask = intensity_band_mask(_model(), _mapping(), CAMERA_RESOLUTION)
    generator = torch.Generator().manual_seed(3)
    noise = 0.3 * torch.randn(
        (4, *CAMERA_RESOLUTION), generator=generator, dtype=torch.float64
    )
    region = torch.zeros(CAMERA_RESOLUTION, dtype=torch.bool)
    region[8:56, 4:60] = True

    level = white_noise_rms(noise, region, mask)

    assert level == pytest.approx(0.3, rel=0.05)
