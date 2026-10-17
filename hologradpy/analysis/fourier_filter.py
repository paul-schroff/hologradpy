"""The Fourier filter of camera frames to the band of the optics, and the white noise
outside the band.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from ..fourier_optics import intensity_nyquist_pixel_size
from ..fourier_transforms import fft_2d, ifft_2d
from ..grids import get_frequency_grid

if TYPE_CHECKING:
    from ..calibration.camera_mapping import CameraMapping
    from ..optics.systems import SLMFourierLensModel


def intensity_band_mask(
    slm_camera_model: SLMFourierLensModel,
    camera_mapping: CameraMapping,
    sensor_shape: tuple[int, int],
) -> Tensor:
    """True at the frequencies of the sensor's FFT grid that an intensity from the SLM
    can hold.

    For an SLM of side ``L`` along an axis, the autocorrelation of its aperture reaches
    ``L / (wavelength * focal_length)`` in the image plane, which is
    ``1 / (2 * intensity_nyquist_pixel_size)``.

    Args:
        slm_camera_model: The model, for the wavelength, the focal length and the SLM
            aperture.
        camera_mapping: How the camera sits relative to the model's image plane.
        sensor_shape: ``(height, width)`` of the frames.

    Returns:
        Tensor: A boolean mask of ``sensor_shape``, in the centred layout of
        :func:`~hologradpy.fourier_transforms.fft_2d`.
    """
    geometry = slm_camera_model.input_geometry
    wavelength = float(geometry.wavelength.reshape(-1)[0])
    pitch_y, pitch_x = (float(size) for size in geometry.pixel_size.reshape(-1, 2)[0])
    rows, columns = geometry.resolution
    focal_length = slm_camera_model.focal_length
    band_x = 0.5 / float(
        intensity_nyquist_pixel_size(wavelength, focal_length, pitch_x, columns)
    )
    band_y = 0.5 / float(
        intensity_nyquist_pixel_size(wavelength, focal_length, pitch_y, rows)
    )

    sensor_per_metre = camera_mapping.sensor_pixels_per_metre
    # The sensor's frequencies in cycles per pixel.
    frequency_columns, frequency_rows = (
        frequency / (2 * torch.pi)
        for frequency in get_frequency_grid(
            (int(sensor_shape[0]), int(sensor_shape[1])),
            torch.ones(2, dtype=torch.float64),
        )
    )
    # A sensor frequency k in cycles per pixel is the image-plane frequency A^T k, with
    # A the sensor pixels per image-plane metre.
    frequency_x = (
        float(sensor_per_metre[0, 0]) * frequency_rows
        + float(sensor_per_metre[1, 0]) * frequency_columns
    )
    frequency_y = (
        float(sensor_per_metre[0, 1]) * frequency_rows
        + float(sensor_per_metre[1, 1]) * frequency_columns
    )
    return (frequency_x.abs() <= band_x) & (frequency_y.abs() <= band_y)


def fourier_filter(image: Tensor, band_mask: Tensor) -> Tensor:
    """``image`` with its spectrum set to zero outside the band.

    Args:
        image: The image ``(..., H, W)``.
        band_mask: The mask of :func:`intensity_band_mask` for the same grid.

    Returns:
        Tensor: The filtered image, real.
    """
    spectrum = fft_2d(image) * band_mask.to(image.device)
    return ifft_2d(spectrum).real.to(image.dtype)


def white_noise_rms(
    images: Tensor, region: Tensor, band_mask: Tensor
) -> float | None:
    """The standard deviation of the white noise in ``region``, from the power of the
    FFT bins outside the band, over every image.

    The images are set to zero outside the region. Their bins outside the band hold
    noise only. White noise of variance ``sigma^2`` over ``n`` pixels puts
    ``n sigma^2`` into every bin of an unnormalized FFT. Structure outside the band,
    such as a sharp step at the edge of the region, raises the estimate.

    Args:
        images: One frame ``(H, W)`` or several ``(N, H, W)``, in the units the
            standard deviation is wanted in.
        region: True inside the region.
        band_mask: The mask of :func:`intensity_band_mask` for the same grid.

    Returns:
        float | None: The standard deviation per pixel of the region, or None when the
        band covers every bin and leaves no bin to measure the noise.
    """
    band_mask = band_mask.to(images.device)
    outside_band = ~band_mask
    if not bool(outside_band.any()):
        return None

    inside = region.to(device=images.device, dtype=torch.bool)
    spectrum = fft_2d(images * inside.to(images.dtype))
    power_per_bin = float((spectrum.abs() ** 2)[..., outside_band].mean())
    return float(np.sqrt(power_per_bin / int(inside.sum())))


def filtered_noise_rms(
    image: Tensor, region: Tensor, band_mask: Tensor
) -> float | None:
    """The RMS of the white noise that :func:`fourier_filter` leaves in ``region``.

    The filter keeps the band's share of the bins, so the RMS left is the standard
    deviation of the noise (:func:`white_noise_rms`) times the square root of that
    share.

    Args:
        image: The frame ``(H, W)``, in the units the RMS is wanted in.
        region: True inside the region.
        band_mask: The mask of :func:`intensity_band_mask` for the same grid.

    Returns:
        float | None: The RMS per pixel of the region, or None when the band covers
        every bin.
    """
    noise = white_noise_rms(image, region, band_mask)
    if noise is None:
        return None
    share = float(band_mask.sum()) / band_mask.numel()
    return noise * float(np.sqrt(share))
