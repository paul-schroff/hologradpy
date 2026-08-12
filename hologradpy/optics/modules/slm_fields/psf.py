"""The SLM-plane field parameterised by a compact camera-plane point spread function."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import Parameter

from .abstract import SLMField
from ...complex_amplitude import ComplexAmplitude
from ....fourier_transforms import ChirpZPartialAffine
from ....fourier_optics import fourier_lens_magnification
from ....grids import get_spatial_grid
from ....profiles.amplitude import gaussian_beam_intensity


def kernel_size_from_waist(
    waist: float, camera_pixel_size: float, extent_in_waists: float = 10.0
) -> int:
    """Kernel side, in camera pixels, that contains a focal spot of this waist.

    The waist is the one fitted to the *measured* focal spot during camera mapping, so
    it already carries the aberration: a worse wavefront gives a broader spot and
    therefore a larger kernel, which is exactly when the fit needs more freedom. Rounded
    up to an odd number so the kernel has a centre pixel.

    Args:
        waist: Focal spot waist in metres.
        camera_pixel_size: Camera pitch in metres, the grid the kernel is sampled on.
        extent_in_waists: How many waists the kernel should span.

    Returns:
        int: The kernel side in camera pixels, odd and at least 3.
    """
    size = math.ceil(extent_in_waists * waist / camera_pixel_size)
    return max(3, size + 1 - size % 2)


def waist_from_camera_mapping(camera_mapping) -> float:
    """The focal spot waist measure by a camera mapping."""
    fitted_waist = getattr(camera_mapping, "average_waist", None)
    if fitted_waist is None:
        fitted_waist = camera_mapping.focal_spot_radius
    return float(fitted_waist)


def _as_kernel_size(psf_kernel_size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(psf_kernel_size, int):
        psf_kernel_size = (psf_kernel_size, psf_kernel_size)
    return int(psf_kernel_size[0]), int(psf_kernel_size[1])


class PSFSLMField(SLMField):
    """The SLM-plane field, parameterised by a compact camera-plane PSF.

    Stands in for :class:`PixelwiseSLMField` and fills the same slot in an
    :class:`~hologradpy.optics.systems.SLMFourierLensModel`, but carries a small complex
    kernel in Fourier space instead of a full-resolution field.
    """

    def __init__(
        self: PSFSLMField,
        focal_length: float,
        camera_pixel_size: tuple[float, float],
        psf_kernel_size: int | tuple[int, int],
        psf_gaussian_waist: float | None = None,
        learnable: bool = True,
        init_psf_kernel: Tensor | None = None,
    ) -> None:
        """
        Args:
            focal_length: Fourier lens focal length in metres.
            camera_pixel_size: Camera pitch ``(y, x)`` in metres, the grid the
                kernel is sampled on.
            psf_kernel_size: PSF kernel side in camera pixels, as an int or ``(y, x)``.
                :func:`kernel_size_from_waist` sizes one around a measured focal spot.
            psf_gaussian_waist: Gaussian waist in metres used to seed the PSF kernel.
            learnable: Whether the kernel receives gradients.
            init_psf_kernel: Starting PSF kernel, ``(kernel_y, kernel_x)``. Pass the
                measured focal spot in amplitude, which is the point spread function
                itself, to start the fit from the real spot rather than an idealised
                Gaussian. Overrides ``psf_gaussian_waist``.
        """
        super().__init__()
        self.focal_length: float = float(focal_length)
        self.camera_pixel_size: tuple[float, float] = (
            float(camera_pixel_size[0]),
            float(camera_pixel_size[1]),
        )
        self.psf_kernel_size: tuple[int, int] = _as_kernel_size(psf_kernel_size)
        self.psf_waist: float | None = psf_gaussian_waist
        self.learnable: bool = learnable
        self.init_psf_kernel: Tensor | None = (
            None if init_psf_kernel is None else torch.as_tensor(init_psf_kernel)
        )

    @classmethod
    def from_camera_mapping(
        cls,
        camera_mapping,
        focal_length: float,
        camera_pixel_size: tuple[float, float],
        kernel_size: int | None = None,
        extent_in_waists: float = 10.0,
        init_psf_kernel: Tensor | None = None,
    ) -> PSFSLMField:
        """Size a kernel from a camera mapping's fitted focal spot.

        The kernel side comes from the mapping's fitted waist, falling back to the spot
        radius when the mapping carries no waist fit. The waist is measured, so it
        already carries the aberration, and a worse wavefront buys a larger kernel,
        which is when the fit needs more freedom.

        Pure by design: pass ``init_psf_kernel`` to seed from a captured image of the
        focal spot, which
        :func:`~hologradpy.calibration.spot_detection.capture_focal_spot` produces. An
        optics module should not be reaching for a live camera itself.

        Args:
            camera_mapping: The fitted mapping, for its waist or spot radius.
            focal_length: Fourier lens focal length in metres.
            camera_pixel_size: Camera pitch ``(y, x)`` in metres.
            kernel_size: Kernel side in camera pixels. Sized from the waist if omitted.
            extent_in_waists: How many waists the kernel should span when sized here.
            init_psf_kernel: Starting kernel, overriding the Gaussian seed.
        """
        waist: float = waist_from_camera_mapping(camera_mapping)

        if kernel_size is None:
            kernel_size = kernel_size_from_waist(
                waist, float(camera_pixel_size[1]), extent_in_waists
            )

        return cls(
            focal_length=focal_length,
            camera_pixel_size=camera_pixel_size,
            psf_kernel_size=kernel_size,
            psf_gaussian_waist=waist,
            init_psf_kernel=init_psf_kernel,
        )

    def lazy_init(
        self: PSFSLMField, complex_amplitude: ComplexAmplitude
    ) -> None:
        if complex_amplitude.number_of_wavelengths != 1:
            raise NotImplementedError(
                "PSFSLMField supports a single wavelength. The kernel to SLM "
                "mapping scales with the wavelength, so several wavelengths "
                "would need one kernel each."
            )

        device = complex_amplitude.device
        real_dtype = complex_amplitude.dtype_r
        self._complex_dtype = complex_amplitude.dtype

        wavelength = float(complex_amplitude.wavelength.reshape(-1)[0])
        pixel_size = complex_amplitude.pixel_size.reshape(-1)
        slm_pitch = (float(pixel_size[0]), float(pixel_size[1]))
        kernel_y, kernel_x = self.psf_kernel_size

        magnification = (
            fourier_lens_magnification(
                wavelength,
                self.focal_length,
                self.camera_pixel_size[1],
                kernel_x,
                slm_pitch[1],
            ),
            fourier_lens_magnification(
                wavelength,
                self.focal_length,
                self.camera_pixel_size[0],
                kernel_y,
                slm_pitch[0],
            ),
        )
        self._chirp_z = ChirpZPartialAffine(
            self.psf_kernel_size,
            self.resolution_in,
            magnification=magnification,
            shift=(0.0, 0.0),
            angle=0.0,
            device=device,
        )

        if self.init_psf_kernel is not None:
            init_psf_kernel = self._normalized(
                self.init_psf_kernel, device=device, real_dtype=real_dtype
            )
        else:
            waist = self.psf_waist
            if waist is None:
                aperture_half_width = 0.5 * self.resolution_in[1] * slm_pitch[1]
                waist = (
                    wavelength * self.focal_length / (torch.pi * aperture_half_width)
                )
            grid_x, grid_y = get_spatial_grid(
                self.psf_kernel_size, self.camera_pixel_size, device=device
            )
            # The kernel is a field, so take the amplitude of that intensity.
            init_psf_kernel = (
                gaussian_beam_intensity(grid_x, grid_y, beam_radius=waist)
                .sqrt()
                .to(real_dtype)
            )

        self.psf_kernel = Parameter(
            init_psf_kernel.to(self._complex_dtype), requires_grad=self.learnable
        )

    def _normalized(
        self: PSFSLMField, kernel: Tensor, device, real_dtype: torch.dtype
    ) -> Tensor:
        """A kernel checked for shape and scaled to unit peak."""
        if tuple(kernel.shape[-2:]) != self.psf_kernel_size:
            raise ValueError(
                f"kernel {tuple(kernel.shape[-2:])} does not match psf_kernel_size "
                f"{self.psf_kernel_size}"
            )
        kernel = kernel.to(device=device, dtype=real_dtype)
        peak = kernel.abs().max()
        return kernel / peak if float(peak) > 0 else kernel

    def get_psf_kernel(self: PSFSLMField) -> Tensor:
        """The complex camera-plane PSF kernel."""
        return self.psf_kernel

    def get_transmission(self: PSFSLMField) -> Tensor:
        """The SLM-plane field the kernel maps to, as a diagonal transmission."""
        kernel = self.get_psf_kernel().to(self._complex_dtype)
        field = self._chirp_z(kernel.unsqueeze(0))
        return field.reshape(1, *self.resolution_in)

    def get_wavefront(self: PSFSLMField) -> Tensor:
        """The recovered SLM-plane complex field, on the SLM grid."""
        return self.get_transmission().squeeze(0)

