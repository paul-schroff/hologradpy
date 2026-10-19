"""The SLM-plane field parameterized by a compact image-plane point spread function."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import Parameter

from .abstract import SLMField
from ...complex_amplitude import ComplexAmplitude
from ....fourier_transforms import ChirpZPartialAffine
from ....fourier_optics import (
    fourier_lens_magnification,
    intensity_nyquist_pixel_size,
)
from ....grids import get_spatial_grid, odd_sample_count
from ....optimization import solve_linear_least_squares
from ....profiles.amplitude import gaussian_beam_intensity
from ....utils import as_pair

if TYPE_CHECKING:
    from ....calibration.camera_mapping.mapping import CameraMapping
    from ....calibration.wavefront.abstract import WavefrontCalibrationData


class PSFSLMField(SLMField):
    """The SLM-plane field, parameterized by a compact image-plane PSF.

    Stands in for :class:`PixelwiseSLMField` and fills the same slot in an
    :class:`~hologradpy.optics.systems.SLMFourierLensModel`. Its complex kernel is the
    point spread function in the image plane of the Fourier lens. A chirp-z transform
    maps the kernel to the SLM grid.

    The kernel is sampled at the intensity Nyquist pitch of the SLM,
    ``lambda f / (2 L)`` along an axis on which the SLM spans ``L``
    (:func:`~hologradpy.fourier_optics.intensity_nyquist_pixel_size`). The SLM-plane
    field of the kernel repeats every ``lambda f`` divided by the pitch, ``2 L``, so
    the SLM holds a single copy of the beam. The pitch and the number of samples of the
    kernel are set from the SLM grid on the first forward pass.

    The kernel is seeded by a Gaussian focal spot, a measured focal spot
    (:meth:`from_camera_mapping`), or a measured SLM-plane field
    (:meth:`from_calibration_data`).
    """

    def __init__(
        self: PSFSLMField,
        focal_length: float,
        psf_extent: float | tuple[float, float],
        psf_gaussian_waist: float | None = None,
        learnable: bool = True,
        init_psf_kernel: Tensor | None = None,
        init_psf_kernel_pixel_size: tuple[float, float] | None = None,
        init_wavefront: Tensor | None = None,
    ) -> None:
        """
        Args:
            focal_length: Fourier lens focal length in metres.
            psf_extent: The width of the image plane the kernel spans, in metres, as
                one value or ``(y, x)``. Each axis takes the odd number of samples that
                covers it. Ten waists of the measured focal spot is a start
                (:meth:`from_camera_mapping`).
            psf_gaussian_waist: Gaussian waist in metres that seeds the PSF kernel.
                Defaults to the focal waist of a Gaussian beam whose waist is half the
                SLM width, ``wavelength * focal_length / (pi * half_width)``.
            learnable: Whether the kernel receives gradients.
            init_psf_kernel: A measured focal spot in amplitude, ``(rows, columns)`` at
                the pitch ``init_psf_kernel_pixel_size``, in the model's orientation. A
                crop that
                :func:`~hologradpy.calibration.wavefront.speckle_calibration.capture_focal_spot`
                returns has to be turned into the model's orientation with
                ``camera_mapping.sensor_crop_to_model`` before it is passed. The spot is
                the point spread function itself, and the kernel is fitted to its
                SLM-plane field. Overrides ``psf_gaussian_waist``.
            init_psf_kernel_pixel_size: The ``(y, x)`` pitch of ``init_psf_kernel`` in
                metres, the pitch of the camera mapping's output plane for a turned
                captured spot.
            init_wavefront: A complex SLM-plane field ``(H, W)`` the kernel is fitted
                to once the geometry is known (:meth:`from_calibration_data`).
                Overrides the other seeds.

        Raises:
            ValueError: ``init_psf_kernel`` comes without its pitch.
        """
        super().__init__()
        if init_psf_kernel is not None and init_psf_kernel_pixel_size is None:
            raise ValueError(
                "init_psf_kernel is a measured spot at its own pitch, so it needs "
                "init_psf_kernel_pixel_size, the output pitch of the camera mapping "
                "for a captured spot."
            )
        self.focal_length: float = float(focal_length)
        self.psf_extent: tuple[float, float] = as_pair(psf_extent)
        # The kernel's pitch and number of samples, (y, x), set by lazy_init.
        self.psf_pixel_size: tuple[float, float] | None = None
        self.psf_kernel_size: tuple[int, int] | None = None
        self.psf_waist: float | None = psf_gaussian_waist
        self.learnable: bool = learnable
        self.init_psf_kernel: Tensor | None = (
            None if init_psf_kernel is None else torch.as_tensor(init_psf_kernel)
        )
        self.init_psf_kernel_pixel_size: tuple[float, float] | None = (
            None
            if init_psf_kernel_pixel_size is None
            else as_pair(init_psf_kernel_pixel_size)
        )
        self.init_wavefront: Tensor | None = (
            None if init_wavefront is None else torch.as_tensor(init_wavefront)
        )
        # The relative RMS error of the fitted kernel's field against the field it was
        # fitted to, init_wavefront or the SLM-plane field of init_psf_kernel.
        self.wavefront_fit_residual: float | None = None
        self._held_transmission: Tensor | None = None
        self._held_kernel: Tensor | None = None

    @classmethod
    def from_camera_mapping(
        cls,
        camera_mapping: CameraMapping,
        focal_length: float,
        extent_in_waists: float = 10.0,
        init_psf_kernel: Tensor | None = None,
        init_psf_kernel_pixel_size: tuple[float, float] | None = None,
    ) -> PSFSLMField:
        """Size a kernel from the focal spot a camera mapping fitted.

        The kernel's extent and the Gaussian seed both come from the waist of the
        mapping's spot fit, ``camera_mapping.spot_fit.waist``. The waist is measured on
        the camera and includes the aberration, so a worse wavefront gives a larger
        kernel. The camera is not read here.

        A given ``init_psf_kernel`` seeds the kernel unchanged.
        :class:`~hologradpy.calibration.wavefront.speckle_calibration.PSFSpeckleCalibrator`
        turns a focal spot captured on the camera
        (:func:`~hologradpy.calibration.wavefront.speckle_calibration.capture_focal_spot`)
        into the model's orientation with ``camera_mapping.sensor_crop_to_model``
        before it passes the spot. The turned spot is sampled at the pitch of the
        mapping's output plane, ``camera_mapping.output_pixel_size``.

        Args:
            camera_mapping: The fitted mapping, for the waist of its spot fit.
            focal_length: Fourier lens focal length in metres.
            extent_in_waists: The width of the image plane the kernel spans, in waists.
            init_psf_kernel: A measured focal spot in amplitude, in the model's
                orientation. It overrides the Gaussian seed.
            init_psf_kernel_pixel_size: The ``(y, x)`` pitch of ``init_psf_kernel`` in
                metres.
        """
        waist = float(camera_mapping.spot_fit.waist)
        return cls(
            focal_length=focal_length,
            psf_extent=extent_in_waists * waist,
            psf_gaussian_waist=waist,
            init_psf_kernel=init_psf_kernel,
            init_psf_kernel_pixel_size=init_psf_kernel_pixel_size,
        )

    @classmethod
    def from_calibration_data(
        cls,
        calibration_data: WavefrontCalibrationData,
        focal_length: float,
        psf_extent: float | tuple[float, float],
        learnable: bool = True,
    ) -> PSFSLMField:
        """A kernel fitted to the SLM-plane field a wavefront calibration measured.

        Args:
            calibration_data: The calibration. Its ``complex_amplitude`` is the
                field on the SLM grid.
            focal_length: Fourier lens focal length in metres.
            psf_extent: The width of the image plane the kernel spans, in metres, as
                one value or ``(y, x)``. Ten waists of the focal spot is a start.
            learnable: Whether the kernel receives gradients.

        Returns:
            PSFSLMField: The field, which fits its kernel on its first forward pass.
        """
        return cls(
            focal_length=focal_length,
            psf_extent=psf_extent,
            learnable=learnable,
            init_wavefront=_single_wavefront(calibration_data.complex_amplitude),
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
        self._complex_dtype = complex_amplitude.dtype

        wavelength = float(complex_amplitude.wavelength.reshape(-1)[0])
        pixel_size = complex_amplitude.pixel_size.reshape(-1)
        slm_pitch = (float(pixel_size[0]), float(pixel_size[1]))
        self.psf_pixel_size = tuple(
            float(
                intensity_nyquist_pixel_size(
                    wavelength, self.focal_length, pitch, count
                )
            )
            for pitch, count in zip(slm_pitch, self.resolution_in)
        )
        self.psf_kernel_size = tuple(
            odd_sample_count(extent, pitch)
            for extent, pitch in zip(self.psf_extent, self.psf_pixel_size)
        )
        self._chirp_z = self._transform_to_slm_grid(
            self.psf_kernel_size, self.psf_pixel_size, wavelength, slm_pitch, device
        )

        if self.init_wavefront is not None:
            wavefront = self.init_wavefront.to(device=device, dtype=self._complex_dtype)
            if tuple(wavefront.shape) != tuple(self.resolution_in):
                raise ValueError(
                    f"The wavefront to fit is {tuple(wavefront.shape)} but the "
                    f"SLM-plane field is {tuple(self.resolution_in)}."
                )
            kernel, self.wavefront_fit_residual = self._fitted_kernel(wavefront)
        elif self.init_psf_kernel is not None:
            spot = self.init_psf_kernel.to(device=device, dtype=self._complex_dtype)
            spot_shape = tuple(spot.shape[-2:])
            spot_to_slm = self._transform_to_slm_grid(
                spot_shape,
                self.init_psf_kernel_pixel_size,
                wavelength,
                slm_pitch,
                device,
            )
            spot_field = spot_to_slm(spot.reshape(1, *spot_shape)).reshape(
                self.resolution_in
            )
            kernel, self.wavefront_fit_residual = self._fitted_kernel(spot_field)
        else:
            waist = self.psf_waist
            if waist is None:
                aperture_half_width = 0.5 * self.resolution_in[1] * slm_pitch[1]
                waist = (
                    wavelength * self.focal_length / (torch.pi * aperture_half_width)
                )
            grid_x, grid_y = get_spatial_grid(
                self.psf_kernel_size, self.psf_pixel_size, device=device
            )
            # The kernel is a field, so take the amplitude of that intensity.
            kernel = (
                gaussian_beam_intensity(grid_x, grid_y, beam_radius_major=waist)
                .sqrt()
                .to(self._complex_dtype)
            )

        self.psf_kernel = Parameter(kernel, requires_grad=self.learnable)

    def _transform_to_slm_grid(
        self: PSFSLMField,
        shape: tuple[int, int],
        pitch: tuple[float, float],
        wavelength: float,
        slm_pitch: tuple[float, float],
        device: torch.device,
    ) -> ChirpZPartialAffine:
        """The chirp-z transform through the Fourier lens, from an image-plane grid of
        ``shape`` and ``(y, x)`` ``pitch`` to the SLM grid. The image-plane grid is
        centred on the optical axis.
        """
        # Negative, so the chirp-z is the inverse transform (see get_transmission).
        magnification = (
            -fourier_lens_magnification(
                wavelength, self.focal_length, pitch[1], shape[1], slm_pitch[1]
            ),
            -fourier_lens_magnification(
                wavelength, self.focal_length, pitch[0], shape[0], slm_pitch[0]
            ),
        )
        return ChirpZPartialAffine(
            shape,
            self.resolution_in,
            magnification=magnification,
            shift=(0.0, 0.0),
            angle=0.0,
            device=device,
        )

    def _fitted_kernel(self: PSFSLMField, wavefront: Tensor) -> tuple[Tensor, float]:
        """The kernel with the SLM-plane field closest to ``wavefront``
        (:func:`~hologradpy.optimization.solve_linear_least_squares`), and the relative
        RMS error left.
        """

        def to_field(kernel: Tensor) -> Tensor:
            return self._chirp_z(kernel.unsqueeze(0)).reshape(wavefront.shape)

        def to_kernel(field: Tensor) -> Tensor:
            return self._chirp_z.adjoint(field.unsqueeze(0)).reshape(
                self.psf_kernel_size
            )

        kernel, relative_error = solve_linear_least_squares(
            to_field, to_kernel, wavefront
        )
        peak = kernel.abs().max()
        if float(peak) > 0:
            kernel = kernel / peak
        return kernel, relative_error

    def get_psf_kernel(self: PSFSLMField) -> Tensor:
        """The complex image-plane PSF kernel."""
        return self.psf_kernel

    def get_transmission(self: PSFSLMField) -> Tensor:
        """The SLM-plane field the kernel maps to, as a diagonal transmission.

        The Fourier lens takes the SLM plane to the camera with a forward transform.
        The negative magnification of the chirp-z point-reflects its window, so the
        chirp-z is the inverse transform. The lens then images the kernel upright, in
        the model's orientation.

        A phase retrieval evaluates it on every forward pass, so it is held while the
        kernel takes no gradient and has not changed.
        """
        kernel = self.get_psf_kernel()
        if kernel.requires_grad:
            self._held_transmission = None
            return self._transmission(kernel)

        # The kernel is small, so comparing its values costs little next to the
        # transform. The comparison catches every way the kernel can change.
        unchanged = (
            self._held_kernel is not None
            and self._held_kernel.device == kernel.device
            and torch.equal(self._held_kernel, kernel.detach())
        )
        if self._held_transmission is None or not unchanged:
            with torch.no_grad():
                self._held_transmission = self._transmission(kernel)
            self._held_kernel = kernel.detach().clone()
        return self._held_transmission

    def _transmission(self: PSFSLMField, kernel: Tensor) -> Tensor:
        """The SLM-plane field of ``kernel``, shaped ``(1, H, W)``."""
        field = self._chirp_z(kernel.to(self._complex_dtype).unsqueeze(0))
        return field.reshape(1, *self.resolution_in)

    def get_wavefront(self: PSFSLMField) -> Tensor:
        """The recovered SLM-plane complex field, on the SLM grid."""
        return self.get_transmission().squeeze(0)


def _single_wavefront(complex_amplitude: ComplexAmplitude) -> Tensor:
    """The one ``(H, W)`` field a calibration holds."""
    flat, _ = complex_amplitude.flatten_batch()
    if flat.shape[0] != 1 or flat.shape[1] != 1:
        raise ValueError(
            "A PSF kernel describes one field at one wavelength, but the calibration "
            f"holds a field of shape {tuple(flat.shape)}."
        )
    return flat[0, 0]

