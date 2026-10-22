from __future__ import annotations

from ....fourier_transforms import NUFFTPartialAffine
from .partial_affine_fourier_lens import PartialAffineFourierLens
from ...complex_amplitude import ComplexAmplitude


class FourierLensNUFFT(PartialAffineFourierLens):
    """Fourier lens via the non-uniform FFT, with a learnable focal-plane partial
    affine (:class:`~.partial_affine_fourier_lens.PartialAffineFourierLens`).

    The NUFFT samples the window of the partial affine along a k-space trajectory
    (:class:`~hologradpy.fourier_transforms.NUFFTPartialAffine`), so it rotates by any
    angle. It interpolates, so the output and its power match the exact transform to
    the tolerance it was asked for (``eps`` in ``nufft_kwargs``, ``1e-6`` by default).
    The transform runs on the grid of the input field.
    """

    def __init__(
        self: FourierLensNUFFT,
        focal_length: float,
        resolution_out: tuple[int, int],
        pixel_size_out: tuple[float, float],
        shift: tuple[float, float] = (0, 0),
        angle: float = 0,
        learnable: bool = True,
        nufft_kwargs: dict = {},
        power_normalized: bool = True,
    ) -> None:
        """
        Args:
            focal_length: The focal length in metres.
            resolution_out: The ``(height, width)`` of the focal plane in pixels.
            pixel_size_out: The ``(height, width)`` pitch of the focal plane in metres.
            shift: The starting shift of the focal plane on the output plane, in output
                pixels along ``(x, y)``. The zeroth order lands this far from the centre
                pixel.
            angle: The starting turn of the focal plane on the output plane, in degrees
                from +x towards +y.
            learnable: Whether the parameters of the partial affine require a gradient.
            nufft_kwargs: Keyword arguments for FINUFFT, such as ``eps``.
            power_normalized: Whether the output carries the optical power of the
                input, through the prefactor ``(du * dv) / (lambda * f)``.
        """
        super().__init__(
            focal_length,
            resolution_out,
            pixel_size_out,
            shift=shift,
            angle=angle,
            learnable=learnable,
            power_normalized=power_normalized,
        )
        self.nufft_kwargs = nufft_kwargs

    def lazy_init(self: FourierLensNUFFT, complex_amplitude: ComplexAmplitude) -> None:
        self.focal_plane_partial_affine.to(
            device=complex_amplitude.device, dtype=complex_amplitude.dtype_r
        )

    def _build_transform(
        self: FourierLensNUFFT, complex_amplitude: ComplexAmplitude
    ) -> NUFFTPartialAffine:
        """The NUFFT between the input plane and the window the partial affine sets, on
        the device and in the real dtype of ``complex_amplitude``.
        """
        resolution_in = tuple(self.resolution_in)
        magnification, offset, angle = self._sampled_window(resolution_in)
        return NUFFTPartialAffine(
            resolution=resolution_in,
            resolution_out=self.resolution_out,
            magnification=magnification,
            shift=offset,
            angle=angle,
            dtype=complex_amplitude.dtype_r,
            device=complex_amplitude.device,
            output_roi=self.output_roi,
            **self.nufft_kwargs,
        )

    def forward(
        self: FourierLensNUFFT,
        complex_amplitude: ComplexAmplitude,
    ) -> ComplexAmplitude:
        """Propagate a field of arbitrary batch rank ``(*batch, n_wl, H, W)``.

        All leading batch dimensions are collapsed into a single axis and the scaled +
        shifted + rotated NUFFT is delegated to :class:`NUFFTPartialAffine`, then the
        original rank is restored.

        The transform is rebuilt from :attr:`focal_plane_partial_affine` on every call.
        """
        flat_field, batch_spec = complex_amplitude.flatten_batch()
        output_field = self._build_transform(complex_amplitude).forward(flat_field)
        if self.power_normalized:
            output_field = output_field * self._power_prefactor()
        return ComplexAmplitude.unflatten_batch(
            output_field,
            batch_spec,
            complex_amplitude.wavelength,
            self.pixel_size_out,
        )

    def adjoint(
        self: FourierLensNUFFT,
        complex_amplitude: ComplexAmplitude,
    ) -> ComplexAmplitude:
        """The conjugate transpose of :meth:`forward`, not its inverse.

        The transform is the ``NUFFTPartialAffine`` adjoint on the same trajectory. It
        maps a field on the output grid ``(*batch, n_wl, H_out, W_out)`` back to the
        input plane, at ``resolution_in`` and ``pixel_size_in``. The module must be
        initialized first, by ``forward`` or ``initialize_from_geometry``.
        """
        flat_field, batch_spec = complex_amplitude.flatten_batch()
        input_field = self._build_transform(complex_amplitude).adjoint(flat_field)
        if self.power_normalized:
            input_field = input_field * self._power_prefactor()
        return ComplexAmplitude.unflatten_batch(
            input_field,
            batch_spec,
            complex_amplitude.wavelength,
            self.pixel_size_in,
        )
