from __future__ import annotations

from jaxtyping import Float

import torch
from torch import Tensor
from torch.nn import Parameter

from ....fourier_optics import (
    fourier_lens_magnification,
    fourier_lens_power_prefactor,
)
from ....fourier_transforms import (
    KbNufftPartialAffine,
    window_offset_from_pixels,
)

from ..abstract import OpticsModule
from ...complex_amplitude import ComplexAmplitude, pixel_area


class FourierLensNUFFT(OpticsModule):
    def __init__(
        self: FourierLensNUFFT,
        focal_length: float,
        resolution_out: tuple[int, int],
        pixel_size_out: tuple[float, float],
        padded_resolution: tuple[int, int] | None = None,
        shift: tuple[float, float] = (0, 0),
        angle: float = 0,
        nufft_kwargs: dict = {},
        power_normalized: bool = True,
    ) -> None:
        super().__init__(pixel_size_out, resolution_out)

        self.focal_length: float = focal_length
        self._padded_resolution_init: tuple[int, int] | None = padded_resolution
        self.shift_init: tuple[float, float] = shift
        self.angle_init: float = angle
        self.nufft_kwargs = nufft_kwargs
        self.power_normalized: bool = power_normalized

    def lazy_init(self: FourierLensNUFFT, complex_amplitude: ComplexAmplitude) -> None:
        # Output geometry (pixel_size_out / resolution_out) is set from the
        # constructor args by the base before this runs.
        if self._padded_resolution_init is not None:
            # check if the provided padded resolution is at least
            # as large as the input resolution
            if (
                self._padded_resolution_init[0] < complex_amplitude.resolution[0]
                or self._padded_resolution_init[1] < complex_amplitude.resolution[1]
            ):
                raise ValueError(
                    "Padded resolution must be at least as large as input resolution."
                )

            # Check if padded_resolution is even
            parity = tuple(self._padded_resolution_init[i] % 2 for i in range(2))

            if parity[0] != 0 or parity[1] != 0:
                raise ValueError("Padded resolution must be even.")
        else:
            self._padded_resolution_init = tuple(
                2 * complex_amplitude.resolution[i] for i in range(2)
            )

        self._padded_resolution: tuple[int, int] = self._padded_resolution_init
        self._padded_resolution_tensor = torch.tensor(
            self._padded_resolution,
            device=complex_amplitude.device,
            dtype=complex_amplitude.dtype_r,
        )

        # Per-wavelength focal-plane zoom in (x, y). Built from the (y, x) array-axis
        # pixel_size / padded resolution and flipped once here into the (x, y)
        # focal-plane convention the transform and the learnable params share. This is
        # the single boundary between the array-axis and focal-plane conventions.
        self._scale: Float[Tensor, "n_wavelengths 2"] = (  # noqa: F722
            fourier_lens_magnification(
                complex_amplitude.wavelength.unsqueeze(-1),
                self.focal_length,
                complex_amplitude.pixel_size,
                self._padded_resolution_tensor.unsqueeze(0),
                self._pixel_size_out.unsqueeze(0),
            ).flip(-1)
        )

        # scale_factor and shift are (x, y), matching the geometry / GeometricWarp
        # convention and the (x, y) internal scale, so they combine directly.
        self.scale_factor: Float[Tensor, "2"] = Parameter(
            torch.ones(
                2,
                dtype=complex_amplitude.dtype_r,
                device=complex_amplitude.device,
            ),
            requires_grad=False,
        )

        # Shift of the focal plane in output pixels, (x, y).
        self.shift: Float[Tensor, "2"] = Parameter(
            torch.tensor(
                self.shift_init,
                dtype=complex_amplitude.dtype_r,
                device=complex_amplitude.device,
            ),
            requires_grad=False,
        )

        # Rotation angle in degrees
        self.angle: Float[Tensor, "1"] = Parameter(
            torch.tensor(
                self.angle_init,
                dtype=complex_amplitude.dtype_r,
                device=complex_amplitude.device,
            ),
            requires_grad=False,
        )

        self._transform = self._build_transform(complex_amplitude)

    def _build_transform(
        self: FourierLensNUFFT, complex_amplitude: ComplexAmplitude
    ) -> KbNufftPartialAffine:
        """Map the per-wavelength ``lambda*f`` geometry to the unit-free
        ``(magnification, shift, angle)`` of a :class:`KbNufftPartialAffine`.

        ``scale_factor``, ``shift`` and ``self._scale`` are all (x, y), matching the
        transform, so they combine directly with no axis swap. The base sample grid
        has bin spacing
        ``2*pi / padded_resolution`` (rad/sample on the padded grid), divided by
        the scale -- which is exactly ``get_zoom_frequency_grid`` with
        ``resolution = padded_resolution`` and ``magnification = scale``. The
        in-pixel ``shift`` becomes the window offset ``-2*pi * shift / (padded *
        scale)`` (the rotation then mixes the axes identically for grid and
        offset, matching the legacy ``scale -> rotate -> shift`` ordering).
        """
        scale: Tensor = (
            self.scale_factor.abs().unsqueeze(0) * self._scale
        )  # (n_wl, 2): (x, y)

        magnification = scale  # (x, y)

        shift_x, shift_y = window_offset_from_pixels(
            self.shift, self._padded_resolution, (scale[:, 0], scale[:, 1])
        )
        shift = torch.stack((shift_x, shift_y), dim=-1)  # (n_wl, 2): (x, y)

        # Negated on purpose to be consistent with the other modules.
        angle_radians = -float(torch.deg2rad(self.angle))

        return KbNufftPartialAffine(
            resolution=tuple(complex_amplitude.resolution),
            resolution_out=self.resolution_out,
            magnification=magnification,
            shift=shift,
            angle=angle_radians,
            grid_size=self._padded_resolution,
            dtype=complex_amplitude.dtype_r,
            device=complex_amplitude.device,
            **self.nufft_kwargs,
        )

    @property
    def pixel_size_out(self: FourierLensNUFFT) -> tuple[float, float]:
        return self._pixel_size_out

    @property
    def resolution_out(self: FourierLensNUFFT) -> tuple[int, int]:
        return self._resolution_out

    def _power_prefactor(self: FourierLensNUFFT) -> Tensor:
        """Fourier-lens amplitude prefactor ``(du*dv) / (lambda*f)`` per
        wavelength, so the transform carries physical optical power. The KbNufft
        is approximate, so power is conserved only up to interpolation error.
        Shaped ``(1, n_wl, 1, 1)`` for the flattened field."""
        pixel_size_in = self.pixel_size_in
        area = pixel_area(pixel_size_in)
        wavelength = self.input_geometry.wavelength.to(torch.float64).reshape(-1)
        prefactor = fourier_lens_power_prefactor(
            area, wavelength, self.focal_length
        )
        return prefactor.to(pixel_size_in.dtype).reshape(1, -1, 1, 1)

    def forward(
        self: FourierLensNUFFT,
        complex_amplitude: ComplexAmplitude,
    ) -> ComplexAmplitude:
        """Propagate a field of arbitrary batch rank ``(*batch, n_wl, H, W)``.

        All leading batch dimensions are collapsed into a single axis and the
        scaled + shifted + rotated NUFFT is delegated to
        :class:`KbNufftPartialAffine`, then the original rank is restored.
        """
        flat_field, batch_spec = complex_amplitude.flatten_batch()
        output_field = self._transform.forward(flat_field)
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
        """Adjoint NUFFT mapping an output-plane field back to the input plane.

        This is the conjugate transpose of :meth:`forward` (the
        ``KbNufftPartialAffine`` adjoint on the same trajectory), not its inverse.
        The input is a field sampled on the output grid ``(*batch, n_wl, H_out,
        W_out)``; the output lives in the input plane with ``resolution_in`` /
        ``pixel_size_in``. The module must be initialized first (via ``forward``
        or ``initialize_from_geometry``).
        """
        flat_field, batch_spec = complex_amplitude.flatten_batch()
        input_field = self._transform.adjoint(flat_field)
        if self.power_normalized:
            input_field = input_field * self._power_prefactor()
        return ComplexAmplitude.unflatten_batch(
            input_field,
            batch_spec,
            complex_amplitude.wavelength,
            self.pixel_size_in,
        )
