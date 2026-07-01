from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Parameter

from ..fourier import ChirpZZoom, shear_rotate

from ..optics_module import OpticsModule
from ..complex_amplitude import ComplexAmplitude


class FourierLensCZT(OpticsModule):
    """Exact Fourier lens via the chirp-z zoom, with learnable focal-plane affine
    parameters (scale, shift, angle).

    Unlike :class:`FourierLensNUFFT` (which interpolates with the Kaiser-Bessel
    NUFFT) this evaluates the *exact* DFT at the chosen focal-plane sample points
    (:class:`ChirpZZoom`), so the sampled amplitudes carry no interpolation error
    and the optical power is represented faithfully. No zero-padding is needed --
    the chirp-z samples the exact spectrum at any spacing.

    ``scale_factor`` (per-axis zoom multiplier), ``shift`` (focal-plane offset in
    output pixels) and ``angle`` (rotation in degrees) are ``nn.Parameter`` s
    (``requires_grad=learnable``), so the focal-plane affine map can be calibrated
    by gradient descent. ``angle`` rotates the input field with a differentiable
    three-shear FFT rotation; ``scale`` / ``shift`` enter the chirp-z directly.

    The geometry is per-wavelength: the base magnification is ``lambda * f /
    (pixel_in * resolution_in * pixel_out)``, so that with the parameters at their
    identity values (scale 1, shift 0, angle 0) a focal pixel measures
    ``pixel_out``.
    """

    def __init__(
        self: FourierLensCZT,
        focal_length: float,
        resolution_out: tuple[int, int],
        pixel_size_out: tuple[float, float],
        shift: tuple[float, float] = (0.0, 0.0),
        angle: float = 0.0,
        learnable: bool = True,
        power_normalized: bool = True,
    ) -> None:
        super().__init__(pixel_size_out, resolution_out)

        self.focal_length: float = focal_length
        self.shift_init: tuple[float, float] = shift
        self.angle_init: float = angle  # degrees
        self.learnable: bool = learnable
        self.power_normalized: bool = power_normalized

    def lazy_init(self: FourierLensCZT, complex_amplitude: ComplexAmplitude) -> None:
        # Output geometry (pixel_size_out / resolution_out) is set from the
        # constructor args by the base before this runs.
        self._input_resolution: tuple[int, int] = tuple(complex_amplitude.resolution)
        resolution_in = torch.tensor(
            self._input_resolution,
            device=complex_amplitude.device,
            dtype=complex_amplitude.dtype_r,
        )

        # Per-wavelength base magnification, indexed (axis0 = y, axis1 = x).
        self._base_magnification: Tensor = (
            complex_amplitude.wavelength.unsqueeze(-1)
            * self.focal_length
            / (
                complex_amplitude.pixel_size
                * resolution_in.unsqueeze(0)
                * self._pixel_size_out.unsqueeze(0)
            )
        )  # (n_wl, 2)

        real_dtype = complex_amplitude.dtype_r
        device = complex_amplitude.device
        self.scale_factor = Parameter(
            torch.ones(2, dtype=real_dtype, device=device),
            requires_grad=self.learnable,
        )
        self.shift = Parameter(
            torch.tensor(self.shift_init, dtype=real_dtype, device=device),
            requires_grad=self.learnable,
        )
        self.angle = Parameter(
            torch.tensor(self.angle_init, dtype=real_dtype, device=device),
            requires_grad=self.learnable,
        )

    def _power_prefactor(self: FourierLensCZT) -> Tensor:
        """Fourier-lens amplitude prefactor ``(du*dv) / (lambda*f)`` per
        wavelength (input pixel area over ``lambda*f``), so the exact chirp-z
        transform conserves optical power (``integral|E_focal|^2 dx ==
        integral|E_slm|^2 du`` over the captured window). Computed in float64 and
        cast to the field's real dtype; the global ``1/i`` phase is omitted as it
        does not affect power. Shaped ``(1, n_wl, 1, 1)`` for the flattened field.
        """
        pixel_size_in = self.pixel_size_in
        pixel_area = (
            (pixel_size_in[:, 0] * pixel_size_in[:, 1]).to(torch.float64).reshape(-1)
        )
        wavelength = self.input_geometry.wavelength.to(torch.float64).reshape(-1)
        prefactor = pixel_area / (wavelength * self.focal_length)  # (n_wl,)
        return prefactor.to(pixel_size_in.dtype).reshape(1, -1, 1, 1)

    def _apply_rotation(
        self: FourierLensCZT, field: Tensor, inverse: bool
    ) -> Tensor:
        """Rotate the (last two) field axes by the learnable angle (or its
        negative for the adjoint). ``self.angle`` is in degrees and is converted to
        radians for the shear rotation. When the parameters are learnable the
        differentiable tensor path is always taken so a gradient flows even at
        ``angle == 0``."""
        angle = torch.deg2rad(-self.angle if inverse else self.angle)
        if self.learnable:
            return shear_rotate(field, angle)
        angle_value = float(angle)
        if angle_value == 0.0:
            return field
        return shear_rotate(field, angle_value)

    def _chirp_z(self: FourierLensCZT, scale: Tensor) -> ChirpZZoom:
        """Build the per-wavelength scale + shift chirp-z (rotation is applied to
        the field separately). ``scale`` is the effective per-axis magnification
        ``(axis0 = y, axis1 = x)``; the chirp-z window is offset by ``shift``
        output pixels."""
        magnification = (scale[1], scale[0])  # (x, y)

        height, width = self._input_resolution
        step_x = (2 * torch.pi / width) / scale[1]
        step_y = (2 * torch.pi / height) / scale[0]
        shift = (self.shift[1] * step_x, self.shift[0] * step_y)  # (x, y)

        return ChirpZZoom(
            self._input_resolution,
            self.resolution_out,
            magnification=magnification,
            shift=shift,
            angle=0.0,
            device=scale.device,
        )

    def forward(
        self: FourierLensCZT, complex_amplitude: ComplexAmplitude
    ) -> ComplexAmplitude:
        flat_field, batch_spec = complex_amplitude.flatten_batch()  # (N, n_wl, H, W)
        field = self._apply_rotation(flat_field, inverse=False)

        scale = self.scale_factor.abs() * self._base_magnification  # (n_wl, 2)
        outputs = [
            self._chirp_z(scale[wavelength]).forward(field[:, wavelength])
            for wavelength in range(field.shape[1])
        ]
        output = torch.stack(outputs, dim=1)  # (N, n_wl, H_out, W_out)
        if self.power_normalized:
            output = output * self._power_prefactor()

        return ComplexAmplitude.unflatten_batch(
            output,
            batch_spec,
            complex_amplitude.wavelength,
            self.pixel_size_out,
        )

    def adjoint(
        self: FourierLensCZT, complex_amplitude: ComplexAmplitude
    ) -> ComplexAmplitude:
        """Conjugate transpose of :meth:`forward`: the chirp-z adjoint followed by
        the inverse rotation."""

        flat_field, batch_spec = complex_amplitude.flatten_batch()
        scale = self.scale_factor.abs() * self._base_magnification
        inputs = [
            self._chirp_z(scale[wavelength]).adjoint(flat_field[:, wavelength])
            for wavelength in range(flat_field.shape[1])
        ]
        field = torch.stack(inputs, dim=1)  # (N, n_wl, H, W)
        field = self._apply_rotation(field, inverse=True)
        if self.power_normalized:
            field = field * self._power_prefactor()

        return ComplexAmplitude.unflatten_batch(
            field,
            batch_spec,
            complex_amplitude.wavelength,
            self.pixel_size_in,
        )
