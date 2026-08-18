from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Parameter

from ....fourier_optics import (
    fourier_lens_magnification,
    fourier_lens_power_prefactor,
)
from ....fourier_transforms import (
    ChirpZPartialAffine,
    padded_resolution_for_rotation,
    window_offset_from_pixels,
)

from ....geometry import PartialAffineTransform
from ....utils import to_canvas
from ..abstract import OpticsModule
from ...complex_amplitude import ComplexAmplitude, pixel_area
from ....geometry import recalibrated_partial_affine


class FourierLensCZT(OpticsModule):
    """Exact Fourier lens via the chirp-z zoom, with learnable focal-plane affine
    parameters (scale, shift, angle).

    Unlike :class:`FourierLensNUFFT` (which interpolates with the Kaiser-Bessel
    NUFFT) this evaluates the *exact* DFT at the chosen focal-plane sample points
    (:class:`ChirpZPartialAffine`), so the sampled amplitudes carry no interpolation
    error and the optical power is represented faithfully.

    ``scale_factor`` (per-axis zoom multiplier, ``(x, y)``), ``shift`` (focal-plane
    offset in output pixels, ``(x, y)``) and ``angle`` (rotation in degrees) are
    ``nn.Parameter`` s (``requires_grad=learnable``), so the focal-plane affine map
    can be calibrated by gradient descent. All three are handed to the transform, which
    samples the scaled, shifted, rotated window directly.

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
        padded_resolution: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(pixel_size_out, resolution_out)

        self.focal_length: float = focal_length
        self.shift_init: tuple[float, float] = shift
        self.angle_init: float = angle  # degrees
        self.learnable: bool = learnable
        self.power_normalized: bool = power_normalized
        self.padded_resolution: tuple[int, int] | None = padded_resolution

    def lazy_init(self: FourierLensCZT, complex_amplitude: ComplexAmplitude) -> None:
        # Output geometry (pixel_size_out / resolution_out) is set from the
        # constructor args by the base before this runs.
        self._input_resolution: tuple[int, int] = tuple(complex_amplitude.resolution)
        self._padded_resolution: tuple[int, int] = self._resolve_padding()

        resolution_in = torch.tensor(
            self._padded_resolution,
            device=complex_amplitude.device,
            dtype=complex_amplitude.dtype_r,
        )

        # Per-wavelength base magnification in (x, y). Built from the (y, x)
        # array-axis pixel_size / resolution and flipped once here into the (x, y)
        # focal-plane convention the chirp-z and the learnable params share. This is
        # the single boundary between the array-axis and focal-plane conventions.
        self._base_magnification: Tensor = fourier_lens_magnification(
            complex_amplitude.wavelength.unsqueeze(-1),
            self.focal_length,
            complex_amplitude.pixel_size,
            resolution_in.unsqueeze(0),
            self._pixel_size_out.unsqueeze(0),
        ).flip(-1)  # (n_wl, 2): (x, y)

        real_dtype = complex_amplitude.dtype_r
        device = complex_amplitude.device
        # scale_factor and shift are (x, y), matching the geometry / GeometricWarp
        # convention and the (x, y) base magnification, so they combine directly.
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

    def apply_partial_affine(self, transform: PartialAffineTransform) -> None:
        """Seed the learnable ``scale_factor`` / ``shift`` / ``angle`` from a fitted
        camera -> model similarity, composing it as a residual onto the current
        values (see :func:`~hologradpy.geometry.partial_affine\
        .recalibrated_partial_affine`).

        The lens stores ``shift`` and ``scale_factor`` as (x, y), matching the
        transform's point convention, so no axis swap is needed. ``shift`` is an image
        translation in output pixels, as it is on
        :class:`~hologradpy.optics.modules.geometric_transforms.GeometricWarp`, so the
        residual is stored as it arrives.
        """
        if not hasattr(self, "scale_factor"):
            raise RuntimeError(
                "FourierLensCZT must be initialized before apply_partial_affine "
                "(run the system once)."
            )
        center = (self.resolution_out[1] // 2, self.resolution_out[0] // 2)
        scale, angle_deg, shift = recalibrated_partial_affine(
            float(self.scale_factor.mean()),
            float(self.angle),
            (float(self.shift[0]), float(self.shift[1])),
            transform,
            center,
        )
        with torch.no_grad():
            self.scale_factor.copy_(
                torch.tensor(
                    [scale, scale],
                    dtype=self.scale_factor.dtype,
                    device=self.scale_factor.device,
                )
            )
            self.shift.copy_(
                torch.tensor(
                    [shift[0], shift[1]],
                    dtype=self.shift.dtype,
                    device=self.shift.device,
                )
            )
            self.angle.copy_(torch.as_tensor(angle_deg, dtype=self.angle.dtype))

    def _power_prefactor(self: FourierLensCZT) -> Tensor:
        """Fourier-lens amplitude prefactor ``(du*dv) / (lambda*f)`` per
        wavelength (input pixel area over ``lambda*f``), so the exact chirp-z
        transform conserves optical power (``integral|E_focal|^2 dx ==
        integral|E_slm|^2 du`` over the captured window). Computed in float64 and
        cast to the field's real dtype; the global ``1/i`` phase is omitted as it
        does not affect power. Shaped ``(1, n_wl, 1, 1)`` for the flattened field.
        """
        pixel_size_in = self.pixel_size_in
        area = pixel_area(pixel_size_in)
        wavelength = self.input_geometry.wavelength.to(torch.float64).reshape(-1)
        prefactor = fourier_lens_power_prefactor(
            area, wavelength, self.focal_length
        )
        return prefactor.to(pixel_size_in.dtype).reshape(1, -1, 1, 1)

    def _resolve_padding(self: FourierLensCZT) -> tuple[int, int]:
        if self.padded_resolution is None:
            return padded_resolution_for_rotation(
                self._input_resolution, float(self.angle_init)
            )

        padded = tuple(int(length) for length in self.padded_resolution)
        if any(
            padded[axis] < self._input_resolution[axis] for axis in (0, 1)
        ):
            raise ValueError(
                f"padded_resolution {padded} is smaller than the input "
                f"{self._input_resolution} on at least one axis, which would crop the "
                "field rather than give the rotation room."
            )
        return padded

    def _chirp_z(self: FourierLensCZT, scale: Tensor) -> ChirpZPartialAffine:
        """Build the per-wavelength scale + shift + rotate chirp-z.

        ``scale`` is the effective per-axis magnification ``(x, y)``; the window is
        offset by ``shift`` output pixels and turned by ``angle``. The transform folds
        the rotation into its own sampling rather than turning the field first, which is
        both cheaper and exact where three shears of the field are not.

        The same object serves both directions: the transform's ``adjoint`` reverses
        the rotation itself, so the angle is not negated here.
        """
        magnification = (scale[0], scale[1])  # (x, y)
        angle = torch.deg2rad(self.angle)
        if not self.learnable:
            # A plain float lets the transform skip the rotation entirely at zero. When
            # the parameters are learnable the tensor is kept so a gradient flows, even
            # at zero.
            angle = float(angle)

        shift = window_offset_from_pixels(
            self.shift, self._padded_resolution, (scale[0], scale[1])
        )  # (x, y)

        return ChirpZPartialAffine(
            self._padded_resolution,
            self.resolution_out,
            magnification=magnification,
            shift=shift,
            angle=angle,
            device=scale.device,
        )

    def forward(
        self: FourierLensCZT, complex_amplitude: ComplexAmplitude
    ) -> ComplexAmplitude:
        flat_field, batch_spec = complex_amplitude.flatten_batch()  # (N, n_wl, H, W)
        field = to_canvas(flat_field, self._padded_resolution)

        scale = self.scale_factor.abs() * self._base_magnification  # (n_wl, 2): (x, y)
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
        """Conjugate transpose of :meth:`forward`: the chirp-z adjoint, the inverse
        rotation, then crop.
        """
        flat_field, batch_spec = complex_amplitude.flatten_batch()
        scale = self.scale_factor.abs() * self._base_magnification
        inputs = [
            self._chirp_z(scale[wavelength]).adjoint(flat_field[:, wavelength])
            for wavelength in range(flat_field.shape[1])
        ]
        field = torch.stack(inputs, dim=1)  # (N, n_wl, H, W)
        field = to_canvas(field, self._input_resolution)
        if self.power_normalized:
            field = field * self._power_prefactor()

        return ComplexAmplitude.unflatten_batch(
            field,
            batch_spec,
            complex_amplitude.wavelength,
            self.pixel_size_in,
        )
