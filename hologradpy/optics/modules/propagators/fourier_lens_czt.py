from __future__ import annotations

import math

import torch
from torch import Tensor

from ....fourier_optics import (
    fourier_lens_magnification,
    fourier_lens_power_prefactor,
)
from ....fourier_transforms import (
    ChirpZPartialAffine,
    padded_resolution_for_rotation,
    window_offset_from_pixels,
)

from ....utils import to_canvas
from ..abstract import OpticsModule
from ..learnable_partial_affine import LearnablePartialAffine
from ...complex_amplitude import ComplexAmplitude, pixel_area


class FourierLensCZT(OpticsModule):
    """Exact Fourier lens via the chirp-z zoom, with a learnable focal-plane partial
    affine.

    The partial affine is :attr:`focal_plane_partial_affine`, a
    :class:`~hologradpy.optics.modules.learnable_partial_affine.LearnablePartialAffine`.
    The transform applies its zoom, shift and rotation by sampling the scaled, shifted
    and rotated window directly. Its parameters require a gradient when ``learnable``
    is set, so the focal-plane geometry can be calibrated by gradient descent.

    The geometry is per-wavelength: the base magnification is ``lambda * f /
    (pixel_in * resolution_in * pixel_out)``, so that with the parameters at their
    identity values (scale 1, shift 0, angle 0) a focal pixel measures
    ``pixel_out``.

    The field is padded before the transform, so the rotation keeps its corners. With
    ``padded_resolution`` left at None, the padding follows the angle of the partial
    affine and grows whenever a larger angle needs more room, including an angle set by
    a calibration after the first pass. An explicit ``padded_resolution`` is used as
    given.
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
        self.learnable: bool = learnable
        self.focal_plane_partial_affine = LearnablePartialAffine(
            resolution_out, shift=shift, angle=angle, learnable=learnable
        )
        self.power_normalized: bool = power_normalized
        self.padded_resolution: tuple[int, int] | None = padded_resolution

    def lazy_init(self: FourierLensCZT, complex_amplitude: ComplexAmplitude) -> None:
        # Output geometry (pixel_size_out / resolution_out) is set from the
        # constructor args by the base before this runs.
        self._input_resolution: tuple[int, int] = tuple(complex_amplitude.resolution)

        # scale_factor and shift are (x, y), matching the geometry / GeometricWarp
        # convention and the (x, y) base magnification, so they combine directly.
        self.focal_plane_partial_affine.to(
            device=complex_amplitude.device, dtype=complex_amplitude.dtype_r
        )
        self._set_padding(self._resolve_padding())

    def _set_padding(self: FourierLensCZT, padded_resolution: tuple[int, int]) -> None:
        """Transform on ``padded_resolution``, with the base magnification that keeps
        the focal plane sampled at ``pixel_size_out`` on that frame.
        """
        self._padded_resolution: tuple[int, int] = padded_resolution
        # On the device and in the dtype of the scale factor it multiplies in forward.
        scale_factor = self.focal_plane_partial_affine.scale_factor
        resolution_in = torch.tensor(
            padded_resolution, device=scale_factor.device, dtype=scale_factor.dtype
        )
        # The output pitch is (2,) during lazy_init and (n_wavelengths, 2) after it.
        pixel_size_out = self._pixel_size_out
        if pixel_size_out.ndim == 1:
            pixel_size_out = pixel_size_out.unsqueeze(0)

        # Per-wavelength base magnification in (x, y). Built from the (y, x)
        # array-axis pixel_size / resolution and flipped once here into the (x, y)
        # focal-plane convention the chirp-z and the learnable params share. This is
        # the single boundary between the array-axis and focal-plane conventions.
        self._base_magnification: Tensor = fourier_lens_magnification(
            self.input_geometry.wavelength.unsqueeze(-1),
            self.focal_length,
            self.pixel_size_in,
            resolution_in.unsqueeze(0),
            pixel_size_out,
        ).flip(-1)  # (n_wl, 2): (x, y)

    def _grow_padding_for_angle(self: FourierLensCZT) -> None:
        """Grow the automatic padding until it holds the rotation of the current angle.

        The padding never shrinks, and an explicit ``padded_resolution`` is used as
        given.

        Raises:
            ValueError: The angle lies more than 60 degrees from both 0 and 180
                degrees.
        """
        angle = float(self.focal_plane_partial_affine.angle.detach())
        if abs(math.cos(math.radians(angle))) < 0.5:
            raise ValueError(
                f"The focal-plane partial affine rotates by {angle:.1f} degrees, and "
                "the chirp-z lens only rotates by up to 60 degrees from 0 or 180 "
                "degrees. Set the orientation the camera mapping suggests with "
                "Camera.set_orientation, or model the system with SLMNUFFT."
            )
        if self.padded_resolution is not None:
            return
        needed = padded_resolution_for_rotation(self._input_resolution, angle)
        grown = tuple(
            max(current, need)
            for current, need in zip(self._padded_resolution, needed)
        )
        if grown != self._padded_resolution:
            self._set_padding(grown)

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
                self._input_resolution,
                float(self.focal_plane_partial_affine.angle.detach()),
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
        partial_affine = self.focal_plane_partial_affine
        magnification = (scale[0], scale[1])  # (x, y)
        angle = torch.deg2rad(partial_affine.angle)
        if not self.learnable:
            # A plain float lets the transform skip the rotation entirely at zero. When
            # the parameters are learnable the tensor is kept so a gradient flows, even
            # at zero.
            angle = float(angle)

        shift = window_offset_from_pixels(
            partial_affine.shift, self._padded_resolution, (scale[0], scale[1])
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
        self._grow_padding_for_angle()
        flat_field, batch_spec = complex_amplitude.flatten_batch()  # (N, n_wl, H, W)
        field = to_canvas(flat_field, self._padded_resolution)

        scale_factor = self.focal_plane_partial_affine.scale_factor
        scale = scale_factor.abs() * self._base_magnification  # (n_wl, 2): (x, y)
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
        self._grow_padding_for_angle()
        flat_field, batch_spec = complex_amplitude.flatten_batch()
        scale_factor = self.focal_plane_partial_affine.scale_factor
        scale = scale_factor.abs() * self._base_magnification
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
