from __future__ import annotations

import math

import torch
from torch import Tensor

from ....fourier_transforms import (
    ChirpZPartialAffine,
    padded_resolution_for_rotation,
)

from ....utils import to_canvas
from .partial_affine_fourier_lens import PartialAffineFourierLens
from ...complex_amplitude import ComplexAmplitude


class FourierLensCZT(PartialAffineFourierLens):
    """Exact Fourier lens via the chirp-z zoom, with a learnable focal-plane partial
    affine (:class:`~.partial_affine_fourier_lens.PartialAffineFourierLens`).

    The field is padded before the transform, so the rotation keeps its corners. With
    ``padded_resolution`` left at None, the padding follows the angle of the partial
    affine. It grows whenever a larger angle needs more room, including an angle set
    by a calibration after the first pass.

    With :attr:`output_roi` set, the cost of the transform follows the rectangle
    (:class:`ChirpZPartialAffine`).
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
        """
        Args:
            focal_length: The focal length in metres.
            resolution_out: The ``(height, width)`` of the focal plane in pixels.
            pixel_size_out: The ``(height, width)`` pitch of the focal plane in metres.
            shift: The starting translation of the partial affine, in output pixels
                along ``(x, y)``.
            angle: The starting rotation of the partial affine, in degrees.
            learnable: Whether the parameters of the partial affine require a gradient.
            power_normalized: Whether the output carries the optical power of the
                input, through the prefactor ``(du * dv) / (lambda * f)``.
            padded_resolution: The ``(height, width)`` the field is padded to before
                the transform, or None for the padding the angle of the partial affine
                needs.
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
        self._padded_resolution: tuple[int, int] = self._resolve_padding()

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
        self._padded_resolution = tuple(
            max(current, need)
            for current, need in zip(self._padded_resolution, needed)
        )

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
                f"{self._input_resolution} on at least one axis, so it crops the field "
                "and leaves the rotation no room."
            )
        return padded

    def _chirp_z(
        self: FourierLensCZT,
        magnification: Tensor,
        offset: Tensor,
        angle: float | Tensor,
    ) -> ChirpZPartialAffine:
        """The chirp-z of one wavelength, from its magnification ``(x, y)`` and window
        offset ``(x, y)`` (:meth:`_sampled_window`).

        The ``adjoint`` of the transform also reverses the rotation, so the same object
        serves both directions.
        """
        return ChirpZPartialAffine(
            self._padded_resolution,
            self.resolution_out,
            magnification=(magnification[0], magnification[1]),
            shift=(offset[0], offset[1]),
            angle=angle,
            device=magnification.device,
            output_roi=self.output_roi,
        )

    def forward(
        self: FourierLensCZT, complex_amplitude: ComplexAmplitude
    ) -> ComplexAmplitude:
        self._grow_padding_for_angle()
        flat_field, batch_spec = complex_amplitude.flatten_batch()  # (N, n_wl, H, W)
        field = to_canvas(flat_field, self._padded_resolution)

        magnification, offset, angle = self._sampled_window(self._padded_resolution)
        outputs = [
            self._chirp_z(magnification[wavelength], offset[wavelength], angle).forward(
                field[:, wavelength]
            )
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
        magnification, offset, angle = self._sampled_window(self._padded_resolution)
        inputs = [
            self._chirp_z(magnification[wavelength], offset[wavelength], angle).adjoint(
                flat_field[:, wavelength]
            )
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
