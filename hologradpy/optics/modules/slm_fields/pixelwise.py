"""The SLM-plane field stored directly, one complex value per SLM pixel."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import Parameter

from .abstract import SLMField
from ..abstract import capture_init
from ...complex_amplitude import ComplexAmplitude

if TYPE_CHECKING:
    from ....calibration.wavefront.abstract import WavefrontCalibrationData


class PixelwiseSLMField(SLMField):
    """The SLM-plane field held one free complex value per pixel. As many free
    parameters as the SLM has pixels.
    """

    @capture_init
    def __init__(
        self: PixelwiseSLMField,
        init_field: ComplexAmplitude | None = None,
    ) -> None:
        """
        Args:
            init_field: The field to start from. Defaults to a uniform one, built on the
                first forward pass once the geometry is known.
        """
        super().__init__()
        self.init_field: ComplexAmplitude | None = init_field

    def lazy_init(self: PixelwiseSLMField, complex_amplitude: ComplexAmplitude) -> None:
        if self.init_field is None:
            number_of_wavelengths = complex_amplitude.number_of_wavelengths
            # A uniform default field is wavelength-independent, but the
            # ComplexAmplitude layout requires an explicit wavelength axis when more
            # than one wavelength is present.
            default_shape = (
                self.resolution_in
                if number_of_wavelengths == 1
                else (number_of_wavelengths, *self.resolution_in)
            )
            self.init_field = ComplexAmplitude(
                data=torch.ones(
                    default_shape,
                    dtype=complex_amplitude.dtype,
                    device=complex_amplitude.device,
                ),
                wavelength=complex_amplitude.wavelength,
                pixel_size=complex_amplitude.pixel_size,
            )

        self.phase = Parameter(
            self.init_field.phase.detach().clone().to(
                dtype=complex_amplitude.dtype_r, device=complex_amplitude.device
            ),
            requires_grad=False,
        )

        self.amplitude = Parameter(
            self.init_field.amplitude.detach().clone().to(
                dtype=complex_amplitude.dtype_r, device=complex_amplitude.device
            ),
            requires_grad=False,
        )

    @classmethod
    def from_calibration_data(
        cls, calibration_data: WavefrontCalibrationData
    ) -> PixelwiseSLMField:
        return cls(init_field=calibration_data.complex_amplitude)

    def get_transmission(self: PixelwiseSLMField) -> Tensor:
        """Complex transmission ``amplitude * exp(i * phase)``. The stored constant
        field, applied as a per-pixel diagonal multiply.
        """
        return self.amplitude * torch.exp(1j * self.phase)

    def get_wavefront(self: PixelwiseSLMField) -> Tensor:
        """The SLM-plane field this module represents.

        The same thing as the transmission here, since the field is stored directly.
        Named alongside :meth:`PSFSLMField.get_wavefront`, where the two differ, so a
        caller can ask any SLM-plane field module for its wavefront without knowing how
        it is parameterized.
        """
        return self.get_transmission()


