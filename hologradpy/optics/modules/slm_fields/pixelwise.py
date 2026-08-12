"""The SLM-plane field stored directly, one complex value per SLM pixel."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import Parameter

from .abstract import SLMField
from ..abstract import SaveDict
from ...complex_amplitude import ComplexAmplitude

if TYPE_CHECKING:
    from ....calibration.wavefront.abstract import WavefrontCalibrationData


class PixelwiseSLMField(SLMField):
    """The SLM-plane field held one free complex value per pixel. As many free
    parameters as the SLM has pixels.

    Args:
        init_field: The field to start from. Defaults to a uniform one, built on the
            first forward pass once the geometry is known.
    """

    def __init__(
        self: PixelwiseSLMField,
        init_field: ComplexAmplitude | None = None,
    ) -> None:
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
            torch.tensor(
                self.init_field.phase,
                dtype=complex_amplitude.dtype_r,
                device=complex_amplitude.device,
            ),
            requires_grad=False,
        )

        self.amplitude = Parameter(
            torch.tensor(
                self.init_field.amplitude,
                dtype=complex_amplitude.dtype_r,
                device=complex_amplitude.device,
            ),
            requires_grad=False,
        )

    @classmethod
    def from_file(cls, path: str, device: torch.device = "cpu") -> PixelwiseSLMField:
        state: SaveDict = torch.load(path, map_location=device, weights_only=False)
        state_dict = state["state_dict"]
        geometry = state["input_geometry"]

        init_field_data: Tensor = state_dict["amplitude"] * torch.exp(
            1j * state_dict["phase"]
        )

        init_field = ComplexAmplitude(
            data=init_field_data.to(device),
            wavelength=geometry.wavelength.to(device),
            pixel_size=geometry.pixel_size.to(device),
        )
        return cls(init_field=init_field)

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
        it is parameterised.
        """
        return self.get_transmission()


