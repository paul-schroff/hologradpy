"""The SLM-plane field stored directly, one complex value per SLM pixel."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from jaxtyping import Complex
from torch import Tensor
from torch.nn import Parameter

from .abstract import SLMField
from ..abstract import capture_init
from ...complex_amplitude import SCALAR, ComplexAmplitude

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
            self.init_field = ComplexAmplitude.from_geometry(
                replace(complex_amplitude.geometry, number_of_components=SCALAR),
                dtype=complex_amplitude.dtype,
            )

        wavefront = self._stored_wavefront(self.init_field).to(
            dtype=complex_amplitude.dtype, device=complex_amplitude.device
        )
        self.phase = Parameter(
            torch.angle(wavefront).detach().clone(), requires_grad=False
        )
        self.amplitude = Parameter(
            wavefront.abs().detach().clone(), requires_grad=False
        )

    @staticmethod
    def _stored_wavefront(
        init_field: ComplexAmplitude,
    ) -> Complex[Tensor, "H W"] | Complex[Tensor, "n_wavelengths H W"]:
        """The values of ``init_field`` in the layout the parameters keep, which is
        ``(H, W)`` for one wavelength and ``(n_wavelengths, H, W)`` otherwise, whatever
        rank the field arrived with.
        """
        flat, _ = init_field.flatten_batch()  # (N, n_wavelengths, H, W)
        if flat.shape[0] != 1:
            raise ValueError(
                "PixelwiseSLMField stores one SLM-plane field, but init_field "
                f"carries {flat.shape[0]} (shape {tuple(init_field.shape)})."
            )
        wavefront = flat[0]
        return wavefront[0] if wavefront.shape[0] == 1 else wavefront

    @classmethod
    def from_calibration_data(
        cls, calibration_data: WavefrontCalibrationData
    ) -> PixelwiseSLMField:
        return cls(init_field=calibration_data.complex_amplitude)

    def get_transmission(
        self: PixelwiseSLMField,
    ) -> Complex[Tensor, "H W"] | Complex[Tensor, "n_wavelengths H W"]:
        """Complex transmission ``amplitude * exp(i * phase)``. The stored constant
        field, applied as a per-pixel diagonal multiply.
        """
        return self.amplitude * torch.exp(1j * self.phase)

    def get_wavefront(
        self: PixelwiseSLMField,
    ) -> Complex[Tensor, "H W"] | Complex[Tensor, "n_wavelengths H W"]:
        """The SLM-plane field this module represents.

        The same thing as the transmission here, since the field is stored directly.
        Named alongside :meth:`PSFSLMField.get_wavefront`, where the two differ, so a
        caller can ask any SLM-plane field module for its wavefront without knowing how
        it is parameterized.
        """
        return self.get_transmission()


