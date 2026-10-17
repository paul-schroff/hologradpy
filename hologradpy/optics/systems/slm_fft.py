from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..complex_amplitude import ComplexAmplitude, FieldGeometry

from ..modules.propagators import FourierLensFFT
from ..modules.slm_fields import PixelwiseSLMField, SLMField
from ..modules.virtual_slms.abstract import VirtualSLM

from .abstract import SLMFourierLensModel, slm_stages, upscaled_padding
from ..modules.abstract import capture_init

if TYPE_CHECKING:
    from ...hardware import SLM

class SLMFFT(SLMFourierLensModel):
    virtual_slm: VirtualSLM
    slm_field: SLMField
    fourier_lens: FourierLensFFT

    @capture_init
    def __init__(
        self,
        input_geometry: FieldGeometry,
        virtual_slm: VirtualSLM,
        slm_field: SLMField,
        focal_length: float,
        padded_resolution: tuple[int, int] = (2048, 2048),
        pointing_focal_shift_std: float | tuple[float, float] | None = None,
        pointing_seed: int | None = None,
        grid_cache: bool = False,
        pixel_size_out: tuple[float, float] | None = None,
    ) -> None:
        # A pixel_size_out sizes the FFT itself, and padded_resolution is then unused.
        fourier_lens = (
            FourierLensFFT(focal_length, pixel_size_out=pixel_size_out)
            if pixel_size_out is not None
            else FourierLensFFT(
                focal_length,
                padded_resolution=upscaled_padding(padded_resolution, virtual_slm),
            )
        )
        super().__init__(
            input_geometry=input_geometry,
            focal_length=focal_length,
            pointing_focal_shift_std=pointing_focal_shift_std,
            pointing_seed=pointing_seed,
            **slm_stages(virtual_slm, slm_field, grid_cache),
            fourier_lens=fourier_lens,
        )

    @classmethod
    def from_slm(
        cls,
        slm: SLM,
        focal_length: float,
        pixel_size_out: tuple[float, float],
        device: torch.device | None = None,
    ) -> SLMFFT:
        """An ideal model of ``slm`` behind a Fourier lens, illuminated by a uniform
        beam.

        Args:
            slm: The SLM, for its geometry and its wavelength.
            focal_length: The focal length of the Fourier lens in metres.
            pixel_size_out: The ``(y, x)`` pitch in metres to sample the focal plane
                near, such as the pitch of a camera. The first pass raises
                ``ValueError`` for a pitch coarser than the SLM reaches,
                ``lambda f / (N dx)`` along an axis of ``N`` pixels of pitch ``dx``.
            device: The torch device of the model, the CPU for None.

        Returns:
            SLMFFT: The model.
        """
        geometry = FieldGeometry.from_slm(slm, device)
        beam = ComplexAmplitude(
            torch.ones(tuple(slm.resolution), dtype=torch.complex64, device=device),
            wavelength=geometry.wavelength,
            pixel_size=geometry.pixel_size,
        )
        return cls(
            input_geometry=geometry,
            virtual_slm=VirtualSLM(full_scale_cycles=1.0),
            slm_field=PixelwiseSLMField(beam),
            focal_length=focal_length,
            pixel_size_out=pixel_size_out,
        )
