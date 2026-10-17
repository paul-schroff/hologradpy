"""The field of a Fourier lens at a set of focal spots (``phase_retrieval.spots``),
which weighted Gerchberg-Saxton and linear superposition share.
"""

from __future__ import annotations

import numpy as np
import torch

from hologradpy.holography.phase_retrieval import LinearSuperpositionPhaseRetriever
from hologradpy.holography.phase_retrieval.spots import (
    propagate_from_spots,
    propagate_to_spots,
    spot_phasors,
)
from hologradpy.optics.complex_amplitude import ComplexAmplitude, FieldGeometry
from hologradpy.optics.modules.slm_fields import PixelwiseSLMField
from hologradpy.optics.modules.virtual_slms import VirtualSLM
from hologradpy.optics.systems import SLMFFT
from hologradpy.profiles.phase import linear_phase

RESOLUTION = (24, 32)
PITCH = 12.5e-6
WAVELENGTH = 1e-6
FOCAL_LENGTH = 0.1
POSITIONS = np.array([[-6.0e-4, 2.5e-4], [1.5e-4, -4.0e-4], [3.0e-4, 3.5e-4]])
INTENSITIES = np.array([1.0, 2.0, 0.5])
PHASES = np.array([0.3, -1.2, 2.0])


def _model() -> SLMFFT:
    geometry = FieldGeometry(
        resolution=RESOLUTION,
        pixel_size=torch.tensor([PITCH, PITCH]),
        wavelength=torch.tensor(WAVELENGTH),
    )
    beam = ComplexAmplitude(
        torch.ones(RESOLUTION, dtype=torch.complex64),
        wavelength=geometry.wavelength,
        pixel_size=geometry.pixel_size,
    )
    return SLMFFT(
        input_geometry=geometry,
        virtual_slm=VirtualSLM(full_scale_cycles=1.0),
        slm_field=PixelwiseSLMField(beam),
        focal_length=FOCAL_LENGTH,
    )


def test_propagating_from_the_spots_is_the_adjoint_of_propagating_to_them() -> None:
    phasors = spot_phasors(_model(), POSITIONS, RESOLUTION, torch.complex128)
    generator = torch.Generator().manual_seed(0)
    field = torch.randn(RESOLUTION, dtype=torch.complex128, generator=generator)
    spot_field = torch.randn(
        len(POSITIONS), dtype=torch.complex128, generator=generator
    )

    forward = torch.sum(propagate_to_spots(field, *phasors).conj() * spot_field)
    adjoint = torch.sum(field.conj() * propagate_from_spots(spot_field, *phasors))

    torch.testing.assert_close(forward, adjoint, rtol=1e-12, atol=0.0)


def test_the_linear_superposition_is_the_sum_of_its_gratings() -> None:
    model = _model()
    retriever = LinearSuperpositionPhaseRetriever(
        model, POSITIONS, INTENSITIES, PHASES
    )

    phase = retriever.retrieve_phase().double().cpu()

    grid_x, grid_y = model.input_geometry.get_spatial_grid()
    gratings = sum(
        np.sqrt(intensity)
        * torch.exp(
            1j
            * (
                linear_phase(
                    grid_x.double(),
                    grid_y.double(),
                    float(x),
                    float(y),
                    wavenumber=2 * np.pi / WAVELENGTH,
                    focal_length=FOCAL_LENGTH,
                )
                + target_phase
            )
        )
        for (x, y), intensity, target_phase in zip(POSITIONS, INTENSITIES, PHASES)
    )
    expected = torch.remainder(gratings.angle(), 2 * torch.pi)
    error = torch.remainder(phase - expected + torch.pi, 2 * torch.pi) - torch.pi
    # The SLM holds the phase in float32.
    assert float(error.abs().max()) < 1e-5
