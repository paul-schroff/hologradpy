"""Sign-convention invariants: VirtualSLM.set_phase takes the desired optical
phase, like slmsuite's SLM.set_phase.

The same linear-phase grating pushed through the hardware path
(slm.set_phase -> slmsuite grayscale conversion -> SimulatedSLMTorch) and set
directly on the model's virtual SLM must place the focal spot at the same,
positive target position -- for both the FFT and the CZT Fourier lens models.
Retrieved phases must be valid on both paths as well.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hologradpy.hardware import SimulatedSLMTorch
from hologradpy.holography.phase_retrieval import LinearSuperpositionPhaseRetriever
from hologradpy.propagation.amplitude_profiles import gaussian_beam_intensity
from hologradpy.propagation.complex_amplitude import ComplexAmplitude, FieldGeometry
from hologradpy.propagation.diagonal_elements import StaticSLMField
from hologradpy.propagation.optical_systems import SLMCZT, SLMFFT
from hologradpy.propagation.phase_profiles import linear_phase
from hologradpy.utils import gpu_to_numpy

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

FOCAL_LENGTH = 0.25
WAVELENGTH = 0.630e-6
CZT_PIXEL = 30e-6
TARGET = (500e-6, 300e-6)  # (x, y) metres in the focal plane


@pytest.fixture(scope="module")
def setup():
    geometry = FieldGeometry(
        resolution=(256, 320),
        pixel_size=torch.tensor([12.5e-6, 12.5e-6]),
        wavelength=torch.tensor(WAVELENGTH),
    )
    slm = SimulatedSLMTorch(input_geometry=geometry, bitdepth=8)
    intensity = gaussian_beam_intensity(*geometry.get_spatial_grid(), beam_radius=1e-3)
    beam = ComplexAmplitude(
        intensity.sqrt() + 0j,
        wavelength=geometry.wavelength,
        pixel_size=geometry.pixel_size,
    )
    # Both models share the (simulated) hardware's virtual SLM, so the same
    # instance is driven through either path.
    fft_model = SLMFFT(
        input_geometry=geometry,
        virtual_slm=slm.virtual_slm,
        static_slm_field=StaticSLMField(beam),
        focal_length=FOCAL_LENGTH,
        padded_resolution=(512, 512),
    )
    czt_model = SLMCZT(
        input_geometry=geometry,
        virtual_slm=slm.virtual_slm,
        camera_resolution=(240, 320),
        camera_pixel_size=(CZT_PIXEL, CZT_PIXEL),
        focal_length=FOCAL_LENGTH,
        static_slm_field=StaticSLMField(beam),
    )
    fft_model()
    czt_model()

    grating = linear_phase(
        *geometry.get_spatial_grid(),
        *TARGET,
        focal_length=FOCAL_LENGTH,
        wavenumber=float(2 * np.pi / WAVELENGTH),
    )
    return slm, fft_model, czt_model, grating


def _fft_pixel(fft_model) -> float:
    return float(fft_model[-1].pixel_size_out.tolist()[0][0])


def _assert_peak_at_target(model, pixel_size: float) -> None:
    image = gpu_to_numpy(model().intensity)
    row, column = np.unravel_index(int(np.argmax(image)), image.shape)
    x = (column - image.shape[1] // 2) * pixel_size
    y = (row - image.shape[0] // 2) * pixel_size
    assert x == pytest.approx(TARGET[0], abs=pixel_size)
    assert y == pytest.approx(TARGET[1], abs=pixel_size)


def test_direct_set_phase_places_spot_at_target(setup):
    slm, fft_model, czt_model, grating = setup
    slm.virtual_slm.set_phase(grating)
    _assert_peak_at_target(fft_model, _fft_pixel(fft_model))
    _assert_peak_at_target(czt_model, CZT_PIXEL)


def test_hardware_set_phase_places_spot_at_target(setup):
    slm, fft_model, czt_model, grating = setup
    slm.set_phase(gpu_to_numpy(grating))
    _assert_peak_at_target(fft_model, _fft_pixel(fft_model))
    _assert_peak_at_target(czt_model, CZT_PIXEL)


def test_retrieved_phase_is_valid_on_both_paths(setup):
    slm, fft_model, czt_model, _ = setup
    phase = LinearSuperpositionPhaseRetriever(
        fft_model, target_positions=torch.tensor([[TARGET[0], TARGET[1]]])
    ).retrieve_phase()

    slm.virtual_slm.set_phase(phase)
    _assert_peak_at_target(fft_model, _fft_pixel(fft_model))
    _assert_peak_at_target(czt_model, CZT_PIXEL)

    slm.set_phase(gpu_to_numpy(phase))
    _assert_peak_at_target(fft_model, _fft_pixel(fft_model))
    _assert_peak_at_target(czt_model, CZT_PIXEL)
