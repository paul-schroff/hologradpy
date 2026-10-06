"""WeightedGerchbergSaxtonPhaseRetriever on a small FFT model, which simulates the field
of each hologram.

The positions lie on pixels of the model's focal plane, so the model shows each spot on
its pixel. One test puts the positions between the pixels of a coarser grid.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hologradpy.grids import coordinates_to_indices
from hologradpy.holography.phase_retrieval import WeightedGerchbergSaxtonPhaseRetriever
from hologradpy.optics.complex_amplitude import ComplexAmplitude, FieldGeometry
from hologradpy.optics.modules.slm_fields import PixelwiseSLMField
from hologradpy.optics.modules.virtual_slms import VirtualSLM
from hologradpy.optics.systems import SLMFFT
from hologradpy.profiles.amplitude import gaussian_beam_intensity
from hologradpy.utils import as_image, gpu_to_numpy

SLM_RESOLUTION = (64, 80)
PITCH = 12.5e-6
WAVELENGTH = 1e-6
FOCAL_LENGTH = 0.1
PADDED = (256, 256)
FOCAL_PITCH = WAVELENGTH * FOCAL_LENGTH / (PADDED[0] * PITCH)  # 31.25 um
# Not symmetric, so a mirrored or transposed hologram misses the sites.
POSITIONS = FOCAL_PITCH * np.array([[-20.0, -12.0], [-8.0, -12.0], [-20.0, 5.0]])
WEIGHTS = np.array([1.0, 2.0, 1.0])


def _model(aberrated: bool = False, padded_resolution=PADDED) -> SLMFFT:
    geometry = FieldGeometry(
        resolution=SLM_RESOLUTION,
        pixel_size=torch.tensor([PITCH, PITCH]),
        wavelength=torch.tensor(WAVELENGTH),
    )
    x, y = geometry.get_spatial_grid()
    amplitude = gaussian_beam_intensity(x, y, beam_radius=0.3e-3).sqrt()
    phase = 40 * ((x / 0.5e-3) ** 2 - (y / 0.4e-3) ** 3) if aberrated else 0 * x
    beam = ComplexAmplitude(
        amplitude * torch.exp(1j * phase),
        wavelength=geometry.wavelength,
        pixel_size=geometry.pixel_size,
    )
    return SLMFFT(
        input_geometry=geometry,
        virtual_slm=VirtualSLM(full_scale_cycles=1.0),
        slm_field=PixelwiseSLMField(beam),
        focal_length=FOCAL_LENGTH,
        padded_resolution=padded_resolution,
    )


def _site_intensities(
    model: SLMFFT, positions: np.ndarray = POSITIONS
) -> tuple[np.ndarray, bool]:
    """The intensity at each site, and whether each site is the peak around it."""
    with torch.no_grad():
        intensity = gpu_to_numpy(as_image(model().intensity))
    x, y = model.fourier_lens.get_spatial_grid_output()
    sites = [(int(r), int(c)) for r, c in coordinates_to_indices(x, y, positions)]
    peaked = all(
        intensity[row, column]
        == intensity[row - 3 : row + 4, column - 3 : column + 4].max()
        for row, column in sites
    )
    return np.array([intensity[site] for site in sites]), peaked


@pytest.mark.parametrize("aberrated", [False, True], ids=["flat", "aberrated"])
def test_spots_land_on_their_pixels_in_the_ratios_asked_for(aberrated) -> None:
    """With an aberrated beam too, whose phase the retriever corrects on the SLM."""
    model = _model(aberrated)
    WeightedGerchbergSaxtonPhaseRetriever(model, POSITIONS, WEIGHTS).retrieve_phase(
        50, verbose=False
    )

    intensities, peaked = _site_intensities(model)
    assert peaked
    np.testing.assert_allclose(
        intensities / intensities.mean(), WEIGHTS / WEIGHTS.mean(), rtol=0.02
    )


def test_spots_between_the_pixels_of_an_fft_grid_land_where_asked() -> None:
    """Half a pixel of the 256 grid off its pixels, on a model four times finer."""
    offsets = FOCAL_PITCH * np.array([[0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])
    positions = POSITIONS + offsets
    model = _model(padded_resolution=(1024, 1024))
    WeightedGerchbergSaxtonPhaseRetriever(model, positions, WEIGHTS).retrieve_phase(
        50, verbose=False
    )

    intensities, peaked = _site_intensities(model, positions)
    assert peaked
    np.testing.assert_allclose(
        intensities / intensities.mean(), WEIGHTS / WEIGHTS.mean(), rtol=0.02
    )


def test_a_hologram_is_corrected_from_where_it_was() -> None:
    """Started from a hologram with new weights and the focal-plane phase held, as in a
    feedback step.
    """
    model = _model()
    phase = WeightedGerchbergSaxtonPhaseRetriever(
        model, POSITIONS, WEIGHTS
    ).retrieve_phase(50, verbose=False)
    corrected = np.array([1.0, 1.0, 2.0])
    WeightedGerchbergSaxtonPhaseRetriever(
        model,
        POSITIONS,
        corrected,
        init_slm_phase=phase,
        focal_phase_iterations=1,
    ).retrieve_phase(20, verbose=False)

    intensities, _ = _site_intensities(model)
    np.testing.assert_allclose(
        intensities / intensities.mean(), corrected / corrected.mean(), rtol=0.05
    )


def test_the_random_start_is_seeded() -> None:
    def retrieve(seed):
        return WeightedGerchbergSaxtonPhaseRetriever(
            _model(), POSITIONS, WEIGHTS, seed=seed
        ).retrieve_phase(3, verbose=False)

    assert torch.equal(retrieve(0), retrieve(0))
    assert not torch.equal(retrieve(0), retrieve(1))


def test_the_record_holds_a_falling_loss() -> None:
    record = WeightedGerchbergSaxtonPhaseRetriever(
        _model(), POSITIONS, WEIGHTS
    ).retrieve(30, verbose=False)

    assert record.phase.shape == SLM_RESOLUTION
    assert len(record.loss_history) == 30
    assert record.loss_history[-1] < record.loss_history[0]


def test_the_recorded_steps_follow_the_iterations(tmp_path) -> None:
    """Each step is read off the SLM, so the last one holds the levels the retrieval
    ends on.
    """
    model = _model()
    record = WeightedGerchbergSaxtonPhaseRetriever(model, POSITIONS, WEIGHTS).retrieve(
        4, verbose=False, step_stride=2, step_directory=tmp_path
    )

    assert record.step_iterations == [2, 4]
    first, last = (record.load_step(i, tmp_path) for i in record.step_iterations)
    assert not np.array_equal(first, last)
    np.testing.assert_array_equal(last, gpu_to_numpy(model.virtual_slm.levels))


def test_positions_beyond_the_addressable_field_are_refused() -> None:
    with pytest.raises(ValueError, match="beyond the field the SLM addresses"):
        WeightedGerchbergSaxtonPhaseRetriever(_model(), [[200 * FOCAL_PITCH, 0.0]])


def test_repeated_positions_are_refused() -> None:
    positions = [[0.0, 0.0], [FOCAL_PITCH, 0.0], [0.0, 0.0]]
    with pytest.raises(ValueError, match=r"Positions \[0, 2\] each repeat"):
        WeightedGerchbergSaxtonPhaseRetriever(_model(), positions)


@pytest.mark.parametrize(
    "intensity", [0.0, -1.0, np.inf, np.nan], ids=["zero", "negative", "inf", "nan"]
)
def test_intensities_that_are_not_positive_and_finite_are_refused(intensity) -> None:
    with pytest.raises(ValueError, match=r"Intensities \[1\]"):
        WeightedGerchbergSaxtonPhaseRetriever(
            _model(), POSITIONS, [1.0, intensity, 1.0]
        )


def test_focal_phase_iterations_below_one_are_refused() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        WeightedGerchbergSaxtonPhaseRetriever(
            _model(), POSITIONS, focal_phase_iterations=0
        )
