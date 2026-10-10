"""The physical frame and the pixel frame of HoloGradPy.

Positions, displacements and angles are physical. Arrays are indexed in the pixel
frame, by ``(column, row)``. Only :mod:`hologradpy.grids` converts between the two.

The consistency tests hold whichever way y points. Each way of putting light at a
position has to agree with the model's output grid: a linear phase and the weighted
Gerchberg-Saxton retriever put a spot on the pixel of its position. The
conjugate-gradient retriever is left out, since it optimises the image on that grid
itself. The vortex detector reads a vortex with the handedness that
:func:`~hologradpy.profiles.phase.vortex_phase` gives it, so its anti-vortex cancels it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from hologradpy.grids import coordinates_to_indices
from hologradpy.holography.phase_retrieval import WeightedGerchbergSaxtonPhaseRetriever
from hologradpy.holography.vortices import VortexDetector
from hologradpy.holography.vortices.vortex_detection import find_vortex_charge
from hologradpy.optics.complex_amplitude import ComplexAmplitude, FieldGeometry
from hologradpy.optics.modules.slm_fields import PixelwiseSLMField
from hologradpy.optics.modules.virtual_slms import VirtualSLM
from hologradpy.optics.systems import SLMFFT
from hologradpy.profiles.amplitude import gaussian_beam_intensity
from hologradpy.profiles.phase import linear_phase, vortex_field
from hologradpy.utils import as_image

SLM_RESOLUTION = (64, 80)
PITCH = 12.5e-6
WAVELENGTH = 1e-6
FOCAL_LENGTH = 0.1
PADDED = (256, 256)
FOCAL_PITCH = WAVELENGTH * FOCAL_LENGTH / (PADDED[0] * PITCH)  # 31.25 um
# Off both axes and off the diagonals, so a mirror or a transpose misses its pixel.
POSITION = (-20 * FOCAL_PITCH, 7 * FOCAL_PITCH)
VORTEX_RESOLUTION = (32, 32)


def _model() -> SLMFFT:
    """A Gaussian beam on a small SLM, focused by an FFT lens."""
    geometry = FieldGeometry(
        resolution=SLM_RESOLUTION,
        pixel_size=torch.tensor([PITCH, PITCH]),
        wavelength=torch.tensor(WAVELENGTH),
    )
    x, y = geometry.get_spatial_grid()
    amplitude = gaussian_beam_intensity(x, y, 0.3e-3).sqrt()
    model = SLMFFT(
        input_geometry=geometry,
        virtual_slm=VirtualSLM(full_scale_cycles=1.0),
        slm_field=PixelwiseSLMField(
            ComplexAmplitude.from_geometry(geometry, data=amplitude + 0j)
        ),
        focal_length=FOCAL_LENGTH,
        padded_resolution=PADDED,
    )
    model()  # Builds the parameters of the virtual SLM.
    return model


def _peak(model: SLMFFT) -> tuple[int, int]:
    """The ``(row, column)`` of the brightest output pixel."""
    with torch.no_grad():
        intensity = as_image(model().intensity)
    row, column = np.unravel_index(int(intensity.argmax()), tuple(intensity.shape))
    return int(row), int(column)


def _pixel_of(model: SLMFFT, position: tuple[float, float]) -> tuple[int, int]:
    """The ``(row, column)`` of ``position`` on the output grid of the model."""
    x, y = model.fourier_lens.get_spatial_grid_output()
    ((row, column),) = coordinates_to_indices(
        x, y, torch.tensor([position], dtype=x.dtype)
    )
    return int(row), int(column)


def _linear_phase(model: SLMFFT) -> None:
    x, y = model.input_geometry.get_spatial_grid()
    model.virtual_slm.set_phase(
        linear_phase(
            x,
            y,
            *POSITION,
            wavenumber=2 * math.pi / WAVELENGTH,
            focal_length=FOCAL_LENGTH,
        )
    )


def _weighted_gerchberg_saxton(model: SLMFFT) -> None:
    WeightedGerchbergSaxtonPhaseRetriever(
        model, np.array([POSITION]), np.array([1.0])
    ).retrieve_phase(10, verbose=False)


@pytest.mark.parametrize(
    "place",
    [_linear_phase, _weighted_gerchberg_saxton],
    ids=["linear phase", "weighted Gerchberg-Saxton"],
)
def test_each_way_of_placing_a_spot_puts_it_on_the_pixel_of_its_position(place):
    model = _model()

    place(model)

    assert _peak(model) == _pixel_of(model, POSITION)


def _field(data: torch.Tensor) -> ComplexAmplitude:
    geometry = FieldGeometry(
        resolution=VORTEX_RESOLUTION,
        pixel_size=torch.tensor([1e-5, 1e-5]),
        wavelength=torch.tensor(1e-6),
    )
    return ComplexAmplitude.from_geometry(geometry, data=data.to(torch.complex64))


def test_a_vortex_of_charge_one_is_read_as_charge_one_and_cancelled():
    """The detector reads the charge on its own grid, so the anti-vortex it builds from
    that charge cancels the winding instead of doubling it.
    """
    detector = VortexDetector(VORTEX_RESOLUTION)
    x, y = detector.pixel_grid
    # Half a pixel off the samples, where the detector can see the zero.
    field = vortex_field(x, y, torch.tensor([[3.5, -6.5]]), torch.tensor([1]))

    detector.detect_vortices(
        _field(field), target_intensity=torch.ones(VORTEX_RESOLUTION)
    )
    charges = detector.charges.reshape(-1)
    centers = [
        center for center, charge in zip(detector.center_indices, charges) if charge
    ]

    assert charges[charges != 0].tolist() == [1]
    cancelled = field * detector.generate_anti_vortex_field()
    assert find_vortex_charge(_field(cancelled), x, y, centers).tolist() == [0]
