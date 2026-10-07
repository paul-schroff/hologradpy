"""Detecting optical vortices, and telling them apart from ordinary amplitude zeros.

The distinction is the whole point and it is easy to lose. ``VortexDetector`` labels
every place the real and imaginary parts of the field both cross zero, and reports how
many such places it found. Most of them are not vortices: an interference null is a
zero the phase does not wind around, and ``find_vortex_charge`` scores it zero. Reading
``number_of_vortices`` as a vortex count therefore overcounts, sometimes by an order of
magnitude, and anything that loops until it reaches zero never terminates.
"""

from __future__ import annotations

import pytest
import torch

from hologradpy.holography.phase_retrieval import PixelwisePhaseRetriever
from hologradpy.holography.vortices import VortexAnnihilator, VortexDetector
from hologradpy.optics.complex_amplitude import ComplexAmplitude, FieldGeometry
from hologradpy.optics.modules.slm_fields import PixelwiseSLMField
from hologradpy.optics.modules.virtual_slms import VirtualSLM
from hologradpy.optics.systems import SLMFFT
from hologradpy.profiles.amplitude import gaussian_beam_intensity
from hologradpy.profiles.phase import linear_phase

RESOLUTION = (64, 64)
PIXEL_SIZE = (1e-5, 1e-5)
WAVELENGTH = 1e-6
# A small SLM and lens for the annihilator, as in test_first_order_optimizers.py.
SLM_PITCH = 12.5e-6
SLM_WAVELENGTH = 780e-9
FOCAL_LENGTH = 0.25
PADDED_RESOLUTION = (128, 128)
OUTPUT_PITCH = SLM_WAVELENGTH * FOCAL_LENGTH / (PADDED_RESOLUTION[0] * SLM_PITCH)


def _field(data: torch.Tensor) -> ComplexAmplitude:
    geometry = FieldGeometry(
        resolution=RESOLUTION,
        pixel_size=torch.tensor(list(PIXEL_SIZE)),
        wavelength=torch.tensor(WAVELENGTH),
    )
    return ComplexAmplitude.from_geometry(geometry, data=data.to(torch.complex64))


def _grid() -> tuple[torch.Tensor, torch.Tensor]:
    """Index coordinates centred where the detector puts its origin.

    Offset by half a pixel so no sample lands exactly on a zero. The detector tests for
    a sign change with a strict ``product < 0``, which an exact zero fails, so a field
    whose null sits precisely on a pixel is invisible to it.
    """
    rows = torch.arange(RESOLUTION[0], dtype=torch.float32) - RESOLUTION[0] // 2 + 0.5
    columns = (
        torch.arange(RESOLUTION[1], dtype=torch.float32) - RESOLUTION[1] // 2 + 0.5
    )
    return columns[None, :].expand(RESOLUTION), rows[:, None].expand(RESOLUTION)


def _charges(data: torch.Tensor) -> torch.Tensor:
    """The charges the detector finds, over a target that is lit everywhere."""
    detector = VortexDetector(RESOLUTION)
    detector.detect_vortices(
        _field(data), target_intensity=torch.ones(RESOLUTION), threshold=0.2
    )
    if detector.number_of_vortices == 0:
        return torch.zeros(0)
    return detector.charges.reshape(-1)


def test_a_single_vortex_is_found_and_carries_one_charge():
    """``x + iy`` winds once around the origin, which is the textbook charge +1."""
    x, y = _grid()

    charges = _charges(x + 1j * y)

    assert int((charges != 0).sum()) == 1
    assert abs(float(charges[charges != 0][0])) == 1


def test_the_opposite_winding_gets_the_opposite_sign():
    """Conjugating the field reverses the direction the phase turns."""
    x, y = _grid()

    forward = _charges(x + 1j * y)
    reversed_ = _charges(x - 1j * y)

    assert float(forward[forward != 0][0]) == -float(reversed_[reversed_ != 0][0])


def test_an_interference_null_is_a_zero_but_not_a_vortex():
    """The failure this module exists to prevent.

    Two beams crossing give a field that is a real fringe pattern times one overall
    phase, so its real and imaginary parts vanish together along whole lines and the
    detector labels plenty of components. None of them is a vortex: the phase jumps by
    pi across a null rather than winding around a point.
    """
    x, _ = _grid()
    fringes = torch.cos(x * torch.pi / 8).to(torch.complex64)
    interference = fringes * torch.exp(torch.tensor(1j * torch.pi / 4))

    detector = VortexDetector(RESOLUTION)
    detector.detect_vortices(
        _field(interference),
        target_intensity=torch.ones(RESOLUTION),
        threshold=0.2,
    )

    assert detector.number_of_vortices > 0, "expected zero crossings to be labelled"
    assert int((detector.charges.reshape(-1) != 0).sum()) == 0


def test_a_smooth_beam_has_neither():
    x, y = _grid()
    gaussian = torch.exp(-(x**2 + y**2) / (2 * 12.0**2)) + 0j

    charges = _charges(gaussian)

    assert int((charges != 0).sum()) == 0


@pytest.mark.parametrize("offset", [(0, 0), (7, -5)])
def test_a_vortex_is_found_where_it_was_put(offset: tuple[int, int]):
    """Pins the row/column convention of ``center_indices``, which a plot depends on."""
    x, y = _grid()
    shift_x, shift_y = offset

    detector = VortexDetector(RESOLUTION)
    detector.detect_vortices(
        _field((x - shift_x) + 1j * (y - shift_y)),
        target_intensity=torch.ones(RESOLUTION),
        threshold=0.2,
    )

    charged = detector.charges.reshape(-1) != 0
    rows, columns = zip(
        *[
            (int(row), int(column))
            for (row, column), keep in zip(detector.center_indices, charged)
            if keep
        ]
    )
    assert rows[0] == pytest.approx(RESOLUTION[0] // 2 + shift_y, abs=1)
    assert columns[0] == pytest.approx(RESOLUTION[1] // 2 + shift_x, abs=1)


def _slm_geometry() -> FieldGeometry:
    return FieldGeometry(
        resolution=RESOLUTION,
        pixel_size=torch.tensor([SLM_PITCH, SLM_PITCH]),
        wavelength=torch.tensor(SLM_WAVELENGTH),
    )


def _spiral() -> torch.Tensor:
    """A charge-1 spiral on the SLM, which puts a vortex in the focal plane.

    It is tilted by half an output pixel, so the vortex does not land exactly on a
    pixel, where the detector cannot see it.
    """
    x, y = _slm_geometry().get_spatial_grid()
    tilt = linear_phase(
        x,
        y,
        OUTPUT_PITCH / 2,
        OUTPUT_PITCH / 2,
        wavenumber=2 * torch.pi / SLM_WAVELENGTH,
        focal_length=FOCAL_LENGTH,
    )
    return (torch.atan2(y, x) + tilt) % (2 * torch.pi)


def _annihilated(
    beam_phase: torch.Tensor, slm_phase: torch.Tensor, target_scale: float = 1.0
) -> tuple[list[int], torch.Tensor]:
    """One round of annihilation, with two CG iterations after it.

    The model's beam is a Gaussian with ``beam_phase``, and the SLM starts at
    ``slm_phase``. The target is a spot over the vortex, times ``target_scale``.

    Returns:
        The vortex count of each round, and the focal intensity after.
    """
    geometry = _slm_geometry()
    x, y = geometry.get_spatial_grid()
    amplitude = gaussian_beam_intensity(x, y, beam_radius=2e-4).sqrt()
    beam = ComplexAmplitude.from_geometry(
        geometry, data=amplitude * torch.exp(1j * beam_phase)
    )
    model = SLMFFT(
        input_geometry=beam.geometry,
        virtual_slm=VirtualSLM(full_scale_cycles=1.0),
        slm_field=PixelwiseSLMField(beam),
        focal_length=FOCAL_LENGTH,
        padded_resolution=PADDED_RESOLUTION,
    )
    model()
    x_out, y_out = model[-1].get_spatial_grid_output()
    target = gaussian_beam_intensity(x_out, y_out, beam_radius=6 * OUTPUT_PITCH)
    retriever = PixelwisePhaseRetriever(
        model, target_scale * target.to(torch.float32), init_slm_phase=slm_phase
    )
    data = VortexAnnihilator(retriever).annihilate_vortices(
        max_iterations=1, cg_iterations=2, verbose=False
    )
    with torch.no_grad():
        return data.counts, model().intensity.squeeze()


def test_the_beam_phase_does_not_change_the_annihilation():
    """The model applies the beam's own phase, so the annihilator sets the SLM to the
    rest. Two models that differ only in the beam's phase, started from SLM phases that
    give the same focal field, end with the same focal field.
    """
    x, y = _slm_geometry().get_spatial_grid()
    beam_phase = 40 * ((x / 4e-4) ** 2 + (y / 4e-4) ** 2)  # A strong defocus.
    spiral = _spiral()

    flat_counts, flat = _annihilated(torch.zeros_like(x), spiral)
    counts, aberrated = _annihilated(beam_phase, (spiral - beam_phase) % (2 * torch.pi))

    assert flat_counts[0] == counts[0] == 1
    torch.testing.assert_close(aberrated, flat, rtol=0, atol=1e-4 * float(flat.max()))


def test_the_threshold_is_a_fraction_of_the_target_peak():
    """A target in other units, here a millionth of the first, finds the same vortex."""
    beam_phase = torch.zeros(RESOLUTION)

    counts, _ = _annihilated(beam_phase, _spiral())
    small_counts, _ = _annihilated(beam_phase, _spiral(), target_scale=1e-6)

    assert counts[0] == small_counts[0] == 1
