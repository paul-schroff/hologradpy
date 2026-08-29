"""The optimal-transport retriever, and the physics that turns a map into a phase.

The physics tests propagate the phase and measure what arrives, rather than comparing
phases, because a phase is only defined up to a constant and up to how it wraps. Two
measurements pin the map between them: the radius of a Gaussian target fixes the
quadratic part, and the position of a shifted target fixes the linear part. Both are
needed: a sign error inverts the image, so it reaches the right size and the wrong
position, while a spurious lens reaches the right position and the wrong size.

The contract tests pin that the phase reaches the model, so ``retrieve()`` reports the
hologram that was just found rather than the one that was there before.
"""

from __future__ import annotations

import math

import pytest
import torch

from hologradpy.fourier_optics import get_focal_spot_radius
from hologradpy.holography.phase_retrieval import (
    OptimalTransportPhaseRetriever,
    optimal_transport,
)
from hologradpy.optics.complex_amplitude import ComplexAmplitude, FieldGeometry
from hologradpy.optics.modules.slm_fields import PixelwiseSLMField
from hologradpy.optics.modules.virtual_slms import VirtualSLM
from hologradpy.optics.systems import SLMCZT
from hologradpy.profiles.amplitude import gaussian_beam_intensity

WAVELENGTH = 1039e-9
FOCAL_LENGTH = 0.3
BEAM_RADIUS = 2.0e-3
SLM_RESOLUTION = (128, 128)
SLM_PIXEL_SIZE = (50e-6, 50e-6)
CAMERA_RESOLUTION = (64, 64)
CAMERA_PIXEL_SIZE = (14e-6, 14e-6)
# Measured: the transport keeps improving as this falls until the kernel underflows,
# which on a photograph is just below 1e-3.
REGULARIZATION = 1e-3


def _model() -> SLMCZT:
    """A small model.

    The camera pitch is set independently of the SLM's, so the lattice the transport
    needs cannot be the model's own grid and has to be built.
    """
    geometry = FieldGeometry(
        resolution=SLM_RESOLUTION,
        pixel_size=torch.tensor(list(SLM_PIXEL_SIZE)),
        wavelength=torch.tensor(WAVELENGTH),
    )
    grid_x, grid_y = geometry.get_spatial_grid()
    beam = ComplexAmplitude.from_geometry(
        geometry,
        data=gaussian_beam_intensity(grid_x, grid_y, beam_radius=BEAM_RADIUS).sqrt()
        + 0j,
    )
    model = SLMCZT(
        input_geometry=geometry,
        virtual_slm=VirtualSLM(phase_scaling=1.0),
        slm_field=PixelwiseSLMField(beam),
        camera_resolution=CAMERA_RESOLUTION,
        camera_pixel_size=CAMERA_PIXEL_SIZE,
        focal_length=FOCAL_LENGTH,
        padded_resolution=(256, 256),
    )
    model()
    return model


def _gaussian_target(
    model: SLMCZT, radius: float, shift: tuple[float, float] = (0.0, 0.0)
) -> torch.Tensor:
    camera_x, camera_y = model[-1].get_spatial_grid_output()
    return gaussian_beam_intensity(
        camera_x, camera_y, beam_radius=radius, shift_x=shift[0], shift_y=shift[1]
    ).to(torch.float32)


def _retriever(model, target, **options) -> OptimalTransportPhaseRetriever:
    return OptimalTransportPhaseRetriever(
        model, target=target, regularization=REGULARIZATION, **options
    )


def _achieved_radius(model: SLMCZT) -> float:
    """The 1/e^2 radius of the focal spot the SLM is currently producing."""
    with torch.no_grad():
        image = model().intensity.squeeze().detach()
    axis = (
        torch.arange(image.shape[0], dtype=torch.float64) - image.shape[0] // 2
    ) * CAMERA_PIXEL_SIZE[0]
    profile = image.sum(dim=1).to(torch.float64)
    profile = (profile - profile.min()).clamp(min=0)
    centre = (profile * axis).sum() / profile.sum()
    variance = (profile * (axis - centre) ** 2).sum() / profile.sum()
    return float(2 * variance.sqrt())


# --- physics ------------------------------------------------------------------------


def test_a_gaussian_target_is_reached_at_the_radius_asked_for():
    """The decisive test: a wrong mapping convention misses this by a factor.

    The transport map is ray optics, so what arrives is the ray radius and the
    diffraction-limited spot added in quadrature. Here the limit is comparable to the
    target, so it dominates, and asserting the ray radius alone would be asserting
    something the physics does not offer. The same quadrature is why
    :func:`~hologradpy.fourier_optics.beam_shaping_focal_length` takes a wavelength.
    """
    model = _model()
    wanted = 60e-6
    target = _gaussian_target(model, wanted)

    _retriever(model, target).retrieve_phase()

    focal_spot = get_focal_spot_radius(BEAM_RADIUS, WAVELENGTH, FOCAL_LENGTH)
    assert _achieved_radius(model) == pytest.approx(
        math.hypot(wanted, focal_spot), rel=0.2
    )


def test_a_shifted_target_moves_the_spot_there():
    """Pins the linear part of the map, which the quadratic tests cannot see."""
    model = _model()
    shift = (120e-6, -80e-6)
    target = _gaussian_target(model, 60e-6, shift=shift)

    _retriever(model, target).retrieve_phase()

    with torch.no_grad():
        image = model().intensity.squeeze().detach().to(torch.float64)
    camera_x, camera_y = model[-1].get_spatial_grid_output()
    centre_x = float((image * camera_x).sum() / image.sum())
    centre_y = float((image * camera_y).sum() / image.sum())

    tolerance = 0.25 * math.hypot(*shift)
    assert math.hypot(centre_x - shift[0], centre_y - shift[1]) < tolerance


# --- the contract with PhaseRetrieverBase -------------------------------------------


def test_the_phase_reaches_the_model():
    """A computed phase must be left on the model.

    Otherwise retrieve() reports the hologram that was already there rather than the
    one just found.
    """
    model = _model()
    target = _gaussian_target(model, 60e-6)
    before = model.virtual_slm.get_phase().clone()

    returned = _retriever(model, target).retrieve_phase()

    assert not torch.allclose(model.virtual_slm.get_phase(), before)
    torch.testing.assert_close(returned, model.virtual_slm.get_phase())


def test_retrieve_produces_a_record():
    """A record is still worth having from a retriever that scores nothing.

    The transport aims at the whole target, so there is no region to measure over and
    the record carries the phase and the timing rather than metrics.
    """
    model = _model()
    target = _gaussian_target(model, 60e-6)

    record = _retriever(model, target).retrieve(200, name="transport")

    assert record.name == "transport"
    assert record.phase.shape == tuple(model.virtual_slm.slm_resolution)
    assert record.metrics == {}
    assert record.visualization_data is not None


def test_the_target_is_set_on_its_own():
    """The signature says the transport aims at a whole target."""
    model = _model()
    target = _gaussian_target(model, 60e-6)
    retriever = _retriever(model, target)

    retriever.set_target(_gaussian_target(model, 80e-6))

    assert retriever.signal_region is None
    with pytest.raises(TypeError):
        retriever.set_target(target, torch.ones_like(target, dtype=torch.bool))


def test_the_phase_has_the_slm_resolution():
    model = _model()
    target = _gaussian_target(model, 60e-6)

    phase = _retriever(model, target).retrieve_phase()

    assert phase.shape == tuple(model.virtual_slm.slm_resolution)


def test_an_explicit_source_intensity_is_used():
    """Callers with a measured beam should not be forced through the model's copy."""
    model = _model()
    target = _gaussian_target(model, 60e-6)
    grid_x, grid_y = model.virtual_slm.get_slm_grid()
    narrow = gaussian_beam_intensity(grid_x, grid_y, beam_radius=BEAM_RADIUS / 2)

    default_phase = _retriever(model, target).retrieve_phase().clone()
    narrow_phase = _retriever(
        model, target, source_intensity=narrow
    ).retrieve_phase()

    assert not torch.allclose(default_phase, narrow_phase)


# --- the separable solver, against the plan it never builds -------------------------


def test_the_separable_map_matches_the_plan_it_never_forms():
    """The one test that would catch the two axes being swapped.

    The solver reads the map off products of two scalings and two one-dimensional
    kernels, never holding the plan. That plan does exist, and on grids small enough to
    build it the map has to agree with what it says. A weight applied on the wrong side
    of the scaling still gives a plausible-looking map, so nothing short of forming the
    plan pins which target axis each component belongs to.

    All four lengths differ, so a transpose that happens to work on a square grid is
    caught here too.
    """
    torch.manual_seed(0)
    epsilon = 5e-2
    source = torch.rand(5, 7, dtype=torch.float64) + 0.1
    target = torch.rand(4, 6, dtype=torch.float64) + 0.1

    source_y = torch.linspace(-1.0, 1.0, 5, dtype=torch.float64)
    source_x = torch.linspace(-1.0, 1.0, 7, dtype=torch.float64)
    target_y = torch.linspace(-0.5, 0.5, 4, dtype=torch.float64)
    target_x = torch.linspace(-0.5, 0.5, 6, dtype=torch.float64)

    kernel_y = optimal_transport._kernel(source_y, target_y, epsilon)
    kernel_x = optimal_transport._kernel(source_x, target_x, epsilon)
    assert kernel_y.shape == (5, 4) and kernel_x.shape == (7, 6)

    map_y, map_x = optimal_transport._separable_map(
        source, target, kernel_y, kernel_x, target_y, target_x, 60
    )

    beam = source / source.sum()
    scaling, other = _scalings(beam, target / target.sum(), kernel_y, kernel_x, 60)
    # plan[j, k, L, M] = scaling[j, k] kernel_y[j, L] kernel_x[k, M] other[L, M]
    plan = torch.einsum("jk,jl,km,lm->jklm", scaling, kernel_y, kernel_x, other)

    # The first moment over each target axis. Compared before dividing by the beam,
    # which is what the solver divides by and is only the plan's marginal once the
    # iteration has converged.
    torch.testing.assert_close(map_y * beam, torch.einsum("jklm,l->jk", plan, target_y))
    torch.testing.assert_close(map_x * beam, torch.einsum("jklm,m->jk", plan, target_x))


def _scalings(beam, wanted, kernel_y, kernel_x, iterations):
    """The pair of scalings after the same sweeps the solver runs."""
    scaling = torch.full_like(beam, 1 / beam.numel())
    other = torch.full_like(wanted, 1 / wanted.numel())
    for _ in range(iterations):
        scaling = beam / (kernel_y @ other @ kernel_x.T)
        scaling = scaling / scaling.max()
        other = wanted / (kernel_y.T @ scaling @ kernel_x)
    return scaling, other


# --- rectangular geometry -----------------------------------------------------------


def _rectangular_model() -> SLMCZT:
    """A model whose two planes agree on neither shape nor sample count."""
    geometry = FieldGeometry(
        resolution=(96, 128),
        pixel_size=torch.tensor([50e-6, 50e-6]),
        wavelength=torch.tensor(WAVELENGTH),
    )
    grid_x, grid_y = geometry.get_spatial_grid()
    beam = ComplexAmplitude.from_geometry(
        geometry,
        data=gaussian_beam_intensity(grid_x, grid_y, beam_radius=1.2e-3).sqrt() + 0j,
    )
    model = SLMCZT(
        input_geometry=geometry,
        virtual_slm=VirtualSLM(phase_scaling=1.0),
        slm_field=PixelwiseSLMField(beam),
        camera_resolution=(48, 80),
        camera_pixel_size=(14e-6, 11e-6),
        focal_length=FOCAL_LENGTH,
        padded_resolution=(256, 256),
    )
    model()
    return model


def test_a_rectangular_model_reaches_the_radius_asked_for():
    """Neither plane is square and they hold different numbers of samples.

    The kernel is one rectangular matrix per axis, so each plane keeps its own shape.
    Squaring either of them, or sharing one grid between them, would shear the map and
    miss this radius.
    """
    model = _rectangular_model()
    wanted = 60e-6
    target = _gaussian_target(model, wanted)

    _retriever(model, target).retrieve_phase()

    focal_spot = get_focal_spot_radius(1.2e-3, WAVELENGTH, FOCAL_LENGTH)
    with torch.no_grad():
        image = model().intensity.squeeze().detach()
    axis = (
        torch.arange(image.shape[0], dtype=torch.float64) - image.shape[0] // 2
    ) * 14e-6
    profile = (image.sum(dim=1).to(torch.float64) - image.sum(dim=1).min()).clamp(min=0)
    centre = (profile * axis).sum() / profile.sum()
    radius = 2 * ((profile * (axis - centre) ** 2).sum() / profile.sum()).sqrt()

    assert float(radius) == pytest.approx(math.hypot(wanted, focal_spot), rel=0.3)


# --- progress reporting -------------------------------------------------------------


class _CountingBar:
    """Stands in for a :class:`~hologradpy.utils.ProgressBar`, counting what it gets."""

    def __init__(self) -> None:
        self.resets: list[int | None] = []
        self.steps = 0

    def reset(self, total: int | None = None) -> None:
        self.resets.append(total)

    def update(self, steps: int = 1, **postfix: object) -> None:
        self.steps += steps


def test_a_borrowed_progress_bar_advances_once_per_sinkhorn_sweep():
    """A caller's bar has to move.

    The retriever used to accept one and quietly ignore it, so anything driving it saw
    nothing happen through the longest part of the run.
    """
    model = _model()
    target = _gaussian_target(model, 60e-6)
    bar = _CountingBar()

    _retriever(model, target).retrieve_phase(12, progress_bar=bar)

    assert bar.resets == [12]
    assert bar.steps == 12


def test_a_silent_retrieval_still_runs():
    model = _model()
    target = _gaussian_target(model, 60e-6)

    phase = _retriever(model, target).retrieve_phase(8, verbose=False)

    assert phase.shape == tuple(model.virtual_slm.slm_resolution)
