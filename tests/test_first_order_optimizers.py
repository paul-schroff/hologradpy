"""Tests for driving a retrieval with a torch.optim optimizer.

torchmin runs a whole search inside one ``step`` call and line searches within it, so a
first-order method needs its own loop. These pin that the loop records what every other
retrieval records, that the step size reaches the optimizer, and that a search which
runs away cannot write to a parameter it was told to leave alone.
"""

from __future__ import annotations

import pytest
import torch

from hologradpy.holography.phase_retrieval import PixelwisePhaseRetriever
from hologradpy.holography.phase_retrieval.abstract import (
    DEFAULT_LEARNING_RATE,
    FIRST_ORDER_OPTIMIZERS,
)
from hologradpy.optics.complex_amplitude import ComplexAmplitude, FieldGeometry
from hologradpy.optics.modules.slm_fields import PixelwiseSLMField
from hologradpy.optics.modules.virtual_slms import VirtualSLM
from hologradpy.optics.systems import SLMFFT
from hologradpy.profiles.amplitude import gaussian_beam_intensity
from hologradpy.profiles.masks import rectangular_mask

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

RESOLUTION = (64, 64)
PITCH = 12.5e-6
WAVELENGTH = 780e-9
FOCAL_LENGTH = 0.25
ITERATIONS = 6


def _model() -> SLMFFT:
    geometry = FieldGeometry(
        resolution=RESOLUTION,
        pixel_size=torch.tensor([PITCH, PITCH]),
        wavelength=torch.tensor(WAVELENGTH),
    )
    x, y = geometry.get_spatial_grid()
    beam = ComplexAmplitude.from_geometry(
        geometry, data=gaussian_beam_intensity(x, y, beam_radius=2e-4).sqrt() + 0j
    )
    return SLMFFT(
        input_geometry=beam.geometry,
        virtual_slm=VirtualSLM(phase_scaling=1.0),
        slm_field=PixelwiseSLMField(beam),
        focal_length=FOCAL_LENGTH,
        padded_resolution=(128, 128),
    )


def _target(model):
    """A top hat wide enough to span many output pixels.

    The padded transform puts the output pitch at ``wavelength * focal_length /
    (padded * pitch)``, which is 122 um here, so the shape has to be measured in
    millimetres to exist at all.
    """
    model()
    x, y = model[-1].get_spatial_grid_output()
    target = rectangular_mask(x, y, 1.5e-3, 1.5e-3, 0.0, 0.0).to(torch.float32)
    region = rectangular_mask(x, y, 4e-3, 4e-3, 0.0, 0.0)
    assert float(target.sum()) > 0, "the target has to land on the output grid"
    return target, region


@pytest.mark.parametrize("method", sorted(FIRST_ORDER_OPTIMIZERS))
def test_every_first_order_method_runs_and_records(method) -> None:
    """One objective evaluation per iteration, and the record built as usual."""
    model = _model()
    target, region = _target(model)
    retriever = PixelwisePhaseRetriever(model, target, region)

    record = retriever.retrieve(
        ITERATIONS, name=method, method=method, verbose=False
    )

    assert len(record.loss_history) == ITERATIONS
    assert record.name == method
    assert record.metrics["rmse"]
    assert torch.isfinite(model.virtual_slm.get_phase()).all()


def test_the_learning_rate_reaches_the_optimizer() -> None:
    """The step size is the knob a first-order method lives or dies by."""
    model = _model()
    target, region = _target(model)
    retriever = PixelwisePhaseRetriever(model, target, region)

    retriever.retrieve(1, method="adam", learning_rate=0.5, verbose=False)
    assert retriever.optimizer.param_groups[0]["lr"] == 0.5

    retriever.retrieve(1, method="adam", verbose=False)
    assert retriever.optimizer.param_groups[0]["lr"] == DEFAULT_LEARNING_RATE


def test_a_first_order_search_is_handed_only_what_it_varies() -> None:
    """This path is handed the one parameter it varies.

    torchmin is given every parameter instead, held back by ``requires_grad``, which
    stops holding once a step goes non-finite.
    """
    model = _model()
    target, region = _target(model)
    retriever = PixelwisePhaseRetriever(model, target, region)

    retriever.retrieve(1, method="adam", verbose=False)

    handed = retriever.optimizer.param_groups[0]["params"]
    assert len(handed) == 1
    assert handed[0] is model.virtual_slm.levels


def test_a_first_order_search_leaves_the_other_parameters_where_they_were() -> None:
    """The camera geometry is not what a phase retrieval is allowed to move."""
    model = _model()
    target, region = _target(model)
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name != "virtual_slm.levels"
    }

    PixelwisePhaseRetriever(model, target, region).retrieve(
        ITERATIONS, method="adam", learning_rate=0.1, verbose=False
    )

    for name, parameter in model.named_parameters():
        if name in before:
            assert torch.equal(parameter, before[name]), name


def test_the_cost_comes_down() -> None:
    """A search that records a history but never descends would pass the rest."""
    model = _model()
    target, region = _target(model)

    record = PixelwisePhaseRetriever(model, target, region).retrieve(
        40, method="adam", verbose=False
    )

    assert record.loss_history[-1] < record.loss_history[0]
