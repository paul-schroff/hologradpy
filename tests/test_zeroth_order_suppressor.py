"""Holograms changed to cancel the zeroth order the model predicts.

The model carries a known zeroth-order field, a reflection and the light at the wraps,
and the holograms are judged on that model. A spot array designed by CG without the
field leaves its zeroth order. The suppressor's steps cancel it in the disc about the
zeroth order and keep the light of the spots.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from hologradpy.calibration.model_fit import field_on_frames
from hologradpy.grids import plane_center
from hologradpy.holography.phase_retrieval import (
    LinearSuperpositionPhaseRetriever,
    PixelwisePhaseRetriever,
)
from hologradpy.holography.zeroth_order import ZerothOrderSuppressor
from hologradpy.loss_functions import LossAbsoluteIntensityMSE
from hologradpy.optics.complex_amplitude import ComplexAmplitude
from hologradpy.optics.modules import VirtualSLM, ZerothOrderField
from hologradpy.optics.modules.zeroth_order import ZEROTH_ORDER_LAYER
from hologradpy.profiles.amplitude import gaussian_beam_intensity

from .test_zeroth_order_calibrator import (
    CAMERA_PITCH,
    CAMERA_RESOLUTION,
    FOCAL_LENGTH,
    FOCAL_WAIST,
    REFLECTION_PEAK,
    WRAP_WEIGHT,
    _reflection_kernel,
    _system,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

# A 2 x 2 array of spots at 0.7 mm from the zeroth order along each axis.
SPOT_POSITIONS = np.array(
    [(0.7e-3, 0.7e-3), (-0.7e-3, 0.7e-3), (0.7e-3, -0.7e-3), (-0.7e-3, -0.7e-3)]
)
ITERATIONS = 40


def _model():
    model = _system(VirtualSLM(full_scale_cycles=1.0))
    model.build()
    return model


def _zeroth_order_field(model) -> ZerothOrderField:
    """The reflection at REFLECTION_PEAK of a spot's peak, and the light at the wraps
    at the calibrator test's weight.
    """
    field = ZerothOrderField(
        FOCAL_LENGTH, wrap_weight=WRAP_WEIGHT, kernel=_reflection_kernel()
    )
    probe = _model()
    probe.virtual_slm.set_phase(probe.steering_phase(tuple(SPOT_POSITIONS[0])))
    with torch.no_grad():
        spot_peak = float(field_on_frames(probe()).abs().square().max())
    alone = _model()
    reflection = ZerothOrderField(FOCAL_LENGTH, kernel=_reflection_kernel())
    alone.insert_after("virtual_slm", ZEROTH_ORDER_LAYER, reflection)
    incident = alone.init_field
    dark = ComplexAmplitude(
        torch.zeros_like(incident.as_tensor()), incident.wavelength, incident.pixel_size
    )
    with torch.no_grad():
        reflection_peak = float(field_on_frames(alone(dark)).abs().square().max())
        field.kernel.mul_(math.sqrt(REFLECTION_PEAK * spot_peak / reflection_peak))
    return field


def _disc(radius: float, center: tuple[float, float]) -> torch.Tensor:
    """A disc of ``radius`` metres about ``center``, ``(x, y)`` in metres from the
    zeroth order, on the output plane.
    """
    rows, columns = torch.meshgrid(
        torch.arange(CAMERA_RESOLUTION[0], dtype=torch.float64),
        torch.arange(CAMERA_RESOLUTION[1], dtype=torch.float64),
        indexing="ij",
    )
    center_column, center_row = plane_center(CAMERA_RESOLUTION)
    x = (columns - center_column) * CAMERA_PITCH
    y = -(rows - center_row) * CAMERA_PITCH
    return (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2


def _retriever(model) -> PixelwisePhaseRetriever:
    """CG towards the spots with all the incident power in them, as the lab designs
    them, from blazed gratings with random phases.
    """
    x, y = model.focal_plane_positions(*model.fourier_lens.get_spatial_grid_output())
    target = sum(
        gaussian_beam_intensity(x, y, FOCAL_WAIST, None, *position)
        for position in SPOT_POSITIONS
    )
    target = target * (
        model.incident_power() / model.output_pixel_area() / target.sum()
    )
    start = LinearSuperpositionPhaseRetriever(
        model,
        SPOT_POSITIONS,
        target_intensities=[1.0] * len(SPOT_POSITIONS),
        target_phases="random",
        seed=0,
    ).retrieve_phase(verbose=False)
    retriever = PixelwisePhaseRetriever(model, target, init_slm_phase=start)
    retriever.set_loss_factory(LossAbsoluteIntensityMSE)
    return retriever


def _powers(model, field: ZerothOrderField, region, spots) -> tuple[float, float]:
    """The power the model with ``field`` puts in the region and in the spots."""
    if ZEROTH_ORDER_LAYER not in model.layers():
        model.insert_after("virtual_slm", ZEROTH_ORDER_LAYER, field)
    field.update_wraps(model)
    with torch.no_grad():
        intensity = field_on_frames(model())[0].abs().square()
    return float(intensity[region].sum()), float(intensity[spots].sum())


@pytest.fixture(scope="module")
def designs():
    """The powers in the disc and in the spots of the array designed by CG, before and
    after the steps, and the record of the steps.
    """
    region = _disc(4 * FOCAL_WAIST, (0.0, 0.0))
    spots = sum(_disc(2 * FOCAL_WAIST, position) for position in SPOT_POSITIONS) > 0

    model = _model()
    _retriever(model).retrieve_phase(3 * ITERATIONS, verbose=False)
    field = _zeroth_order_field(model)
    before = _powers(model, field, region, spots)
    suppression = ZerothOrderSuppressor(model, field, region).suppress_zeroth_order(
        verbose=False
    )
    after = _powers(model, field, region, spots)
    return before, after, suppression


def test_the_steps_leave_little_of_the_zeroth_order_and_keep_the_spots(
    designs,
) -> None:
    (before_region, before_spots), (after_region, after_spots), _ = designs

    # On this bench the steps left 0.014 % at 0.5 % of the spots' power.
    assert after_region < 2e-3 * before_region
    assert after_spots == pytest.approx(before_spots, rel=0.01)


def test_every_step_is_recorded(designs) -> None:
    (before_region, _), _, suppression = designs

    assert suppression.region_power[0] == pytest.approx(before_region, rel=1e-4)
    assert len(suppression.region_power) == suppression.metadata["steps"] + 1
    assert len(suppression.step_costs) == suppression.metadata["steps"]
    assert suppression.step_costs[-1] < suppression.step_costs[0]


def test_the_model_keeps_the_light_at_the_wraps_of_the_design(designs) -> None:
    _, (after_region, _), suppression = designs

    assert suppression.region_power[-1] == pytest.approx(after_region, rel=1e-4)


def test_another_zeroth_order_field_in_the_model_is_refused() -> None:
    model = _model()
    model.insert_after("virtual_slm", "earlier", ZerothOrderField(FOCAL_LENGTH))

    with pytest.raises(ValueError, match="another name"):
        ZerothOrderSuppressor(
            model, ZerothOrderField(FOCAL_LENGTH), _disc(1e-4, (0.0, 0.0))
        )
