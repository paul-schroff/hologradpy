"""CameraInTheLoopCorrector: gradient steps on the hologram, with the error measured on
the camera and its gradient taken through the model.

The steps run on the 64 by 64 bench of test_model_learning_feedback.py. The model
carries the bench's own aberrated beam, since the steps need a good model. The start
is a short CG retrieval, which leaves the pattern 3 % from the target.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hologradpy.hardware import SimulatedCameraTorch, SimulatedSLMTorch, open_slm
from hologradpy.holography.camera_feedback import (
    CameraFeedbackData,
    CameraInTheLoopCorrector,
)
from hologradpy.optics.modules.slm_fields import PixelwiseSLMField
from hologradpy.optics.modules.virtual_slms import VirtualSLM
from tests.test_model_learning_feedback import (
    CAMERA_PIXEL_SIZE,
    LOSS_SCALE,
    TARGET_POSITION,
    _aberration,
    _beam,
    _geometry,
    _identity_mapping,
    _init_phase,
    _model,
    _target,
)

# Ten times the default, which was chosen on a 1272 by 1024 SLM. On this 64 by 64 one,
# 20 steps at 1e-3 lowered the error from 3.0 to 2.1 %, and 20 steps at 1e-4 to 2.6 %.
LEARNING_RATE = 1e-3
STEPS = 20
START_CG_STEPS = 20


def _bench(camera_shift: tuple[float, float] = (0.0, 0.0)) -> SimpleNamespace:
    """A simulated bench, and a model that carries the bench's own beam."""
    torch.manual_seed(0)
    geometry = _geometry()
    slm = open_slm(SimulatedSLMTorch, input_geometry=geometry, bitdepth=8)
    hardware = _model(
        geometry,
        slm.virtual_slm,
        PixelwiseSLMField(_beam(geometry, _aberration(geometry))),
        camera_shift=camera_shift,
    )
    camera = SimulatedCameraTorch(hardware, bitdepth=12, read_noise=0.0)
    model = _model(
        geometry,
        VirtualSLM.from_slm(slm),
        PixelwiseSLMField(_beam(geometry, _aberration(geometry))),
    )
    patch, region = _target()
    return SimpleNamespace(
        slm=slm, camera=camera, model=model, patch=patch, region=region
    )


def _started(bench: SimpleNamespace) -> CameraInTheLoopCorrector:
    """The corrector, with the model holding a short CG retrieval to start from."""
    corrector = CameraInTheLoopCorrector(
        slm=bench.slm,
        camera=bench.camera,
        slm_camera_model=bench.model,
        target=bench.patch,
        signal_region=bench.region,
        target_position=TARGET_POSITION,
        init_slm_phase=_init_phase(),
        loss_scale=LOSS_SCALE,
        camera_mapping=_identity_mapping(),
    )
    corrector.phase_retriever.retrieve(START_CG_STEPS, method="cg", verbose=False)
    return corrector


def _run(corrector: CameraInTheLoopCorrector, **kwargs) -> CameraFeedbackData:
    kwargs.setdefault("steps", STEPS)
    kwargs.setdefault("learning_rate", LEARNING_RATE)
    kwargs.setdefault("averages", 10)
    return corrector.run(verbose=False, **kwargs)


@pytest.fixture(scope="module")
def citl_run() -> SimpleNamespace:
    """One run, shared by the tests that only read it."""
    bench = _bench()
    corrector = _started(bench)
    model = bench.model
    held_values = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter is not model.virtual_slm.levels
    }
    held_gradients = {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }
    data = _run(corrector)
    return SimpleNamespace(
        bench=bench,
        data=data,
        held_values=held_values,
        held_gradients=held_gradients,
    )


def test_the_steps_lower_the_error(citl_run: SimpleNamespace) -> None:
    start, end = citl_run.data.metrics["rmse"]
    assert end < 0.85 * start


def test_only_the_levels_change(citl_run: SimpleNamespace) -> None:
    """The other parameters of the model keep their values, and every parameter keeps
    its ``requires_grad`` flag.
    """
    for name, parameter in citl_run.bench.model.named_parameters():
        assert parameter.requires_grad == citl_run.held_gradients[name]
        if name in citl_run.held_values:
            assert torch.equal(parameter.detach(), citl_run.held_values[name])


def test_the_slm_shows_the_last_hologram(citl_run: SimpleNamespace) -> None:
    bench = citl_run.bench
    last = citl_run.data.retrievals[-1].phase
    assert np.array_equal(bench.slm.display, bench.slm.phase_to_levels(last))


def test_the_record_holds_the_first_and_the_last_hologram(
    citl_run: SimpleNamespace,
) -> None:
    data = citl_run.data
    assert len(data.measured_images) == 2
    assert len(data.retrievals) == 2
    for values in data.metrics.values():
        assert len(values) == 2
    assert len(data.metadata["step_errors"]) == STEPS
    assert data.metadata["overexposed_steps"] == []


def test_a_displaced_camera_leaves_the_steps_working() -> None:
    """The camera is displaced by 1.5 pixels along x. Every frame is moved onto the
    prediction, and the target onto the frame. The displacement is then not taken for
    an error of the pattern.
    """
    bench = _bench(camera_shift=(1.5 * CAMERA_PIXEL_SIZE, 0.0))
    data = _run(_started(bench))

    start, end = data.metrics["rmse"]
    assert end < 0.85 * start


def test_an_overexposed_step_is_reported() -> None:
    """At the longest exposure the camera accepts, every frame is overexposed."""
    bench = _bench()
    corrector = _started(bench)
    longest = bench.camera.exposure_bounds[1]

    with pytest.warns(UserWarning, match="are overexposed"):
        data = _run(corrector, steps=2, autoexpose=False, exposure=longest)

    assert data.metadata["overexposed_steps"] == [0, 1]
