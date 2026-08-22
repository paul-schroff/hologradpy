"""Camera feedback: the closed loop, the retargeting contract it rests on, and the
arithmetic of the target update.

The loop is run against a deliberate model mismatch, a simulated bench whose beam
carries an aberration the retriever's model does not have. That mismatch is the thing
feedback exists to absorb, so a test where the two models agree would pass while proving
nothing.
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from hologradpy.calibration.camera_mapping import (
    CameraMapping,
    FocalSpotFit,
)
from hologradpy.datasets import CaptureStore, RetrievalStepStore
from hologradpy.grids import get_spatial_grid
from hologradpy.analysis.error_metrics import (
    DEFAULT_INTENSITY_METRICS,
    IntensityMetric,
    efficiency,
    normalize,
)
from hologradpy.hardware import (
    SimulatedCameraTorch,
    SimulatedSLMTorch,
    open_slm,
)
from hologradpy.holography.camera_feedback import (
    CameraFeedbackData,
    CameraFeedbackVisualizer,
    SimpleFeedbackCorrector,
)
from hologradpy.holography.phase_retrieval import (
    MODEL_CHECKPOINT_NAME,
    RETRIEVAL_STEPS_NAME,
    GradientPhaseRetriever,
    PhaseRetrievalData,
    LinearSuperpositionPhaseRetriever,
    RetrievalStepWriter,
    ZernikePhaseRetriever,
)
from hologradpy.optics.complex_amplitude import (
    ComplexAmplitude,
    FieldGeometry,
)
from hologradpy.optics.modules.hardware_models import (
    BackgroundScatter,
    PointingInstability,
)
from hologradpy.optics.modules.slm_fields import PixelwiseSLMField
from hologradpy.optics.modules.virtual_slms import VirtualSLM
from hologradpy.optics.systems import SLMCZT
from hologradpy.profiles.amplitude import gaussian_beam_intensity
from hologradpy.profiles.amplitude import super_gaussian
from hologradpy.profiles.masks import rectangular_mask
from hologradpy.profiles.phase import linear_phase
from hologradpy.roi import ROI
from hologradpy.utils import gpu_to_numpy


pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

SLM_RESOLUTION = (64, 64)
SLM_PIXEL_SIZE = 12.5e-6
CAMERA_RESOLUTION = (64, 64)
CAMERA_PIXEL_SIZE = 20e-6
WAVELENGTH = 0.670e-6
FOCAL_LENGTH = 0.1
# The default is sized for a megapixel camera, where the unit-sum normalization leaves
# the cost far smaller than it is on this 64 by 64 one. At the default the line search
# overshoots into nan on the first step.
LOSS_SCALE = 1e8


def _geometry() -> FieldGeometry:
    return FieldGeometry(
        resolution=SLM_RESOLUTION,
        pixel_size=torch.tensor([SLM_PIXEL_SIZE, SLM_PIXEL_SIZE]),
        wavelength=torch.tensor(WAVELENGTH),
    )


def _beam(
    geometry: FieldGeometry, phase: torch.Tensor | None = None
) -> ComplexAmplitude:
    amplitude = gaussian_beam_intensity(
        *geometry.get_spatial_grid(), beam_radius=0.4e-3
    ).sqrt()
    if phase is None:
        field = amplitude + 0j
    else:
        field = amplitude * torch.exp(1j * phase)
    return ComplexAmplitude(
        field,
        wavelength=geometry.wavelength,
        pixel_size=geometry.pixel_size,
        power=1e-3,
    )


def _aberration(geometry: FieldGeometry) -> torch.Tensor:
    """A smooth low-order wavefront error, of the size a real bench carries.

    Built by hand rather than from :class:`Zernike` so the test states exactly what the
    retriever's model is missing.
    """
    grid_x, grid_y = geometry.get_spatial_grid()
    normalized_x = grid_x / grid_x.abs().max()
    normalized_y = grid_y / grid_y.abs().max()
    return 1.2 * (normalized_x**2 - normalized_y**2) + 0.8 * normalized_x * normalized_y


def _model(
    geometry: FieldGeometry,
    virtual_slm: VirtualSLM,
    field: ComplexAmplitude,
) -> SLMCZT:
    model = SLMCZT(
        input_geometry=geometry,
        virtual_slm=virtual_slm,
        camera_resolution=CAMERA_RESOLUTION,
        camera_pixel_size=(CAMERA_PIXEL_SIZE, CAMERA_PIXEL_SIZE),
        focal_length=FOCAL_LENGTH,
        slm_field=PixelwiseSLMField(field),
    )
    # Sizes the output grid, which the target is built on.
    model()
    return model


# A patch smaller than the sensor, placed off axis. The offset matters physically: the
# undiffracted zeroth order sits at the center of the sensor and cannot be controlled by
# the hologram, so leaving it inside the signal region would have the loop trying to
# correct light it has no say over, and would pollute every metric with it.
PATCH_RESOLUTION = (24, 24)
TARGET_POSITION = (250e-6, 0.0)


def _target(model: SLMCZT = None) -> tuple[torch.Tensor, torch.Tensor]:
    """The target patch and its signal region, on their own grid at the camera pitch.

    A flat top with a soft edge rather than a rectangle. A finite aperture cannot
    make an infinitely sharp edge, so a rectangle asks the loop for something
    unreachable, and it is not band limited: placing it between samples rings, which
    put 24% overshoot on the target being chased. The order sets the softness, and
    this one rolls off over about two camera pixels, its half maximum still at the
    rectangle's own edge.
    """
    grid = get_spatial_grid(PATCH_RESOLUTION, (CAMERA_PIXEL_SIZE, CAMERA_PIXEL_SIZE))
    target = super_gaussian(*grid, 0.0, 0.0, 6, 6, 90e-6, 90e-6).float()
    signal_region = rectangular_mask(*grid, 300e-6, 300e-6, 0.0, 0.0)
    return target, signal_region


def _init_phase() -> torch.Tensor:
    """A random start plus the grating that steers to the target.

    Random, because a flat phase would put every photon in one central pixel, where the
    normalized cost has almost nothing to descend. Steered, because the target sits away
    from the zeroth order and a start with no light near it converges badly. The
    corrector does not add this itself, so that a caller writing it here does not get it
    twice.
    """
    generator = torch.Generator().manual_seed(1)
    speckle = 2 * torch.pi * torch.rand(SLM_RESOLUTION, generator=generator)

    geometry = _geometry()
    grating = linear_phase(
        *geometry.get_spatial_grid(),
        tilt_x=TARGET_POSITION[0],
        tilt_y=TARGET_POSITION[1],
        wavenumber=geometry.wavenumber.reshape(()),
        focal_length=FOCAL_LENGTH,
    )
    return speckle + grating.to(speckle.dtype)


def _placed(
    target: torch.Tensor, signal_region: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The patch pasted onto a sensor-sized frame at TARGET_POSITION.

    What the corrector does in place_target, done here so the tests that drive a bare
    retriever have a full-frame target to work with.
    """
    frame_target = torch.zeros(CAMERA_RESOLUTION, dtype=target.dtype)
    frame_region = torch.zeros(CAMERA_RESOLUTION, dtype=signal_region.dtype)

    offset_column = int(round(TARGET_POSITION[0] / CAMERA_PIXEL_SIZE))
    offset_row = int(round(TARGET_POSITION[1] / CAMERA_PIXEL_SIZE))
    top = CAMERA_RESOLUTION[0] // 2 + offset_row - PATCH_RESOLUTION[0] // 2
    left = CAMERA_RESOLUTION[1] // 2 + offset_column - PATCH_RESOLUTION[1] // 2

    frame_target[top : top + PATCH_RESOLUTION[0], left : left + PATCH_RESOLUTION[1]] = (
        target
    )
    frame_region[top : top + PATCH_RESOLUTION[0], left : left + PATCH_RESOLUTION[1]] = (
        signal_region
    )
    return frame_target, frame_region


def _bench() -> tuple:
    """A simulated bench, and a retriever whose model does not know its aberration."""
    torch.manual_seed(0)
    geometry = _geometry()
    slm = open_slm(SimulatedSLMTorch, input_geometry=geometry, bitdepth=8)

    hardware_model = _model(
        geometry, slm.virtual_slm, _beam(geometry, _aberration(geometry))
    )
    camera = SimulatedCameraTorch(hardware_model, bitdepth=12, noise_level=0.0)

    model = _model(geometry, VirtualSLM.from_slm(slm), _beam(geometry))
    target, signal_region = _target(model)

    # Targeted on the full frame so the retriever can be exercised on its own. The
    # corrector replaces this with the placed patch in place_target().
    placed_target, placed_region = _placed(target, signal_region)
    retriever = GradientPhaseRetriever(
        slm_camera_model=model,
        target=placed_target,
        signal_region=placed_region,
        init_slm_phase=_init_phase(),
        loss_scale=LOSS_SCALE,
    )

    # Exposed on a hologram of the shape the loop will work with, as a bench would be,
    # and with headroom: a railed frame carries no discrepancy to feed back, and the
    # potential only gets brighter as the feedback sharpens it.
    slm.set_phase(gpu_to_numpy(retriever.retrieve_phase(15, method="cg")))
    camera.autoexpose(set_fraction=0.35, raise_on_rail=False)
    return slm, camera, retriever, target, signal_region


def _identity_mapping(resolution=CAMERA_RESOLUTION) -> CameraMapping:
    """A mapping that says the camera and the model already agree.

    Passed throughout so register() has nothing to measure: a real spot-array mapping
    needs resolvable spots, which this 64 by 64 bench does not have. The point pairs are
    identical, so the fitted transform is the identity and calibrate_from_mapping is a
    no-op.
    """
    points = [(10.0, 12.0), (40.0, 15.0), (20.0, 45.0), (50.0, 50.0)]
    return CameraMapping(
        timestamp=datetime(2026, 1, 1),
        name="identity",
        transform=np.eye(3),
        detected_points=points,
        calculated_points=points,
        zeroth_order_position=(resolution[0] / 2, resolution[1] / 2),
        spot_fit=FocalSpotFit(waist=2.0),
    )


def _feedback(**kwargs) -> SimpleFeedbackCorrector:
    slm, camera, retriever, target, signal_region = _bench()
    kwargs.setdefault("camera_mapping", _identity_mapping())
    kwargs.setdefault("target_position", TARGET_POSITION)
    return SimpleFeedbackCorrector(
        phase_retriever=retriever,
        camera=camera,
        slm=slm,
        target=target,
        signal_region=signal_region,
        **kwargs,
    )


# --- The closed loop ---------------------------------------------------------------


ITERATIONS = 8


@pytest.fixture(scope="module")
def feedback_run() -> CameraFeedbackData:
    """One run of the loop, shared by the tests that only read its record.

    The corrector builds its own retriever here, which is both the common path and the
    one that adds the grating steering to the off-axis target.
    """
    slm, camera, retriever, patch, region = _bench()
    feedback = SimpleFeedbackCorrector(
        slm=slm,
        camera=camera,
        slm_camera_model=retriever.slm_camera_model,
        target=patch,
        signal_region=region,
        target_position=TARGET_POSITION,
        init_slm_phase=_init_phase(),
        loss_scale=LOSS_SCALE,
        camera_mapping=_identity_mapping(),
    )
    # Gain below one because a short optimization cannot deliver the whole correction
    # in one go, and asking for it makes the loop oscillate rather than settle. The
    # off-axis target needs a longer search than a centered one to get anywhere.
    return feedback.run(
        retriever_iterations=[40] * ITERATIONS,
        gain=0.7,
        averages=1,
        retrieve_options={"method": "cg"},
        verbose=False,
    )


def test_feedback_reduces_the_error(feedback_run: CameraFeedbackData) -> None:
    """The whole point: the potential the aberrated bench delivers gets closer to the
    target the model alone could not reach.

    The rms is what the loop drives and falls all the way down. The psnr is compared at
    its best rather than at the end, because a gain this loop cannot lower mid-run
    overshoots: it peaks partway and comes back, which is the whole reason the
    visualizer marks a best iteration rather than the last one.
    """
    rms_history = feedback_run.metrics["rmse"]
    psnr_history = feedback_run.metrics["psnr [dB]"]
    assert rms_history[-1] < 0.6 * rms_history[0]
    assert max(psnr_history) > psnr_history[0]


def test_records_every_iteration(feedback_run: CameraFeedbackData) -> None:
    assert feedback_run.number_of_iterations == ITERATIONS
    for series in (
        feedback_run.measured_images,
        feedback_run.corrected_targets,
        feedback_run.retrievals,
        *feedback_run.metrics.values(),
    ):
        assert len(series) == ITERATIONS

    assert feedback_run.retrievals[0].phase.shape == SLM_RESOLUTION
    assert feedback_run.final_camera_image.shape == CAMERA_RESOLUTION


def test_per_iteration_frames_are_cropped_to_the_region(
    feedback_run: CameraFeedbackData,
) -> None:
    """Only the region is kept per iteration, since that is all the loop measures.

    One whole-sensor frame is kept, and putting a cropped one back is exact for the
    corrected targets because the update is gated by the region.
    """
    cropped = feedback_run.signal_roi.crop(feedback_run.signal_region)
    assert cropped.shape != CAMERA_RESOLUTION
    assert feedback_run.measured_images[0].shape == cropped.shape
    assert feedback_run.corrected_targets[0].shape == cropped.shape

    restored = feedback_run.full_corrected_target(0)
    assert restored.shape == CAMERA_RESOLUTION
    assert np.allclose(restored, np.where(feedback_run.signal_region, restored, 0.0))


def test_corrected_target_compensates_the_measurement(
    feedback_run: CameraFeedbackData,
) -> None:
    """Where the bench under-delivers, the served target is raised, and vice versa.

    This is the sign of the discrepancy, the defect a short run's rms test would not
    reliably catch.
    """
    roi = feedback_run.signal_roi
    inside = roi.crop(feedback_run.signal_region).astype(bool)
    measured = normalize(
        feedback_run.full_measured_image(0), feedback_run.signal_region
    )
    shortfall = roi.crop(feedback_run.target) - roi.crop(measured)
    correction = (
        feedback_run.corrected_targets[1] - feedback_run.corrected_targets[0]
    )

    # Correlated rather than equal: the update is clamped at zero and gated.
    correlation = np.corrcoef(shortfall[inside], correction[inside])[0, 1]
    assert correlation > 0.5


def test_first_corrected_target_is_the_original(
    feedback_run: CameraFeedbackData,
) -> None:
    """Nothing to correct before the first measurement, so the retriever gets the
    target as given (normalized over the signal region).
    """
    original = feedback_run.target
    served = feedback_run.full_corrected_target(0)
    assert np.allclose(served, np.where(feedback_run.signal_region, original, 0.0))


def test_run_leaves_the_retriever_on_the_corrected_target() -> None:
    feedback = _feedback()
    feedback.run(retriever_iterations=[5] * 2, averages=1, verbose=False)

    assert feedback.corrected_target is not feedback.target
    assert torch.equal(
        feedback.phase_retriever.target,
        torch.as_tensor(
            feedback.corrected_target,
            dtype=feedback.target.dtype,
            device=feedback.target.device,
        ),
    )


def test_run_returns_a_saveable_record(feedback_run: CameraFeedbackData) -> None:
    assert isinstance(feedback_run, CameraFeedbackData)
    assert feedback_run.retrievals[-1].phase.shape == SLM_RESOLUTION
    assert len(feedback_run.retrievals) == ITERATIONS
    assert all(
        isinstance(retrieval, PhaseRetrievalData)
        for retrieval in feedback_run.retrievals
    )
    # The corrected target has drifted away from the wanted one, which is the whole
    # point: the difference is the compensation for what the model does not know.
    assert len(feedback_run.corrected_targets) == ITERATIONS
    assert not np.allclose(
        feedback_run.corrected_targets[-1],
        feedback_run.signal_roi.crop(feedback_run.target),
    )


def test_record_survives_a_round_trip(
    feedback_run: CameraFeedbackData, tmp_path
) -> None:
    """A save keeps every measurement, since those are the part of a run that cannot be
    recreated without the bench, and the reloaded record still draws.
    """
    path = tmp_path / "feedback.asdf"
    feedback_run.save(path)
    reloaded = CameraFeedbackData.load(path)

    assert np.allclose(
        reloaded.retrievals[-1].phase, feedback_run.retrievals[-1].phase
    )
    assert len(reloaded.corrected_targets) == len(feedback_run.corrected_targets)
    assert np.allclose(
        reloaded.corrected_targets[-1], feedback_run.corrected_targets[-1]
    )
    assert len(reloaded.measured_images) == ITERATIONS
    assert np.allclose(reloaded.measured_images[-1], feedback_run.measured_images[-1])
    assert np.allclose(reloaded.final_camera_image, feedback_run.final_camera_image)
    assert reloaded.metrics == feedback_run.metrics
    assert reloaded.lower_is_better == feedback_run.lower_is_better
    assert len(reloaded.retrievals) == len(feedback_run.retrievals)

    figure = reloaded.visualizer().render()
    assert figure is not None
    plt.close(figure)


def test_a_run_stores_its_target_once(feedback_run: CameraFeedbackData) -> None:
    """The target and the signal region belong to the run, which holds them once. A
    nested retrieval carries only the phase it produced.
    """
    assert feedback_run.retrievals
    for retrieval in feedback_run.retrievals:
        assert retrieval.target is None
        assert retrieval.signal_region is None
        assert retrieval.phase is not None


def test_no_dataset_without_a_path(tmp_path) -> None:
    """The cropped series on the record is the only copy unless a dataset is asked
    for.
    """
    feedback = _feedback()
    feedback.run(retriever_iterations=[5] * 2, averages=1, verbose=False)

    assert list(tmp_path.iterdir()) == []


def test_a_run_writes_a_dataset_when_asked(tmp_path) -> None:
    """A feedback run is a set of (camera image, phase pattern) pairs, the same shape a
    speckle capture produces, so it reads back as one.
    """
    path = tmp_path / "run.asdf"
    feedback = _feedback()
    data = feedback.run(
        retriever_iterations=[5] * 3,
        averages=1,
        verbose=False,
        dataset_path=path,
    )

    with CaptureStore.open(path) as store:
        assert len(store) == 3
        sample = store.read(0)
        # Full frames, unlike the cropped series the record keeps for plotting.
        assert sample["camera_image"].shape == CAMERA_RESOLUTION
        assert sample["slm_levels"].shape == SLM_RESOLUTION

        # Patterns are the levels the SLM displayed, so a quarter of the width and
        # exactly what it held.
        assert store.phase_bitdepth == feedback.slm.bitdepth
        assert sample["slm_levels"].dtype == np.uint8
        assert np.array_equal(
            sample["slm_levels"],
            feedback.slm.phase_to_levels(data.retrievals[0].phase),
        )
        assert np.array_equal(
            store.read(2)["slm_levels"],
            feedback.slm.phase_to_levels(data.retrievals[-1].phase),
        )


def test_a_written_dataset_describes_itself(tmp_path) -> None:
    """The record goes inside the file, so one path is the whole dataset and moving it
    changes nothing.
    """
    path = tmp_path / "run.asdf"
    feedback = _feedback()
    data = feedback.run(
        retriever_iterations=[5] * 2,
        averages=1,
        verbose=False,
        dataset_path=path,
    )

    moved = tmp_path / "moved.asdf"
    path.rename(moved)

    with CaptureStore.open(moved) as store:
        assert len(store) == 2
        assert isinstance(store.record(), CameraFeedbackData)
        assert np.allclose(
            store.record().retrievals[-1].phase, data.retrievals[-1].phase
        )


def test_zeroth_order_is_outside_the_signal_region() -> None:
    """The undiffracted spot must not be measured.

    It sits wherever the optics put it and no hologram can move or remove it, so a
    signal region containing it has the loop chasing light it has no say over, and every
    metric carries that spot's brightness rather than the potential's.
    """
    feedback = _feedback()
    placement = feedback.placement_data()

    zeroth_row, zeroth_column = placement.zeroth_order
    region = np.asarray(placement.signal_region, dtype=bool)
    assert not region[int(round(zeroth_row)), int(round(zeroth_column))]

    # And not merely off by a pixel: a few pixels of margin either side.
    rows = np.flatnonzero(region.any(axis=1))
    columns = np.flatnonzero(region.any(axis=0))
    outside = (
        zeroth_row < rows[0] - 2
        or zeroth_row > rows[-1] + 2
        or zeroth_column < columns[0] - 2
        or zeroth_column > columns[-1] + 2
    )
    assert outside, (
        f"the zeroth order at ({zeroth_row:.0f}, {zeroth_column:.0f}) is inside or on "
        f"the edge of the signal region spanning rows {rows[0]}-{rows[-1]}, "
        f"columns {columns[0]}-{columns[-1]}"
    )


def test_placement_reports_addressability() -> None:
    """A target inside the Nyquist limit is reported as producible."""
    feedback = _feedback()
    placement = feedback.placement_data()
    assert placement.is_addressable
    assert placement.overshoot <= 0.0

    figure = placement.visualizer().render()
    assert figure is not None
    plt.close(figure)


def test_patch_is_placed_at_the_requested_offset() -> None:
    """The patch center lands TARGET_POSITION from the zeroth order, converted through
    the camera pitch.
    """
    feedback = _feedback()
    placement = feedback.placement_data()

    expected_column = TARGET_POSITION[0] / CAMERA_PIXEL_SIZE
    expected_row = TARGET_POSITION[1] / CAMERA_PIXEL_SIZE
    row_offset = placement.target_center[0] - placement.zeroth_order[0]
    column_offset = placement.target_center[1] - placement.zeroth_order[1]

    assert np.isclose(row_offset, expected_row, atol=0.5)
    assert np.isclose(column_offset, expected_column, atol=0.5)


# --- Registration ------------------------------------------------------------------


def test_mapping_seeds_the_model(monkeypatch) -> None:
    """A rotated or displaced camera is what this is for, and the loop must apply the
    mapping before it measures anything.
    """
    slm, camera, retriever, target, signal_region = _bench()
    applied = []
    monkeypatch.setattr(
        type(retriever.slm_camera_model),
        "calibrate_from_mapping",
        lambda self, mapping: applied.append(mapping),
    )

    sentinel = _identity_mapping()
    feedback = SimpleFeedbackCorrector(
        phase_retriever=retriever,
        camera=camera,
        slm=slm,
        target=target,
        signal_region=signal_region,
        target_position=TARGET_POSITION,
        camera_mapping=sentinel,
    )
    data = feedback.run(retriever_iterations=[5] * 1, averages=1, verbose=False)

    assert applied == [sentinel]
    # Recorded without its diagnostic frames, so it is a copy rather than the object.
    assert data.camera_mapping.timestamp == sentinel.timestamp
    assert np.array_equal(data.camera_mapping.transform, sentinel.transform)
    assert data.camera_mapping.visualization_data is None


def test_a_mapping_is_measured_when_none_is_given(monkeypatch) -> None:
    """With no mapping supplied, one is measured with a spot array.

    The mapper is stubbed: a real spot array needs resolvable spots, which this 64 by 64
    bench does not have.
    """
    slm, camera, retriever, target, signal_region = _bench()
    measured = _identity_mapping()

    class StubMapper:
        instances = []

        def __init__(self, *arguments):
            StubMapper.instances.append(arguments)

        def map_camera(self):
            return measured

    monkeypatch.setattr(
        "hologradpy.holography.camera_feedback.abstract.SpotArrayMapper", StubMapper
    )
    applied = []
    monkeypatch.setattr(
        type(retriever.slm_camera_model),
        "calibrate_from_mapping",
        lambda self, mapping: applied.append(mapping),
    )

    feedback = SimpleFeedbackCorrector(
        phase_retriever=retriever,
        camera=camera,
        slm=slm,
        target=target,
        signal_region=signal_region,
    )
    feedback.run(retriever_iterations=[5] * 1, averages=1, verbose=False)

    assert len(StubMapper.instances) == 1
    assert StubMapper.instances[0] == (slm, camera, retriever.slm_camera_model)
    assert applied == [measured]


def test_registration_is_applied_only_once(monkeypatch) -> None:
    """calibrate_from_mapping composes a residual, so applying it twice would put the
    rotation and shift in twice over.
    """
    slm, camera, retriever, target, signal_region = _bench()
    applied = []
    monkeypatch.setattr(
        type(retriever.slm_camera_model),
        "calibrate_from_mapping",
        lambda self, mapping: applied.append(mapping),
    )

    mapping = _identity_mapping()
    feedback = SimpleFeedbackCorrector(
        phase_retriever=retriever,
        camera=camera,
        slm=slm,
        target=target,
        signal_region=signal_region,
        camera_mapping=mapping,
    )
    feedback.register()
    feedback.register()
    feedback.run(retriever_iterations=[5] * 1, averages=1, verbose=False)

    assert applied == [mapping]


# --- Building the retriever --------------------------------------------------------


def test_builds_its_own_retriever() -> None:
    """The usual case: a model and a target, no search run by hand first."""
    torch.manual_seed(0)
    geometry = _geometry()
    slm = open_slm(SimulatedSLMTorch, input_geometry=geometry, bitdepth=8)
    hardware = _model(geometry, slm.virtual_slm, _beam(geometry, _aberration(geometry)))
    camera = SimulatedCameraTorch(hardware, bitdepth=12, noise_level=0.0)

    model = _model(geometry, VirtualSLM.from_slm(slm), _beam(geometry))
    target, signal_region = _target(model)

    feedback = SimpleFeedbackCorrector(
        slm=slm,
        camera=camera,
        slm_camera_model=model,
        target=target,
        signal_region=signal_region,
        init_slm_phase=_init_phase(),
        loss_scale=LOSS_SCALE,
        camera_mapping=_identity_mapping(),
    )

    assert isinstance(feedback.phase_retriever, GradientPhaseRetriever)
    assert feedback.slm_camera_model is model

    data = feedback.run(
        retriever_iterations=[10] * 2, averages=1, verbose=False
    )
    assert data.retrievals[-1].phase.shape == SLM_RESOLUTION


def test_uses_the_retriever_it_is_given() -> None:
    slm, camera, retriever, target, signal_region = _bench()
    feedback = SimpleFeedbackCorrector(
        phase_retriever=retriever,
        camera=camera,
        slm=slm,
        target=target,
        signal_region=signal_region,
        camera_mapping=_identity_mapping(),
    )
    assert feedback.phase_retriever is retriever


def test_no_model_and_no_retriever_raises() -> None:
    slm, camera, _, target, signal_region = _bench()
    with pytest.raises(ValueError, match="slm_camera_model"):
        SimpleFeedbackCorrector(
            slm=slm, camera=camera, target=target, signal_region=signal_region
        )


# --- The retargeting contract ------------------------------------------------------


def test_set_target_changes_the_loss_for_cg() -> None:
    _, _, retriever, patch, region_patch = _bench()
    target, signal_region = _placed(patch, region_patch)
    field = retriever.slm_camera_model()

    before = float(retriever.loss_function(field))
    retriever.set_target(torch.roll(target, shifts=6, dims=1))
    after = float(retriever.loss_function(field))

    assert not np.isclose(before, after)


def test_set_target_changes_the_loss_for_zernike() -> None:
    geometry = _geometry()
    slm = open_slm(SimulatedSLMTorch, input_geometry=geometry, bitdepth=8)
    model = _model(geometry, VirtualSLM.from_slm(slm), _beam(geometry))
    target, signal_region = _placed(*_target(model))

    retriever = ZernikePhaseRetriever(
        slm_camera_model=model,
        target=target,
        signal_region=signal_region,
        loss_scale=LOSS_SCALE,
    )
    field = retriever.slm_camera_model()

    before = float(retriever.loss_function(field))
    retriever.set_target(torch.roll(target, shifts=6, dims=1))
    after = float(retriever.loss_function(field))

    assert not np.isclose(before, after)


def test_set_target_keeps_the_signal_region() -> None:
    """A retarget need only pass the target: feedback calls it every iteration with the
    same region.
    """
    _, _, retriever, patch, region_patch = _bench()
    target, signal_region = _placed(patch, region_patch)
    retriever.set_target(target * 0.5)
    assert torch.equal(retriever.signal_region, signal_region)


def test_linear_superposition_refuses_to_be_retargeted() -> None:
    geometry = _geometry()
    slm = open_slm(SimulatedSLMTorch, input_geometry=geometry, bitdepth=8)
    model = _model(geometry, VirtualSLM.from_slm(slm), _beam(geometry))
    target, _ = _placed(*_target(model))

    retriever = LinearSuperpositionPhaseRetriever(
        slm_camera_model=model,
        target_positions=torch.tensor([[0.0, 0.0], [100e-6, 0.0]]),
    )
    with pytest.raises(NotImplementedError, match="target_positions"):
        retriever.set_target(target)


def test_target_without_signal_region_raises() -> None:
    geometry = _geometry()
    slm = open_slm(SimulatedSLMTorch, input_geometry=geometry, bitdepth=8)
    model = _model(geometry, VirtualSLM.from_slm(slm), _beam(geometry))
    target, _ = _placed(*_target(model))

    retriever = GradientPhaseRetriever(
        slm_camera_model=model,
        target=target,
        signal_region=torch.ones(CAMERA_RESOLUTION, dtype=torch.bool),
        init_slm_phase=_init_phase(),
        loss_scale=LOSS_SCALE,
    )
    retriever.signal_region = None
    with pytest.raises(ValueError, match="signal region"):
        retriever.set_target(target)


# --- The result record and the recorded steps --------------------------------------


def test_retrieve_returns_a_record() -> None:
    _, _, retriever, target, _ = _bench()
    record = retriever.retrieve(8, method="cg", verbose=False)

    assert record.phase.shape == SLM_RESOLUTION
    assert record.target.shape == CAMERA_RESOLUTION
    assert record.name == "GradientPhaseRetriever"
    assert set(record.metrics) == {"rmse", "psnr [dB]"}
    # One entry per objective evaluation, which the line search makes several of per
    # iteration, so this is at least as long as the iteration count.
    assert len(record.loss_history) >= 1
    assert record.loss_history[-1] < record.loss_history[0]


def test_retrieve_phase_still_returns_a_tensor() -> None:
    """The bare tensor is what the feedback loop and the examples use, so it stays."""
    _, _, retriever, _, _ = _bench()
    phase = retriever.retrieve_phase(5, method="cg", verbose=False)
    assert torch.is_tensor(phase)
    assert tuple(phase.shape) == SLM_RESOLUTION


def test_record_survives_a_save_load_round_trip(tmp_path) -> None:
    _, _, retriever, _, _ = _bench()
    record = retriever.retrieve(5, method="cg", verbose=False)

    path = tmp_path / "retrieval.asdf"
    record.save(path)
    reloaded = PhaseRetrievalData.load(path)

    assert np.allclose(reloaded.phase, record.phase)
    assert reloaded.metrics == record.metrics
    assert reloaded.loss_history == record.loss_history
    assert reloaded.timestamp == record.timestamp


def test_a_loaded_record_finds_the_steps_beside_it(tmp_path) -> None:
    """A record whose steps live in a sibling file used to need the caller to remember
    where they were, so a record opened on its own loaded and then failed at replay.
    """
    _, _, retriever, _, _ = _bench()
    record = retriever.retrieve(
        6, method="cg", verbose=False, step_stride=2, step_directory=tmp_path
    )
    assert record.step_iterations, "the retrieval recorded no steps to look for"

    # Built rather than loaded, so there is nowhere to look and it says so.
    with pytest.raises(FileNotFoundError, match="not loaded from a file"):
        record.load_step(record.step_iterations[0])

    path = tmp_path / "retrieval.asdf"
    record.save(path)
    reloaded = PhaseRetrievalData.load(path)

    assert reloaded.source_directory == tmp_path
    step = reloaded.load_step(reloaded.step_iterations[0])
    assert step.shape == tuple(SLM_RESOLUTION)

    # An explicit directory still wins, which is what a record split from its steps
    # needs.
    np.testing.assert_allclose(
        reloaded.load_step(reloaded.step_iterations[0], tmp_path), step
    )


def test_no_steps_recorded_by_default(tmp_path) -> None:
    """Off unless asked: a megapixel SLM writes about 5 MB per step."""
    _, _, retriever, _, _ = _bench()
    record = retriever.retrieve(6, method="cg", verbose=False)

    assert record.step_stride is None
    assert record.step_iterations == []
    assert list(tmp_path.iterdir()) == []


def test_steps_are_recorded_at_the_stride(tmp_path) -> None:
    _, _, retriever, _, _ = _bench()
    record = retriever.retrieve(
        8,
        method="cg",
        verbose=False,
        step_stride=2,
        step_directory=tmp_path,
    )

    assert record.step_stride == 2
    assert record.step_iterations, "the retrieval ran no iterations to record"
    assert all(index % 2 == 0 for index in record.step_iterations)

    for index in record.step_iterations:
        assert record.load_step(index, tmp_path).shape == SLM_RESOLUTION

    # The pattern moves as the retrieval proceeds, so consecutive steps differ.
    if len(record.step_iterations) > 1:
        first = record.load_step(record.step_iterations[0], tmp_path)
        last = record.load_step(record.step_iterations[-1], tmp_path)
        assert not np.allclose(first, last)


def test_steps_are_patterns_and_one_checkpoint(tmp_path) -> None:
    """Two files: the search's own parameter in one store, and the model that turns it
    back into predicted images. The images themselves are derived, so they are not
    stored.
    """
    _, _, retriever, _, _ = _bench()
    record = retriever.retrieve(
        8, method="cg", verbose=False, step_stride=2, step_directory=tmp_path
    )

    written = sorted(path.name for path in tmp_path.iterdir())
    assert written == sorted([MODEL_CHECKPOINT_NAME, RETRIEVAL_STEPS_NAME])
    assert record.model_checkpoint == MODEL_CHECKPOINT_NAME

    with RetrievalStepStore.open(tmp_path / RETRIEVAL_STEPS_NAME) as store:
        assert len(store) == len(record.step_iterations)
        assert set(store.read(0)) == {"slm_fraction"}


def test_replay_reproduces_the_prediction(tmp_path, monkeypatch) -> None:
    """The claim the storage saving rests on: a saved phase and the model give back
    exactly what the search saw, on the device it ran on.
    """
    _, _, retriever, _, _ = _bench()
    model = retriever.slm_camera_model

    predictions = {}
    original_record = RetrievalStepWriter.record

    def record_prediction(self, iteration: int, step_model) -> None:
        original_record(self, iteration, step_model)
        if iteration in self.iterations:
            predictions[iteration] = retriever.predicted_intensity()

    monkeypatch.setattr(RetrievalStepWriter, "record", record_prediction)
    record = retriever.retrieve(
        8, method="cg", verbose=False, step_stride=2, step_directory=tmp_path
    )

    assert predictions, "the retrieval ran no iterations to record"
    for iteration, expected in predictions.items():
        _, replayed = record.replay(iteration, tmp_path, model=model)
        assert np.array_equal(replayed, expected)


def test_replay_rebuilds_the_model_from_the_checkpoint(tmp_path) -> None:
    """No live model needed: the checkpoint beside the steps is enough, and it does
    not have to be told which system wrote it.
    """
    _, _, retriever, _, _ = _bench()
    record = retriever.retrieve(
        6, method="cg", verbose=False, step_stride=3, step_directory=tmp_path
    )
    iteration = record.step_iterations[-1]

    _, through_the_model = record.replay(
        iteration, tmp_path, model=retriever.slm_camera_model
    )
    _, from_the_checkpoint = record.replay(iteration, tmp_path)
    assert np.array_equal(from_the_checkpoint, through_the_model)


def test_replay_leaves_the_model_alone(tmp_path) -> None:
    """Replaying an old iteration must not quietly roll the model back to it."""
    _, _, retriever, _, _ = _bench()
    record = retriever.retrieve(
        6, method="cg", verbose=False, step_stride=2, step_directory=tmp_path
    )
    model = retriever.slm_camera_model
    before = gpu_to_numpy(model.virtual_slm.get_phase())

    record.replay(record.step_iterations[0], tmp_path, model=model)

    assert np.array_equal(gpu_to_numpy(model.virtual_slm.get_phase()), before)


def test_replay_without_a_checkpoint_raises() -> None:
    record = PhaseRetrievalData(
        timestamp=datetime.now(),
        name="no checkpoint",
        phase=np.zeros(SLM_RESOLUTION),
        target=np.zeros(CAMERA_RESOLUTION),
        signal_region=np.zeros(CAMERA_RESOLUTION),
        step_iterations=[0],
    )
    with pytest.raises(ValueError, match="no model checkpoint"):
        record.replay(0, ".")


def test_recording_on_a_stochastic_model_raises() -> None:
    """A model that draws randomness per forward pass cannot have its images rebuilt,
    and saying so beats saving an image the search never saw.
    """
    slm, camera, retriever, _, _ = _bench()
    retriever.slm_camera_model.insert_after(
        "slm_field", "pointing", PointingInstability(1e-6)
    )

    with pytest.raises(ValueError, match="pointing"):
        retriever.retrieve(4, step_stride=2, step_directory=".")


def test_a_deterministic_hardware_module_does_not_block_recording(tmp_path) -> None:
    """The guard reads the module's state, not its class: noise switched off is
    deterministic, and a speckle background drawn once into a buffer always was.
    """
    _, _, retriever, _, _ = _bench()
    retriever.slm_camera_model.insert_after(
        "slm_field", "scatter", BackgroundScatter(power=1e-9, seed=0)
    )

    record = retriever.retrieve(
        4, method="cg", verbose=False, step_stride=2, step_directory=tmp_path
    )
    assert record.step_iterations


def test_a_later_retrieval_does_not_inherit_the_steps(tmp_path) -> None:
    """Recorded steps belong to the retrieval that asked for them.

    They live on a recorder built per search rather than on the retriever, so a later
    search that asked for none cannot overwrite the files or extend the bookkeeping.
    """
    _, _, retriever, _, _ = _bench()
    record = retriever.retrieve(
        4, method="cg", verbose=False, step_stride=2, step_directory=tmp_path
    )
    written = {path.name: path.stat().st_mtime_ns for path in tmp_path.iterdir()}

    retriever.retrieve_phase(4, method="cg", verbose=False)

    assert {
        path.name: path.stat().st_mtime_ns for path in tmp_path.iterdir()
    } == written
    assert record.step_iterations == [2, 4]


def test_missing_step_raises(tmp_path) -> None:
    _, _, retriever, _, _ = _bench()
    record = retriever.retrieve(
        4, method="cg", verbose=False, step_stride=2, step_directory=tmp_path
    )
    with pytest.raises(KeyError, match="No step recorded for iteration 99"):
        record.load_step(99, tmp_path)
    with pytest.raises(KeyError, match="No step recorded for iteration 99"):
        record.replay(99, tmp_path)


def test_step_stride_without_a_directory_raises() -> None:
    """Fails before the search rather than after it, so no time is wasted."""
    _, _, retriever, _, _ = _bench()
    with pytest.raises(ValueError, match="step_directory"):
        retriever.retrieve(4, step_stride=2)


# --- The target update arithmetic --------------------------------------------------


def test_correction_is_confined_to_the_signal_region() -> None:
    signal_region = np.zeros((4, 4))
    signal_region[1:3, 1:3] = 1
    corrected = np.ones((4, 4))
    discrepancy = np.full((4, 4), 0.5)

    updated = SimpleFeedbackCorrector._corrected_target_for(
        corrected, discrepancy, 1.0, signal_region, 0.0
    )
    assert np.all(updated[signal_region == 0] == 0)
    assert np.allclose(updated[signal_region == 1], 1.5)


def test_correction_clamps_negatives() -> None:
    signal_region = np.ones((3, 3))
    updated = SimpleFeedbackCorrector._corrected_target_for(
        np.full((3, 3), 0.2), np.full((3, 3), -1.0), 1.0, signal_region, 0.0
    )
    assert np.all(updated >= 0)


def test_zero_gain_reproduces_the_target() -> None:
    signal_region = np.ones((3, 3))
    corrected = np.linspace(0, 1, 9).reshape(3, 3)
    updated = SimpleFeedbackCorrector._corrected_target_for(
        corrected, np.full((3, 3), 0.5), 0.0, signal_region, 0.0
    )
    assert np.allclose(updated, corrected)


def test_iteration_count_comes_from_the_retriever_iterations() -> None:
    """The list length is the run length, so there is no separate count to keep in
    step with it.
    """
    feedback = _feedback()
    data = feedback.run(
        retriever_iterations=[6, 5, 4], averages=1, verbose=False
    )

    assert data.number_of_iterations == 3
    assert len(data.retrievals) == 3
    # Each search got the budget it was given.
    assert [len(r.loss_history) > 0 for r in data.retrievals] == [True] * 3


def test_gain_takes_a_scalar_or_one_per_iteration() -> None:
    """Either form, since a flat gain is the common case and a decaying one is what
    stops the loop overshooting at the end.
    """
    feedback = _feedback()
    flat = feedback.run(
        retriever_iterations=[5] * 3, gain=0.7, averages=1, verbose=False
    )
    assert flat.number_of_iterations == 3

    feedback = _feedback()
    decaying = feedback.run(
        retriever_iterations=[5] * 3, gain=[1.0, 0.6, 0.3], averages=1, verbose=False
    )
    assert decaying.number_of_iterations == 3


def test_a_mismatched_gain_sequence_raises() -> None:
    feedback = _feedback()
    with pytest.raises(ValueError, match="2 entries but there are 3"):
        feedback.run(
            retriever_iterations=[5] * 3, gain=[1.0, 0.5], averages=1, verbose=False
        )


def test_a_scalar_retriever_iterations_raises() -> None:
    """A bare int is refused, since the sequence length sets the run length."""
    feedback = _feedback()
    with pytest.raises(TypeError, match=r"\[20\] \* n"):
        feedback.run(retriever_iterations=20, averages=1, verbose=False)


def test_empty_retriever_iterations_raises() -> None:
    feedback = _feedback()
    with pytest.raises(ValueError, match="no feedback iterations"):
        feedback.run(retriever_iterations=[], averages=1, verbose=False)


def test_per_iteration_accepts_a_scalar_or_a_sequence() -> None:
    assert SimpleFeedbackCorrector._per_iteration(0.5, 3, "gain") == [0.5, 0.5, 0.5]
    assert SimpleFeedbackCorrector._per_iteration([1, 2, 3], 3, "gain") == [1, 2, 3]
    with pytest.raises(ValueError, match="2 entries but there are 3"):
        SimpleFeedbackCorrector._per_iteration([1, 2], 3, "gain")


# --- Preconditions -----------------------------------------------------------------


def test_mismatched_grids_raise() -> None:
    slm, camera, retriever, target, signal_region = _bench()
    feedback = SimpleFeedbackCorrector(
        phase_retriever=retriever,
        camera=camera,
        slm=slm,
        target=target,
        signal_region=signal_region,
        camera_mapping=_identity_mapping(),
    )
    # Reaching past the property, which has no setter. Building a genuinely mismatched
    # pair would mean a second bench, and this provokes the same check.
    retriever.slm_camera_model[-1]._resolution_out = (16, 16)

    with pytest.raises(ValueError, match=r"\(16, 16\).*\(64, 64\)"):
        feedback.run(retriever_iterations=[20], verbose=False)


def test_signal_region_defaults_to_the_whole_patch() -> None:
    """A caller who sized the patch themselves gets the whole of it measured."""
    slm, camera, retriever, target, _ = _bench()
    retriever.signal_region = None
    feedback = SimpleFeedbackCorrector(
        phase_retriever=retriever,
        camera=camera,
        slm=slm,
        target=target,
        camera_mapping=_identity_mapping(),
    )
    assert torch.all(feedback.signal_region_patch == 1)
    assert feedback.signal_region_patch.shape == target.shape


def test_mismatched_patch_shapes_raise() -> None:
    """The target and the region are placed together, so they travel as a pair."""
    slm, camera, retriever, target, _ = _bench()
    with pytest.raises(ValueError, match="the same shape"):
        SimpleFeedbackCorrector(
            phase_retriever=retriever,
            camera=camera,
            slm=slm,
            target=target,
            signal_region=torch.ones((8, 8)),
        )


# --- The visualizer ----------------------------------------------------------------


def _panel_titles(panels) -> str:
    """Every title the panels set, as one string, by drawing them onto a scratch
    figure.
    """
    titles = []
    for panel in panels.values():
        figure, axes = plt.subplots()
        panel(axes)
        titles.append(axes.get_title())
        plt.close(figure)
    return " | ".join(titles)


def _synthetic_data(
    iterations: int = 3,
    metric_names: Sequence[str] = ("rmse", "psnr [dB]"),
    with_initial_guess: bool = True,
) -> CameraFeedbackData:
    """A record shaped like a run without paying for one, for the drawing tests."""
    rng = np.random.default_rng(0)
    signal_region = np.zeros((12, 16))
    signal_region[3:9, 4:12] = 1
    roi = ROI.detect(signal_region, threshold=0.0, pad=0)
    data = CameraFeedbackData(
        timestamp=datetime.now(),
        name="synthetic",
        target=rng.random((12, 16)),
        signal_region=signal_region,
        final_camera_image=rng.random((12, 16)) * 100,
        initial_guess=rng.random((12, 16)) * 400 if with_initial_guess else None,
        lower_is_better={"rmse": True, "psnr [dB]": False},
    )
    for iteration in range(iterations):
        data.measured_images.append(roi.crop(rng.random((12, 16)) * 100))
        data.corrected_targets.append(roi.crop(rng.random((12, 16))))
        # Leaned, as a real run stores them.
        data.retrievals.append(
            PhaseRetrievalData(
                timestamp=datetime.now(),
                name="synthetic",
                phase=rng.random((10, 10)) * 2 * np.pi,
            ).lean()
        )
        for offset, name in enumerate(metric_names):
            data.metrics.setdefault(name, []).append(offset + 0.5 / (iteration + 1))
    return data


# Each row, and how many cells it draws with the synthetic record above.
ROW_METHODS = {
    "render_target": 2,
    "render_hologram": 2,
    "render_signal_region": 3,
    "render_convergence": 2,
}


@pytest.mark.parametrize("method,cells", ROW_METHODS.items())
def test_each_row_renders_on_its_own(method: str, cells: int) -> None:
    """Every row is callable separately, which is the point of the split."""
    visualizer = _synthetic_data().visualizer()
    figure = getattr(visualizer, method)()

    assert figure is not None
    # Image cells carry a colorbar axes each, line cells do not.
    assert len(figure.axes) >= cells
    plt.close(figure)


def test_the_rows_and_the_whole_figure_agree() -> None:
    """default_layout and panels are built from the same section methods, so the
    combined figure cannot drift away from the rows drawn individually.
    """
    visualizer = _synthetic_data().visualizer()

    from_rows = set()
    for section in ("_target", "_hologram", "_region", "_convergence"):
        from_rows |= set(getattr(visualizer, f"{section}_panels")())

    assert set(visualizer.panels()) == from_rows
    assert len(visualizer.default_layout()._rows) == 4


def test_target_row_falls_back_without_an_initial_guess() -> None:
    """A record written before initial_guess was carried still renders, one cell
    narrower, rather than raising.
    """
    data = _synthetic_data(with_initial_guess=False)
    visualizer = data.visualizer()

    assert len(visualizer._target_cells()) == 1
    assert set(visualizer._target_panels()) == {"target"}

    figure = visualizer.render()
    assert figure is not None
    plt.close(figure)


def test_signal_region_shows_the_best_iteration() -> None:
    """Not the last: the loop can overshoot and finish worse than it managed earlier."""
    data = _synthetic_data(iterations=5)
    data.metrics["rmse"] = [0.5, 0.2, 0.05, 0.3, 0.4]
    visualizer = data.visualizer()

    assert visualizer.best_iteration() == 2
    assert "iteration 3" in _panel_titles(visualizer._region_panels())


def test_best_iteration_does_not_depend_on_the_metric_being_called_rms() -> None:
    """Keyed on the run's first metric, not on a name, so relabeling the default
    metric does not quietly send this back to the last iteration.
    """
    data = _synthetic_data(iterations=5, metric_names=("rmse",))
    data.metrics = {"rmse": [0.5, 0.2, 0.05, 0.3, 0.4]}
    data.lower_is_better = {"rmse": True}

    assert data.visualizer().best_iteration() == 2


def test_best_iteration_honours_a_figure_of_merit_first_metric() -> None:
    """A run whose first metric is better when higher is ranked that way."""
    data = _synthetic_data(iterations=4, metric_names=("psnr [dB]",))
    data.metrics = {"psnr [dB]": [25.0, 31.0, 33.0, 30.0]}
    data.lower_is_better = {"psnr [dB]": False}

    assert data.visualizer().best_iteration() == 2


def test_best_index_follows_each_metric_own_direction() -> None:
    """Rms is best at its minimum, psnr at its maximum, and the record says which."""
    data = _synthetic_data()
    data.lower_is_better = {"rmse": True, "psnr [dB]": False}
    best = data.visualizer()._best_index

    assert best("rmse", [0.11, 0.06, 0.031, 0.056]) == 2
    assert best("psnr [dB]", [25.0, 30.0, 33.0, 31.0]) == 2


def test_best_index_defaults_to_lower_is_better() -> None:
    """A record written before the flag existed, or a metric the run did not describe,
    is treated as an error rather than a figure of merit.
    """
    data = _synthetic_data()
    data.lower_is_better = {}
    assert data.visualizer()._best_index("anything", [5.0, 1.0, 9.0]) == 1


def test_run_records_each_metric_direction() -> None:
    """The direction comes from the metric, so the visualizer never has to guess."""
    feedback = _feedback()
    throughput = IntensityMetric(
        "throughput",
        lambda signal, target, measured: efficiency(signal, measured),
        lower_is_better=False,
    )
    data = feedback.run(
        retriever_iterations=[5] * 2,
        averages=1,
        metrics=(*DEFAULT_INTENSITY_METRICS, throughput),
        verbose=False,
    )

    assert data.lower_is_better == {
        "rmse": True,
        "psnr [dB]": False,
        "throughput": False,
    }


def test_signal_region_row_is_cropped() -> None:
    """All three panels are drawn on the region rather than the full frame."""
    data = _synthetic_data()
    visualizer = data.visualizer()

    cropped = data.signal_roi.crop(np.asarray(data.signal_region))
    assert visualizer._difference().shape == cropped.shape
    assert cropped.shape != data.target.shape
    assert cropped.size < data.target.size


@pytest.mark.parametrize("iterations", [1, 3])
def test_visualizer_renders(iterations: int) -> None:
    figure = _synthetic_data(iterations).visualizer().render()
    assert figure is not None
    plt.close(figure)


@pytest.mark.parametrize(
    "metric_names",
    [("rmse",), ("rmse", "psnr [dB]"), ("rmse", "psnr [dB]", "efficiency")],
)
def test_visualizer_grows_a_panel_per_metric(metric_names: Sequence[str]) -> None:
    """The convergence row is built from the metrics dict, so a caller's own metric
    plots itself without the visualizer knowing what it is.
    """
    data = _synthetic_data(metric_names=metric_names)
    visualizer = data.visualizer()

    panels = visualizer.panels()
    for name in metric_names:
        assert visualizer._metric_key(name) in panels

    figure = visualizer.render_convergence()
    assert figure is not None
    plt.close(figure)


def test_visualizer_renders_without_metrics() -> None:
    """A run told to measure nothing still draws its images."""
    data = _synthetic_data(metric_names=())
    figure = data.visualizer().render()
    assert figure is not None
    plt.close(figure)


def test_custom_metric_is_recorded() -> None:
    """A metric the corrector has never heard of goes through by name."""
    feedback = _feedback()
    # Named apart from the imported efficiency(), which the lambda calls: a local of
    # the same name would rebind it and the lambda would recurse into itself.
    efficiency_metric = IntensityMetric(
        "efficiency", lambda signal, target, measured: efficiency(signal, measured)
    )
    data = feedback.run(
        retriever_iterations=[5] * 2,
        averages=1,
        metrics=(*DEFAULT_INTENSITY_METRICS, efficiency_metric),
        verbose=False,
    )

    assert set(data.metrics) == {"rmse", "psnr [dB]", "efficiency"}
    assert len(data.metrics["efficiency"]) == 2
    assert all(0.0 <= value <= 1.0 for value in data.metrics["efficiency"])


def test_visualizer_renders_a_chosen_iteration() -> None:
    visualizer = _synthetic_data().visualizer(iteration=0)
    figure = visualizer.render()
    plt.close(figure)

    figure = visualizer.render_iteration(1)
    assert isinstance(visualizer, CameraFeedbackVisualizer)
    plt.close(figure)


def test_run_result_renders(feedback_run: CameraFeedbackData) -> None:
    figure = feedback_run.visualizer().render()
    assert figure is not None
    plt.close(figure)


def test_phase_reaches_the_slm() -> None:
    """The loop displays what it retrieved, which is what makes the next measurement
    mean anything.
    """
    feedback = _feedback()
    data = feedback.run(retriever_iterations=[5] * 1, averages=1, verbose=False)

    retrieved = gpu_to_numpy(
        feedback.phase_retriever.slm_camera_model.virtual_slm.get_phase()
    )
    assert np.allclose(data.retrievals[-1].phase, retrieved)
    # Quantized to the SLM bit depth on the way to the display, and shifted by one
    # level there, so equal only to within a couple of levels.
    displayed = feedback.slm.virtual_slm.get_phase().numpy()
    wrapped_error = np.angle(np.exp(1j * (displayed - retrieved)))
    assert np.abs(wrapped_error).max() < 3 * 2 * np.pi / 2**8
