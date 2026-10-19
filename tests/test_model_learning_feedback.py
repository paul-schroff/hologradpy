"""ModelLearningFeedbackCorrector: measure, learn the model, step.

The loop runs on the 64 by 64 bench of test_camera_feedback.py. Its beam carries an
aberration the retriever's model does not have. The model's SLM field is a PSF kernel
fitted to the unaberrated beam, as a wavefront calibration hands it over. The learning
therefore has the aberration to find.
"""

from __future__ import annotations

import itertools
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hologradpy.analysis.error_metrics import normalize, remove_linear_phase
from hologradpy.analysis.fitting import fit_translation
from hologradpy.calibration.camera_mapping import CameraMapping, FocalSpotFit
from hologradpy.fourier_transforms import fft_2d, ifft_2d
from hologradpy.grids import get_frequency_grid, get_pixel_grid, get_spatial_grid
from hologradpy.hardware import SimulatedCameraTorch, SimulatedSLMTorch, open_slm
from hologradpy.holography.camera_feedback import (
    CameraFeedbackData,
    ModelLearningFeedbackCorrector,
    corrected_target_intensity,
)
from hologradpy.holography.camera_feedback.model_learning_feedback import _plateau_gap
from hologradpy.optics.complex_amplitude import ComplexAmplitude, FieldGeometry
from hologradpy.optics.modules.propagators import AngularSpectrumMethod
from hologradpy.optics.modules.slm_fields import (
    PixelwiseSLMField,
    PSFSLMField,
    SLMField,
)
from hologradpy.optics.modules.virtual_slms import VirtualSLM
from hologradpy.optics.systems import SLMCZT
from hologradpy.profiles.amplitude import gaussian_beam_intensity, top_hat_2D
from hologradpy.profiles.phase import defocus_phase, linear_phase

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

SLM_RESOLUTION = (64, 64)
SLM_PIXEL_SIZE = 12.5e-6
CAMERA_RESOLUTION = (64, 64)
CAMERA_PIXEL_SIZE = 20e-6
WAVELENGTH = 0.670e-6
FOCAL_LENGTH = 0.1
# The intensity cost on this 64 by 64 camera needs a smaller weight than the default,
# as in test_camera_feedback.py.
LOSS_SCALE = 1e8
# The focal spot of the 0.4 mm beam, the shoulders of the top hat.
SPOT_RADIUS = WAVELENGTH * FOCAL_LENGTH / (np.pi * 0.4e-3)
# A top hat of 200 x 200 um, four spot radii across, as the lab's top hat is. The
# patch holds it with a margin of two spot radii on every side, and the whole patch is
# the signal region.
PLATEAU_WIDTH = 200e-6
PATCH_RESOLUTION = (21, 21)
# Off axis by more than the patch's half width plus a spot radius, and inside the
# camera's 640 um.
TARGET_POSITION = (350e-6, 0.0)
# Ten focal-spot waists of the 0.4 mm beam, 13 samples at the kernel's pitch.
PSF_EXTENT = 10 * SPOT_RADIUS
ITERATIONS = 8


def _geometry() -> FieldGeometry:
    return FieldGeometry(
        resolution=SLM_RESOLUTION,
        pixel_size=torch.tensor([SLM_PIXEL_SIZE, SLM_PIXEL_SIZE]),
        wavelength=torch.tensor(WAVELENGTH),
    )


def _beam(
    geometry: FieldGeometry,
    phase: torch.Tensor | None = None,
    beam_radius: float = 0.4e-3,
) -> ComplexAmplitude:
    amplitude = gaussian_beam_intensity(
        *geometry.get_spatial_grid(), beam_radius_major=beam_radius
    ).sqrt()
    field = amplitude + 0j if phase is None else amplitude * torch.exp(1j * phase)
    return ComplexAmplitude(
        field,
        wavelength=geometry.wavelength,
        pixel_size=geometry.pixel_size,
        power=1e-3,
    )


def _aberration(geometry: FieldGeometry) -> torch.Tensor:
    """The low-order wavefront error of test_camera_feedback.py."""
    grid_x, grid_y = geometry.get_spatial_grid()
    normalized_x = grid_x / grid_x.abs().max()
    normalized_y = grid_y / grid_y.abs().max()
    return 1.2 * (normalized_x**2 - normalized_y**2) + 0.8 * normalized_x * normalized_y


def _model(
    geometry: FieldGeometry,
    virtual_slm: VirtualSLM,
    slm_field: SLMField,
    camera_shift: tuple[float, float] = (0.0, 0.0),
) -> SLMCZT:
    model = SLMCZT(
        input_geometry=geometry,
        virtual_slm=virtual_slm,
        camera_resolution=CAMERA_RESOLUTION,
        camera_pixel_size=(CAMERA_PIXEL_SIZE, CAMERA_PIXEL_SIZE),
        focal_length=FOCAL_LENGTH,
        slm_field=slm_field,
        camera_shift=camera_shift,
    )
    model()
    return model


def _target() -> tuple[torch.Tensor, torch.Tensor]:
    grid = get_spatial_grid(PATCH_RESOLUTION, (CAMERA_PIXEL_SIZE, CAMERA_PIXEL_SIZE))
    target = top_hat_2D(
        *grid, 0.0, 0.0, PLATEAU_WIDTH, PLATEAU_WIDTH, SPOT_RADIUS, SPOT_RADIUS, 1.0
    ).float()
    return target, torch.ones_like(target)


def _init_phase() -> torch.Tensor:
    """A random start plus the grating that steers to the target."""
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


def _identity_mapping(
    output_pixel_size: tuple[float, float] = (CAMERA_PIXEL_SIZE, CAMERA_PIXEL_SIZE),
) -> CameraMapping:
    points = [(10.0, 12.0), (40.0, 15.0), (20.0, 45.0), (50.0, 50.0)]
    return CameraMapping(
        timestamp=datetime(2026, 1, 1),
        name="identity",
        transform=np.eye(3),
        detected_points=points,
        calculated_points=points,
        spot_fit=FocalSpotFit(waist=2.0),
        output_pixel_size=output_pixel_size,
        output_resolution=CAMERA_RESOLUTION,
    )


def _bench(
    aberrated: bool = True,
    camera_shift: tuple[float, float] = (0.0, 0.0),
    psf_field: bool = True,
) -> SimpleNamespace:
    """A simulated bench, and a model with the unaberrated beam as its field."""
    torch.manual_seed(0)
    geometry = _geometry()
    slm = open_slm(SimulatedSLMTorch, input_geometry=geometry, bitdepth=8)
    phase = _aberration(geometry) if aberrated else None
    hardware = _model(
        geometry,
        slm.virtual_slm,
        PixelwiseSLMField(_beam(geometry, phase)),
        camera_shift=camera_shift,
    )
    camera = SimulatedCameraTorch(hardware, bitdepth=12, read_noise=0.0)

    if psf_field:
        calibration = SimpleNamespace(complex_amplitude=_beam(geometry))
        field = PSFSLMField.from_calibration_data(
            calibration, FOCAL_LENGTH, PSF_EXTENT
        )
    else:
        field = PixelwiseSLMField(_beam(geometry))
    model = _model(geometry, VirtualSLM.from_slm(slm), field)
    patch, region = _target()
    return SimpleNamespace(
        slm=slm,
        camera=camera,
        hardware=hardware,
        model=model,
        patch=patch,
        region=region,
    )


def _corrector(bench: SimpleNamespace) -> ModelLearningFeedbackCorrector:
    return ModelLearningFeedbackCorrector(
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


def _run(corrector: ModelLearningFeedbackCorrector, **kwargs) -> CameraFeedbackData:
    kwargs.setdefault("retriever_iterations", [40] * ITERATIONS)
    kwargs.setdefault("averages", 1)
    kwargs.setdefault("retrieve_options", {"method": "cg"})
    return corrector.run(verbose=False, **kwargs)


def _phase_residual(field: torch.Tensor, reference: torch.Tensor) -> float:
    """The RMS phase between two fields over the beam, weighted by its intensity, with
    the piston and the tilt removed.
    """
    weight = reference.abs() ** 2
    beam = weight > weight.max() * np.exp(-2)
    phasor = remove_linear_phase(field * reference.conj(), beam)
    angle = torch.angle(phasor * torch.exp(-1j * torch.angle((phasor * beam).sum())))
    return float(
        torch.sqrt((angle**2 * weight * beam).sum() / (weight * beam).sum())
    )


@pytest.fixture(scope="module", params=["levenberg_marquardt", "adam"])
def learning_run(request) -> SimpleNamespace:
    """One run with each learning method, shared by the tests that only read it. Adam
    takes its 50 default steps, and Levenberg-Marquardt its 5.
    """
    bench = _bench()
    start_field = bench.model.slm_field.get_wavefront().detach().clone()
    data = _run(_corrector(bench), learning_method=request.param)
    return SimpleNamespace(
        data=data,
        start_field=start_field,
        learned_field=bench.model.slm_field.get_wavefront().detach(),
        true_field=bench.hardware.slm_field.get_wavefront().detach(),
        kernel_size=bench.model.slm_field.psf_kernel_size,
    )


# --- The closed loop -----------------------------------------------------------------


def test_learning_reduces_the_error(learning_run: SimpleNamespace) -> None:
    rmse = learning_run.data.metrics["rmse"]
    assert rmse[-1] < 0.3 * rmse[0]


def test_the_model_learns_to_predict_the_frames(learning_run: SimpleNamespace) -> None:
    """The "innovation" is the error of predicting a frame before learning from it. It
    falls as the model learns the aberration.
    """
    innovation = learning_run.data.metadata["innovation"]
    assert innovation[-1] < 0.5 * innovation[0]


def test_the_learned_field_moves_toward_the_bench(
    learning_run: SimpleNamespace,
) -> None:
    before = _phase_residual(learning_run.start_field, learning_run.true_field)
    after = _phase_residual(learning_run.learned_field, learning_run.true_field)
    assert after < 0.75 * before


@pytest.fixture(scope="module")
def shifted_run() -> CameraFeedbackData:
    """A run on the bench with its camera displaced by 1.5 pixels along x, so the
    first pattern lands 1.5 pixels from where the model puts it and from the target.
    """
    bench = _bench(camera_shift=(1.5 * CAMERA_PIXEL_SIZE, 0.0))
    return _run(_corrector(bench))


def test_the_model_learns_where_the_pattern_lands(
    shifted_run: CameraFeedbackData,
) -> None:
    """The frames keep the displacement, so the learning fits it into the model. Later
    frames then land where the model predicts them.
    """
    first_column, first_row = shifted_run.metadata["translations"][0][0]
    last_column, last_row = shifted_run.metadata["translations"][-1][0]
    assert np.hypot(first_column, first_row) > 1.0
    assert np.hypot(last_column, last_row) < 0.2


def test_frames_are_normalized_by_their_bright_pixels(
    learning_run: SimpleNamespace,
) -> None:
    """No scale is fitted to the model. Each frame's bright pixels therefore average to
    one, as the target's do.
    """
    data = learning_run.data
    region = data.signal_roi.crop(np.asarray(data.signal_region)).astype(float)
    for frame in data.measured_images:
        assert np.allclose(normalize(frame, region), frame)


def test_records_every_iteration(learning_run: SimpleNamespace) -> None:
    data = learning_run.data
    assert data.number_of_iterations == ITERATIONS
    for key in (
        "exposures",
        "translations",
        "pattern_offsets",
        "noise_floor",
        "innovation",
        "residual",
        "response_ratio",
        "learning_cost",
        "psf_kernels",
    ):
        assert len(data.metadata[key]) == ITERATIONS
    assert data.metadata["response_ratio"][0] is None
    assert data.metadata["psf_kernels"][0].shape == learning_run.kernel_size
    assert learning_run.kernel_size == (13, 13)


# --- The parts of a step --------------------------------------------------------------


def test_learning_keeps_the_retrievers_hologram() -> None:
    bench = _bench()
    corrector = _corrector(bench)
    learn = corrector._learn
    held: list[tuple[torch.Tensor, torch.Tensor]] = []

    def learn_and_compare(*args, **kwargs):
        before = bench.model.virtual_slm.get_phase().detach().clone()
        cost = learn(*args, **kwargs)
        held.append((before, bench.model.virtual_slm.get_phase().detach().clone()))
        return cost

    corrector._learn = learn_and_compare
    _run(corrector, retriever_iterations=[10] * 2, learning_steps=5)

    assert len(held) == 2
    for before, after in held:
        assert torch.equal(before, after)


def test_the_pattern_returns_to_the_target(shifted_run: CameraFeedbackData) -> None:
    """Once the model has learned where the pattern lands, the next hologram brings it
    back onto the target. Its shape converges at the same time. The recorded offsets
    are where the patterns landed, the registered frames' own translations included.
    """
    offsets = np.hypot(*np.asarray(shifted_run.metadata["pattern_offsets"]).T)
    assert offsets[0] > 1.0
    assert offsets[-1] < 0.25
    rmse = shifted_run.metrics["rmse"]
    assert rmse[-1] < 0.3 * rmse[0]


def test_frames_after_the_first_iteration_sit_on_the_prediction() -> None:
    """Without learning, the model never finds the camera's 1.5-pixel displacement. The
    first frame keeps it, and the second is moved onto the prediction. The recorded
    offsets are where both patterns landed.
    """
    bench = _bench(aberrated=False, camera_shift=(1.5 * CAMERA_PIXEL_SIZE, 0.0))
    data = _run(_corrector(bench), retriever_iterations=[40, 40], learning_steps=0)

    roi = data.signal_roi
    region = torch.as_tensor(roi.crop(np.asarray(data.signal_region)).astype(bool))
    target = torch.as_tensor(roi.crop(np.asarray(data.target, dtype=float)))
    first, second = (
        fit_translation(torch.as_tensor(np.asarray(frame, dtype=float)), target, region)
        .shift
        for frame in data.measured_images
    )
    assert abs(first[0]) == pytest.approx(1.5, abs=0.15)
    assert np.hypot(*second) < 0.15
    landed = np.hypot(*np.asarray(data.metadata["pattern_offsets"]).T)
    assert landed == pytest.approx([1.5, 1.5], abs=0.15)


def test_first_defocused_frames_sit_where_the_frame_in_focus_landed() -> None:
    """The bench's pointing changes with every displayed hologram. In the first
    iteration, the defocused frames are moved to where the frame in focus landed. All
    three frames then reach the learning with one displacement. In the second
    iteration, the frames reach the learning on their predictions.

    The learning is one Adam step, which leaves the model at the bench's own beam. The
    defocused patterns therefore keep the shape of their predictions, and their fits
    come back within 0.1 pixel.
    """
    bench = _bench(aberrated=False)
    corrector = _corrector(bench)
    # A new pointing for every display, in camera pixels along (x, y). An iteration
    # displays its hologram, the two defocused holograms and its hologram again.
    pointings = itertools.cycle([(1.5, 0.0), (-1.0, 1.0), (0.5, -1.5), (0.0, 0.0)])
    camera_shift = bench.hardware.focal_plane_partial_affine.shift
    display = bench.slm.set_phase

    def display_with_new_pointing(phase, *args, **kwargs):
        with torch.no_grad():
            camera_shift.copy_(torch.tensor(next(pointings), dtype=camera_shift.dtype))
        return display(phase, *args, **kwargs)

    bench.slm.set_phase = display_with_new_pointing
    learn = corrector._learn
    translations: list[list[tuple[float, float]]] = []

    def learn_and_fit(hologram_frames, *args):
        predictions = [
            corrector._predict_intensity(levels) for levels in hologram_frames.levels
        ]
        translations.append(
            [
                fit_translation(frame, prediction, hologram_frames.region).shift
                for prediction, frame in zip(predictions, hologram_frames.frames)
            ]
        )
        return learn(hologram_frames, *args)

    corrector._learn = learn_and_fit
    _run(
        corrector,
        retriever_iterations=[10] * 2,
        learning_method="adam",
        learning_steps=1,
        defocus_distances=(-5e-3, 5e-3),
    )

    first, second = translations
    assert np.hypot(*first[0]) > 0.5
    for shift in first[1:]:
        assert shift == pytest.approx(first[0], abs=0.1)
    for shift in second[3:]:
        assert np.hypot(*shift) < 0.1


def test_the_start_hologram_moves_the_pattern_by_the_shift() -> None:
    """The linear phase added to the hologram moves the model's focal spot on the
    sensor by the requested shift.
    """
    bench = _bench(aberrated=False)
    corrector = _corrector(bench)
    corrector.apply_camera_mapping(verbose=False)
    geometry = _geometry()
    grating = linear_phase(
        *geometry.get_spatial_grid(),
        tilt_x=TARGET_POSITION[0],
        tilt_y=TARGET_POSITION[1],
        wavenumber=geometry.wavenumber.reshape(()),
        focal_length=FOCAL_LENGTH,
    ).numpy()
    bench.model.virtual_slm.set_phase(grating)
    before = torch.as_tensor(corrector.phase_retriever.predicted_intensity())
    corrector._start_from_moved_hologram(grating, (1.0, -2.0))
    after = torch.as_tensor(corrector.phase_retriever.predicted_intensity())

    fit = fit_translation(after, before, torch.ones_like(before, dtype=torch.bool))
    assert fit.shift == pytest.approx((1.0, -2.0), abs=0.05)


def test_records_a_known_pointing_offset() -> None:
    """A camera displaced by 1.5 pixels along x shows the pattern displaced by as much
    from where the model puts it.
    """
    bench = _bench(aberrated=False, camera_shift=(1.5 * CAMERA_PIXEL_SIZE, 0.0))
    data = _run(_corrector(bench), retriever_iterations=[40], learning_steps=0)

    column, row = data.metadata["translations"][0][0]
    assert abs(row) < 0.1
    assert abs(column) == pytest.approx(1.5, abs=0.1)


def test_every_retrieval_runs_on_the_absolute_cost() -> None:
    """Every retrieval of a run takes the absolute intensity cost, and the retriever is
    left with the cost it had.
    """
    bench = _bench()
    corrector = _corrector(bench)
    retriever = corrector.phase_retriever
    retrieve = retriever.retrieve
    costs: list[str] = []

    def retrieve_and_note(*args, **kwargs):
        costs.append(type(retriever.loss_function).__name__)
        return retrieve(*args, **kwargs)

    retriever.retrieve = retrieve_and_note
    _run(corrector, retriever_iterations=[10] * 2, learning_steps=0)

    assert costs == ["LossAbsoluteIntensityMSE", "LossAbsoluteIntensityMSE"]
    assert retriever.loss_factory is None


def test_a_band_that_keeps_every_bin_leaves_the_simple_update() -> None:
    """The update of the simple feedback is the prediction plus the gain times the
    error, clipped at zero in the region.
    """
    generator = torch.Generator().manual_seed(0)
    corrected = torch.rand(CAMERA_RESOLUTION, generator=generator, dtype=torch.float64)
    frame = torch.rand(CAMERA_RESOLUTION, generator=generator, dtype=torch.float64)
    target = torch.rand(CAMERA_RESOLUTION, generator=generator, dtype=torch.float64)
    region = torch.zeros(CAMERA_RESOLUTION, dtype=torch.bool)
    region[16:48, 20:52] = True
    every_bin = torch.ones(CAMERA_RESOLUTION, dtype=torch.bool)

    step = corrected_target_intensity(
        corrected, frame, target, region, 0.7, band_mask=every_bin
    )
    simple = torch.clamp(corrected + 0.7 * (target - frame) * region, min=0.0) * region

    assert torch.allclose(step, simple)


def test_an_exposure_given_with_autoexposure_leaves_the_autoexposure() -> None:
    """The exposure argument holds while autoexposure is off. With autoexposure on, the
    frames are measured at the exposure the autoexposure found.
    """
    bench = _bench()
    longest = bench.camera.exposure_search_bounds[1]

    data = _run(
        _corrector(bench),
        retriever_iterations=[10],
        learning_steps=0,
        autoexpose=True,
        exposure=longest,
    )

    assert data.metadata["exposures"][0] < longest
    assert data.metadata["overexposed_iterations"] == []


def test_the_defocus_phase_propagates_the_pattern() -> None:
    """A defocus of minus one Rayleigh range moves the focus towards the SLM, so the
    camera's plane lies one Rayleigh range past the focus. The model's field of the
    defocused hologram is then its field in focus propagated forward over one Rayleigh
    range by the angular spectrum method, and not backward.

    The beam is narrow, so its focal spot of 142 um and the spot one Rayleigh range
    away both fit on the camera. One Rayleigh range either side of focus, the fields
    carry opposite curvatures, and their overlap is 1/sqrt(2).
    """
    beam_radius = 0.15e-3
    geometry = _geometry()
    slm = open_slm(SimulatedSLMTorch, input_geometry=geometry, bitdepth=8)
    model = _model(
        geometry,
        VirtualSLM.from_slm(slm),
        PixelwiseSLMField(_beam(geometry, beam_radius=beam_radius)),
    )
    spot_radius = WAVELENGTH * FOCAL_LENGTH / (np.pi * beam_radius)
    rayleigh_range = np.pi * spot_radius**2 / WAVELENGTH

    with torch.no_grad():
        in_focus = model()
        model.virtual_slm.set_phase(
            defocus_phase(
                *geometry.get_spatial_grid(),
                -rayleigh_range,
                FOCAL_LENGTH,
                2 * np.pi / WAVELENGTH,
            )
        )
        defocused = model().as_tensor().flatten()
        overlaps = []
        for distance in (rayleigh_range, -rayleigh_range):
            propagated = AngularSpectrumMethod(distance)(in_focus).as_tensor().flatten()
            overlap = (defocused.conj() * propagated).sum().abs()
            overlaps.append(
                float(overlap / (defocused.norm() * propagated.norm()))
            )

    assert overlaps[0] > 0.999
    assert overlaps[1] == pytest.approx(1 / np.sqrt(2), abs=0.02)


def test_the_defocused_frames_join_the_learning() -> None:
    """Each iteration adds its hologram in focus and both defocused holograms to the
    learning, and the SLM is left showing the hologram measured in focus.
    """
    bench = _bench()
    corrector = _corrector(bench)
    learn = corrector._learn
    samples: list[int] = []

    def learn_and_count(hologram_frames, *args, **kwargs):
        samples.append(len(hologram_frames))
        assert len(hologram_frames.frames) == len(hologram_frames.levels)
        return learn(hologram_frames, *args, **kwargs)

    corrector._learn = learn_and_count
    distance = 5e-3
    data = _run(
        corrector,
        retriever_iterations=[10] * 2,
        learning_steps=5,
        defocus_distances=(-distance, distance),
    )

    assert samples == [3, 6]
    assert data.metadata["defocus_distances"] == [-distance, distance]
    assert [len(values) for values in data.metadata["defocused_innovation"]] == [2, 2]
    assert np.array_equal(
        bench.slm.display, bench.slm.phase_to_levels(data.retrievals[-1].phase)
    )


# --- Random diversity and the exposure of the defocused frames ----------------------

DEFOCUS_DISTANCES = (-5e-3, 5e-3)


def _counted_autoexposures(corrector: ModelLearningFeedbackCorrector) -> list[int]:
    """Count the corrector's autoexposures on the signal region from here on."""
    autoexposures: list[int] = []
    autoexpose = corrector._autoexpose

    def autoexpose_and_count(sensor_region):
        autoexposures.append(1)
        return autoexpose(sensor_region)

    corrector._autoexpose = autoexpose_and_count
    return autoexposures


def _outside_the_band_up_to_a_tilt(phase: torch.Tensor, band: torch.Tensor) -> float:
    """The part of ``phase`` outside ``band`` that no tilt accounts for, as a fraction
    of the phase. A tilt is not band-limited on a finite grid, so the tilt's own part
    outside the band is fitted away first.
    """

    def outside(image: torch.Tensor) -> torch.Tensor:
        return ifft_2d(fft_2d(image + 0j) * ~band).real.flatten()

    grid_x, grid_y = get_pixel_grid(tuple(phase.shape), dtype=torch.float64)
    tilts = torch.stack((outside(grid_x), outside(grid_y)), dim=1)
    residual = outside(phase)
    fit = torch.linalg.solve(tilts.T @ tilts, tilts.T @ residual)
    return float((residual - tilts @ fit).norm() / phase.norm())


def test_random_diversity_adds_a_new_phase_in_its_band_to_every_defocused_hologram():
    """Every defocused hologram is its hologram plus its defocus plus a phase drawn
    anew. The phase lies in the band the run records, and its RMS over the model's
    beam is the defocus RMS.
    """
    bench = _bench()
    corrector = _corrector(bench)
    displayed: list[np.ndarray] = []
    display = bench.slm.set_phase

    def display_and_note(phase, *args, **kwargs):
        displayed.append(np.array(phase, dtype=np.float64))
        return display(phase, *args, **kwargs)

    bench.slm.set_phase = display_and_note
    data = _run(
        corrector,
        retriever_iterations=[10] * 2,
        learning_steps=0,
        defocus_distances=DEFOCUS_DISTANCES,
        random_diversity=True,
    )

    # Four displays per iteration, with the defocused holograms second and third.
    defocus = corrector._defocus_phases(DEFOCUS_DISTANCES)
    added = [
        displayed[4 * iteration + 1 + index] - displayed[4 * iteration] - defocus[index]
        for iteration in range(2)
        for index in range(2)
    ]
    slm_pitch = torch.tensor((SLM_PIXEL_SIZE, SLM_PIXEL_SIZE), dtype=torch.float64)
    frequency_x, frequency_y = get_frequency_grid(SLM_RESOLUTION, slm_pitch)
    band = torch.hypot(frequency_x, frequency_y) <= (
        2 * np.pi * data.metadata["random_diversity_band_radius"]
        / (WAVELENGTH * FOCAL_LENGTH)
    )
    weights = (bench.model.slm_field.get_wavefront().detach().abs() ** 2).double()
    weights = weights.reshape(SLM_RESOLUTION)

    def weighted_rms(phase: torch.Tensor) -> float:
        mean = (weights * phase).sum() / weights.sum()
        return float(torch.sqrt((weights * (phase - mean) ** 2).sum() / weights.sum()))

    rms = data.metadata["random_diversity_rms"]
    defocus_rms = max(
        weighted_rms(torch.as_tensor(phase, dtype=torch.float64)) for phase in defocus
    )
    assert data.metadata["random_diversity"] is True
    assert rms == pytest.approx(defocus_rms, rel=1e-12)
    for phase in added:
        phase = torch.as_tensor(phase)
        assert weighted_rms(phase) == pytest.approx(rms, rel=1e-9)
        assert _outside_the_band_up_to_a_tilt(phase, band) < 1e-9
    for first, second in itertools.combinations(added, 2):
        assert not np.allclose(first, second)


def _plateau_corrector(
    bench: SimpleNamespace, rows: slice, columns: slice
) -> ModelLearningFeedbackCorrector:
    """A corrector with the target's plateau on ``rows`` and ``columns`` of the
    21 x 21 patch. The identity mapping places the patch on whole sensor pixels.
    """
    patch = torch.full(PATCH_RESOLUTION, 0.3)
    patch[rows, columns] = 1.0
    return ModelLearningFeedbackCorrector(
        slm=bench.slm,
        camera=bench.camera,
        slm_camera_model=bench.model,
        target=patch,
        signal_region=torch.ones_like(patch),
        target_position=(18 * CAMERA_PIXEL_SIZE, 0.0),
        init_slm_phase=_init_phase(),
        loss_scale=LOSS_SCALE,
        camera_mapping=_identity_mapping(),
    )


def test_the_band_radius_is_the_gap_between_plateau_and_signal_region() -> None:
    """A plateau 6 pixels from the top and bottom of its window and 5 pixels from its
    sides gives 5 pixels of 20 um.
    """
    corrector = _plateau_corrector(_bench(), slice(6, 15), slice(5, 16))

    data = _run(
        corrector,
        retriever_iterations=[5],
        learning_steps=0,
        defocus_distances=DEFOCUS_DISTANCES,
        random_diversity=True,
    )

    assert data.metadata["random_diversity_band_radius"] == pytest.approx(
        5 * CAMERA_PIXEL_SIZE, rel=1e-12
    )


def test_a_plateau_nearer_the_edge_than_the_lowest_ripple_reaches_is_refused() -> None:
    """The lowest ripple of the 0.8 mm SLM deflects light by lambda f / 0.8 mm, 84 um,
    so a plateau 4 pixels of 20 um from the edge leaves the random phase no band.
    """
    corrector = _plateau_corrector(_bench(), slice(5, 16), slice(4, 17))

    with pytest.raises(ValueError, match="margin around the plateau"):
        _run(
            corrector,
            retriever_iterations=[5],
            learning_steps=0,
            defocus_distances=DEFOCUS_DISTANCES,
            random_diversity=True,
        )


def test_the_plateau_gap_is_measured_across_the_image_of_each_edge() -> None:
    """On a sheared camera, a column step moves 20 um along x and 5 um towards -y, and
    a row step 10 um towards -y. Rows of the sensor then lie 200 / hypot(20, 5) um
    apart in the image plane and columns 20 um apart, so 5 rows reach less far than 4
    columns.
    """
    patch = torch.full(PATCH_RESOLUTION, 0.3)
    patch[5:16, 4:17] = 1.0
    # Its columns are the image-plane (x, y) steps of one sensor column and one row.
    metres_per_pixel = np.array([[20e-6, 0.0], [-5e-6, -10e-6]])

    gap = _plateau_gap(patch, np.linalg.inv(metres_per_pixel))

    assert gap == pytest.approx(5 * 200e-12 / np.hypot(20e-6, 5e-6), rel=1e-12)


def test_the_plateau_gap_counts_rows_and_columns_at_their_own_pitch() -> None:
    """Through the identity mapping onto a model plane with a pitch of 10 um along y
    and 20 um along x, a sensor row is 10 um high and a sensor column 20 um wide. A
    plateau 6 rows from the top and bottom of its window and 4 columns from its sides
    then lies 60 um from the top and bottom edges and 80 um from the sides.
    """
    mapping = _identity_mapping(output_pixel_size=(10e-6, 20e-6))
    patch = torch.full(PATCH_RESOLUTION, 0.3)
    patch[6:15, 4:17] = 1.0

    gap = _plateau_gap(patch, mapping.sensor_pixels_per_metre)

    assert gap == pytest.approx(6 * 10e-6, rel=1e-12)


def test_random_diversity_needs_defocus_distances() -> None:
    bench = _bench()
    with pytest.raises(ValueError, match="needs defocus_distances"):
        _run(_corrector(bench), retriever_iterations=[5], random_diversity=True)


def test_the_defocused_frames_are_measured_at_the_exposure_in_focus() -> None:
    """An iteration autoexposes once, on its hologram in focus, and measures its
    defocused frames at that exposure.
    """
    bench = _bench()
    corrector = _corrector(bench)
    autoexposures = _counted_autoexposures(corrector)
    exposures: list[float] = []
    aligned_frame = corrector._aligned_frame

    def note_exposure(*args, **kwargs):
        exposures.append(float(bench.camera.get_exposure()))
        return aligned_frame(*args, **kwargs)

    corrector._aligned_frame = note_exposure
    data = _run(
        corrector,
        retriever_iterations=[10] * 2,
        learning_steps=0,
        defocus_distances=DEFOCUS_DISTANCES,
    )

    assert len(autoexposures) == 2
    assert exposures[0] == exposures[1] == exposures[2]
    assert exposures[3] == exposures[4] == exposures[5]
    assert data.metadata["reexposed_defocused_frames"] == []
    assert data.metadata["random_diversity"] is False


def test_an_overexposed_defocused_frame_is_autoexposed_and_measured_again() -> None:
    """The first defocused frame reports overexposure, so it is autoexposed and measured
    again. The next defocused frame starts from the exposure in focus.
    """
    bench = _bench()
    corrector = _corrector(bench)
    autoexposures = _counted_autoexposures(corrector)
    exposures: list[float] = []
    aligned_frame = corrector._aligned_frame

    def overexpose_the_first_defocused_frame(*args, **kwargs):
        exposures.append(float(bench.camera.get_exposure()))
        frame, fits, overexposed, captured = aligned_frame(*args, **kwargs)
        # The second measurement of the iteration is its first defocused frame.
        return frame, fits, overexposed or len(exposures) == 2, captured

    corrector._aligned_frame = overexpose_the_first_defocused_frame
    data = _run(
        corrector,
        retriever_iterations=[10],
        learning_steps=0,
        defocus_distances=DEFOCUS_DISTANCES,
    )

    # In focus, the first defocused frame twice, and the second defocused frame.
    assert len(exposures) == 4
    assert len(autoexposures) == 2
    assert exposures[2] != exposures[0]
    assert exposures[3] == exposures[0]
    assert data.metadata["reexposed_defocused_frames"] == [(0, 0)]
    assert data.metadata["overexposed_iterations"] == []


def test_a_pixelwise_field_is_not_learned() -> None:
    bench = _bench(psf_field=False)
    with pytest.raises(ValueError, match="PSFSLMField.from_calibration_data"):
        _run(_corrector(bench), retriever_iterations=[5], learning_steps=5)


def test_an_unknown_learning_method_is_refused() -> None:
    bench = _bench()
    with pytest.raises(ValueError, match="levenberg_marquardt, adam"):
        _run(_corrector(bench), retriever_iterations=[5], learning_method="sgd")


def test_unknown_parameter_names_are_refused() -> None:
    bench = _bench()
    with pytest.raises(ValueError, match="No parameter of the model matches"):
        _run(
            _corrector(bench),
            retriever_iterations=[5],
            learning_steps=5,
            learned_parameters=("no_such_module",),
        )
