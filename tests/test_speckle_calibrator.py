"""Smoke test for the speckle wavefront calibrator on a simulated setup.

Runs the full pipeline (generate band-limited phase patterns -> capture simulated
camera speckle -> fit the learnable SLM-plane field) end-to-end on a small
``SimulatedSLMTorch`` / ``SimulatedCameraTorch`` and checks it returns a valid
field-only ``WavefrontCalibrationData``. It is a run-only smoke test (no quantitative
recovery bar): the calibrator uses its default ``SLMNUFFT`` model against an
``SLMFFTAffine`` "hardware", a deliberate model mismatch.
"""

from __future__ import annotations

import shutil
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from hologradpy.hardware import SimulatedSLMTorch, SimulatedCameraTorch
from hologradpy.optics.complex_amplitude import (
    ComplexAmplitude,
    FieldGeometry,
)
from hologradpy.optics.systems import (
    SLMCZT,
    SLMFFTAffine,
    SLMNUFFT,
)
from hologradpy.optics.modules.slm_fields import (
    PSFSLMField,
    PixelwiseSLMField,
    kernel_size_from_waist,
    waist_from_camera_mapping,
)
from hologradpy.optics.modules.virtual_slms import VirtualSLM
from hologradpy.profiles.amplitude import gaussian_beam_intensity
from hologradpy.analysis.fitting import remove_tilt
from hologradpy.geometry import PartialAffineTransform
from hologradpy.calibration.camera_mapping import (
    CameraMapping,
    FocalSpotFit,
)
from hologradpy.calibration.wavefront.speckle_calibration import (
    PSFSpeckleCalibrator,
    PixelwiseSpeckleCalibrator,
    SpeckleCalibratorVisualizer,
    SpeckleVisualizationData,
)
from hologradpy.calibration.speckle.records import (
    SpeckleCaptureData,
)
from hologradpy.datasets import CaptureStore
from hologradpy.calibration.speckle.dataset_generator import (
    DatasetGenerator,
)
from hologradpy.loss_functions import (
    gradient_loss,
    normalize_to_unit_sum,
)
from hologradpy.calibration.wavefront.abstract import (
    WavefrontCalibrationData,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

DEVICE = torch.device("cpu")
SLM_RESOLUTION = (64, 64)
CAMERA_RESOLUTION = (48, 48)
CAMERA_PIXEL_SIZE = (30e-6, 30e-6)
FOCAL_LENGTH = 0.1


def _build_hardware():
    """A small simulated SLM + camera whose 'hardware' carries a known non-flat
    SLM-plane wavefront (gaussian amplitude + a mild quadratic phase).
    """
    geometry = FieldGeometry(
        resolution=SLM_RESOLUTION,
        pixel_size=torch.tensor([12.5e-6, 12.5e-6], device=DEVICE),
        wavelength=torch.tensor(0.63e-6, device=DEVICE),
    )
    slm = SimulatedSLMTorch(input_geometry=geometry, bitdepth=8)

    grid_x, grid_y = geometry.get_spatial_grid()
    amplitude = gaussian_beam_intensity(grid_x, grid_y, beam_radius=3e-4).sqrt()
    phase_bump = 4.0e5 * (grid_x**2 + grid_y**2)  # a mild defocus-like aberration
    beam = ComplexAmplitude(
        amplitude * torch.exp(1j * phase_bump),
        wavelength=geometry.wavelength,
        pixel_size=geometry.pixel_size,
        power=1e-3,
    )

    hardware = SLMFFTAffine(
        input_geometry=geometry,
        virtual_slm=slm.virtual_slm,
        camera_resolution=CAMERA_RESOLUTION,
        camera_pixel_size=CAMERA_PIXEL_SIZE,
        focal_length=FOCAL_LENGTH,
        slm_field=PixelwiseSLMField(beam),
        padded_resolution=(128, 128),
        camera_angle=0.0,
        camera_shift=(0.0, 0.0),
    )
    camera = SimulatedCameraTorch(hardware, noise_level=0.0)
    camera.set_exposure(1e-3)
    camera.get_image()
    return slm, camera


def _build_model(slm, camera, focal_length=FOCAL_LENGTH, slm_field=None) -> SLMCZT:
    """The model a calibrator fits, on the devices' own geometry.

    Shared by the speckle test modules, which all need one and none of which is about
    building it. The examples construct theirs inline, which is where a reader should
    look for how this is done on a real bench.

    One real dtype across the geometry, deliberately. An SLM reports its pixel size as
    numpy float64 and its wavelength as a plain float, so building each tensor from the
    value alone gives a geometry that is float64 on one axis and float32 on the next.
    The model then returns complex128 while its own parameters stay float32, and the
    only symptom is a backward pass refusing to run.
    """
    dtype = torch.get_default_dtype()
    geometry = FieldGeometry(
        resolution=tuple(slm.resolution),
        pixel_size=torch.tensor(tuple(slm.pixel_size), dtype=dtype, device=DEVICE),
        wavelength=torch.tensor(slm.wavelength, dtype=dtype, device=DEVICE),
    )

    return SLMCZT(
        input_geometry=geometry,
        virtual_slm=VirtualSLM.from_slm(slm),
        camera_resolution=tuple(camera.resolution),
        camera_pixel_size=tuple(camera.pixel_size),
        focal_length=focal_length,
        slm_field=PixelwiseSLMField() if slm_field is None else slm_field,
    )


def _synthetic_mapping(
    zeroth_order_position: tuple[float, float] | None = None,
) -> CameraMapping:
    """An identity camera -> model mapping with the zeroth order at the (square)
    camera center, so the affine seed is near identity.

    Args:
        zeroth_order_position: Where the zeroth order sits, ``(y, x)`` in camera pixels.
            Defaults to the sensor center. Pass one to model an off-axis camera.
    """
    truth = PartialAffineTransform.from_components(scale=1.0)
    detected = np.random.default_rng(0).uniform(-50, 50, size=(12, 2))
    calculated = truth.transform_points(detected)
    # zeroth_order_position is stored (y, x) = (row, col).
    center = zeroth_order_position or (
        CAMERA_RESOLUTION[0] / 2,
        CAMERA_RESOLUTION[1] / 2,
    )
    return CameraMapping(
        timestamp=datetime.now(),
        name="synthetic",
        transform=truth.as_matrix(homogeneous=False),
        detected_points=detected.tolist(),
        calculated_points=calculated.tolist(),
        zeroth_order_position=center,
        spot_fit=FocalSpotFit(waist=CAMERA_PIXEL_SIZE[0] * 2),
    )


def test_speckle_calibrator_runs_end_to_end(tmp_path) -> None:
    slm, camera = _build_hardware()
    mapping = _synthetic_mapping()

    calibrator = PixelwiseSpeckleCalibrator(
        slm=slm,
        camera=camera,
        camera_mapping=mapping,
        slm_camera_model=_build_model(
            slm, camera, FOCAL_LENGTH,
        ),
        dataset_path=tmp_path / "dataset.asdf",
        number_of_random_patterns=3,
    )

    result = calibrator.calibrate(
        speckle_pattern_extent=(5e-4, 5e-4),
        number_of_epochs=2,
        batch_size=1,
    )

    assert isinstance(result, WavefrontCalibrationData)
    assert result.complex_amplitude.resolution == SLM_RESOLUTION
    assert torch.isfinite(result.complex_amplitude.as_tensor()).all()
    assert "beam_radius" in result.metadata
    assert result.metadata["number_of_random_patterns"] == 3


def test_visualization_data_populated_and_visualizer_renders(tmp_path) -> None:
    """The calibration carries visualization data and the visualizer draws it.

    Follows the repo-wide VisualizationData pattern: the calibrator records the
    payload, the visualizer renders it, and the payload travels on the saved
    WavefrontCalibrationData.
    """
    slm, camera = _build_hardware()

    calibrator = PixelwiseSpeckleCalibrator(
        slm=slm,
        camera=camera,
        camera_mapping=_synthetic_mapping(),
        slm_camera_model=_build_model(
            slm, camera, FOCAL_LENGTH,
        ),
        dataset_path=tmp_path / "dataset.asdf",
        number_of_random_patterns=3,
    )
    result = calibrator.calibrate(
        speckle_pattern_extent=(5e-4, 5e-4), number_of_epochs=2, batch_size=1
    )

    data = result.visualization_data
    assert isinstance(data, SpeckleVisualizationData)
    assert data.camera_image.shape == CAMERA_RESOLUTION
    assert data.roi_mask.shape == CAMERA_RESOLUTION
    # The two ROI panels must be comparable, so they share a shape.
    assert data.measured_roi.shape == data.predicted_roi.shape
    assert data.recovered_phase.shape == SLM_RESOLUTION
    assert len(data.loss_history) == 2

    # Each term of the cost recorded alongside the total, which is what lets the
    # convergence panel show a prior climbing while the data term falls. They are the
    # same additions the total is, so they have to add up to it epoch by epoch.
    assert set(data.loss_component_history) == {
        "intensity mse",
        "phase smoothness",
        "amplitude smoothness",
    }
    for epoch, total in enumerate(data.loss_history):
        parts = [values[epoch] for values in data.loss_component_history.values()]
        assert len(parts) == 3
        assert sum(parts) == pytest.approx(total, rel=1e-6)


def test_a_dataset_can_be_inspected_before_a_fit_is_spent_on_it(tmp_path) -> None:
    """A capture is worth checking before paying for a fit: the region should sit on the
    speckle and the frame should be exposed rather than saturated. That needs a payload
    carrying only what a dataset holds, so none of the fitted panels can be required.
    """
    slm, camera = _build_hardware()
    calibrator = PixelwiseSpeckleCalibrator(
        slm=slm,
        camera=camera,
        camera_mapping=_synthetic_mapping(),
        slm_camera_model=_build_model(slm, camera, FOCAL_LENGTH),
        dataset_path=tmp_path / "dataset.asdf",
        number_of_random_patterns=3,
    )
    capture_data = calibrator.dataset_generator.generate_dataset((5e-4, 5e-4))

    data = calibrator.dataset_visualization_data(capture_data)

    assert data.slm_pattern.shape == SLM_RESOLUTION
    assert data.camera_image.shape == CAMERA_RESOLUTION
    # Nothing has been fitted, so none of these can be filled in yet.
    assert data.measured_roi is None
    assert data.recovered_phase is None
    assert not data.loss_history

    figure = data.visualizer().render_dataset()
    titles = [axs.get_title() for axs in figure.axes if axs.get_title()]
    # The panel says which units it drew, since a captured pattern is levels from a
    # device that quantizes and radians from one that does not.
    assert titles == ["SLM pattern [levels]", "camera + ROI"]
    plt.close(figure)


def test_asking_to_inspect_a_dataset_that_was_never_captured_says_so(tmp_path) -> None:
    """Rather than an attribute error deep in the loader."""
    slm, camera = _build_hardware()
    calibrator = PixelwiseSpeckleCalibrator(
        slm=slm,
        camera=camera,
        camera_mapping=_synthetic_mapping(),
        slm_camera_model=_build_model(slm, camera, FOCAL_LENGTH),
        dataset_path=tmp_path / "dataset.asdf",
        number_of_random_patterns=3,
    )

    with pytest.raises(RuntimeError, match="No dataset to show"):
        calibrator.dataset_visualization_data()


def _fitted_calibrator(tmp_path):
    """A calibrator that has captured and fitted, ready to extract a calibration.

    Split from the extraction so a test can extract more than once from one fit, which
    is both faster and the only way to vary an extraction argument in isolation.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    slm, camera = _build_hardware()
    calibrator = PixelwiseSpeckleCalibrator(
        slm=slm,
        camera=camera,
        camera_mapping=_synthetic_mapping(),
        slm_camera_model=_build_model(slm, camera, FOCAL_LENGTH),
        dataset_path=tmp_path / "dataset.asdf",
        number_of_random_patterns=3,
    )
    capture_data = calibrator.dataset_generator.generate_dataset((5e-4, 5e-4))
    calibrator.fit_wavefront(
        number_of_epochs=2,
        batch_size=1,
        capture_data=capture_data,
        verbose=False,
    )
    return calibrator


def _calibrate(tmp_path, **kwargs):
    """A short calibration on the shared smoke geometry."""
    return _fitted_calibrator(tmp_path).generate_slm_beam_calibration(**kwargs)


def test_a_simulated_bench_records_what_was_injected(tmp_path) -> None:
    """Only simulated hardware can say what the calibration was trying to recover, and
    recording it is what makes the comparison panels and the residual metrics possible
    without the caller keeping the truth to hand.
    """
    result = _calibrate(tmp_path)
    data = result.visualization_data

    assert data.injected_field is not None
    assert data.injected_field.shape == SLM_RESOLUTION
    assert np.iscomplexobj(data.injected_field)
    # The metrics come from the same truth, so a run is judgeable from the record alone.
    for key in ("residual_phase_rms", "residual_fraction"):
        assert key in result.metadata
        assert np.isfinite(result.metadata[key])


def test_the_comparison_mask_is_cut_from_the_injected_beam(tmp_path) -> None:
    """A mask cut from the recovery moves with the fit's own errors: a run that flattens
    the beam would be judged on a region its own mistake chose, and the panels would
    change shape between runs that ought to be comparable. One definition, so the figure
    and the reported numbers describe the same pixels.
    """
    result = _calibrate(tmp_path)
    data = result.visualization_data

    intensity = np.abs(data.injected_field) ** 2
    expected = intensity > result.metadata["beam_mask_threshold"] * intensity.max()

    assert data.beam_mask is not None
    assert np.array_equal(data.beam_mask, expected)


def test_a_real_camera_falls_back_to_the_recovered_beam(tmp_path, monkeypatch) -> None:
    """There is no truth to cut a mask from on a real bench, so the region reverts to
    the recovered beam, which is also the region tilt was fitted over.
    """
    monkeypatch.delattr(SimulatedCameraTorch, "static_slm_field")

    result = _calibrate(tmp_path)
    data = result.visualization_data

    intensity = np.abs(result.complex_amplitude.as_tensor().numpy()) ** 2
    intensity /= intensity.max()
    expected = intensity > result.metadata["beam_mask_threshold"]

    assert np.array_equal(data.beam_mask, expected)


def test_the_beam_mask_threshold_is_settable(tmp_path) -> None:
    """It was hard-coded at 0.005 under a standing TODO. Raising it must actually shrink
    the region, since it decides how much of the dim wings a calibration is judged on.

    One fit, two extractions, so the threshold is the only thing that differs.
    """
    calibrator = _fitted_calibrator(tmp_path)

    generous = calibrator.generate_slm_beam_calibration(beam_mask_threshold=0.0)
    # Taken from the beam the mask is now cut from, so it bites whatever that beam
    # actually looks like. A fixed value cannot, since a threshold below the peak of a
    # near-flat profile keeps the whole aperture.
    intensity = np.abs(generous.visualization_data.injected_field) ** 2
    intensity /= intensity.max()
    strict = calibrator.generate_slm_beam_calibration(
        beam_mask_threshold=float(np.median(intensity))
    )

    assert strict.metadata["beam_mask_threshold"] == pytest.approx(
        float(np.median(intensity))
    )
    assert strict.visualization_data.beam_mask.sum() < (
        generous.visualization_data.beam_mask.sum()
    )


def test_a_real_camera_leaves_the_comparison_out(tmp_path, monkeypatch) -> None:
    """A camera that cannot answer for the truth must not break the calibration, or the
    diagnostics figure. It simply gets no comparison, and asking for one says why.
    """
    # A real camera has no such attribute. Taking it off is the closest stand-in that
    # leaves every other behavior of the simulated bench identical.
    monkeypatch.delattr(SimulatedCameraTorch, "static_slm_field")

    result = _calibrate(tmp_path)

    data = result.visualization_data
    assert data.injected_field is None
    assert "residual_phase_rms" not in result.metadata
    # The diagnostics figure is unaffected.
    data.visualizer().render()
    with pytest.raises(RuntimeError, match="static_slm_field"):
        data.visualizer().render_comparison()

    figure = SpeckleCalibratorVisualizer(data).render()
    assert figure is not None


def test_dataset_survives_being_moved(tmp_path) -> None:
    """A captured dataset stays readable after it is moved.

    Nothing in the file points at where it lives, and what describes the capture is
    inside it, so one path is the whole dataset.
    """
    slm, camera = _build_hardware()

    original = tmp_path / "original"
    original.mkdir()
    calibrator = PixelwiseSpeckleCalibrator(
        slm=slm,
        camera=camera,
        camera_mapping=_synthetic_mapping(),
        slm_camera_model=_build_model(
            slm, camera, FOCAL_LENGTH,
        ),
        dataset_path=original / "dataset.asdf",
        number_of_random_patterns=2,
    )
    calibrator.dataset_generator.generate_dataset((5e-4, 5e-4))

    with CaptureStore.open(original / "dataset.asdf") as store:
        expected = store.read(0)

    moved = tmp_path / "moved"
    shutil.move(str(original), str(moved))

    with CaptureStore.open(moved / "dataset.asdf") as store:
        sample = store.read(0)
        capture_data = store.record()

    np.testing.assert_array_equal(sample["camera_image"], expected["camera_image"])
    np.testing.assert_array_equal(sample["slm_levels"], expected["slm_levels"])
    assert isinstance(capture_data, SpeckleCaptureData)


def test_fitting_without_a_dataset_fails_loudly(tmp_path) -> None:
    """Skipping the capture must name the missing step, not fail on a None attribute
    somewhere inside the region-of-interest helper.
    """
    slm, camera = _build_hardware()
    calibrator = PixelwiseSpeckleCalibrator(
        slm=slm,
        camera=camera,
        camera_mapping=_synthetic_mapping(),
        slm_camera_model=_build_model(slm, camera, FOCAL_LENGTH),
        dataset_path=tmp_path / "dataset.asdf",
        number_of_random_patterns=2,
    )

    with pytest.raises(RuntimeError, match="generate_dataset"):
        calibrator.fit_wavefront(number_of_epochs=1, batch_size=1)


def test_a_dataset_can_be_captured_once_and_refitted(tmp_path) -> None:
    """The two-phase workflow the removed wrapper used to serve: capture through the
    generator, then fit as many times as wanted without recapturing.
    """
    slm, camera = _build_hardware()
    calibrator = PixelwiseSpeckleCalibrator(
        slm=slm,
        camera=camera,
        camera_mapping=_synthetic_mapping(),
        slm_camera_model=_build_model(slm, camera, FOCAL_LENGTH),
        dataset_path=tmp_path / "dataset.asdf",
        number_of_random_patterns=3,
    )

    captured = calibrator.dataset_generator.generate_dataset((5e-4, 5e-4), seed=0)

    # batch_size equal to the dataset size puts every pattern in the single batch, so an
    # epoch loss is over the same data every time. At a smaller batch, drop_last leaves
    # one batch of a randomly drawn subset and the epoch loss depends on which patterns
    # it happened to draw, which makes the comparison below a coin flip.
    first = calibrator.fit_wavefront(
        number_of_epochs=3, batch_size=3, verbose=False, capture_data=captured
    )
    # The second call needs no capture_data: the first one is remembered.
    second = calibrator.fit_wavefront(number_of_epochs=3, batch_size=3, verbose=False)

    assert len(first) == 3 and len(second) == 3
    assert calibrator.capture_data is captured
    # Refitting continues from the field the first fit left rather than restarting, so
    # the second run opens below where the first one did.
    assert second[0] < first[0]


def test_a_coarse_mapping_is_run_when_none_is_supplied(tmp_path) -> None:
    """The mapping is what seeds the model's affine and places the region of interest,
    so a calibrator without one is useless. Rather than requiring the caller to run a
    CoarseMapper first, it runs one itself.

    Uses the coarse mapper's own setup: this repo's speckle fixtures are a 64x64 SLM
    onto a 48x48 sensor, which is too small for the probe spots to be found.
    """
    from .test_coarse_mapper import _build_setup

    slm, camera, _ = _build_setup(camera_angle=4.0, camera_shift=(-12, 7))

    calibrator = PixelwiseSpeckleCalibrator(
        slm=slm,
        camera=camera,
        slm_camera_model=SLMCZT(
            input_geometry=slm.input_geometry,
            virtual_slm=VirtualSLM.from_slm(slm),
            camera_resolution=tuple(camera.resolution),
            camera_pixel_size=tuple(float(pitch) for pitch in camera.pixel_size),
            focal_length=0.25,
            slm_field=PixelwiseSLMField(),
        ),
        dataset_path=tmp_path / "dataset.asdf",
        number_of_random_patterns=2,
    )

    assert isinstance(calibrator.camera_mapping, CameraMapping)
    # It recovered the rotation the simulated camera was built with, so the mapping is a
    # real measurement and not an identity placeholder. Negated because the mapping runs
    # camera -> model, the opposite of the angle the camera was built with.
    assert calibrator.camera_mapping.rotation_degrees == pytest.approx(-4.0, abs=0.5)
    # And the zeroth order landed somewhere on the sensor.
    row, column = calibrator.camera_mapping.zeroth_order_position
    assert 0 <= row < camera.resolution[0] and 0 <= column < camera.resolution[1]


def _plain_model(slm, camera, focal_length: float = 0.25) -> SLMCZT:
    """A model carrying nothing but an unmodulated field, as a caller would hand over
    without having decided how the SLM-plane field should be parameterized.
    """
    return SLMCZT(
        input_geometry=slm.input_geometry,
        virtual_slm=VirtualSLM.from_slm(slm),
        camera_resolution=tuple(camera.resolution),
        camera_pixel_size=tuple(float(pitch) for pitch in camera.pixel_size),
        focal_length=focal_length,
        slm_field=PixelwiseSLMField(),
    )


def test_the_calibrator_builds_the_field_it_fits_from_its_own_mapping(tmp_path) -> None:
    """The whole point of the arrangement.

    A PSF kernel's size follows from the fitted focal spot, so it cannot be chosen until
    a camera mapping exists, and the mapping needs a model. Building the field after the
    mapping rather than before breaks that loop: the caller hands over a plain model and
    the calibrator installs a kernel that is both sized and seeded from the measurement.
    """
    from .test_coarse_mapper import _build_setup

    slm, camera, _ = _build_setup()
    model = _plain_model(slm, camera)
    assert isinstance(model.slm_field, PixelwiseSLMField)

    calibrator = PSFSpeckleCalibrator(
        slm=slm,
        camera=camera,
        slm_camera_model=model,
        dataset_path=tmp_path / "dataset.asdf",
        number_of_random_patterns=2,
    )

    field = calibrator.slm_camera_model.slm_field
    assert isinstance(field, PSFSLMField)

    # Sized from the waist the mapping fitted, not from a guess.
    expected = kernel_size_from_waist(
        waist_from_camera_mapping(calibrator.camera_mapping),
        float(camera.pixel_size[1]),
    )
    assert field.psf_kernel_size == (expected, expected)
    assert tuple(field.get_psf_kernel().shape) == (expected, expected)

    # And seeded from the captured spot rather than the Gaussian fallback. lazy_init
    # normalizes by the peak, so compare against that.
    assert field.init_psf_kernel is not None, "nothing was measured"
    seed = field.init_psf_kernel.to(torch.complex64)
    seed = seed / seed.abs().max()
    assert torch.allclose(field.get_psf_kernel().detach(), seed, atol=1e-5)


def test_the_swap_leaves_no_ghost_of_the_replaced_field(tmp_path) -> None:
    """The old field must leave the model entirely, or the optimizer would carry
    parameters that no longer affect the forward pass.
    """
    from .test_coarse_mapper import _build_setup

    slm, camera, _ = _build_setup()
    calibrator = PSFSpeckleCalibrator(
        slm=slm,
        camera=camera,
        slm_camera_model=_plain_model(slm, camera),
        dataset_path=tmp_path / "dataset.asdf",
        number_of_random_patterns=2,
    )

    names = [name for name, _ in calibrator.slm_camera_model.named_parameters()]
    assert "slm_field.psf_kernel" in names
    assert not any(name.startswith("slm_field.amplitude") for name in names)
    assert not any(name.startswith("slm_field.phase") for name in names)

    # The replacement is what the forward pass actually runs, so gradients reach it.
    calibrator.slm_camera_model().as_tensor().abs().sum().backward()
    kernel = calibrator.slm_camera_model.slm_field.psf_kernel
    assert kernel.grad is not None and float(kernel.grad.abs().sum()) > 0


def test_a_supplied_field_of_the_right_type_is_kept(tmp_path) -> None:
    """Supplying one is how a fit is warm-started from an earlier calibration, so it
    must be used exactly as given rather than rebuilt from a fresh measurement.
    """
    from .test_coarse_mapper import _build_setup

    slm, camera, _ = _build_setup()
    psf_field = PSFSLMField(
        focal_length=0.25,
        camera_pixel_size=tuple(float(pitch) for pitch in camera.pixel_size),
        psf_kernel_size=11,
    )
    model = _plain_model(slm, camera)
    model.slm_field = psf_field

    calibrator = PSFSpeckleCalibrator(
        slm=slm,
        camera=camera,
        slm_camera_model=model,
        dataset_path=tmp_path / "dataset.asdf",
        number_of_random_patterns=2,
    )

    assert calibrator.slm_camera_model.slm_field is psf_field
    assert psf_field.psf_kernel_size == (11, 11)


def test_a_supplied_mapping_skips_the_coarse_mapping(tmp_path) -> None:
    """Passing one must use it as given, so a mapping saved from an earlier session is
    honoured rather than silently remeasured.
    """
    slm, camera = _build_hardware()
    mapping = _synthetic_mapping()

    calibrator = PixelwiseSpeckleCalibrator(
        slm=slm,
        camera=camera,
        slm_camera_model=_build_model(slm, camera, FOCAL_LENGTH),
        dataset_path=tmp_path / "dataset.asdf",
        camera_mapping=mapping,
        number_of_random_patterns=2,
    )

    assert calibrator.camera_mapping is mapping


def test_dataset_manifest_rejects_the_wrong_record_type(tmp_path) -> None:
    """The versioned envelope makes a mismatched file a clear error."""
    mapping = _synthetic_mapping()
    path = tmp_path / "mapping.asdf"
    mapping.save(path)

    with pytest.raises(TypeError, match="CameraMapping"):
        SpeckleCaptureData.load(path)


def test_capturing_before_generating_patterns_fails_loudly(tmp_path) -> None:
    """Capturing without a region-of-interest mask raises instead of metering the
    exposure against the whole sensor, zeroth order included.
    """
    slm, camera = _build_hardware()
    generator = DatasetGenerator(
        slm=slm,
        camera=camera,
        camera_mapping=_synthetic_mapping(),
        focal_length=FOCAL_LENGTH,
        dataset_path=tmp_path / "dataset.asdf",
        number_of_random_patterns=1,
    )

    with pytest.raises(RuntimeError, match="generate_phase_patterns"):
        generator.capture_camera_images()


def _generator_for(camera_mapping, slm, camera, tmp_path) -> DatasetGenerator:
    return DatasetGenerator(
        slm=slm,
        camera=camera,
        camera_mapping=camera_mapping,
        focal_length=FOCAL_LENGTH,
        dataset_path=tmp_path / "dataset.asdf",
        number_of_random_patterns=1,
    )


def test_the_speckle_extent_is_a_width_not_a_radius(tmp_path) -> None:
    """A requested extent must measure across the speckle, so it can be compared
    against the sensor size. Passing it through as a radius made every region twice the
    size asked for, which silently overran the sensor.
    """
    slm, camera = _build_hardware()
    generator = _generator_for(_synthetic_mapping(), slm, camera, tmp_path)

    extent = (6e-4, 6e-4)
    generator.generate_phase_patterns(extent, seed=0)

    # The zeroth-order exclusion punches out the middle, so measure the outside of the
    # region rather than counting the pixels in it.
    rows, columns = np.nonzero(generator.roi_mask)
    height = (rows.max() - rows.min() + 1) * camera.pixel_size[0]
    width = (columns.max() - columns.min() + 1) * camera.pixel_size[1]

    assert height == pytest.approx(extent[0], rel=0.1)
    assert width == pytest.approx(extent[1], rel=0.1)


def test_the_default_extent_is_the_largest_that_fits_an_off_axis_camera(
    tmp_path,
) -> None:
    """The speckle is centered on the zeroth order, not on the sensor, so the default
    has to be measured from the mapping. Assuming a centered beam puts a third of the
    region off the sensor, where it contributes nothing but skews the autoexposure.
    """
    slm, camera = _build_hardware()
    # A quarter of the way down, so the top edge is the nearest one.
    zeroth = (CAMERA_RESOLUTION[0] / 4, CAMERA_RESOLUTION[1] / 2)
    generator = _generator_for(_synthetic_mapping(zeroth), slm, camera, tmp_path)

    assert generator.largest_extent_on_sensor() == pytest.approx(
        (
            2 * zeroth[0] * camera.pixel_size[0],
            CAMERA_RESOLUTION[1] * camera.pixel_size[1],
        )
    )

    generator.generate_phase_patterns(seed=0)

    rows, columns = np.nonzero(generator.roi_mask)
    # Reaching the nearest edge, to within the pixel the grid convention costs, is what
    # makes it the largest one that fits, and it stops well short of the far edge.
    assert rows.min() <= 1
    assert rows.max() == pytest.approx(2 * zeroth[0], abs=2)
    assert rows.max() < CAMERA_RESOLUTION[0] - 1


def test_a_zeroth_order_off_the_sensor_refuses_to_pick_an_extent(tmp_path) -> None:
    """Coarse mapping extrapolates the zeroth order through the affine, so it can land
    off the sensor. Nothing centered there fits, and the arithmetic would otherwise hand
    back a negative width and an empty region.
    """
    slm, camera = _build_hardware()
    off_sensor = (-5.0, CAMERA_RESOLUTION[1] / 2)
    generator = _generator_for(_synthetic_mapping(off_sensor), slm, camera, tmp_path)

    with pytest.raises(ValueError, match="Pass an extent explicitly"):
        generator.largest_extent_on_sensor()


# The recovery test runs on its own, larger geometry. The smoke geometry above is
# deliberately tiny and cannot recover a wavefront at any smoothness weight (the
# residual aberration stays within a few percent of 1.0, meaning the calibration
# cancels nothing), so a recovery bar on it would only measure noise.
RECOVERY_SLM_RESOLUTION = (128, 128)
RECOVERY_CAMERA_RESOLUTION = (96, 96)
RECOVERY_CAMERA_PIXEL_SIZE = (20e-6, 20e-6)
RECOVERY_BEAM_RADIUS = 9e-4


def _recovery_mapping() -> CameraMapping:
    """Identity mapping for the recovery geometry, zeroth order at the center.

    The chirp-z model samples the camera grid natively, so identity is the right
    affine seed whatever the pixel pitch is.
    """
    truth = PartialAffineTransform.from_components(scale=1.0)
    detected = np.random.default_rng(0).uniform(-50, 50, size=(12, 2))
    calculated = truth.transform_points(detected)
    return CameraMapping(
        timestamp=datetime.now(),
        name="synthetic",
        transform=truth.as_matrix(homogeneous=False),
        detected_points=detected.tolist(),
        calculated_points=calculated.tolist(),
        zeroth_order_position=(
            RECOVERY_CAMERA_RESOLUTION[0] / 2,
            RECOVERY_CAMERA_RESOLUTION[1] / 2,
        ),
        spot_fit=FocalSpotFit(waist=RECOVERY_CAMERA_PIXEL_SIZE[0] * 2),
    )


def test_speckle_calibrator_recovers_injected_wavefront(tmp_path) -> None:
    """With developed speckle, the calibrator recovers a known SLM-plane aberration.

    The forward model matches the hardware, so with a properly sized speckle the
    recovered phase correlates with the injected one. Intensity-only speckle
    sensing recovers the wavefront up to a conjugate (global sign) ambiguity,
    since the two give identical intensities, so the metric is the absolute
    correlation over the illuminated region with piston and tilt removed.

    The dataset is seeded, so the patterns are reproducible rather than a lottery
    on whatever the generator happens to draw.
    """
    torch.manual_seed(0)
    np.random.seed(0)

    geometry = FieldGeometry(
        resolution=RECOVERY_SLM_RESOLUTION,
        pixel_size=torch.tensor([12.5e-6, 12.5e-6], device=DEVICE),
        wavelength=torch.tensor(0.63e-6, device=DEVICE),
    )
    slm = SimulatedSLMTorch(input_geometry=geometry, bitdepth=8)

    grid_x, grid_y = geometry.get_spatial_grid()
    amplitude = gaussian_beam_intensity(
        grid_x, grid_y, beam_radius=RECOVERY_BEAM_RADIUS
    ).sqrt()
    injected_phase = (grid_x**2 - grid_y**2) / RECOVERY_BEAM_RADIUS**2  # astigmatism

    beam = ComplexAmplitude(
        amplitude * torch.exp(1j * injected_phase),
        wavelength=geometry.wavelength,
        pixel_size=geometry.pixel_size,
        power=1e-3,
    )
    hardware = SLMNUFFT(
        input_geometry=geometry,
        virtual_slm=slm.virtual_slm,
        camera_resolution=RECOVERY_CAMERA_RESOLUTION,
        camera_pixel_size=RECOVERY_CAMERA_PIXEL_SIZE,
        focal_length=FOCAL_LENGTH,
        slm_field=PixelwiseSLMField(beam),
    )
    camera = SimulatedCameraTorch(hardware, noise_level=0.0)
    camera.set_exposure(1e-3)
    camera.get_image()

    calibrator = PixelwiseSpeckleCalibrator(
        slm=slm,
        camera=camera,
        camera_mapping=_recovery_mapping(),
        slm_camera_model=_build_model(
            slm, camera, FOCAL_LENGTH,
        ),
        dataset_path=tmp_path / "dataset.asdf",
        number_of_random_patterns=16,
    )
    result = calibrator.calibrate(
        speckle_pattern_extent=(1.5e-3, 1.5e-3),
        number_of_epochs=80,
        batch_size=4,
        seed=0,
        verbose=False,
    )

    recovered = result.complex_amplitude.as_tensor().detach().cpu().numpy()
    intensity = np.abs(recovered) ** 2
    mask = intensity > 0.1 * intensity.max()

    recovered_flat = remove_tilt(np.angle(recovered), mask=mask)
    injected_flat = remove_tilt(injected_phase.cpu().numpy(), mask=mask)
    a = recovered_flat[mask] - recovered_flat[mask].mean()
    b = injected_flat[mask] - injected_flat[mask].mean()
    correlation = float((a * b).sum() / np.sqrt((a**2).sum() * (b**2).sum()))

    # This bar used to sit at 0.12, because the fit only reached about 0.19 here
    # and the geometry was assumed too small to do better. It was not the
    # geometry: autoexposure was accepting a fully saturated frame as correctly
    # exposed, so the fit was being handed clipped speckle. With that fixed it
    # reaches 0.9994 to 0.9995 across seeds, so the bar can mean something.
    assert abs(correlation) > 0.95


def _quadratic_phase(number_of_pixels: int) -> torch.Tensor:
    """One fixed physical wavefront, sampled on a square grid of a given size."""
    axis = torch.linspace(-1.0, 1.0, number_of_pixels)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    return 1.3 * (x**2 + y**2) + 0.7 * x * y + 0.4 * x


def test_smoothness_penalty_is_resolution_agnostic() -> None:
    """One physical wavefront must give the same value at any SLM sampling.

    The raw per-pixel difference scales with the pitch, so the unscaled penalty
    falls by 4x per doubling of the resolution and a fixed weight would mean
    something different on every device.
    """
    values = [gradient_loss(_quadratic_phase(n)) for n in (256, 512, 1024)]
    for value in values[1:]:
        assert abs(value - values[0]) / values[0] < 0.02

    unscaled = [
        gradient_loss(_quadratic_phase(n), aperture_relative=False)
        for n in (256, 512, 1024)
    ]
    assert unscaled[0] / unscaled[1] == pytest.approx(4.0, rel=0.02)
    assert unscaled[1] / unscaled[2] == pytest.approx(4.0, rel=0.02)


def test_speckle_mismatch_is_region_size_agnostic() -> None:
    """The data term must not depend on how many pixels the region holds.

    Speckle intensity is exponentially distributed, so a unit-sum image puts
    about 1 / N in every pixel and the summed squared error falls off as 1 / N.
    Scaling by the pixel count has to cancel that at any region size.
    """
    generator = torch.Generator().manual_seed(0)
    region_sizes = (2_500, 25_000, 250_000)

    scaled, unscaled = [], []
    for number_of_pixels in region_sizes:
        measured = -torch.log(
            torch.rand(1, number_of_pixels, 1, generator=generator).clamp_min(1e-12)
        )
        predicted = (
            measured
            * (1 + 0.10 * torch.randn(1, number_of_pixels, 1, generator=generator))
        ).clamp_min(0.0)

        mismatch = float(
            (
                (normalize_to_unit_sum(predicted) - normalize_to_unit_sum(measured))
                ** 2
            ).sum()
        )
        scaled.append(number_of_pixels * mismatch)
        unscaled.append(mismatch)

    for value in scaled[1:]:
        assert abs(value - scaled[0]) / scaled[0] < 0.10

    # Without the scaling the term tracked the region size, not the mismatch.
    assert unscaled[0] / unscaled[-1] > 50


def test_unit_sum_normalization_is_invariant_to_image_scale() -> None:
    """Brightness must not move the data term, only the speckle shape.

    Exposure and the laser power set the overall scale of a frame, and neither
    is being fitted, so the loss has to ignore them.
    """
    generator = torch.Generator().manual_seed(1)
    images = torch.rand(3, 16, 16, generator=generator)
    scaled = images * torch.tensor([1.0, 25.0, 1e-3]).reshape(-1, 1, 1)

    assert torch.allclose(
        normalize_to_unit_sum(images), normalize_to_unit_sum(scaled), atol=1e-6
    )
    totals = normalize_to_unit_sum(scaled).sum(dim=(-2, -1))
    assert torch.allclose(totals, torch.ones(3), atol=1e-6)


def test_unit_sum_normalization_survives_an_all_zero_image() -> None:
    """An untrained field can put every photon in the excluded zeroth order."""
    result = normalize_to_unit_sum(torch.zeros(2, 8, 8))
    assert torch.isfinite(result).all()
    assert float(result.abs().max()) == 0.0


def test_phase_patterns_are_reproducible_from_a_seed() -> None:
    """A seeded dataset must be reproducible, and an unseeded one must vary.

    The patterns previously came from opensimplex, which keeps its seed in module
    state and defaults to seeding from time.time_ns(), so a dataset ignored numpy
    and torch seeding entirely. band_limited_random_phase takes a generator
    instead, so the caller owns reproducibility and there is no global to collide
    over.
    """
    from hologradpy.profiles.phase import band_limited_random_phase

    band = torch.zeros((16, 16), dtype=torch.bool)
    band[6:11, 6:11] = True

    def pattern(seed):
        return band_limited_random_phase(
            band, generator=torch.Generator().manual_seed(seed)
        )

    assert torch.equal(pattern(7), pattern(7))
    assert not torch.equal(pattern(7), pattern(8))

    # One generator drawn from repeatedly gives a different pattern each time,
    # which is what makes a dataset of distinct patterns from a single seed.
    generator = torch.Generator().manual_seed(7)
    assert not torch.equal(
        band_limited_random_phase(band, generator=generator),
        band_limited_random_phase(band, generator=generator),
    )


def test_phase_patterns_span_one_2pi_cycle_without_wrapping() -> None:
    """The SLM wraps its phase into one 2 pi cycle, so a pattern spanning more
    than that gains discontinuities it was never meant to have, and a
    discontinuity is broadband: it scatters light right out of the region the
    band limit was chosen to fill.
    """
    from hologradpy.profiles.phase import band_limited_random_phase

    rows, columns = np.indices((64, 64))
    band = torch.as_tensor(((rows - 32) ** 2 + (columns - 32) ** 2) < 10**2)

    phase = band_limited_random_phase(
        band, generator=torch.Generator().manual_seed(0)
    ).numpy()

    assert phase.min() >= 0.0
    assert np.isclose(np.ptp(phase), 2 * np.pi)
    steps = np.concatenate(
        [np.abs(np.diff(phase, axis=0)).ravel(), np.abs(np.diff(phase, axis=1)).ravel()]
    )
    assert steps.max() < np.pi
