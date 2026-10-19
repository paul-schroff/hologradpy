"""Smoke test for the speckle wavefront calibrator on a simulated setup.

Runs the full pipeline (generate band-limited phase patterns -> capture simulated
camera speckle -> fit the learnable SLM-plane field) end-to-end on a small
``SimulatedSLMTorch`` / ``SimulatedCameraTorch`` and checks it returns a valid
field-only ``WavefrontCalibrationData``. It is a run-only smoke test (no quantitative
recovery bar): the calibrator fits the ``SLMCZT`` model of ``_build_model`` against an
``SLMFFTAffine`` "hardware", a deliberate model mismatch.
"""

from __future__ import annotations

import dataclasses
import shutil
from datetime import datetime
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from hologradpy.hardware import SimulatedSLMTorch, SimulatedCameraTorch
from hologradpy.utils import as_image
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
)
from hologradpy.optics.modules.pixel_crosstalk import SuperGaussianCrosstalk
from hologradpy.optics.modules.virtual_slms import VirtualSLM
from hologradpy.profiles.amplitude import gaussian_beam_intensity
from hologradpy.profiles.gaussian_zernike import GaussianZernikeBeam
from hologradpy.profiles.phase import linear_phase
from hologradpy.grids import get_spatial_grid, odd_sample_count, plane_center
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
from hologradpy.calibration.wavefront.speckle_calibration.speckle_calibrator import (
    psf_calibrator,
)
from hologradpy.calibration import CrosstalkSpeckleCalibrator
from hologradpy.calibration.speckle.records import (
    SpeckleCaptureData,
)
from hologradpy.datasets import CaptureStore
from hologradpy.calibration.speckle.dataset_generator import (
    DatasetGenerator,
)
from hologradpy.roi import ROI
from hologradpy.holography.camera_feedback import SimpleFeedbackCorrector
from hologradpy.loss_functions import (
    gradient_loss,
    normalize_to_unit_sum,
)
from hologradpy.calibration.wavefront.abstract import (
    WavefrontCalibrationData,
)

from .registry import requires_nufft

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
    amplitude = gaussian_beam_intensity(grid_x, grid_y, beam_radius_major=3e-4).sqrt()
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
    camera = SimulatedCameraTorch(hardware, read_noise=0.0)
    camera.set_exposure(1e-3)
    camera.get_image()
    return slm, camera


def _build_model(
    slm, camera, focal_length=FOCAL_LENGTH, slm_field=None, pixel_crosstalk=None
) -> SLMCZT:
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
        virtual_slm=VirtualSLM.from_slm(slm, pixel_crosstalk=pixel_crosstalk),
        camera_resolution=tuple(camera.resolution),
        camera_pixel_size=tuple(camera.pixel_size),
        focal_length=focal_length,
        slm_field=PixelwiseSLMField() if slm_field is None else slm_field,
    )


def _synthetic_mapping(
    zeroth_order_position: tuple[float, float] | None = None,
    scale: float = 1.0,
    angle_deg: float = 0.0,
) -> CameraMapping:
    """A camera -> model mapping whose model plane is the camera grid, as in
    :func:`_build_model`. With the defaults it is the identity.

    The transform carries the zeroth order onto the centre of the model's output plane.

    Args:
        zeroth_order_position: Where the zeroth order sits, ``(column, row)`` in camera
            pixels. Defaults to the centre of the (square) sensor. Pass one to model an
            off-axis camera.
        scale: Model pixels per camera pixel.
        angle_deg: Rotation of the camera against the model plane, in degrees.
    """
    zeroth = zeroth_order_position or (
        CAMERA_RESOLUTION[1] / 2,
        CAMERA_RESOLUTION[0] / 2,
    )
    center_x, center_y = plane_center(CAMERA_RESOLUTION)
    truth = PartialAffineTransform.from_components(
        scale=scale,
        angle_deg=angle_deg,
        shift=(center_x - zeroth[0], center_y - zeroth[1]),
        center=(zeroth[0], zeroth[1]),
    )
    detected = np.random.default_rng(0).uniform(-50, 50, size=(12, 2))
    calculated = truth.transform_points(detected)
    return CameraMapping(
        timestamp=datetime.now(),
        name="synthetic",
        transform=truth.as_matrix(homogeneous=False),
        detected_points=detected.tolist(),
        calculated_points=calculated.tolist(),
        spot_fit=FocalSpotFit(waist=CAMERA_PIXEL_SIZE[0] * 2),
        output_pixel_size=CAMERA_PIXEL_SIZE,
        output_resolution=CAMERA_RESOLUTION,
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
    # The fitted beam is a record of its own, not metadata.
    assert isinstance(result.beam, GaussianZernikeBeam)
    assert "beam_radius" not in result.metadata
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
    calibrator.fit(
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

    intensity = np.abs(as_image(result.complex_amplitude).numpy()) ** 2
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
        calibrator.fit(number_of_epochs=1, batch_size=1)


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
    first = calibrator.fit(
        number_of_epochs=3, batch_size=3, verbose=False, capture_data=captured
    )
    # The second call needs no capture_data: the first one is remembered.
    second = calibrator.fit(number_of_epochs=3, batch_size=3, verbose=False)

    assert len(first) == 3 and len(second) == 3
    assert calibrator.capture_data is captured
    # Refitting continues from the field the first fit left rather than restarting, so
    # the second run opens below where the first one did.
    assert second[0] < first[0]


def test_a_spot_array_mapping_is_run_when_none_is_supplied(tmp_path) -> None:
    """The mapping seeds the model's affine and places the region of interest, so a
    calibrator needs one. When none is passed, the calibrator runs a SpotArrayMapper.
    Its spots share one hologram and so one pointing of the beam.

    The test runs on the setup of the coarse mapper tests. The speckle fixtures here put
    a 64x64 SLM onto a 48x48 sensor, which is too small for the spot array.
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
    assert calibrator.camera_mapping.name == "spot_array"
    # It recovered the rotation the simulated camera was built with, so the mapping is a
    # real measurement and not an identity placeholder. Negated because the mapping runs
    # camera -> model, the opposite of the angle the camera was built with.
    assert calibrator.camera_mapping.rotation_degrees == pytest.approx(-4.0, abs=0.5)
    # And the zeroth order landed somewhere on the sensor.
    column, row = calibrator.camera_mapping.zeroth_order_position
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
    """A PSF kernel's extent follows from the fitted focal spot, so it cannot be chosen
    until a camera mapping exists. The mapping in turn needs a model. The calibrator
    therefore builds the field after the mapping. The caller hands over a plain model,
    and the calibrator installs a kernel sized and seeded from the measurement.
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

    # Sampled at the intensity Nyquist pitch of the SLM, lambda f / (2 L) per axis.
    pitch = tuple(
        float(slm.wavelength) * 0.25 / (2 * count * float(slm.pixel_size[0]))
        for count in slm.resolution
    )
    assert field.psf_pixel_size == pytest.approx(pitch, rel=1e-6)
    # Sized from the waist the mapping fitted, ten waists by default.
    extent = 10 * float(calibrator.camera_mapping.spot_fit.waist)
    expected = tuple(odd_sample_count(extent, axis_pitch) for axis_pitch in pitch)
    assert field.psf_kernel_size == expected
    assert tuple(field.get_psf_kernel().shape) == expected

    # And seeded from the captured spot, resampled onto the model's pixels at the
    # mapping's output pitch. The kernel fits the SLM-plane field of the spot closely.
    assert field.init_psf_kernel is not None, "nothing was measured"
    assert field.init_psf_kernel_pixel_size == pytest.approx(
        calibrator.camera_mapping.output_pixel_size
    )
    assert field.wavefront_fit_residual < 1e-2


def test_the_psf_kernel_spans_the_requested_waists(tmp_path) -> None:
    """The default kernel is several times wider than a well-corrected spot, and the
    margin only fits noise. The extent is therefore the caller's to choose.
    """
    from .test_coarse_mapper import _build_setup

    slm, camera, _ = _build_setup()
    calibrator = PSFSpeckleCalibrator(
        slm=slm,
        camera=camera,
        slm_camera_model=_plain_model(slm, camera),
        dataset_path=tmp_path / "dataset.asdf",
        number_of_random_patterns=2,
        psf_kernel_waists=5.0,
    )

    field = calibrator.slm_camera_model.slm_field
    extent = 5.0 * float(calibrator.camera_mapping.spot_fit.waist)
    expected = tuple(
        odd_sample_count(extent, axis_pitch) for axis_pitch in field.psf_pixel_size
    )
    assert field.psf_kernel_size == expected
    assert tuple(field.get_psf_kernel().shape) == expected
    # The captured spot covers the same extent on the camera.
    side = odd_sample_count(extent, float(camera.pixel_size[1]))
    assert tuple(field.init_psf_kernel.shape) == (side, side)


@pytest.mark.parametrize("scale", [2.0, 0.5])
def test_a_spot_from_a_turned_camera_seeds_the_kernel_in_the_model_orientation(
    tmp_path, monkeypatch, scale
) -> None:
    """The captured spot is a crop of the sensor, so it is turned and scaled like the
    camera. The calibrator resamples it onto the model's pixels
    (:meth:`~hologradpy.calibration.camera_mapping.CameraMapping.sensor_crop_to_model`),
    and the seed then sits at the pitch of the mapping's output plane.

    The camera is turned by 30 degrees, and the model samples the focal plane at half
    or at twice the camera pitch. The crop side is counted at the finer of the two
    pitches, so both the camera crop and the resampled seed span the kernel's extent of
    ten waists.
    """
    slm, camera = _build_hardware()
    output_pixel_size = (CAMERA_PIXEL_SIZE[0] / scale, CAMERA_PIXEL_SIZE[1] / scale)
    mapping = dataclasses.replace(
        _synthetic_mapping(scale=scale, angle_deg=30.0),
        output_pixel_size=output_pixel_size,
    )
    model = SLMCZT(
        input_geometry=slm.input_geometry,
        virtual_slm=VirtualSLM.from_slm(slm),
        camera_resolution=CAMERA_RESOLUTION,
        camera_pixel_size=mapping.output_pixel_size,
        focal_length=FOCAL_LENGTH,
        slm_field=PixelwiseSLMField(),
    )
    crops = []

    def capture(slm, camera, camera_mapping, slm_camera_model, kernel_size):
        # An asymmetric crop, which a turn or a mirror changes.
        crop = np.random.default_rng(0).uniform(size=(kernel_size, kernel_size))
        crops.append(crop)
        return crop

    monkeypatch.setattr(psf_calibrator, "capture_focal_spot", capture)
    calibrator = PSFSpeckleCalibrator(
        slm=slm,
        camera=camera,
        slm_camera_model=model,
        dataset_path=tmp_path / "dataset.asdf",
        camera_mapping=mapping,
        number_of_random_patterns=2,
    )

    side = odd_sample_count(
        10 * float(mapping.spot_fit.waist),
        min(CAMERA_PIXEL_SIZE[1], output_pixel_size[1]),
    )
    assert [crop.shape for crop in crops] == [(side, side)]
    field = calibrator.slm_camera_model.slm_field
    torch.testing.assert_close(
        field.init_psf_kernel,
        torch.as_tensor(mapping.sensor_crop_to_model(crops[0]), dtype=torch.float32),
    )
    assert field.init_psf_kernel_pixel_size == pytest.approx(mapping.output_pixel_size)


def test_a_non_positive_psf_kernel_is_refused(tmp_path) -> None:
    slm, camera = _build_hardware()
    with pytest.raises(ValueError, match="must be positive"):
        PSFSpeckleCalibrator(
            slm=slm,
            camera=camera,
            slm_camera_model=_build_model(slm, camera),
            dataset_path=tmp_path / "dataset.asdf",
            camera_mapping=_synthetic_mapping(),
            psf_kernel_waists=0.0,
        )


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
    psf_field = PSFSLMField(focal_length=0.25, psf_extent=2e-4)
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
    assert psf_field.psf_extent == (2e-4, 2e-4)
    assert psf_field.init_psf_kernel is None


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


def _refuse_to_map(*arguments, **options):
    raise AssertionError("a camera mapping was measured although one was supplied")


def test_the_wavefront_fit_holds_the_partial_affine(tmp_path) -> None:
    """The focal-plane partial affine is calibrated from the camera mapping alone, so
    the wavefront fit leaves it frozen, where the mapping put it.
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
    lens = calibrator.slm_camera_model.fourier_lens
    calibrated = {
        name: value.detach().clone() for name, value in lens.state_dict().items()
    }

    capture_data = calibrator.dataset_generator.generate_dataset((5e-4, 5e-4))
    calibrator.fit(
        number_of_epochs=2, batch_size=1, capture_data=capture_data, verbose=False
    )

    for name, value in lens.state_dict().items():
        assert torch.equal(value, calibrated[name]), name
    assert not any(parameter.requires_grad for parameter in lens.parameters())


def test_the_crosstalk_fit_holds_the_partial_affine(tmp_path) -> None:
    """The crosstalk fit also leaves the focal-plane affine frozen, where the camera
    mapping put it.
    """
    slm, camera = _build_hardware()
    model = _build_model(
        slm, camera, pixel_crosstalk=SuperGaussianCrosstalk(upscale_factor=3)
    )
    calibrator = CrosstalkSpeckleCalibrator(
        slm=slm,
        camera=camera,
        camera_mapping=_synthetic_mapping(),
        slm_camera_model=model,
        dataset_path=tmp_path / "dataset.asdf",
        number_of_random_patterns=3,
    )
    lens = model.fourier_lens
    calibrated = {
        name: value.detach().clone() for name, value in lens.state_dict().items()
    }

    capture_data = calibrator.dataset_generator.generate_dataset((5e-4, 5e-4))
    calibrator.fit(
        number_of_epochs=2, batch_size=1, capture_data=capture_data, verbose=False
    )

    for name, value in lens.state_dict().items():
        assert torch.equal(value, calibrated[name]), name
    assert not any(parameter.requires_grad for parameter in lens.parameters())


def test_a_crosstalk_calibration_draws_its_patterns_per_pixel(
    tmp_path, monkeypatch
) -> None:
    """The crosstalk patterns are drawn one pixel at a time."""
    slm, camera = _build_hardware()
    model = _build_model(
        slm, camera, pixel_crosstalk=SuperGaussianCrosstalk(upscale_factor=3)
    )
    calibrator = CrosstalkSpeckleCalibrator(
        slm=slm,
        camera=camera,
        camera_mapping=_synthetic_mapping(),
        slm_camera_model=model,
        dataset_path=tmp_path / "dataset.asdf",
        number_of_random_patterns=1,
    )
    monkeypatch.setattr(calibrator, "fit", lambda **_: None)
    monkeypatch.setattr(calibrator, "generate_crosstalk_calibration", lambda: None)

    calibrator.calibrate(seed=0, verbose=False)

    generator = calibrator.dataset_generator
    assert generator.phase_pattern_type == "uniform"
    assert generator.metadata["speckle_extent"] is None


def test_feedback_keeps_the_partial_affine_of_a_calibration(
    tmp_path, monkeypatch
) -> None:
    """A PSF calibration sets the focal-plane partial affine of the model from its
    camera mapping. Camera feedback handed that mapping leaves the affine where it was.

    The camera is rotated, so an affine that differs from the calibration's shows in
    the angle.
    """
    from .test_coarse_mapper import _build_setup

    slm, camera, _ = _build_setup(camera_angle=4.0, camera_shift=(-12, 7))
    calibrator = PSFSpeckleCalibrator(
        slm=slm,
        camera=camera,
        slm_camera_model=_plain_model(slm, camera),
        dataset_path=tmp_path / "dataset.asdf",
        number_of_random_patterns=2,
    )
    model = calibrator.slm_camera_model
    calibrated = {
        name: value.detach().clone()
        for name, value in model.fourier_lens.state_dict().items()
    }
    monkeypatch.setattr(
        "hologradpy.holography.camera_feedback.abstract.SpotArrayMapper",
        _refuse_to_map,
    )

    feedback = SimpleFeedbackCorrector(
        slm=slm,
        camera=camera,
        slm_camera_model=model,
        target=torch.ones(8, 8),
        target_position=(6e-4, 0.0),
        camera_mapping=calibrator.camera_mapping,
    )
    feedback.apply_camera_mapping()

    for name, value in model.fourier_lens.state_dict().items():
        assert torch.equal(value, calibrated[name]), name


def test_dataset_manifest_rejects_the_wrong_record_type(tmp_path) -> None:
    """The versioned envelope makes a mismatched file a clear error."""
    mapping = _synthetic_mapping()
    path = tmp_path / "mapping.asdf"
    mapping.save(path)

    with pytest.raises(TypeError, match="CameraMapping"):
        SpeckleCaptureData.load(path)


def test_capturing_before_generating_patterns_fails_loudly(tmp_path) -> None:
    """Capturing raises before a region-of-interest mask exists, since the autoexposure
    measures inside the mask.
    """
    slm, camera = _build_hardware()
    generator = DatasetGenerator(
        slm=slm,
        camera=camera,
        camera_mapping=_synthetic_mapping(),
        slm_camera_model=_build_model(slm, camera),
        dataset_path=tmp_path / "dataset.asdf",
        number_of_random_patterns=1,
    )

    with pytest.raises(RuntimeError, match="generate_phase_patterns"):
        generator.capture_camera_images()


def _generator_for(
    camera_mapping, slm, camera, tmp_path, **options
) -> DatasetGenerator:
    return DatasetGenerator(
        slm=slm,
        camera=camera,
        camera_mapping=camera_mapping,
        slm_camera_model=_build_model(slm, camera),
        dataset_path=tmp_path / "dataset.asdf",
        number_of_random_patterns=1,
        **options,
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


def test_the_largest_extent_on_the_zeroth_order_fits_an_off_axis_camera(
    tmp_path,
) -> None:
    """A speckle kept on the zeroth order is centred there, not on the sensor, so its
    largest extent is measured from the zeroth order to the nearest sensor edge.
    """
    slm, camera = _build_hardware()
    # A quarter of the way down, so the top edge is the nearest one.
    zeroth = (CAMERA_RESOLUTION[1] / 2, CAMERA_RESOLUTION[0] / 4)
    generator = _generator_for(_synthetic_mapping(zeroth), slm, camera, tmp_path)

    assert generator.largest_extent_on_sensor(tilt=(0.0, 0.0)) == pytest.approx(
        (
            2 * zeroth[1] * camera.pixel_size[0],
            CAMERA_RESOLUTION[1] * camera.pixel_size[1],
        )
    )

    generator.generate_phase_patterns(tilt=(0.0, 0.0), seed=0)

    rows, columns = np.nonzero(generator.roi_mask)
    # Reaching the nearest edge, to within the pixel the grid convention costs, is what
    # makes it the largest one that fits, and it stops well short of the far edge.
    assert rows.min() <= 1
    assert rows.max() == pytest.approx(2 * zeroth[1], abs=2)
    assert rows.max() < CAMERA_RESOLUTION[0] - 1


@pytest.mark.parametrize("waists", [4.0, 8.0])
def test_the_zeroth_order_mask_is_sized_in_waists(tmp_path, waists) -> None:
    """The model predicts none of the undiffracted light, and on a real SLM its wings
    reach well past a few waists, so the excluded disc has to be sizable on request.
    """
    slm, camera = _build_hardware()
    mapping = _synthetic_mapping()
    generator = DatasetGenerator(
        slm=slm,
        camera=camera,
        camera_mapping=mapping,
        slm_camera_model=_build_model(slm, camera),
        dataset_path=tmp_path / "dataset.asdf",
        number_of_random_patterns=1,
        zeroth_order_mask_waists=waists,
    )
    generator.generate_phase_patterns(seed=0)

    radius = waists * mapping.spot_fit.waist
    radius_pixels = radius / camera.pixel_size[0]
    rows, columns = np.indices(generator.roi_mask.shape)
    distance = np.hypot(
        rows - mapping.zeroth_order_position[1],
        columns - mapping.zeroth_order_position[0],
    )
    # A pixel of slack either side for the grid convention.
    assert not generator.roi_mask[distance < radius_pixels - 1].any()
    assert generator.roi_mask[
        (distance > radius_pixels + 1) & (distance < radius_pixels + 3)
    ].all()
    assert generator.metadata["zeroth_order_mask_radius"] == pytest.approx(radius)


def test_the_calibrator_passes_the_zeroth_order_mask_to_its_generator(
    tmp_path,
) -> None:
    slm, camera = _build_hardware()
    calibrator = PixelwiseSpeckleCalibrator(
        slm=slm,
        camera=camera,
        camera_mapping=_synthetic_mapping(),
        slm_camera_model=_build_model(slm, camera, FOCAL_LENGTH),
        dataset_path=tmp_path / "dataset.asdf",
        number_of_random_patterns=1,
        zeroth_order_mask_waists=10.0,
    )

    assert calibrator.dataset_generator.zeroth_order_mask_waists == 10.0


def test_a_negative_zeroth_order_mask_is_refused(tmp_path) -> None:
    slm, camera = _build_hardware()
    with pytest.raises(ValueError, match="cannot be negative"):
        DatasetGenerator(
            slm=slm,
            camera=camera,
            camera_mapping=_synthetic_mapping(),
            slm_camera_model=_build_model(slm, camera),
            dataset_path=tmp_path / "dataset.asdf",
            zeroth_order_mask_waists=-1.0,
        )


def test_a_zeroth_order_off_the_sensor_refuses_an_extent_around_it(tmp_path) -> None:
    """Coarse mapping extrapolates the zeroth order through the affine, so it can land
    off the sensor. No speckle centred there fits on the sensor.
    """
    slm, camera = _build_hardware()
    off_sensor = (CAMERA_RESOLUTION[1] / 2, -5.0)
    generator = _generator_for(_synthetic_mapping(off_sensor), slm, camera, tmp_path)

    with pytest.raises(ValueError, match="Pass an extent explicitly"):
        generator.largest_extent_on_sensor(tilt=(0.0, 0.0))


# --- Steering the speckle with a tilt -----------------------------------------------


@pytest.mark.parametrize(
    "zeroth",
    [
        (CAMERA_RESOLUTION[1] / 2, CAMERA_RESOLUTION[0] / 2),
        (CAMERA_RESOLUTION[1] / 2, CAMERA_RESOLUTION[0] / 4),
        (CAMERA_RESOLUTION[1] / 2, -5.0),
    ],
    ids=["centre", "near an edge", "off the sensor"],
)
def test_the_default_steers_the_speckle_to_the_sensor_centre(tmp_path, zeroth) -> None:
    """By default the speckle is steered to the sensor centre, so its region spans the
    whole sensor wherever the zeroth order sits, off the sensor included.
    """
    slm, camera = _build_hardware()
    mapping = _synthetic_mapping(zeroth)
    generator = _generator_for(mapping, slm, camera, tmp_path)

    generator.generate_phase_patterns(seed=0, verbose=False)

    assert generator.metadata["speckle_tilt"] == pytest.approx(
        mapping.tilt_to_sensor_center(camera)
    )
    assert generator.metadata["speckle_extent"] == pytest.approx(
        tuple(n * pitch for n, pitch in zip(CAMERA_RESOLUTION, CAMERA_PIXEL_SIZE))
    )
    rows, columns = np.nonzero(generator.roi_mask)
    # The zeroth-order disc can bite into the top edge, so the other three edges
    # show the reach, to within the pixel the grid convention costs.
    assert columns.min() <= 1
    assert columns.max() >= CAMERA_RESOLUTION[1] - 2
    assert rows.max() >= CAMERA_RESOLUTION[0] - 2


def test_a_tilt_adds_its_linear_phase_to_every_pattern(tmp_path) -> None:
    slm, camera = _build_hardware()
    tilt = (4e-4, -2e-4)
    centred = _generator_for(_synthetic_mapping(), slm, camera, tmp_path)
    tilted = _generator_for(_synthetic_mapping(), slm, camera, tmp_path)

    centred.generate_phase_patterns(
        (5e-4, 5e-4), tilt=(0.0, 0.0), seed=0, verbose=False
    )
    tilted.generate_phase_patterns((5e-4, 5e-4), tilt=tilt, seed=0, verbose=False)

    slm_grid = get_spatial_grid(
        SLM_RESOLUTION, torch.tensor(tuple(slm.pixel_size), dtype=torch.float64)
    )
    ramp = linear_phase(
        *slm_grid,
        *tilt,
        tilt_units="metres",
        wavenumber=2 * np.pi / slm.wavelength,
        focal_length=FOCAL_LENGTH,
    ).numpy()
    np.testing.assert_allclose(
        np.exp(1j * (tilted.phase_patterns[0] - centred.phase_patterns[0])),
        np.exp(1j * ramp),
        atol=1e-9,
    )
    assert tilted.metadata["speckle_tilt"] == tilt


def test_a_tilted_speckle_lands_on_its_region_of_interest(tmp_path) -> None:
    """The linear phase and the region of interest agree on where the speckle goes.
    The light falls inside the region, and the region mirrored through the zeroth
    order, where the opposite tilt puts it, stays dark.
    """
    slm, camera = _build_hardware()
    tilt = (6e-4, 0.0)
    generator = _generator_for(_synthetic_mapping(), slm, camera, tmp_path)
    mirrored = _generator_for(_synthetic_mapping(), slm, camera, tmp_path)

    generator.generate_dataset((1.2e-3, 1.2e-3), tilt=tilt, seed=0)
    mirrored.generate_phase_patterns(
        (1.2e-3, 1.2e-3), tilt=(-tilt[0], -tilt[1]), seed=0, verbose=False
    )

    with CaptureStore.open(generator.dataset_path) as store:
        frame = np.asarray(store.read(0)["camera_image"], dtype=np.float64)
    inside = frame[generator.roi_mask].mean()
    opposite = frame[mirrored.roi_mask].mean()
    # A phase-only pattern without crosstalk sends only the light its quantization
    # scatters into the mirrored order.
    assert inside > 20 * opposite


def test_a_tilted_region_keeps_the_speckle_centre(tmp_path) -> None:
    """The patterns form no undiffracted focal spot, so the tilt carries none to the
    speckle centre.
    """
    slm, camera = _build_hardware()
    mapping = _synthetic_mapping()
    generator = _generator_for(mapping, slm, camera, tmp_path)
    tilt = (6e-4, 0.0)

    generator.generate_phase_patterns(
        (1.2e-3, 1.2e-3), tilt=tilt, seed=0, verbose=False
    )

    radius_pixels = (
        generator.metadata["zeroth_order_mask_radius"] / camera.pixel_size[0]
    )
    centre_column, centre_row = mapping.image_plane_to_sensor(tilt)[0]
    rows, columns = np.indices(generator.roi_mask.shape)
    distance = np.hypot(columns - centre_column, rows - centre_row)
    assert generator.roi_mask[distance < radius_pixels].all()


def test_a_speckle_inside_the_zeroth_order_disc_is_refused(tmp_path) -> None:
    slm, camera = _build_hardware()
    generator = _generator_for(_synthetic_mapping(), slm, camera, tmp_path)

    with pytest.raises(ValueError, match="region of interest is empty"):
        generator.generate_phase_patterns(
            (2e-4, 2e-4), tilt=(0.0, 0.0), seed=0, verbose=False
        )


def test_a_magnified_mapping_scales_the_extent_and_keeps_the_region(tmp_path) -> None:
    """The extent is measured in the image plane. A mapping that puts two model pixels
    on every sensor pixel doubles the largest extent and keeps its region on the same
    sensor pixels.
    """
    slm, camera = _build_hardware()
    # A zeroth-order disc of 8.5 px. Its edge stays 0.015 px or more from every pixel,
    # so the 3e-7 px rounding of the magnified mapping's zeroth order moves no pixel
    # across it.
    plain = _generator_for(
        _synthetic_mapping(), slm, camera, tmp_path, zeroth_order_mask_waists=4.25
    )
    magnified = _generator_for(
        _synthetic_mapping(scale=2.0),
        slm,
        camera,
        tmp_path,
        zeroth_order_mask_waists=4.25,
    )

    assert magnified.largest_extent_on_sensor() == pytest.approx(
        tuple(2 * width for width in plain.largest_extent_on_sensor())
    )

    plain.generate_phase_patterns(seed=0, verbose=False)
    magnified.generate_phase_patterns(seed=0, verbose=False)

    np.testing.assert_array_equal(magnified.roi_mask, plain.roi_mask)


def test_the_largest_speckle_on_a_rotated_camera_touches_the_sensor_edge(
    tmp_path,
) -> None:
    """The image of the speckle ellipse turns with the camera, so on a rotated camera
    the ellipse shrinks until its image reaches the nearest sensor edge, and no
    further.
    """
    slm, camera = _build_hardware()
    zeroth = (CAMERA_RESOLUTION[1] / 2, CAMERA_RESOLUTION[0] / 4)
    aligned = _generator_for(_synthetic_mapping(zeroth), slm, camera, tmp_path)
    mapping = _synthetic_mapping(zeroth, angle_deg=30.0)
    rotated = _generator_for(mapping, slm, camera, tmp_path)

    # Kept on the zeroth order, a quarter of the way down, so the ellipse is twice as
    # wide as it is tall and its turn matters.
    height, width = rotated.largest_extent_on_sensor(tilt=(0.0, 0.0))

    angles = np.linspace(0.0, 2 * np.pi, 721)
    boundary = mapping.image_plane_to_sensor(
        np.stack([width / 2 * np.cos(angles), height / 2 * np.sin(angles)], axis=1)
    )
    sensor_size = np.asarray(CAMERA_RESOLUTION)[::-1]  # (columns, rows)
    clearance = np.minimum(boundary, sensor_size - boundary)
    assert clearance.min() == pytest.approx(0.0, abs=1e-3)
    aligned_height, aligned_width = aligned.largest_extent_on_sensor(tilt=(0.0, 0.0))
    assert height < aligned_height
    assert width < aligned_width


def test_a_tilt_leaves_uniform_patterns_unchanged(tmp_path) -> None:
    """A uniform pattern fills the whole addressable field, so a tilt has nothing to
    move.
    """
    slm, camera = _build_hardware()
    centred = _generator_for(_synthetic_mapping(), slm, camera, tmp_path)
    tilted = _generator_for(_synthetic_mapping(), slm, camera, tmp_path)

    centred.generate_phase_patterns(seed=0, verbose=False, pattern="uniform")
    tilted.generate_phase_patterns(
        tilt=(4e-4, 0.0), seed=0, verbose=False, pattern="uniform"
    )

    np.testing.assert_array_equal(tilted.phase_patterns[0], centred.phase_patterns[0])
    assert tilted.metadata["speckle_tilt"] is None


def test_a_speckle_past_the_addressable_field_is_refused(tmp_path) -> None:
    """Past ``wavelength * focal_length / (2 * pixel_size)`` from the zeroth order the
    band of the patterns aliases, and the speckle lands at the wrapped position.
    """
    slm, camera = _build_hardware()
    generator = _generator_for(_synthetic_mapping(), slm, camera, tmp_path)
    half_field = slm.wavelength * FOCAL_LENGTH / (2 * slm.pixel_size[1])

    with pytest.raises(ValueError, match="field the SLM addresses"):
        generator.generate_phase_patterns(
            (5e-4, 5e-4), tilt=(half_field, 0.0), seed=0, verbose=False
        )


def test_the_calibration_steers_its_speckle_with_the_tilt(tmp_path) -> None:
    slm, camera = _build_hardware()
    calibrator = PixelwiseSpeckleCalibrator(
        slm=slm,
        camera=camera,
        camera_mapping=_synthetic_mapping(),
        slm_camera_model=_build_model(slm, camera, FOCAL_LENGTH),
        dataset_path=tmp_path / "dataset.asdf",
        number_of_random_patterns=1,
    )

    calibrator.calibrate(
        speckle_pattern_extent=(1.2e-3, 1.2e-3),
        speckle_pattern_tilt=(6e-4, 0.0),
        number_of_epochs=1,
        batch_size=1,
        verbose=False,
    )

    assert calibrator.capture_data.metadata["speckle_tilt"] == pytest.approx(
        (6e-4, 0.0)
    )


# --- The benchmark ----------------------------------------------------------------


def _benchmark_record(slm) -> tuple[WavefrontCalibrationData, np.ndarray]:
    """A calibration record whose incident field has unit amplitude and a random
    phase, and that phase in radians.
    """
    incident_phase = np.random.default_rng(1).uniform(
        -np.pi, np.pi, tuple(slm.resolution)
    )
    record = WavefrontCalibrationData(
        timestamp=datetime.now(),
        name="benchmark",
        complex_amplitude=ComplexAmplitude(
            torch.exp(1j * torch.as_tensor(incident_phase)).to(torch.complex64),
            wavelength=torch.as_tensor(slm.wavelength),
            pixel_size=torch.as_tensor(tuple(slm.pixel_size)),
        ),
    )
    return record, incident_phase


def test_a_benchmark_adds_the_correction_of_its_record(tmp_path) -> None:
    """A calibration record holds the incident field, so the benchmark adds the
    negative of its phase, the correction, to the pattern.
    """
    slm, camera = _build_hardware()
    record, incident_phase = _benchmark_record(slm)
    plain = _generator_for(_synthetic_mapping(), slm, camera, tmp_path)
    corrected = _generator_for(_synthetic_mapping(), slm, camera, tmp_path)

    plain.generate_phase_patterns((5e-4, 5e-4), seed=0, verbose=False)
    corrected.generate_phase_patterns(
        (5e-4, 5e-4), benchmark_calibration=record, seed=0, verbose=False
    )

    assert len(corrected.phase_patterns) == len(plain.phase_patterns) == 1
    np.testing.assert_allclose(
        np.exp(1j * (corrected.phase_patterns[0] - plain.phase_patterns[0])),
        np.exp(-1j * incident_phase),
        atol=1e-5,
    )


def test_a_benchmark_dataset_displays_the_corrected_pattern(tmp_path) -> None:
    """The levels shown on the SLM and stored for the fit carry the correction, and
    the record keeps the benchmark the dataset was captured with.
    """
    slm, camera = _build_hardware()
    record, incident_phase = _benchmark_record(slm)
    plain = _generator_for(_synthetic_mapping(), slm, camera, tmp_path)
    plain.generate_phase_patterns((5e-4, 5e-4), seed=0, verbose=False)
    generator = _generator_for(_synthetic_mapping(), slm, camera, tmp_path)

    capture = generator.generate_dataset(
        (5e-4, 5e-4), benchmark_calibration=record, seed=0
    )

    assert capture.benchmark_calibration is record
    with CaptureStore.open(generator.dataset_path) as store:
        levels = np.asarray(store.read(0)["slm_levels"])
    displayed_phase = np.asarray(slm.phase_response.to_phase(levels))
    # The incident field and the displayed correction leave the random pattern, to
    # within one level of the SLM.
    np.testing.assert_allclose(
        np.exp(1j * (displayed_phase + incident_phase)),
        np.exp(1j * plain.phase_patterns[0]),
        atol=2 * np.pi / 2**slm.bitdepth,
    )


# --- Autoexposure, overexposure, the background and the camera's state --------------


def _masked_peaks(generator: DatasetGenerator) -> list[float]:
    """The brightest count inside the region of interest of every stored frame."""
    with CaptureStore.open(generator.dataset_path) as store:
        return [
            float(np.max(np.asarray(store.read(index)["camera_image"])[
                generator.roi_mask
            ]))
            for index in range(len(store))
        ]


def _speckle_generator(
    slm, camera, tmp_path, number_of_random_patterns: int
) -> DatasetGenerator:
    return DatasetGenerator(
        slm=slm,
        camera=camera,
        camera_mapping=_synthetic_mapping(),
        slm_camera_model=_build_model(slm, camera),
        dataset_path=tmp_path / "dataset.asdf",
        number_of_random_patterns=number_of_random_patterns,
    )


def test_capture_autoexposes_on_the_first_pattern(tmp_path) -> None:
    """The camera is autoexposed on the first pattern and holds that exposure for every
    frame. The first frame therefore sits at the default set fraction of 0.75, and the
    record holds the exposure of the captured frames.
    """
    # The sensor draws its shot noise from torch's global generator.
    torch.manual_seed(0)
    slm, camera = _build_hardware()
    entry_exposure = camera.get_exposure()
    generator = _speckle_generator(slm, camera, tmp_path, 8)

    capture = generator.generate_dataset((5e-4, 5e-4), seed=0)

    full_scale = camera.max_pixel_value
    first_peak = _masked_peaks(generator)[0]
    assert first_peak == pytest.approx(0.75 * full_scale, abs=0.05 * full_scale)
    assert capture.metadata["exposure_time"] == capture.camera_data.exposure
    assert capture.metadata["exposure_set_fraction"] == 0.75
    assert camera.get_exposure() == entry_exposure


def test_capture_reads_out_the_whole_sensor_with_a_window_left_on_the_camera(
    tmp_path,
) -> None:
    slm, camera = _build_hardware()
    window = ROI(4, 6, 20, 24)
    camera.set_roi(window)
    generator = _speckle_generator(slm, camera, tmp_path, 2)

    capture = generator.generate_dataset((5e-4, 5e-4), seed=0)

    assert generator.roi_mask.shape == CAMERA_RESOLUTION
    assert capture.camera_data.roi == ROI(0, 0, *CAMERA_RESOLUTION)
    with CaptureStore.open(generator.dataset_path) as store:
        assert store.read(0)["camera_image"].shape == CAMERA_RESOLUTION
    assert camera.roi == window


def test_the_fit_subtracts_a_background_evaluated_at_the_capture_exposure(
    tmp_path,
) -> None:
    """Frames are stored raw. A background passed to the fit is evaluated at the
    equivalent exposure recorded by the dataset. It is taken off every frame before the
    frame is normalized.
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

    capture = calibrator.dataset_generator.generate_dataset((5e-4, 5e-4), seed=0)
    evaluated_at: list[float] = []

    def background(exposure: float) -> float:
        evaluated_at.append(exposure)
        return 7.0

    calibrator.fit(
        number_of_epochs=1,
        batch_size=1,
        capture_data=capture,
        background=background,
        verbose=False,
    )

    assert evaluated_at == [capture.camera_data.equivalent_exposure]
    calibration = calibrator.generate_slm_beam_calibration()
    assert calibration.metadata["background_level"] == 7.0
    frames = calibrator.fitter.frames
    with CaptureStore.open(calibrator.dataset_path) as store:
        raw = np.asarray(store.read(0)["camera_image"], dtype=np.float64)
    expected = frames.roi.crop(raw - 7.0)
    expected = expected / expected[frames.region.cpu().numpy()].mean()
    np.testing.assert_allclose(
        frames.frames[0].double().cpu().numpy(), expected, rtol=1e-5, atol=1e-6
    )

    calibrator.fit(number_of_epochs=1, batch_size=1, verbose=False)
    assert calibrator.fitter.frames.background_level is None


def test_the_background_is_evaluated_at_the_equivalent_exposure_of_a_capture(
    tmp_path,
) -> None:
    """A dataset captured with the gain raised records the exposure and the gain. The
    counts above the black level follow the equivalent exposure, so a background
    function is evaluated there.
    """
    slm, camera = _build_hardware()
    calibrator = PixelwiseSpeckleCalibrator(
        slm=slm,
        camera=camera,
        camera_mapping=_synthetic_mapping(),
        slm_camera_model=_build_model(slm, camera, FOCAL_LENGTH),
        dataset_path=tmp_path / "dataset.asdf",
        number_of_random_patterns=1,
    )
    capture = calibrator.dataset_generator.generate_dataset((5e-4, 5e-4), seed=0)
    raised = dataclasses.replace(
        capture, camera_data=dataclasses.replace(capture.camera_data, gain=6.0)
    )
    evaluated_at: list[float] = []

    def background(exposure: float) -> float:
        evaluated_at.append(exposure)
        return 0.0

    calibrator.fit(
        number_of_epochs=1,
        batch_size=1,
        capture_data=raised,
        background=background,
        verbose=False,
    )

    assert evaluated_at == [
        pytest.approx(capture.camera_data.exposure * 10.0 ** (6.0 / 20.0))
    ]


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
        spot_fit=FocalSpotFit(waist=RECOVERY_CAMERA_PIXEL_SIZE[0] * 2),
        output_pixel_size=RECOVERY_CAMERA_PIXEL_SIZE,
        output_resolution=RECOVERY_CAMERA_RESOLUTION,
    )


@requires_nufft
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
        grid_x, grid_y, beam_radius_major=RECOVERY_BEAM_RADIUS
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
    camera = SimulatedCameraTorch(hardware, read_noise=0.0)
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

    recovered = as_image(result.complex_amplitude).detach().cpu().numpy()
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
    # exposed, so the fit was being handed overexposed speckle. With that fixed it
    # reaches 0.9994 to 0.9995 across seeds, so the bar can mean something.
    assert abs(correlation) > 0.95


# The patterns of a PSF calibration on the pointing bench, and the pointing jitter of
# its beam, in metres of the focal plane per axis.
POINTING_PATTERNS = 8
POINTING_STD = 10e-6
# The largest pointing shift reaches 0.7 of a speckle grain, and the fit registers it
# in 6 Levenberg-Marquardt steps.
POINTING_STEPS = 6


def _pointing_bench(pointing_focal_shift_std: float | None):
    """The recovery bench with its astigmatic beam pointed anew for every frame. The
    hardware is the chirp-z model, so the bench is the kind of model the PSF
    calibration fits.
    """
    torch.manual_seed(0)
    geometry = FieldGeometry(
        resolution=RECOVERY_SLM_RESOLUTION,
        pixel_size=torch.tensor([12.5e-6, 12.5e-6], device=DEVICE),
        wavelength=torch.tensor(0.63e-6, device=DEVICE),
    )
    slm = SimulatedSLMTorch(input_geometry=geometry, bitdepth=8)
    grid_x, grid_y = geometry.get_spatial_grid()
    amplitude = gaussian_beam_intensity(
        grid_x, grid_y, beam_radius_major=RECOVERY_BEAM_RADIUS
    ).sqrt()
    injected_phase = (grid_x**2 - grid_y**2) / RECOVERY_BEAM_RADIUS**2
    beam = ComplexAmplitude(
        amplitude * torch.exp(1j * injected_phase),
        wavelength=geometry.wavelength,
        pixel_size=geometry.pixel_size,
        power=1e-3,
    )
    hardware = SLMCZT(
        input_geometry=geometry,
        virtual_slm=slm.virtual_slm,
        camera_resolution=RECOVERY_CAMERA_RESOLUTION,
        camera_pixel_size=RECOVERY_CAMERA_PIXEL_SIZE,
        focal_length=FOCAL_LENGTH,
        slm_field=PixelwiseSLMField(beam),
        pointing_focal_shift_std=pointing_focal_shift_std,
        pointing_seed=1,
    )
    camera = SimulatedCameraTorch(hardware, read_noise=0.0)
    camera.set_exposure(1e-3)
    camera.get_image()
    return slm, camera, hardware


@pytest.fixture(scope="module")
def pointing_calibrations(tmp_path_factory) -> dict:
    """A PSF calibration of the pointing bench without jitter and with it, and the
    bench's pointing of every captured frame, as a focal-plane translation less the
    mean over the frames.
    """
    calibrations = {}
    for label, std in (("steady", None), ("jittered", POINTING_STD)):
        slm, camera, hardware = _pointing_bench(std)
        calibrator = PSFSpeckleCalibrator(
            slm=slm,
            camera=camera,
            camera_mapping=_recovery_mapping(),
            slm_camera_model=_build_model(slm, camera, FOCAL_LENGTH),
            dataset_path=tmp_path_factory.mktemp(label) / "dataset.asdf",
            number_of_random_patterns=POINTING_PATTERNS,
        )
        pointing = getattr(hardware, "pointing_instability", None)
        if pointing is not None:
            pointing.record()
        result = calibrator.calibrate(
            speckle_pattern_extent=(1.5e-3, 1.5e-3),
            steps=POINTING_STEPS,
            chunk_size=4,
            seed=0,
            verbose=False,
        )
        true_translations = None
        if pointing is not None:
            # One frame per pattern, captured after the autoexposure.
            angles = pointing.angle_history.numpy().astype(float)
            shifts = angles[-POINTING_PATTERNS:] * FOCAL_LENGTH
            true_translations = shifts - shifts.mean(axis=0)
        calibrations[label] = SimpleNamespace(
            calibrator=calibrator,
            result=result,
            true_translations=true_translations,
        )
    return calibrations


def test_the_fitted_translations_follow_the_pointing_of_the_beam(
    pointing_calibrations,
) -> None:
    """Each frame's fitted translation is the pointing the bench drew for that frame,
    its angle times the focal length.
    """
    jittered = pointing_calibrations["jittered"]
    fitted = np.asarray(jittered.result.metadata["frame_translations"])
    error = fitted - jittered.true_translations

    assert jittered.true_translations.std() > 0.5 * POINTING_STD
    assert np.sqrt((error**2).mean()) < 0.02 * jittered.true_translations.std()


def test_a_jittered_calibration_recovers_the_field_of_a_steady_one(
    pointing_calibrations,
) -> None:
    """The translations take up the pointing jitter, so the field comes out as close
    to the injected one as on the bench without it.
    """
    steady = pointing_calibrations["steady"].result.metadata["residual_phase_rms"]
    jittered = pointing_calibrations["jittered"].result.metadata["residual_phase_rms"]

    assert steady < 0.02
    assert jittered < 1.5 * steady


def test_the_record_holds_one_translation_per_frame_with_zero_mean(
    pointing_calibrations,
) -> None:
    for calibration in pointing_calibrations.values():
        translations = np.asarray(calibration.result.metadata["frame_translations"])
        assert translations.shape == (POINTING_PATTERNS, 2)
        assert np.abs(translations.mean(axis=0)).max() < 1e-12


def test_the_fit_records_its_cost_before_and_after_each_step(
    pointing_calibrations,
) -> None:
    """The fit records its cost before the first step and after each step, and the
    cost never rises.
    """
    for calibration in pointing_calibrations.values():
        costs = calibration.calibrator.loss_history
        assert len(costs) == POINTING_STEPS + 1
        assert all(later <= earlier for earlier, later in zip(costs, costs[1:]))
        assert costs[-1] < 0.1 * costs[0]
        assert calibration.result.visualization_data.loss_history == costs


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
