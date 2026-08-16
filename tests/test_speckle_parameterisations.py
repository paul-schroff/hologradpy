"""Tests for what each speckle-calibrator subclass changes about the fit.

How the SLM-plane field is parameterised decides three things: what starting point it
wants measured, what it costs to fit, and what is worth plotting afterwards. Each is a
method on a concrete calibrator, so each is asserted here against the class rather than
only through a full calibration, where a wrong cost still produces a plausible result.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from hologradpy.optics.modules.slm_fields import (  # noqa: E402
    PSFSLMField,
    PixelwiseSLMField,
)
from hologradpy.loss_functions import MaskedIntensityMSE, SumOfLosses  # noqa: E402
from hologradpy.calibration.wavefront.speckle_calibration import (  # noqa: E402
    PSFCalibratorVisualizer,
    PSFSpeckleCalibrator,
    PSFSpeckleVisualizationData,
    SpeckleCalibrator,
    SpeckleCalibratorVisualizer,
    SpeckleVisualizationData,
    PixelwiseSpeckleCalibrator,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

MASK = torch.ones((4, 4))


def _setup():
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from test_speckle_calibrator import (  # noqa: E402
        FOCAL_LENGTH,
        _build_hardware,
        _synthetic_mapping,
    )

    slm, camera = _build_hardware()
    return slm, camera, _synthetic_mapping(), FOCAL_LENGTH


def _build_model(*args, **kwargs):
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from test_speckle_calibrator import _build_model as build  # noqa: E402

    return build(*args, **kwargs)


def _psf_field(camera, focal_length, kernel_size: int = 7, init_psf_kernel=None):
    return PSFSLMField(
        focal_length=focal_length,
        camera_pixel_size=tuple(camera.pixel_size),
        psf_kernel_size=kernel_size,
        init_psf_kernel=init_psf_kernel,
    )


def _pixelwise_calibrator(tmp_path):
    slm, camera, mapping, focal_length = _setup()
    return PixelwiseSpeckleCalibrator(
        slm=slm,
        camera=camera,
        camera_mapping=mapping,
        slm_camera_model=_build_model(slm, camera, focal_length),
        dataset_path=tmp_path / "dataset.asdf",
        number_of_random_patterns=2,
    )


def _psf_calibrator(tmp_path, slm_field=None):
    """A PSF calibrator. With no ``slm_field`` the calibrator builds its own, which is
    the normal path: a kernel cannot be sized until a mapping has fitted the spot."""
    slm, camera, mapping, focal_length = _setup()
    return PSFSpeckleCalibrator(
        slm=slm,
        camera=camera,
        camera_mapping=mapping,
        slm_camera_model=_build_model(slm, camera, focal_length, slm_field=slm_field),
        dataset_path=tmp_path / "dataset.asdf",
        number_of_random_patterns=2,
    )


def test_the_base_calibrator_cannot_be_instantiated() -> None:
    """The neutral base has no cost of its own, so picking it must fail rather than
    quietly fit without whatever prior the parameterisation needed."""
    assert "_fit_settings" in SpeckleCalibrator.__abstractmethods__

    with pytest.raises(TypeError, match="abstract"):
        SpeckleCalibrator(
            slm=None,
            camera=None,
            camera_mapping=None,
            slm_camera_model=None,
            dataset_path="dataset.asdf",
        )


def test_a_field_of_the_wrong_type_is_replaced(tmp_path, capsys) -> None:
    """The calibrator owns the field it fits, so a model carrying something else has it
    swapped rather than being refused. It says so, since a field the caller supplied is
    being discarded."""
    slm, camera, mapping, focal_length = _setup()

    calibrator = PSFSpeckleCalibrator(
        slm=slm,
        camera=camera,
        camera_mapping=mapping,
        slm_camera_model=_build_model(slm, camera, focal_length),  # PixelwiseSLMField
        dataset_path=tmp_path / "dataset.asdf",
    )
    assert isinstance(calibrator.slm_camera_model.slm_field, PSFSLMField)
    assert "Replacing the model's PixelwiseSLMField" in capsys.readouterr().out

    # And the other way round.
    calibrator = PixelwiseSpeckleCalibrator(
        slm=slm,
        camera=camera,
        camera_mapping=mapping,
        slm_camera_model=_build_model(
            slm, camera, focal_length,
            slm_field=_psf_field(camera, focal_length),
        ),
        dataset_path=tmp_path / "dataset.asdf",
    )
    assert isinstance(calibrator.slm_camera_model.slm_field, PixelwiseSLMField)
    assert "Replacing the model's PSFSLMField" in capsys.readouterr().out


def test_a_pixelwise_field_asks_for_the_smoothness_prior(tmp_path) -> None:
    """Stored one value per SLM pixel, so it is unconstrained and needs a prior. One
    term per quantity, so each can be weighted and plotted on its own."""
    settings = _pixelwise_calibrator(tmp_path)._fit_settings(MASK)

    assert isinstance(settings.loss, SumOfLosses)
    assert [type(term).__name__ for term in settings.loss.terms] == [
        "MaskedIntensityMSE",
        "PhaseSmoothness",
        "AmplitudeSmoothness",
    ]
    assert settings.learning_rate == pytest.approx(1e-2)


def test_a_psf_field_asks_for_the_bare_mismatch_and_a_larger_step(tmp_path) -> None:
    """Band limited by construction, so no prior, and one kernel pixel moves the whole
    SLM plane, so a larger step."""
    settings = _psf_calibrator(tmp_path)._fit_settings(MASK)

    assert isinstance(settings.loss, MaskedIntensityMSE)
    assert settings.learning_rate == pytest.approx(3e-2)


def test_a_built_kernel_is_measured_from_the_camera(tmp_path) -> None:
    """A kernel the calibrator builds starts from the real focal spot, which carries
    whatever aberration is actually present, rather than an idealised Gaussian."""
    calibrator = _psf_calibrator(tmp_path)
    field = calibrator.slm_camera_model.slm_field
    kernel = field.init_psf_kernel

    assert kernel is not None
    assert tuple(kernel.shape[-2:]) == field.psf_kernel_size
    assert torch.isfinite(kernel).all()
    assert float(kernel.abs().max()) > 0.0


def test_a_supplied_psf_field_is_used_exactly_as_given(tmp_path) -> None:
    """Supplying one is how an already-fitted model is reused, so nothing about it is
    remeasured, including its starting kernel."""
    supplied = torch.full((5, 5), 0.25)
    slm_field = _psf_field(_setup()[1], 0.25, kernel_size=5, init_psf_kernel=supplied)
    calibrator = _psf_calibrator(tmp_path, slm_field=slm_field)

    assert calibrator.slm_camera_model.slm_field is slm_field
    assert torch.equal(slm_field.init_psf_kernel, supplied)


def test_a_pixelwise_calibrator_seeds_nothing(tmp_path) -> None:
    """Most parameterisations want no measured starting point, and the base hook must
    leave them untouched rather than fail."""
    calibrator = _pixelwise_calibrator(tmp_path)

    assert not hasattr(calibrator.slm_camera_model.slm_field, "init_psf_kernel")
    assert calibrator._visualization_extras() == {}


def test_the_psf_calibrator_records_its_kernel_for_the_visualizer(tmp_path) -> None:
    """The extra payload is the whole reason the two visualizers differ."""
    calibrator = _psf_calibrator(tmp_path)

    kernel_size = calibrator.slm_camera_model.slm_field.psf_kernel_size

    extras = calibrator._visualization_extras()
    assert set(extras) == {"psf_kernel"}
    assert extras["psf_kernel"].shape == kernel_size
    assert calibrator.visualization_data_type is PSFSpeckleVisualizationData


def _visualization_payload(**extras):
    shared = dict(
        camera_image=np.random.default_rng(0).uniform(size=(48, 48)),
        roi_mask=np.ones((48, 48), dtype=bool),
        slm_pattern=np.zeros((48, 48)),
        measured_roi=np.random.default_rng(1).uniform(size=(12, 12)),
        predicted_roi=np.random.default_rng(2).uniform(size=(12, 12)),
        recovered_amplitude=np.ones((64, 64)),
        recovered_phase=np.zeros((64, 64)),
        loss_history=[1.0, 0.5, 0.25],
    )
    return shared | extras


def test_each_payload_returns_its_own_visualizer() -> None:
    """So a calibration reloaded from disk can be plotted without remembering how it
    was fitted."""
    static_data = SpeckleVisualizationData(**_visualization_payload())
    psf_data = PSFSpeckleVisualizationData(
        **_visualization_payload(psf_kernel=np.ones((7, 7), dtype=complex))
    )

    assert type(static_data.visualizer()) is SpeckleCalibratorVisualizer
    assert type(psf_data.visualizer()) is PSFCalibratorVisualizer


def test_the_psf_visualizer_adds_two_cells_to_the_shared_figure() -> None:
    """It extends the base rather than replacing it, so the shared diagnostics stay."""
    shared = _visualization_payload()
    static_data = SpeckleVisualizationData(**shared)
    psf_data = PSFSpeckleVisualizationData(
        **_visualization_payload(psf_kernel=np.ones((7, 7), dtype=complex))
    )

    base_panels = set(SpeckleCalibratorVisualizer(static_data).panels())
    psf_panels = set(PSFCalibratorVisualizer(psf_data).panels())

    assert base_panels < psf_panels
    assert psf_panels - base_panels == {"psf_amplitude", "psf_phase"}

    assert PSFCalibratorVisualizer(psf_data).render() is not None


def test_the_psf_visualizer_degrades_without_a_kernel() -> None:
    """The kernel field has to be defaulted, so rendering must not depend on it."""
    psf_data = PSFSpeckleVisualizationData(**_visualization_payload())
    assert psf_data.psf_kernel is None

    visualizer = PSFCalibratorVisualizer(psf_data)
    assert "psf_amplitude" not in visualizer.panels()
    assert visualizer.render() is not None
