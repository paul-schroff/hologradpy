"""Tests for the absolute power reference a chirp-z retrieval is measured against.

A chirp-z model computes only a window of the focal plane, so a cost that renormalises
what it is given cannot tell that the search has pushed light out of that window. The
machinery here is what gives it an absolute reference instead: a power taken *before*
the lens, the pixel area that turns a sum of intensity into power, and the sampling
margin that says whether such a sum means anything.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hologradpy.analysis.error_metrics import captured_power, efficiency_metric
from hologradpy.holography.phase_retrieval import GradientPhaseRetriever
from hologradpy.loss_functions import LossAbsoluteIntensityMSE, LossIntensityMSE
from hologradpy.optics.complex_amplitude import ComplexAmplitude, FieldGeometry
from hologradpy.optics.modules.slm_fields import PixelwiseSLMField
from hologradpy.optics.modules.virtual_slms import VirtualSLM
from hologradpy.optics.systems import SLMCZT
from hologradpy.profiles.amplitude import gaussian_beam_intensity

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

WAVELENGTH = 1039e-9
PITCH = 12.5e-6
RESOLUTION = (128, 128)
FOCAL_LENGTH = 0.3
PADDED = (256, 256)

# Half the width of the diffraction order, the region the SLM can address at all.
ORDER_HALF_EXTENT = WAVELENGTH * FOCAL_LENGTH / (2 * PITCH)


def _geometry() -> FieldGeometry:
    return FieldGeometry(
        resolution=RESOLUTION,
        pixel_size=torch.tensor([PITCH, PITCH]),
        wavelength=torch.tensor(WAVELENGTH),
    )


def _model(camera_resolution=(256, 256), camera_pixel_size=None) -> SLMCZT:
    """A chirp-z model whose window can be made to cover more or less of the order."""
    geometry = _geometry()
    x, y = geometry.get_spatial_grid()
    beam = ComplexAmplitude.from_geometry(
        geometry,
        data=gaussian_beam_intensity(x, y, beam_radius=5e-4).sqrt() + 0j,
    )
    if camera_pixel_size is None:
        # Sized so the window spans the whole diffraction order.
        pitch_out = 2 * ORDER_HALF_EXTENT / camera_resolution[0]
        camera_pixel_size = (pitch_out, pitch_out)

    model = SLMCZT(
        input_geometry=geometry,
        virtual_slm=VirtualSLM(phase_scaling=1.0),
        slm_field=PixelwiseSLMField(beam),
        focal_length=FOCAL_LENGTH,
        camera_resolution=camera_resolution,
        camera_pixel_size=camera_pixel_size,
        padded_resolution=PADDED,
    )
    model()
    return model


def _scatter(model: SLMCZT, seed: int = 0) -> None:
    """Put a random phase on the SLM, which throws light across the whole order."""
    generator = torch.Generator().manual_seed(seed)
    model.virtual_slm.set_phase(
        torch.rand(RESOLUTION, generator=generator) * 2 * torch.pi
    )
    model()


def _window_power(model: SLMCZT) -> float:
    return float(model().intensity.detach().sum()) * model.output_pixel_area()


def test_incident_power_agrees_with_the_power_report() -> None:
    """The cheap single-stage answer must match the module-by-module walk."""
    model = _model()
    report = model.power_report()
    entering_the_lens = next(
        entry for entry in report["modules"] if entry["module"] == "virtual_slm"
    )

    assert model.incident_power() == pytest.approx(
        float(entering_the_lens["power"].sum()), rel=1e-9
    )


def test_incident_power_does_not_move_when_the_slm_phase_does() -> None:
    """The load-bearing invariant.

    A phase-only SLM cannot change how much light arrives at the lens, so the reference
    an efficiency divides by has to stay put. If it moved, the search could improve its
    efficiency by shrinking the denominator instead of by placing light where it was
    asked for.
    """
    model = _model()
    before = model.incident_power()
    _scatter(model)

    # To float32 rounding through exp(i*phase), not bitwise.
    assert model.incident_power() == pytest.approx(before, rel=1e-6)


def test_a_window_covering_the_order_captures_all_the_incident_power() -> None:
    """Parseval, stated in the units the efficiency metric works in."""
    model = _model()
    _scatter(model)

    assert _window_power(model) == pytest.approx(model.incident_power(), rel=1e-2)


def test_shrinking_the_window_loses_power_the_reference_still_counts() -> None:
    """The failure this whole mechanism exists to make visible.

    Light scattered outside the window is absent from the model's output but is still
    in the denominator, so the efficiency falls. Under a renormalising cost the same two
    fields give identical values.
    """
    pitch_out = 2 * ORDER_HALF_EXTENT / 256
    whole = _model((256, 256), (pitch_out, pitch_out))
    window = _model((64, 64), (pitch_out, pitch_out))
    _scatter(whole)
    _scatter(window)

    assert whole.incident_power() == pytest.approx(window.incident_power(), rel=1e-6)
    assert _window_power(window) < 0.5 * _window_power(whole)


def test_the_sampling_margin_falls_below_one_for_a_coarse_camera_pixel() -> None:
    """Below one the window power is aliased, so an efficiency read off it can be
    improved by aliasing rather than by putting light where it was asked for.
    """
    fine = _model()
    coarse = _model((32, 32), (2 * ORDER_HALF_EXTENT / 32,) * 2)

    assert min(fine.focal_plane_sampling_margin()) >= 1.0
    assert min(coarse.focal_plane_sampling_margin()) < 1.0


def test_asking_for_a_layer_the_system_does_not_have_says_so() -> None:
    model = _model()

    with pytest.raises(KeyError, match="fourier_lens"):
        model.power_entering("no_such_stage")


class TestTheEfficiencyMetric:
    """The metric that lets the loss of efficiency be seen rather than inferred."""

    def test_it_gives_the_same_answer_from_numpy_and_from_torch(self) -> None:
        """It is evaluated on numpy arrays once a run ends and on torch tensors as it
        proceeds, so it must not care which it is handed.
        """
        metric = efficiency_metric(incident_power=2.0, pixel_area=0.5)
        region = np.ones((4, 4))
        measured = np.arange(16, dtype=float).reshape(4, 4)

        from_numpy = metric(region, measured, measured)
        from_torch = metric(
            torch.as_tensor(region), torch.as_tensor(measured),
            torch.as_tensor(measured),
        )

        assert from_numpy == pytest.approx(from_torch)

    def test_capturing_the_whole_reference_gives_one(self) -> None:
        measured = np.ones((4, 4))
        metric = efficiency_metric(incident_power=16.0, pixel_area=1.0)

        assert metric(np.ones((4, 4)), measured, measured) == pytest.approx(1.0)

    def test_a_region_holding_nothing_gives_zero(self) -> None:
        measured = np.ones((4, 4))
        metric = efficiency_metric(incident_power=16.0, pixel_area=1.0)

        assert metric(np.zeros((4, 4)), measured, measured) == pytest.approx(0.0)

    def test_the_whole_frame_flavour_ignores_the_region(self) -> None:
        """``in_signal_region=False`` answers 'did the light stay in the window at all',
        which is the question a chirp-z model is asked.
        """
        measured = np.ones((4, 4))
        metric = efficiency_metric(16.0, 1.0, in_signal_region=False)

        assert metric(np.zeros((4, 4)), measured, measured) == pytest.approx(1.0)

    def test_captured_power_is_the_integral_over_the_region(self) -> None:
        measured = np.ones((4, 4))

        assert captured_power(measured, 2.0) == pytest.approx(32.0)
        assert captured_power(measured, 2.0, np.eye(4)) == pytest.approx(8.0)


class TestTheLossFactory:
    """A cost given once must survive the target being set again.

    A feedback loop calls ``set_target`` on every iteration to fold in what the camera
    measured, which rebuilds the cost. Without a factory a custom cost is dropped
    partway through the run, silently.
    """

    def _retriever(self) -> GradientPhaseRetriever:
        model = _model((64, 64), (2 * ORDER_HALF_EXTENT / 256,) * 2)
        target = torch.ones((64, 64))
        region = torch.ones((64, 64), dtype=torch.bool)
        return GradientPhaseRetriever(
            slm_camera_model=model, target=target, signal_region=region
        )

    def test_a_factory_survives_a_retarget(self) -> None:
        retriever = self._retriever()
        retriever.set_loss_factory(
            lambda target, mask: LossAbsoluteIntensityMSE(target, mask)
        )

        retriever.set_target(torch.ones((64, 64)) * 2)

        assert isinstance(retriever.loss_function, LossAbsoluteIntensityMSE)

    def test_a_plain_loss_does_not_survive_a_retarget(self) -> None:
        """The documented behaviour the factory exists to work around, pinned so the
        difference between the two is deliberate.
        """
        retriever = self._retriever()
        retriever.set_loss_function(
            LossAbsoluteIntensityMSE(torch.ones((64, 64)), torch.ones((64, 64)))
        )

        retriever.set_target(torch.ones((64, 64)) * 2)

        assert isinstance(retriever.loss_function, LossIntensityMSE)

    def test_clearing_the_factory_goes_back_to_the_default(self) -> None:
        retriever = self._retriever()
        retriever.set_loss_factory(
            lambda target, mask: LossAbsoluteIntensityMSE(target, mask)
        )

        retriever.set_loss_factory(None)

        assert isinstance(retriever.loss_function, LossIntensityMSE)
