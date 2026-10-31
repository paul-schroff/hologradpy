"""The calibration of the zeroth order on a simulated SLM.

The simulated camera's model carries a reflection, a ``ZerothOrderField`` with a fixed
kernel, and either the light at the wraps of the wrap rule or a crosstalk model. The
calibrator's model has the same beam and neither. The pattern can move by a fresh
translation whenever the SLM shows another hologram, as the beam's pointing moves
between holograms.
"""

from __future__ import annotations

import math
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from hologradpy.calibration.camera_mapping import CameraMapping, FocalSpotFit
from hologradpy.calibration.model_fit import field_on_frames
from hologradpy.calibration.zeroth_order import (
    ZEROTH_ORDER_LAYER,
    ZerothOrderCalibrationData,
    ZerothOrderCalibrator,
    fit_kernel_and_wrap_weight,
    fit_missed_field,
)
from hologradpy.geometry import PartialAffineTransform
from hologradpy.grids import pixel_frame_vector, plane_center
from hologradpy.hardware import SimulatedCameraTorch, SimulatedSLMTorch
from hologradpy.optics.complex_amplitude import ComplexAmplitude, FieldGeometry
from hologradpy.optics.modules import (
    PixelwiseSLMField,
    SuperGaussianCrosstalk,
    VirtualSLM,
    ZerothOrderField,
)
from hologradpy.optics.systems import SLMCZT
from hologradpy.phase_levels import LinearResponse
from hologradpy.roi import ROI

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

SLM_RESOLUTION = (64, 64)
SLM_PITCH = 12.5e-6
WAVELENGTH = 800e-9
FOCAL_LENGTH = 0.1
BEAM_RADIUS = 0.3e-3
BEAM_POWER = 1e-10
CAMERA_RESOLUTION = (96, 96)
CAMERA_PITCH = 30e-6
# The focal waist of the beam, lambda f / (pi w), as a camera mapping fits it.
FOCAL_WAIST = WAVELENGTH * FOCAL_LENGTH / (math.pi * BEAM_RADIUS)
# Three gratings of about 770, 1060 and 1280 wrapped edges. Their spots lie at least
# seven waists outside the disc.
GRATING_TILTS = ((1.0e-3, 0.2e-3), (-1.05e-3, 0.6e-3), (0.9e-3, -1.1e-3))
WRAP_WEIGHT = -0.4 + 0.15j
# The peak of the reflection's intensity, as a fraction of the first grating's peak.
REFLECTION_PEAK = 4e-3


def _geometry() -> FieldGeometry:
    return FieldGeometry(
        resolution=SLM_RESOLUTION,
        pixel_size=torch.tensor([SLM_PITCH, SLM_PITCH]),
        wavelength=torch.tensor(WAVELENGTH),
    )


def _beam(added_phase: np.ndarray | None = None) -> ComplexAmplitude:
    """A Gaussian beam with 0.15 waves of defocus at its radius from the backplane,
    and ``added_phase`` in radians on top, as a calibration that measured the beam
    through the SLM's vendor correction holds it.
    """
    geometry = _geometry()
    x, y = geometry.get_spatial_grid()
    radial = (x**2 + y**2) / BEAM_RADIUS**2
    field = torch.exp(-radial) * torch.exp(1j * 2 * math.pi * 0.15 * radial)
    if added_phase is not None:
        field = field * torch.exp(1j * torch.as_tensor(added_phase, dtype=x.dtype))
    return ComplexAmplitude(
        field.to(torch.complex64),
        wavelength=geometry.wavelength,
        pixel_size=geometry.pixel_size,
        power=BEAM_POWER,
    )


def _system(
    virtual_slm: VirtualSLM, beam: ComplexAmplitude | None = None
) -> SLMCZT:
    return SLMCZT(
        input_geometry=_geometry(),
        virtual_slm=virtual_slm,
        camera_resolution=CAMERA_RESOLUTION,
        camera_pixel_size=(CAMERA_PITCH, CAMERA_PITCH),
        focal_length=FOCAL_LENGTH,
        slm_field=PixelwiseSLMField(_beam() if beam is None else beam),
    )


def _mapping() -> CameraMapping:
    """The identity mapping onto the model's output plane, the whole sensor."""
    center_x, center_y = plane_center(CAMERA_RESOLUTION)
    identity = PartialAffineTransform.from_components(
        scale=1.0, angle_deg=0.0, shift=(0.0, 0.0), center=(center_x, center_y)
    )
    detected = np.random.default_rng(0).uniform(0, 60, size=(12, 2))
    return CameraMapping(
        timestamp=datetime.now(),
        name="synthetic",
        transform=identity.as_matrix(homogeneous=False),
        detected_points=detected.tolist(),
        calculated_points=identity.transform_points(detected).tolist(),
        spot_fit=FocalSpotFit(waist=FOCAL_WAIST),
        output_pixel_size=(CAMERA_PITCH, CAMERA_PITCH),
        output_resolution=CAMERA_RESOLUTION,
    )


def _reflection_kernel() -> torch.Tensor:
    """A round spot of two kernel pixels' waist, a pixel off the zeroth order, at a
    phase of 1 rad.
    """
    rows, columns = torch.meshgrid(
        torch.arange(15) - 7.0, torch.arange(15) - 7.0, indexing="ij"
    )
    spot = torch.exp(-((columns - 1.0) ** 2 + rows**2) / 4.0)
    return (spot * complex(math.cos(1.0), math.sin(1.0))).to(torch.complex64)


class _PointingCamera(SimulatedCameraTorch):
    """A simulated camera. Its pattern moves by a fresh random translation whenever the
    SLM shows another hologram, as the beam's pointing moves between holograms. Each
    translation is kept on :attr:`translations`, ``(columns, rows)`` in pixels.
    """

    def __init__(self, model: SLMCZT, jitter: float, **sensor_kwargs) -> None:
        super().__init__(model, **sensor_kwargs)
        self.jitter = jitter
        self.translations: list[tuple[float, float]] = []
        self._generator = np.random.default_rng(5)
        self._shown: torch.Tensor | None = None

    def _capture_frame(self) -> torch.Tensor:
        levels = self.slm_camera_model.virtual_slm.levels.detach()
        if self._shown is None or not torch.equal(levels, self._shown):
            self._shown = levels.clone()
            columns, rows = self.jitter * self._generator.standard_normal(2)
            self.translations.append((float(columns), float(rows)))
            with torch.no_grad():
                self.slm_camera_model.focal_plane_partial_affine.shift.copy_(
                    torch.tensor(pixel_frame_vector(self.translations[-1]))
                )
        return super()._capture_frame()


def _bench(
    crosstalk: bool = False,
    jitter: float = 0.0,
    vendor_correction: np.ndarray | None = None,
) -> tuple[SimulatedSLMTorch, _PointingCamera, SLMCZT]:
    """The simulated SLM and camera, and the camera's model up to the lens.

    The camera's model adds the reflection and, without a crosstalk model, the light
    at the wraps of the hologram on display. A vendor correction, in gray levels, is
    loaded on the SLM, and the backplane carries minus its phase, which it cancels.
    """
    slm = SimulatedSLMTorch(
        input_geometry=_geometry(),
        bitdepth=8,
        pixel_crosstalk=SuperGaussianCrosstalk(upscale_factor=2) if crosstalk else None,
    )
    backplane = None
    if vendor_correction is not None:
        slm.load_vendor_correction(vendor_correction)
        backplane = -LinearResponse(bitdepth=8).to_phase(vendor_correction)
    truth = _system(slm.virtual_slm, _beam(backplane))
    zeroth_order = ZerothOrderField(
        FOCAL_LENGTH,
        wrap_weight=0.0 if crosstalk else WRAP_WEIGHT,
        kernel=_reflection_kernel(),
    )
    truth.insert_after("virtual_slm", "zeroth_order_truth", zeroth_order)
    zeroth_order.register_forward_pre_hook(
        lambda module, inputs: module.update_wraps(truth)
    )
    camera = _PointingCamera(truth, jitter=jitter, bitdepth=16, add_noise=False)
    camera.set_exposure(1e-3)
    truth.build()
    _scale_reflection(slm, truth, zeroth_order)
    return slm, camera, truth


def _field_ahead_of_the_sensor(
    truth: SLMCZT, incident: ComplexAmplitude | None = None
) -> torch.Tensor:
    """The camera's model run without the sensor the camera appended to it, from the
    centred pointing, ``(N, H, W)``.
    """
    truth.focal_plane_partial_affine.reset()
    field = truth.init_field if incident is None else incident
    for name in list(truth.layers())[:-1]:
        field = truth[name](field)
    return field_on_frames(field)


def _scale_reflection(
    slm: SimulatedSLMTorch, truth: SLMCZT, zeroth_order: ZerothOrderField
) -> None:
    """Scale the reflection's kernel so its peak is REFLECTION_PEAK of the first
    grating's, as the SLM shows the grating.
    """
    with torch.no_grad(), truth.virtual_slm.preserve_levels():
        slm.set_phase(truth.steering_phase(GRATING_TILTS[0]).numpy())
        held_weight = zeroth_order.wrap_weight.clone()
        held_kernel = zeroth_order.kernel.clone()
        zeroth_order.wrap_weight.zero_()
        zeroth_order.kernel.zero_()
        grating = _field_ahead_of_the_sensor(truth)
        zeroth_order.kernel.copy_(held_kernel)
        incident = truth.init_field
        dark = ComplexAmplitude(
            torch.zeros_like(incident.as_tensor()),
            incident.wavelength,
            incident.pixel_size,
        )
        reflection = _field_ahead_of_the_sensor(truth, dark)
        zeroth_order.wrap_weight.copy_(held_weight)
        ratio = float(grating.abs().square().max() / reflection.abs().square().max())
        zeroth_order.kernel.mul_(math.sqrt(REFLECTION_PEAK * ratio))


def _calibrator(
    tmp_path, slm, camera, beam: ComplexAmplitude | None = None, **options
) -> ZerothOrderCalibrator:
    model = _system(VirtualSLM.from_slm(slm), beam)
    return ZerothOrderCalibrator(
        slm,
        camera,
        model,
        _mapping(),
        tmp_path / "zeroth_order.asdf",
        GRATING_TILTS,
        **options,
    )


def _missed_field(
    truth: SLMCZT, calibrator: ZerothOrderCalibrator, phase: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The field the plain model misses for ``phase``, the field the calibrated model
    predicts for it, both on the box of the disc, and the disc on the box.

    The SLM shows the phase with its vendor correction, and the model takes the levels
    of the phase alone, as the calibration's probes do.
    """
    model = calibrator.slm_camera_model
    slm = calibrator.slm
    box = ROI.detect(calibrator.region, pad=0)
    region = box.crop(torch.as_tensor(calibrator.region))
    zeroth_order = calibrator.zeroth_order_field
    with torch.no_grad(), truth.virtual_slm.preserve_levels():
        slm.set_phase(np.asarray(phase))
        model.virtual_slm.set_levels(slm.phase_to_levels(np.asarray(phase)), 8)
        missed = box.crop(_field_ahead_of_the_sensor(truth))
        zeroth_order.update_wraps(model)
        with_field = box.crop(field_on_frames(model()))
        held_kernel = zeroth_order.kernel.clone()
        held_weight = zeroth_order.wrap_weight.clone()
        zeroth_order.kernel.zero_()
        zeroth_order.wrap_weight.zero_()
        plain = box.crop(field_on_frames(model()))
        zeroth_order.kernel.copy_(held_kernel)
        zeroth_order.wrap_weight.copy_(held_weight)
    return (missed - plain)[0], (with_field - plain)[0], region


def _prediction_error(
    truth: SLMCZT, calibrator: ZerothOrderCalibrator, phase: torch.Tensor
) -> float:
    """The power of the prediction's error in the disc, over the missed power."""
    missed, predicted, region = _missed_field(truth, calibrator, phase)
    error = (predicted - missed)[region].abs().square().sum()
    return float(error / missed[region].abs().square().sum())


# --- The fits ----------------------------------------------------------------------


def test_the_missed_field_and_the_floor_are_solved_on_every_pixel() -> None:
    generator = torch.Generator().manual_seed(0)
    shape = (8, 6, 5)
    model = torch.randn(shape, dtype=torch.complex64, generator=generator)
    missed = torch.randn(shape[1:], dtype=torch.complex64, generator=generator)
    floor = torch.rand(shape[1:], generator=generator)
    frames = (model + missed).abs().square() + floor
    region = torch.ones(shape[1:], dtype=torch.bool)

    solved = fit_missed_field(frames, model, region)

    assert bool(solved.resolved.all())
    torch.testing.assert_close(solved.field, missed, atol=1e-4, rtol=0)
    torch.testing.assert_close(solved.floor, floor, atol=1e-4, rtol=0)


def test_left_out_measurements_change_nothing_and_too_few_leave_a_pixel_out() -> None:
    generator = torch.Generator().manual_seed(1)
    shape = (8, 4, 4)
    model = torch.randn(shape, dtype=torch.complex64, generator=generator)
    missed = torch.randn(shape[1:], dtype=torch.complex64, generator=generator)
    frames = (model + missed).abs().square()
    valid = torch.ones(shape, dtype=torch.bool)
    # Corrupted measurements, left out of the solve of their pixels.
    frames[:3, 0, 0] = 1e3
    valid[:3, 0, 0] = False
    valid[:6, 1, 1] = False

    solved = fit_missed_field(
        frames, model, torch.ones(shape[1:], dtype=torch.bool), valid
    )

    torch.testing.assert_close(solved.field[0, 0], missed[0, 0], atol=1e-4, rtol=0)
    assert not bool(solved.resolved[1, 1])
    assert int(solved.resolved.sum()) == 15


def test_the_kernel_and_the_wrap_weight_are_told_apart_by_the_gratings() -> None:
    generator = torch.Generator().manual_seed(2)
    operator = torch.randn(3 * 40, 9, dtype=torch.complex128, generator=generator)
    kernel = torch.randn(3, 3, dtype=torch.complex128, generator=generator)
    wrap_fields = torch.randn(3, 10, 12, dtype=torch.complex128, generator=generator)

    def forward(values: torch.Tensor) -> torch.Tensor:
        return (operator @ values.reshape(-1)).reshape(10, 12)

    def adjoint(field: torch.Tensor) -> torch.Tensor:
        return (operator.conj().T @ field.reshape(-1)).reshape(3, 3)

    weight = -0.3 + 0.2j
    measured = forward(kernel)[None] + weight * wrap_fields
    masks = torch.ones(measured.shape, dtype=torch.bool)

    fit = fit_kernel_and_wrap_weight(measured, wrap_fields, masks, forward, adjoint)

    assert fit.wrap_weight == pytest.approx(weight, abs=1e-5)
    torch.testing.assert_close(fit.kernel, kernel, atol=1e-5, rtol=0)
    assert fit.relative_residual < 1e-5


# --- The calibration on the simulated SLM ---------------------------------------


@pytest.fixture(scope="module")
def calibrated(tmp_path_factory):
    """One calibration on the bench with the wrap rule and a pointing jitter."""
    slm, camera, truth = _bench(jitter=0.25)
    calibrator = _calibrator(tmp_path_factory.mktemp("zeroth_order"), slm, camera)
    calibration = calibrator.calibrate(verbose=False)
    return calibrator, calibration, camera, truth


def test_the_calibration_recovers_the_wrap_weight_and_the_reflection(
    calibrated,
) -> None:
    calibrator, calibration, _, truth = calibrated

    assert calibration.wrap_weight == pytest.approx(WRAP_WEIGHT, rel=0.03)
    # What is left is the change of the light at the wraps from probe to probe, which
    # the solve on every pixel takes as one missed field per grating.
    assert calibration.metadata["relative_residual"] < 0.05
    for tilt in ((0.0, 0.5e-3), (-0.3e-3, -0.6e-3)):
        phase = truth.steering_phase(tilt)
        assert _prediction_error(truth, calibrator, phase) < 1e-3


def test_every_frame_is_registered_on_its_own_translation(calibrated) -> None:
    calibrator, calibration, camera, _ = calibrated
    probes = len(calibrator.capture_data.probes)

    fitted = calibration.metadata["registration_shifts"]
    applied = np.asarray(camera.translations[-probes:])

    np.testing.assert_allclose(fitted, applied, atol=0.02)


def test_the_scale_is_the_sensor_s_counts_per_unit_of_intensity(calibrated) -> None:
    calibrator, calibration, camera, _ = calibrated
    sensor = camera.sensor
    photon_energy = 6.62607015e-34 * 299792458.0 / WAVELENGTH
    counts_per_intensity = (
        CAMERA_PITCH**2
        / photon_energy
        * calibrator.capture_data.metadata["long_exposure"]
        * sensor.quantum_efficiency
        * sensor.gain
        * sensor.max_pixel_value
        / sensor.full_well_capacity
    )

    assert calibration.metadata["scale"] == pytest.approx(
        counts_per_intensity, rel=0.02
    )


def test_the_model_carries_the_fitted_zeroth_order(calibrated) -> None:
    calibrator, calibration, _, _ = calibrated
    field = calibrator.slm_camera_model[ZEROTH_ORDER_LAYER]

    assert field is calibrator.zeroth_order_field
    assert complex(field.wrap_weight) == pytest.approx(
        calibration.wrap_weight, abs=1e-6
    )
    np.testing.assert_allclose(
        field.kernel.detach().numpy(), calibration.kernel, atol=1e-7
    )


def test_a_calibration_rebuilds_its_field_in_another_model(calibrated) -> None:
    calibrator, calibration, _, truth = calibrated
    rebuilt = _system(VirtualSLM.from_slm(calibrator.slm))
    rebuilt.build()
    rebuilt.calibrate_from_mapping(_mapping())
    # Inserted into the built model, since its light at the wraps needs the edges of a
    # hologram before its first forward pass.
    rebuilt.insert_after(
        "virtual_slm",
        ZEROTH_ORDER_LAYER,
        ZerothOrderField.from_calibration_data(calibration),
    )
    phase = truth.steering_phase((0.0, 0.5e-3))

    fields = []
    for model in (calibrator.slm_camera_model, rebuilt):
        with torch.no_grad(), model.virtual_slm.preserve_levels():
            model.virtual_slm.set_phase(phase)
            model[ZEROTH_ORDER_LAYER].update_wraps(model)
            fields.append(field_on_frames(model()))

    torch.testing.assert_close(fields[1], fields[0])


def test_a_calibration_refuses_a_model_of_another_grid(calibrated) -> None:
    _, calibration, _, _ = calibrated
    geometry = FieldGeometry(
        resolution=(32, 32),
        pixel_size=torch.tensor([SLM_PITCH, SLM_PITCH]),
        wavelength=torch.tensor(WAVELENGTH),
    )
    field = ZerothOrderField.from_calibration_data(calibration)
    incident = ComplexAmplitude(
        torch.ones(1, 1, 32, 32, dtype=torch.complex64),
        geometry.wavelength,
        geometry.pixel_size,
    )

    with pytest.raises(ValueError, match="Nyquist pitch"):
        field(incident)


def test_the_record_survives_a_round_trip(calibrated, tmp_path) -> None:
    _, calibration, _, _ = calibrated
    path = tmp_path / "zeroth_order_calibration.asdf"

    calibration.lean().save(path)
    reopened = ZerothOrderCalibrationData.load(path)

    assert reopened.wrap_weight == calibration.wrap_weight
    assert reopened.kernel_pixel_size == calibration.kernel_pixel_size
    np.testing.assert_array_equal(reopened.kernel, calibration.kernel)
    np.testing.assert_array_equal(
        reopened.metadata["registration_shifts"],
        calibration.metadata["registration_shifts"],
    )


def test_the_visualizer_renders(calibrated) -> None:
    _, calibration, _, _ = calibrated

    figure = calibration.visualization_data.visualizer().render()

    assert len(figure.axes) >= 7
    plt.close(figure)


def test_a_crosstalk_model_s_zeroth_order_is_predicted_for_another_hologram(
    tmp_path,
) -> None:
    slm, camera, truth = _bench(crosstalk=True)
    calibrator = _calibrator(tmp_path, slm, camera)

    calibrator.calibrate(verbose=False)

    # On this crosstalk model the rule left 5 to 7 % of the missed power unpredicted on
    # these gratings, against below 0.02 % on a bench that follows the rule itself.
    for tilt in ((0.0, 0.5e-3), (-0.3e-3, -0.6e-3), (1.2e-3, 0.0)):
        phase = truth.steering_phase(tilt)
        assert _prediction_error(truth, calibrator, phase) < 0.1


def test_a_vendor_correction_adds_its_wraps_to_those_the_calibration_fits(
    tmp_path,
) -> None:
    """The device shows each probe with the correction added, as the light at the wraps
    of the camera's model sees it. The calibrated beam is the corrected one, as a
    calibration through the correction measures it.
    """
    rows, columns = np.indices(SLM_RESOLUTION)
    radial = ((rows - 31.5) ** 2 + (columns - 31.5) ** 2) / 32.0**2
    levels = (np.round(256 * 0.6 * radial).astype(np.int64) % 256).astype(np.uint8)
    slm, camera, truth = _bench(vendor_correction=levels)
    calibrator = _calibrator(tmp_path, slm, camera)

    calibration = calibrator.calibrate(verbose=False)

    np.testing.assert_allclose(
        calibration.added_phase, LinearResponse(bitdepth=8).to_phase(levels)
    )
    assert calibration.wrap_weight == pytest.approx(WRAP_WEIGHT, rel=0.03)
    for tilt in ((0.0, 0.5e-3), (-0.3e-3, -0.6e-3)):
        phase = truth.steering_phase(tilt)
        assert _prediction_error(truth, calibrator, phase) < 1e-3


# --- What the calibrator refuses ---------------------------------------------------


def test_one_grating_is_refused(tmp_path) -> None:
    slm, camera, _ = _bench()
    model = _system(VirtualSLM.from_slm(slm))

    with pytest.raises(ValueError, match="at least two"):
        ZerothOrderCalibrator(
            slm, camera, model, _mapping(), tmp_path / "set.asdf", GRATING_TILTS[:1]
        )


def test_two_pistons_are_refused(tmp_path) -> None:
    slm, camera, _ = _bench()

    with pytest.raises(ValueError, match="quadratures"):
        _calibrator(tmp_path, slm, camera, number_of_pistons=2)


def test_a_grating_inside_the_disc_is_refused(tmp_path) -> None:
    slm, camera, _ = _bench()
    model = _system(VirtualSLM.from_slm(slm))

    with pytest.raises(ValueError, match="outside the zeroth order"):
        ZerothOrderCalibrator(
            slm,
            camera,
            model,
            _mapping(),
            tmp_path / "set.asdf",
            ((0.1e-3, 0.0), GRATING_TILTS[1]),
        )


def test_a_model_that_carries_a_zeroth_order_field_is_refused(tmp_path) -> None:
    slm, camera, _ = _bench()
    model = _system(VirtualSLM.from_slm(slm))
    model.insert_after("virtual_slm", "earlier", ZerothOrderField(FOCAL_LENGTH))

    with pytest.raises(ValueError, match="already carries"):
        ZerothOrderCalibrator(
            slm, camera, model, _mapping(), tmp_path / "set.asdf", GRATING_TILTS
        )
