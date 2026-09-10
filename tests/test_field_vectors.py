"""Building a field vector, taking it apart, and carrying it through a whole model.

The component axis exists so that a vector field can travel the same path a scalar one
does. What these pin is that the two agree: a field vector through a model is the three
scalar runs it is made of, and the sensor reads their sum.

The costs that measure a field against a target amplitude and phase refuse one, since a
target has one of each and a field vector has three.
"""

from __future__ import annotations

import pytest
import torch

from hologradpy.loss_functions import (
    LossAbsoluteFidelity,
    LossFidelity,
    LossIntensityMSE,
    LossVorticity,
)
from hologradpy.optics.complex_amplitude import (
    SCALAR,
    VECTOR,
    ComplexAmplitude,
    FieldGeometry,
)
from hologradpy.optics.modules.hardware_models import CameraSensor
from hologradpy.optics.modules.slm_fields import PixelwiseSLMField
from hologradpy.optics.modules.virtual_slms.abstract import VirtualSLM
from hologradpy.optics.systems import SLMCZT, SLMFFT

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

SLM_RESOLUTION = (32, 32)
CAMERA_RESOLUTION = (16, 16)
WAVELENGTH = 800e-9
PIXEL_SIZE = (10e-6, 10e-6)


def _geometry(number_of_components: int = SCALAR) -> FieldGeometry:
    return FieldGeometry(
        wavelength=torch.tensor([WAVELENGTH]),
        pixel_size=torch.tensor([list(PIXEL_SIZE)]),
        resolution=SLM_RESOLUTION,
        number_of_components=number_of_components,
    )


def _scalar_field(value: complex = 1.0) -> ComplexAmplitude:
    return ComplexAmplitude.from_geometry(
        _geometry(), data=torch.full(SLM_RESOLUTION, value, dtype=torch.complex64)
    )


# --- building and taking apart ------------------------------------------------------


def test_a_polarized_scalar_field_keeps_its_power() -> None:
    """A unit Jones vector divides the amplitude between the components, so what the
    field carries in total is unchanged.
    """
    scalar = _scalar_field()

    along_x = scalar.with_polarization((1.0, 0.0, 0.0))
    diagonal = scalar.with_polarization((0.6, 0.8, 0.0))

    assert along_x.is_vector
    assert diagonal.number_of_components == VECTOR
    torch.testing.assert_close(along_x.power(), scalar.power())
    torch.testing.assert_close(diagonal.power(), scalar.power())


def test_a_polarized_field_puts_the_amplitude_where_the_jones_vector_says() -> None:
    scalar = _scalar_field()

    vector = scalar.with_polarization((0.0, 1.0, 0.0))

    assert float(vector.component(0).amplitude.max()) == pytest.approx(0.0)
    assert float(vector.component(1).amplitude.max()) == pytest.approx(1.0)
    assert float(vector.component(2).amplitude.max()) == pytest.approx(0.0)


def test_a_field_vector_is_built_from_three_scalar_fields() -> None:
    parts = [_scalar_field(1.0), _scalar_field(2.0), _scalar_field(3.0)]

    vector = ComplexAmplitude.from_components(*parts)

    assert vector.number_of_components == VECTOR
    for index, part in enumerate(parts):
        torch.testing.assert_close(
            vector.component(index).as_tensor(), part.as_tensor()
        )


def test_stacking_adds_an_axis_where_joining_fills_one() -> None:
    """Three scalar fields can be put together two ways, and they mean different things.
    Stacking adds an axis, giving a batch of three scalar fields, one per element.
    Joining fills the component axis they already have, giving one field vector.
    """
    parts = [_scalar_field(1.0), _scalar_field(2.0), _scalar_field(3.0)]

    stacked = torch.stack(parts, dim=0)
    joined = ComplexAmplitude.from_components(*parts)

    assert stacked.batch_shape == (VECTOR,)
    assert stacked.number_of_components == SCALAR
    assert joined.batch_shape == ()
    assert joined.number_of_components == VECTOR


def test_a_field_vector_cannot_be_polarized_again() -> None:
    vector = _scalar_field().with_polarization((1.0, 0.0, 0.0))

    with pytest.raises(ValueError, match="already carries a field vector"):
        vector.with_polarization((0.0, 1.0, 0.0))


def test_a_jones_vector_is_three_values() -> None:
    with pytest.raises(ValueError, match="Jones vector"):
        _scalar_field().with_polarization((1.0, 0.0))


def test_a_component_out_of_range_says_so() -> None:
    with pytest.raises(IndexError, match="Component 2"):
        _scalar_field().component(2)


# --- through a whole model ----------------------------------------------------------


def _fourier_model(number_of_components: int = SCALAR) -> SLMFFT:
    return SLMFFT(
        input_geometry=_geometry(number_of_components),
        virtual_slm=VirtualSLM(phase_scaling=1.0),
        slm_field=PixelwiseSLMField(),
        focal_length=0.1,
        padded_resolution=(64, 64),
    )


def _chirp_z_model(number_of_components: int = SCALAR) -> SLMCZT:
    return SLMCZT(
        input_geometry=_geometry(number_of_components),
        virtual_slm=VirtualSLM(phase_scaling=1.0),
        slm_field=PixelwiseSLMField(),
        focal_length=0.1,
        camera_resolution=CAMERA_RESOLUTION,
        camera_pixel_size=(20e-6, 20e-6),
    )


MODEL_FACTORIES = {"SLMFFT": _fourier_model, "SLMCZT": _chirp_z_model}


def _showing(model, phase: torch.Tensor | None = None):
    """``model`` with ``phase`` on its SLM.

    The virtual SLM builds its levels on the first pass, so one runs before the phase
    can be set.
    """
    model()
    if phase is not None:
        model.virtual_slm.set_phase(phase)
    return model


def _ramp() -> torch.Tensor:
    values = torch.linspace(0.0, 3.0, SLM_RESOLUTION[0] * SLM_RESOLUTION[1])
    return values.reshape(SLM_RESOLUTION)


@pytest.mark.parametrize("name", list(MODEL_FACTORIES))
def test_a_field_vector_through_a_model_is_its_three_scalar_runs(name) -> None:
    """The acceptance test for the component axis. Nothing in the chain mixes the
    components, so the vector run has to reproduce each scalar one exactly.
    """
    jones = (0.6, 0.0, 0.8)
    phase = _ramp()

    vector_model = _showing(MODEL_FACTORIES[name](), phase)
    vector_out = vector_model(vector_model.init_field.with_polarization(jones))

    scalar_model = _showing(MODEL_FACTORIES[name](), phase)
    scalar_out = scalar_model()

    assert vector_out.number_of_components == VECTOR
    for index, weight in enumerate(jones):
        torch.testing.assert_close(
            vector_out.component(index).as_tensor(),
            scalar_out.as_tensor() * weight,
            rtol=1e-4,
            atol=1e-6,
        )


def test_the_sensor_reads_a_field_vector_as_the_sum_of_its_components() -> None:
    model = _showing(_fourier_model(), _ramp())
    # A deep well, so that nothing clips. Saturation is the one step of the sensor that
    # is not linear, and the components only add below it.
    sensor = CameraSensor(0.5, 1e12, 1e-3, add_noise=False, quantize=False)

    scalar = model()
    vector = model(model.init_field.with_polarization((0.6, 0.0, 0.8)))

    frame = sensor(vector)
    parts = sum(sensor(vector.component(index)) for index in range(VECTOR))

    assert frame.shape == scalar.resolution
    assert float(frame.max()) < sensor.max_pixel_value
    torch.testing.assert_close(frame, parts, rtol=1e-4, atol=1e-4)
    # A unit Jones vector moves no power, so the frame matches the scalar run.
    torch.testing.assert_close(frame, sensor(scalar), rtol=1e-3, atol=1e-3)


def test_a_declared_component_count_reaches_the_modules() -> None:
    """A geometry carrying three components builds a vector input field, and every
    module reports what it passes on.
    """
    model = _fourier_model(number_of_components=VECTOR)

    assert model.init_field.is_vector
    output = model()

    assert output.is_vector
    for module in model.layers().values():
        assert module.number_of_components_out == VECTOR


# --- the costs that measure against a scalar target ---------------------------------


def _image_plane_vector() -> ComplexAmplitude:
    model = _showing(_fourier_model())
    return model(model.init_field.with_polarization((1.0, 0.0, 0.0)))


COSTS_ON_AMPLITUDE_AND_PHASE = {
    "LossFidelity": lambda mask, target: LossFidelity(
        target, torch.zeros_like(target), mask
    ),
    "LossAbsoluteFidelity": lambda mask, target: LossAbsoluteFidelity(
        target, torch.zeros_like(target), mask
    ),
    "LossVorticity": lambda mask, target: LossVorticity(target),
}


@pytest.mark.parametrize("name", list(COSTS_ON_AMPLITUDE_AND_PHASE))
def test_a_cost_measuring_amplitude_and_phase_refuses_a_field_vector(name) -> None:
    field = _image_plane_vector()
    mask = torch.ones(field.resolution)
    cost = COSTS_ON_AMPLITUDE_AND_PHASE[name](mask, torch.ones(field.resolution))

    with pytest.raises(ValueError, match="field vector"):
        cost(field)


def test_an_intensity_cost_accepts_a_field_vector() -> None:
    """An intensity cost measures the irradiance, which sums the components, so a field
    vector needs no special case.
    """
    field = _image_plane_vector()
    mask = torch.ones(field.resolution)

    value = LossIntensityMSE(torch.ones(field.resolution), mask)(field)

    assert torch.isfinite(value)
