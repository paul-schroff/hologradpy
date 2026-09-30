"""Tests for the CameraSensor OpticsModule (focal intensity -> camera pixels).

Covers the intensity -> photon -> electron -> ADU chain: shape / wavelength
summation, full-well saturation, bit-depth quantization, the deterministic
differentiable path, shot and read noise, and the wiring into SimulatedCameraTorch.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.constants import Planck, speed_of_light

from hologradpy.optics.complex_amplitude import ComplexAmplitude, FieldGeometry
from hologradpy.optics.modules.hardware_models import CameraSensor
from hologradpy.optics.modules.slm_fields import PixelwiseSLMField
from hologradpy.optics.modules.virtual_slms.abstract import VirtualSLM
from hologradpy.optics.systems import SLMFFTAffine
from hologradpy.hardware import CameraOrientation, SimulatedCameraTorch
from hologradpy.roi import ROI


pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

PIXEL = (5e-6, 5e-6)
WAVELENGTH = 800e-9


def constant_field(shape, value, wavelength=WAVELENGTH, pixel=PIXEL):
    data = torch.full(shape, value, dtype=torch.complex64)
    return ComplexAmplitude(data, torch.tensor(wavelength), pixel)


def test_output_is_real_image_of_input_shape() -> None:
    field = constant_field((8, 12), 0.5)
    out = CameraSensor(0.5, 1e5, 1e-3, add_noise=False, quantize=False)(field)
    assert out.shape == (8, 12)
    assert not out.is_complex()


def test_deterministic_expected_adu() -> None:
    intensity = 0.3
    field = constant_field((8, 8), math.sqrt(intensity))
    sensor = CameraSensor(
        quantum_efficiency=0.4, full_well_capacity=5e4, exposure_time=2e-3,
        bitdepth=10, add_noise=False, quantize=False,
    )
    out = sensor(field)

    photon_energy = Planck * speed_of_light / WAVELENGTH
    photons = intensity * (PIXEL[0] * PIXEL[1]) / photon_energy * 2e-3
    electrons = photons * 0.4
    adu = electrons / 5e4 * (2**10 - 1)
    torch.testing.assert_close(
        out, torch.full((8, 8), adu, dtype=out.dtype), rtol=1e-3, atol=1e-3
    )


def test_multiwavelength_sums_to_one_image() -> None:
    generator = torch.Generator().manual_seed(0)
    data = (0.1 * torch.rand(2, 6, 6, generator=generator)).to(torch.complex64)
    wavelengths = torch.tensor([800e-9, 900e-9])
    sensor = CameraSensor(0.4, 1e6, 1e-3, add_noise=False, quantize=False)

    out = sensor(ComplexAmplitude(data, wavelengths, PIXEL))
    out0 = sensor(ComplexAmplitude(data[0], torch.tensor(800e-9), PIXEL))
    out1 = sensor(ComplexAmplitude(data[1], torch.tensor(900e-9), PIXEL))

    assert out.shape == (6, 6)
    # No saturation/noise/quantize -> the chain is linear, so the multi-wavelength
    # image is the sum of the per-wavelength images.
    torch.testing.assert_close(out, out0 + out1, rtol=1e-4, atol=1e-4)


def test_full_well_saturation_clips_to_max() -> None:
    field = constant_field((8, 8), 1e3)  # huge intensity -> saturate
    sensor = CameraSensor(0.5, 1e4, 1e-3, bitdepth=8, add_noise=False)
    out = sensor(field)
    assert torch.all(out == sensor.max_pixel_value)


def test_quantize_gives_integer_values_in_range() -> None:
    field = constant_field((8, 8), 0.2)
    sensor = CameraSensor(0.5, 5e4, 1e-3, bitdepth=8, add_noise=False, quantize=True)
    out = sensor(field)
    assert torch.all(out == out.floor())
    assert out.min() >= 0
    assert out.max() <= sensor.max_pixel_value


def test_differentiable_path_has_gradient() -> None:
    generator = torch.Generator().manual_seed(1)
    data = (
        0.3 * (torch.rand(8, 8, generator=generator)
               + 1j * torch.rand(8, 8, generator=generator))
    ).to(torch.complex64).requires_grad_(True)
    field = ComplexAmplitude(data, torch.tensor(WAVELENGTH), PIXEL)

    sensor = CameraSensor(0.5, 1e6, 1e-3, add_noise=False, quantize=False)
    sensor(field).sum().backward()

    assert data.grad is not None
    assert torch.isfinite(data.grad).all()
    assert float(data.grad.abs().sum()) > 0.0


def test_read_noise_increases_variance() -> None:
    field = constant_field((32, 32), 0.0)  # no signal -> isolate the read noise
    noiseless = CameraSensor(
        0.5, 5e4, 1e-3, read_noise=0.0, add_noise=False, quantize=False
    )(field)
    noisy = CameraSensor(
        0.5, 5e4, 1e-3, read_noise=50.0, add_noise=True, quantize=False
    )(field)
    assert float(noiseless.var()) == 0.0   # constant input, no noise -> flat
    assert float(noisy.var()) > 0.0        # read noise adds spread
    assert float(noisy.mean()) > 0.0       # clipped at zero, so the mean is positive


# A constant field bringing about 15000 electrons to each pixel, and a read noise whose
# variance of 1600 electrons squared is about a tenth of the shot noise.
NOISE_FIELD_VALUE = math.sqrt(0.3)
READ_NOISE = 40.0

# The dark current in electrons per second, 100 electrons in a 1 ms exposure.
DARK_CURRENT = 1e5


def _electron_counting_sensor(exposure_time: float = 1e-3, **kwargs) -> CameraSensor:
    """A sensor reading one ADU per electron, far below its full well."""
    return CameraSensor(
        quantum_efficiency=0.5,
        full_well_capacity=2**24 - 1,
        exposure_time=exposure_time,
        bitdepth=24,
        quantize=False,
        **kwargs,
    )


def _expected_electrons(field: ComplexAmplitude) -> float:
    return float(_electron_counting_sensor(add_noise=False)(field).mean())


def test_shot_noise_draws_the_signal_as_a_poisson_count() -> None:
    """Without read noise, the variance of the electrons equals their mean."""
    torch.manual_seed(0)
    field = constant_field((256, 256), NOISE_FIELD_VALUE)
    expected = _expected_electrons(field)

    electrons = _electron_counting_sensor(read_noise=0.0)(field)

    assert float(electrons.mean()) == pytest.approx(expected, rel=1e-3)
    assert float(electrons.var()) == pytest.approx(expected, rel=0.03)


def test_the_read_noise_adds_its_variance_to_the_shot_noise() -> None:
    """The read noise has zero mean, so it leaves the mean electrons alone."""
    torch.manual_seed(1)
    field = constant_field((256, 256), NOISE_FIELD_VALUE)
    expected = _expected_electrons(field)
    read_variance = READ_NOISE**2

    electrons = _electron_counting_sensor(read_noise=READ_NOISE)(field)

    assert float(electrons.mean()) == pytest.approx(expected, rel=1e-3)
    assert float(electrons.var()) == pytest.approx(expected + read_variance, rel=0.03)


def test_without_shot_noise_only_the_read_noise_is_drawn() -> None:
    torch.manual_seed(2)
    field = constant_field((256, 256), NOISE_FIELD_VALUE)
    expected = _expected_electrons(field)

    electrons = _electron_counting_sensor(
        read_noise=READ_NOISE, shot_noise=False
    )(field)

    assert float(electrons.mean()) == pytest.approx(expected, rel=1e-3)
    assert float(electrons.var()) == pytest.approx(READ_NOISE**2, rel=0.03)


@pytest.mark.parametrize("shot_noise", [True, False])
@pytest.mark.parametrize("exposure_time", [1e-3, 4e-3])
def test_the_dark_current_grows_with_the_exposure_time(
    exposure_time: float, shot_noise: bool
) -> None:
    """The dark electrons are a Poisson count of the dark current over the exposure, so
    their mean and their variance both grow with the exposure time. The shot noise of
    the signal has no part in it.
    """
    torch.manual_seed(3)
    field = constant_field((256, 256), 0.0)
    dark_electrons = DARK_CURRENT * exposure_time

    electrons = _electron_counting_sensor(
        exposure_time=exposure_time, dark_current=DARK_CURRENT, shot_noise=shot_noise
    )(field)

    assert float(electrons.mean()) == pytest.approx(dark_electrons, rel=1e-2)
    assert float(electrons.var()) == pytest.approx(dark_electrons, rel=0.03)


def test_the_noisy_electrons_carry_the_gradient_of_the_expected_ones() -> None:
    """torch.poisson passes no gradient, so the draw hands the gradient of the
    expected electrons through.
    """
    generator = torch.Generator().manual_seed(1)
    data = (
        0.3 * (torch.rand(8, 8, generator=generator)
               + 1j * torch.rand(8, 8, generator=generator))
    ).to(torch.complex64)
    gradients = []
    for add_noise in (False, True):
        leaf = data.clone().requires_grad_(True)
        sensor = CameraSensor(
            0.5, 1e6, 1e-3, read_noise=4.0, add_noise=add_noise, quantize=False
        )
        sensor(ComplexAmplitude(leaf, torch.tensor(WAVELENGTH), PIXEL)).sum().backward()
        gradients.append(leaf.grad)

    noiseless, noisy = gradients
    assert float(noiseless.abs().sum()) > 0.0
    torch.testing.assert_close(noisy, noiseless)


def test_any_noise_source_makes_the_sensor_stochastic() -> None:
    assert CameraSensor(read_noise=0.0).is_stochastic
    assert not CameraSensor(read_noise=0.0, shot_noise=False).is_stochastic
    assert CameraSensor(read_noise=4.0, shot_noise=False).is_stochastic
    assert CameraSensor(shot_noise=False, dark_current=DARK_CURRENT).is_stochastic
    assert not CameraSensor(add_noise=False, dark_current=DARK_CURRENT).is_stochastic


# --- Integration with SimulatedCameraTorch -----------------------------------
def _make_model():
    geometry = FieldGeometry(
        wavelength=torch.tensor([800e-9]),
        pixel_size=torch.tensor([[10e-6, 10e-6]]),
        resolution=(32, 32),
    )
    static = PixelwiseSLMField(
        ComplexAmplitude(
            torch.ones((32, 32), dtype=torch.complex64),
            geometry.wavelength,
            geometry.pixel_size,
        )
    )
    return SLMFFTAffine(
        input_geometry=geometry,
        virtual_slm=VirtualSLM(full_scale_cycles=1.0),
        camera_resolution=(24, 24),
        camera_pixel_size=(20e-6, 20e-6),
        focal_length=0.1,
        slm_field=static,
        padded_resolution=(64, 64),
    )


def test_simulated_camera_appends_sensor_and_emits_pixels() -> None:
    model = _make_model()
    camera = SimulatedCameraTorch(
        model, quantum_efficiency=0.5, full_well_capacity=1e5, bitdepth=8
    )

    # A CameraSensor was built from the kwargs and appended as the last module.
    assert isinstance(model[-1], CameraSensor)
    assert camera.sensor is model[-1]

    image = camera._capture_frame()
    assert image.shape == (24, 24)
    assert torch.all(image == image.floor())
    assert image.min() >= 0
    assert image.max() <= camera.sensor.max_pixel_value


def test_camera_exposure_drives_sensor() -> None:
    model = _make_model()
    camera = SimulatedCameraTorch(model, full_well_capacity=1e6)

    camera.set_exposure(5e-3)
    camera._capture_frame()
    assert camera.sensor.exposure_time == float(camera.exposure)


def test_get_image_tensor_matches_get_image() -> None:
    """The tensor path runs the full pipeline (orientation, ROI crop, averaging) on
    tensors and matches the numpy frame value for value.
    """
    model = _make_model()
    camera = SimulatedCameraTorch(
        model,
        orientation=CameraOrientation("90", fliplr=True),
        add_noise=False,
        full_well_capacity=1e6,
    )
    camera.set_exposure(1e-3)
    camera.set_roi(ROI(2, 3, 10, 8))

    tensor_image = camera.get_image_tensor()
    assert isinstance(tensor_image, torch.Tensor)
    assert tuple(tensor_image.shape) == (10, 8)

    numpy_image = camera.get_image()
    np.testing.assert_array_equal(numpy_image, tensor_image.cpu().numpy())

    tensor_summed = camera.get_image_tensor(averaging=3)
    numpy_summed = camera.get_image(averaging=3)
    np.testing.assert_array_equal(numpy_summed, tensor_summed.cpu().numpy())


def test_autoexpose_never_accepts_a_saturated_frame() -> None:
    """An overexposed frame hides the true peak, so it can never count as converged.

    With ``set_fraction`` close to full scale the error of a saturated frame can
    fall inside ``tolerance`` on its own: at 8 bits, 0.95 targets 243.2 and a
    saturated frame reads 255, an error of 0.046 against the default 0.05. The
    loop then exited immediately and left the exposure untouched, so the speckle
    calibrator was handed completely overexposed frames and could not fit anything.
    """
    model = _make_model()
    camera = SimulatedCameraTorch(model, bitdepth=8, read_noise=0.0)

    # Start far enough into saturation that a single gentle step cannot fix it.
    camera.set_exposure(1.0)
    assert float(np.asarray(camera.get_image()).max()) >= camera.adu_levels - 1

    # A budget above the default of 5. An overexposed frame hides the true peak, so the
    # search cuts the exposure until a frame falls below full scale, then steps between
    # the exposures either side of full scale onto the target.
    exposure = camera.autoexpose(
        set_fraction=0.95, tolerance=0.05, max_iterations=25
    )

    image = np.asarray(camera.get_image(), dtype=float)
    assert exposure < 1.0                                  # it actually reduced
    assert image.max() < camera.adu_levels - 1             # and is not overexposed


def test_a_field_vector_reads_as_the_sum_of_its_components() -> None:
    """A sensor measures irradiance, so the three components of a field vector land in
    one frame and add. Nothing about the frame says the field was a vector.
    """
    generator = torch.Generator().manual_seed(3)
    data = (0.1 * torch.rand(3, 1, 6, 6, generator=generator)).to(torch.complex64)
    sensor = CameraSensor(0.4, 1e6, 1e-3, add_noise=False, quantize=False)

    together = sensor(ComplexAmplitude(data, torch.tensor(WAVELENGTH), PIXEL))
    apart = sum(
        sensor(
            ComplexAmplitude(
                data[component : component + 1], torch.tensor(WAVELENGTH), PIXEL
            )
        )
        for component in range(3)
    )

    assert together.shape == (6, 6)
    torch.testing.assert_close(together, apart, rtol=1e-4, atol=1e-4)
