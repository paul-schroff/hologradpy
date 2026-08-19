"""Tests for the CameraSensor OpticsModule (focal intensity -> camera pixels).

Covers the intensity -> photon -> electron -> ADU chain: shape / wavelength
summation, full-well saturation, bit-depth quantization, the deterministic
differentiable path, read noise, and the wiring into SimulatedCameraTorch.
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
        0.5, 5e4, 1e-3, noise_level=0.0, add_noise=False, quantize=False
    )(field)
    noisy = CameraSensor(
        0.5, 5e4, 1e-3, noise_level=50.0, add_noise=True, quantize=False
    )(field)
    assert float(noiseless.var()) == 0.0   # constant input, no noise -> flat
    assert float(noisy.var()) > 0.0        # read noise adds spread
    assert float(noisy.mean()) > 0.0       # and a positive offset


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
        virtual_slm=VirtualSLM(phase_scaling=1.0),
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
    assert camera.sensor.exposure_time == float(camera.exposure_s)


def test_get_image_torch_backend_matches_numpy() -> None:
    """backend="torch" runs the full pipeline (orientation, ROI crop, averaging) on
    tensors and matches the numpy path value for value.
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

    tensor_image = camera.get_image(backend="torch")
    assert isinstance(tensor_image, torch.Tensor)
    assert tuple(tensor_image.shape) == (10, 8)

    numpy_image = camera.get_image()
    np.testing.assert_array_equal(numpy_image, tensor_image.cpu().numpy())

    tensor_summed = camera.get_image(averaging=3, backend="torch")
    numpy_summed = camera.get_image(averaging=3)
    np.testing.assert_array_equal(numpy_summed, tensor_summed.cpu().numpy())

    with pytest.raises(ValueError):
        camera.get_image(backend="nonsense")


def test_autoexpose_never_accepts_a_saturated_frame() -> None:
    """A clipped frame hides the true peak, so it can never count as converged.

    With ``set_fraction`` close to full scale the error of a saturated frame can
    fall inside ``tolerance`` on its own: at 8 bits, 0.95 targets 243.2 and a
    saturated frame reads 255, an error of 0.046 against the default 0.05. The
    loop then exited immediately and left the exposure untouched, so the speckle
    calibrator was handed completely clipped frames and could not fit anything.
    """
    model = _make_model()
    camera = SimulatedCameraTorch(model, bitdepth=8, noise_level=0.0)

    # Start far enough into saturation that a single gentle step cannot fix it.
    camera.set_exposure(1.0)
    assert float(np.asarray(camera.get_image()).max()) >= camera.adu_levels - 1

    # An explicit budget, well above the default of 5. A clipped frame hides the true
    # peak, so there is no step that lands on the target and the descent is geometric:
    # from this starting point it takes about twenty frames, where an underexposed
    # region takes one. That asymmetry is a property of the descent, not of this test.
    exposure = camera.autoexpose(
        set_fraction=0.95, tolerance=0.05, max_iterations=25
    )

    image = np.asarray(camera.get_image(), dtype=float)
    assert exposure < 1.0                                  # it actually reduced
    assert image.max() < camera.adu_levels - 1             # and is no longer clipped
