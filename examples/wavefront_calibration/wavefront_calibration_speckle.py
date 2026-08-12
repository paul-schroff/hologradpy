# %% Imports
import os
from typing import Literal

import matplotlib.pyplot as plt
import torch

from hologradpy.hardware import (
    SimulatedSLMTorch,
    SimulatedCameraTorch,
    open_camera,
    open_slm,
)

from hologradpy.calibration import PSFSpeckleCalibrator, PixelwiseSpeckleCalibrator

from hologradpy.optics.systems import SLMCZT
from hologradpy.optics.modules.slm_fields import PixelwiseSLMField
from hologradpy.optics.modules.virtual_slms import VirtualSLM
from hologradpy.optics.complex_amplitude import ComplexAmplitude, FieldGeometry

from hologradpy.profiles.amplitude import gaussian_beam_intensity
from hologradpy.profiles.zernike import Zernike
from hologradpy.utils import Timer, get_device

device = get_device(verbose=True)

data_directory = "../data/"
os.makedirs(data_directory, exist_ok=True)

FOCAL_LENGTH = 0.25
SEED = 0

torch.manual_seed(SEED)

# %% Simulated SLM
input_geometry = FieldGeometry(
    resolution=(1024, 1280),
    pixel_size=torch.tensor([12.5e-6, 12.5e-6], device=device),
    wavelength=torch.tensor(1039e-9, device=device),
)

slm = open_slm(SimulatedSLMTorch, input_geometry=input_geometry, bitdepth=8)

# %% Setting up Simulated camera

# Gaussian beam amplitude profile
gaussian_amplitude = gaussian_beam_intensity(
    *slm.get_spatial_grid(device=device),
    beam_radius=5e-3,
).sqrt()

# Generating aberrations
zernike = Zernike(
    input_geometry.resolution,
    unit_disk_mode="fill",
    number_of_radial_orders=6,
    device=device,
)

coefficient_generator = torch.Generator().manual_seed(SEED)
zernike_coefficients = 0.5 * torch.rand(
    zernike.number_of_zernikes, generator=coefficient_generator
)
injected_phase = zernike.get_phase(zernike_coefficients.to(device))

aberrated_beam = ComplexAmplitude(
    gaussian_amplitude * torch.exp(1j * injected_phase),
    wavelength=input_geometry.wavelength,
    pixel_size=input_geometry.pixel_size,
    power=1e-3,
)

simulated_camera_model = SLMCZT(
    input_geometry=input_geometry,
    virtual_slm=slm.virtual_slm,
    camera_resolution=(900, 1440),
    camera_pixel_size=(3.45e-6, 3.45e-6),
    focal_length=FOCAL_LENGTH,
    slm_field=PixelwiseSLMField(aberrated_beam),
    camera_angle=5.0,
    camera_shift=(-200, 50),
    padded_resolution=(int(1.2 * 900), int(1.2 * 1440)),
)

camera = open_camera(
    SimulatedCameraTorch,
    slm_camera_model=simulated_camera_model,
    bitdepth=12,
    nd_filter_optical_density=6,
    noise_level=4,
)

# %% Test image capture
camera.set_exposure(1e-4)
test_image = camera.get_image()
plt.figure()
plt.imshow(test_image, cmap="turbo")
plt.colorbar()
plt.title("Simulated camera image")

# %% Setting up the learnable model
slm_camera_model = SLMCZT(
    input_geometry=input_geometry,
    virtual_slm=VirtualSLM.from_slm(slm),
    camera_resolution=tuple(camera.resolution),
    camera_pixel_size=tuple(float(pitch) for pitch in camera.pixel_size),
    focal_length=FOCAL_LENGTH,
    slm_field=PixelwiseSLMField(),
    padded_resolution=(int(1.2 * 900), int(1.2 * 1440)),
)

# %% Calibrate the static SLM field model
PARAMETERIZATION: Literal["psf", "pixel_wise"] = "psf"
NUMBER_OF_PATTERNS = 10

if PARAMETERIZATION == "psf":
    NUMBER_OF_EPOCHS = 50
    BATCH_SIZE = 5

    calibrator = PSFSpeckleCalibrator(
        slm,
        camera,
        slm_camera_model=slm_camera_model,
        dataset_directory=data_directory,
        number_of_random_patterns=NUMBER_OF_PATTERNS,
    )
elif PARAMETERIZATION == "pixel_wise":
    NUMBER_OF_EPOCHS = 1000
    BATCH_SIZE = 10

    calibrator = PixelwiseSpeckleCalibrator(
        slm,
        camera,
        slm_camera_model=slm_camera_model,
        dataset_directory=data_directory,
        number_of_random_patterns=NUMBER_OF_PATTERNS,
    )

with Timer("Wavefront calibration", verbose=True) as timer:
    calibration = calibrator.calibrate(
        seed=SEED,
        number_of_epochs=NUMBER_OF_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=True,
        # speckle_pattern_extent=(2e-3, 2e-3),
    )

total = timer.elapsed_time
print()
print(f"{device.type}: {total:.1f} s total")

# %% Calibration diagnostics
calibration.visualization_data.visualizer().render()

# %% Recovered vs injected SLM-plane wavefront
calibration.visualization_data.visualizer().render_comparison()

metadata = calibration.metadata
residual = metadata["residual_phase_rms"]
print(f"Residual wavefront error: {residual:.4f} rad (lambda/{6.283 / residual:.0f})")
print(f"Aberration left after correction: {metadata['residual_fraction']:.1%}")

plt.show()

# %%
