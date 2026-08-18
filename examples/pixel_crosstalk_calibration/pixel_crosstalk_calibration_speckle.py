# %% Imports
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import torch

from hologradpy.hardware import (
    SimulatedSLMTorch,
    SimulatedCameraTorch,
    open_camera,
    open_slm,
)

from hologradpy.calibration import CrosstalkSpeckleCalibrator

from hologradpy.optics.systems import SLMCZT
from hologradpy.optics.modules.pixel_crosstalk import (
    FreeKernelCrosstalk,
    SuperGaussianCrosstalk,
)
from hologradpy.optics.modules.slm_fields import PixelwiseSLMField
from hologradpy.optics.modules.virtual_slms import VirtualSLM
from hologradpy.optics.complex_amplitude import ComplexAmplitude, FieldGeometry

from hologradpy.profiles.amplitude import gaussian_beam_intensity
from hologradpy.utils import Timer, get_device

device = get_device(verbose=True)

data_directory = Path("../data/")
data_directory.mkdir(parents=True, exist_ok=True)
dataset_path = data_directory / "crosstalk_dataset.asdf"

FOCAL_LENGTH = 0.1
SEED = 0

# Sub-pixels across one SLM pixel.
UPSCALE_FACTOR = 5
EXTENT = 3

INJECTED_ORDER = 1.20
INJECTED_WIDTH = 2.03

torch.manual_seed(SEED)

# %% Simulated SLM
input_geometry = FieldGeometry(
    resolution=(256, 320),
    pixel_size=torch.tensor([12.5e-6, 12.5e-6], device=device),
    wavelength=torch.tensor(1039e-9, device=device),
)

slm = open_slm(SimulatedSLMTorch, input_geometry=input_geometry, bitdepth=8)

# %% Setting up the simulated camera

# Gaussian beam amplitude profile
gaussian_amplitude = gaussian_beam_intensity(
    *slm.get_spatial_grid(device=device),
    beam_radius=1.2e-3,
).sqrt()

gaussian_beam = ComplexAmplitude(
    gaussian_amplitude + 0j,
    wavelength=input_geometry.wavelength,
    pixel_size=input_geometry.pixel_size,
    power=1e-3,
)

CAMERA_RESOLUTION = (400, 640)
CAMERA_PIXEL_SIZE = (3.45e-6, 3.45e-6)
PADDED_RESOLUTION = (int(1.2 * CAMERA_RESOLUTION[0]), int(1.2 * CAMERA_RESOLUTION[1]))

simulated_camera_model = SLMCZT(
    input_geometry=input_geometry,
    virtual_slm=slm.virtual_slm,
    camera_resolution=CAMERA_RESOLUTION,
    camera_pixel_size=CAMERA_PIXEL_SIZE,
    focal_length=FOCAL_LENGTH,
    slm_field=PixelwiseSLMField(gaussian_beam),
    camera_angle=5.0,
    camera_shift=(-300e-6, 100e-6),
    padded_resolution=PADDED_RESOLUTION,
)

# The fringing field goes on the simulated camera only. The model being fitted starts
# from a different kernel, so the calibration has something to recover.
camera = open_camera(
    SimulatedCameraTorch,
    slm_camera_model=simulated_camera_model,
    bitdepth=12,
    nd_filter_optical_density=6,
    noise_level=4,
    crosstalk_upscale_factor=UPSCALE_FACTOR,
    crosstalk_extent=EXTENT,
    crosstalk_order=INJECTED_ORDER,
    crosstalk_width=INJECTED_WIDTH,
)

# %% Test image capture
camera.set_exposure(1e-4)
test_image = camera.get_image()
plt.figure()
plt.imshow(test_image, cmap="turbo")
plt.colorbar()
plt.title("Simulated camera image")

# %% Setting up the learnable model
# The crosstalk fit holds the SLM-plane beam fixed, so the wavefront has to be
# calibrated first. The simulated bench carries an unaberrated beam, so there is no
# wavefront to recover and the model can be given the same flat one.
slm_camera_model = SLMCZT(
    input_geometry=input_geometry,
    virtual_slm=VirtualSLM.from_slm(slm),
    camera_resolution=tuple(camera.resolution),
    camera_pixel_size=tuple(float(pitch) for pitch in camera.pixel_size),
    focal_length=FOCAL_LENGTH,
    slm_field=PixelwiseSLMField(gaussian_beam),
    padded_resolution=PADDED_RESOLUTION,
)

# %% Calibrate the pixel-crosstalk model
PARAMETERIZATION: Literal["super_gaussian", "free_kernel"] = "free_kernel"
NUMBER_OF_PATTERNS = 10

if PARAMETERIZATION == "super_gaussian":
    NUMBER_OF_EPOCHS = 100
    BATCH_SIZE = 5

    # Started well away from the injected values, at a near-Gaussian kernel wide enough
    # in frequency to be almost no crosstalk at all.
    pixel_crosstalk = SuperGaussianCrosstalk(
        upscale_factor=UPSCALE_FACTOR, extent=EXTENT, order=2.0, width=4.0
    )
elif PARAMETERIZATION == "free_kernel":
    NUMBER_OF_EPOCHS = 100
    BATCH_SIZE = 5

    # Every sample learned on its own, starting from a center delta.
    pixel_crosstalk = FreeKernelCrosstalk(
        upscale_factor=UPSCALE_FACTOR, extent=EXTENT
    )

calibrator = CrosstalkSpeckleCalibrator(
    slm,
    camera,
    slm_camera_model=slm_camera_model,
    dataset_path=dataset_path,
    pixel_crosstalk=pixel_crosstalk,
    number_of_random_patterns=NUMBER_OF_PATTERNS,
)

with Timer("Pixel crosstalk calibration", verbose=True) as timer:
    calibration = calibrator.calibrate(
        seed=SEED,
        number_of_epochs=NUMBER_OF_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=True,
    )

total = timer.elapsed_time
print()
print(f"{device.type}: {total:.1f} s total")

# %% Calibration diagnostics
calibration.visualization_data.visualizer().render()

# %% Recovered vs injected kernel
calibration.visualization_data.visualizer().render_comparison()

print(f"Model: {calibration.model}")
print(f"Light kept inside its own pixel: {calibration.central_pixel_weight:.1%}")
for name, value in calibration.parameters.items():
    if isinstance(value, float):
        print(f"  {name}: {value:.4f}")

metadata = calibration.metadata
if "kernel_relative_rms_error" in metadata:
    print(f"Kernel error: {metadata['kernel_relative_rms_error']:.1%} rms")

plt.show()

# %%
