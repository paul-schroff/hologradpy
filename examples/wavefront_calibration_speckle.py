# %% Imports
import matplotlib.pyplot as plt
import torch

from hologradpy.hardware import SimulatedSLMTorch, SimulatedCameraTorch

from hologradpy.calibration import (
    SpeckleCalibrator, CameraMapping
)

from hologradpy.propagation.optical_systems import SLMFFTAffine

from hologradpy.propagation.utils.optics_utils import (
    gaussian_beam_intensity,
)
from hologradpy.propagation.utils.tensor_utils import (
    check_device,
)

device = check_device(verbose=True)

# %% Initializing simulated SLM and camera
slm = SimulatedSLMTorch(
    resolution=(1280, 1024),
    pitch_um=12.5,
    wav_um=0.630,
    torch_device=device,
    bitdepth=8,
)

gaussian_beam = gaussian_beam_intensity(
    *slm.virtual_slm.get_spatial_grid_input(),
    beam_radius=5e-3,
)

slm_fft_affine_args = {
    "focal_length": 0.25,
    "constant_field_slm": torch.tensor(gaussian_beam, dtype=torch.complex64),
    "device": device,
    "padded_resolution": (2048, 2048),
}

camera = SimulatedCameraTorch(
    slm,
    resolution=(1440, 960),
    pitch_um=(3.75, 3.75),
    slm_camera_model_cls=SLMFFTAffine,
    slm_camera_model_args=slm_fft_affine_args,
)

# %% Test image capture
camera.set_exposure(0.001)
test_image = camera.get_image()
plt.figure()
plt.imshow(test_image)
plt.colorbar()

# %% Loading camera mapping data
camera_mapping = CameraMapping.load("data/camera_mapping.pkl")

# %%
directory = "data/"

calibrator = SpeckleCalibrator(
    slm,
    camera,
    camera_mapping,
    focal_length=0.25,
    directory=directory,
    number_of_random_patterns=10,
)

# %%
calibrator.calibrate_wavefront(
    speckle_pattern_extent=(1.5e-3, 1.5e-3),
    number_of_epochs=50,
    batch_size=5,
)

# %%
