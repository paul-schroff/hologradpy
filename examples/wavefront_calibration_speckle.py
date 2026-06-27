# %% Imports
import matplotlib.pyplot as plt
import torch

from hologradpy.hardware import SimulatedSLMTorch, SimulatedCameraTorch

from hologradpy.calibration import SpeckleCalibrator, CameraMapping

from hologradpy.propagation.optical_systems import SLMFFTAffine, SLMNUFFTAffine
from hologradpy.propagation.diagonal_elements import StaticSLMField
from hologradpy.propagation.complex_amplitude import (
    ComplexAmplitude, FieldGeometry,
)

from hologradpy.propagation.amplitude_profiles import gaussian_beam_intensity
from hologradpy.utils import (
    get_device,
)

device = get_device(verbose=True)

# %% Initializing simulated SLM and camera
input_geometry = FieldGeometry(
    resolution=(1280, 1024),
    pixel_size=torch.tensor([12.5e-6, 12.5e-6], device=device),
    wavelength=torch.tensor(630e-9, device=device),
)

slm = SimulatedSLMTorch(input_geometry=input_geometry, bitdepth=8)

gaussian_beam_amplitude = gaussian_beam_intensity(
    *slm.get_spatial_grid(device=device),
    beam_radius=5e-3,
) ** 0.5

constant_field_slm = ComplexAmplitude(
    gaussian_beam_amplitude + 0j,
    wavelength=input_geometry.wavelength,
    pixel_size=input_geometry.pixel_size,
)

simulated_camera_model = SLMNUFFTAffine(
    input_geometry=input_geometry,
    virtual_slm=slm.virtual_slm,
    camera_resolution=(960, 1440),
    camera_pixel_size=(3.45e-6, 3.45e-6),
    focal_length=0.25,
    static_slm_field=StaticSLMField(constant_field_slm),
    camera_angle=2.0,
    camera_shift=(-20, 10),
)

camera = SimulatedCameraTorch(simulated_camera_model, bitdepth=12)

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
