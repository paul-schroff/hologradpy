"""
Conjugate-gradient phase retrieval
==================================

Finds the SLM phase that forms a target light potential, by minimising a cost function
based on the output of a model that simluates the propagation of light from the SLM to 
the camera.
"""

# %% Imports
import matplotlib.pyplot as plt

from hologradpy.holography.phase_retrieval import CGPhaseRetriever
from hologradpy.profiles.phase import lens_phase
from hologradpy.profiles.amplitude import (
    gaussian_beam_intensity,
    gaussian_blur,
)
from hologradpy.profiles.masks import rectangular_mask
from hologradpy.utils import (
    get_device,
    gpu_to_numpy,
)

from hologradpy.optics.complex_amplitude import (
    ComplexAmplitude,
    FieldGeometry,
)
from hologradpy.optics.systems import SLMFFT
from hologradpy.optics.modules.slm_fields import PixelwiseSLMField
from hologradpy.optics.modules.virtual_slms import VirtualSLM

from hologradpy.holography.vortices import VortexAnnihilator

from hologradpy.hardware import SimulatedSLMTorch, open_slm

import torch

# %% Set up the SLM and camera devices
device = get_device(verbose=True)

slm_geometry = FieldGeometry(
    resolution=(1024, 1280),
    pixel_size=torch.tensor([12.5e-6, 12.5e-6], device=device),
    wavelength=torch.tensor(0.670e-6, device=device),
)

slm = open_slm(SimulatedSLMTorch, input_geometry=slm_geometry, bitdepth=8)

# %% Set up the SLM and camera modules
beam_radius = 4e-3  # beam radius in mm
focal_length = 500e-3  # focal length in mm
padded_resolution = (2048, 2048)  # padded resolution for the FFT

slm_grid = slm_geometry.get_spatial_grid()
slm_intensity = gaussian_beam_intensity(*slm_grid, beam_radius=beam_radius)
slm_field = ComplexAmplitude.from_geometry(
    slm_geometry, data=slm_intensity.sqrt() + 0j
)

plt.figure()
plt.imshow(gpu_to_numpy(slm_intensity), cmap="turbo")
plt.title("SLM Intensity Pattern")
plt.colorbar(label="Intensity [a.u.]")

# TODO: Adapt the previous initial phase guess function.
init_slm_phase = lens_phase(
    *slm_grid,
    focal_length=1.5,
    wavenumber=2 * torch.pi / slm.wavelength,
).to(torch.float32)

slm_camera_model = SLMFFT(
    input_geometry=slm_field.geometry,
    virtual_slm=VirtualSLM(phase_scaling=1.0, init_phase=init_slm_phase),
    slm_field=PixelwiseSLMField(slm_field),
    focal_length=focal_length,
    padded_resolution=padded_resolution,
)

# %% Plot initial simulated output
init_electric_field = slm_camera_model()
init_intensity = init_electric_field.intensity

plt.figure()
plt.imshow(gpu_to_numpy(init_intensity), cmap="turbo")
plt.title("Initial Simulated Camera Image")
plt.colorbar(label="Intensity (a.u.)")

slm_power = slm_camera_model.slm_field.amplitude**2
image_power = init_intensity.sum()

print(f"SLM Power: {slm_power.sum().item()}")
print(f"Image Power: {image_power.item()}")

# %% Setting up the target potential and signal region
camera_grid = slm_camera_model[-1].get_spatial_grid_output()

top_hat_width = 1500e-6
top_hat_height = 700e-6
target_top_hat = rectangular_mask(
    *camera_grid,
    top_hat_width,
    top_hat_height,
    0e-6,
    0e-6,
)

target_top_hat = gaussian_blur(target_top_hat.float(), beam_radius=4)

signal_region = rectangular_mask(
    *camera_grid,
    2 * top_hat_width,
    2 * top_hat_height,
    shift_x=0,
    shift_y=0,
)

plt.figure()
plt.imshow(gpu_to_numpy(target_top_hat), cmap="turbo")
plt.title("Target Top Hat Pattern")
plt.colorbar(label="Intensity (a.u.)")


# %% Setting up the phase retrieval module
phase_retriever = CGPhaseRetriever(
    slm_camera_model=slm_camera_model,
    target=target_top_hat,
    signal_region=signal_region,
    init_slm_phase=init_slm_phase,
)

# %% Phase retrieval
phase = phase_retriever.retrieve_phase(20, method="cg")

# %% Plotting the results
complex_amplitude = phase_retriever.slm_camera_model()
intensity_out = complex_amplitude.intensity
phase_out = complex_amplitude.phase

plt.figure()
plt.imshow(gpu_to_numpy(phase_out % (2 * torch.pi)), cmap="magma")
plt.title("Final Retrieved Phase")
plt.colorbar(label="Phase [radians]")

plt.figure()
plt.imshow(gpu_to_numpy(intensity_out), cmap="turbo")
plt.title("Final Retrieved Intensity")
plt.colorbar(label="Intensity [a.u.]")

# %% Vortex detection
vortex_annihilator = VortexAnnihilator(phase_retriever)
vortex_annihilator.annihilate_vortices(
    target_intensity_threshold=0.1, max_iterations=5, cg_iterations=20
)

# %% Refine SLM phase after vortex annihilation
phase = phase_retriever.retrieve_phase(100, method="l-bfgs")

# %% Plotting the results
complex_amplitude = phase_retriever.slm_camera_model()
intensity_out = complex_amplitude.intensity
phase_out = complex_amplitude.phase

plt.figure()
plt.imshow(gpu_to_numpy(phase) % (2 * torch.pi), cmap="magma")
plt.title("Final Retrieved Phase")
plt.colorbar(label="Phase [radians]")

plt.figure()
plt.imshow(gpu_to_numpy(intensity_out)[700:-700, 700:-700], cmap="turbo")
plt.title("Final Retrieved Intensity")
plt.colorbar(label="Intensity [a.u.]")
# %%
