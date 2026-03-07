# %% Imports
from collections import OrderedDict
import matplotlib.pyplot as plt

from hologradpy.holography.phase_retrieval import CGPhaseRetriever
from hologradpy.propagation.utils.optics_utils import (
    gaussian_beam_intensity,
    lens_phase,
    rectangular_mask,
    gaussian_blur,
)
from hologradpy.propagation.utils.tensor_utils import (
    check_device,
    gpu_to_numpy,
)
from hologradpy.propagation.optical_systems import SLMFFTAffine, SLMCameraModel
from hologradpy.propagation.virtual_slms import VirtualSLM

from hologradpy.holography.vortices import VortexAnnihilator

from slmsuite.hardware.slms.slm import SLM
from slmsuite.hardware.cameras.simulated import SimulatedCamera as Camera

import torch

# %% Set up the SLM and camera devices
device = check_device(verbose=True)

slm = SLM(
    resolution=(1280, 1024),
    wav_um=0.670,
    pitch_um=12.5,
)

virtual_slm = VirtualSLM(slm, device=device)

camera = Camera(
    slm=slm,
    resolution=(1440, 900),
    pitch_um=3.45,
)

# %% Set up the SLM and camera modules
beam_radius = 4e-3  # beam radius in mm
focal_length = 500e-3  # focal length in mm
padded_resolution = (2048, 2048)  # padded resolution for the FFT

slm_grid = virtual_slm.get_spatial_grid_input()
slm_intensity = gaussian_beam_intensity(*slm_grid, beam_radius=beam_radius)
slm_field = slm_intensity.sqrt() + 0j

plt.figure()
plt.imshow(gpu_to_numpy(slm_intensity), cmap='turbo')
plt.title('SLM Intensity Pattern')
plt.colorbar(label='Intensity [a.u.]')

# TODO: Adapt the previous initial phase guess function.
init_slm_phase = lens_phase(
    *slm_grid,
    focal_length=2.5,
    wavenumber=2 * torch.pi / (slm.wav_um * 1e-6),
).to(torch.float32)

slm_camera_model = SLMFFTAffine(
    virtual_slm=virtual_slm,
    camera=camera,
    focal_length=focal_length,
    constant_field_slm=slm_field,
    padded_resolution=padded_resolution,
    device=device,
)

slm_camera_model = SLMCameraModel(
    OrderedDict([
        ("virtual_slm", virtual_slm),
        ("constant_field", slm_camera_model.constant_field),
        ("fourier_lens", slm_camera_model.fourier_lens),
        # ("affine_transform", slm_camera_model.affine_transform),
    ])
)

# %% Plot initial simulated output
slm_camera_model.virtual_slm.set_phase(init_slm_phase)
init_electric_field = slm_camera_model()
init_intensity = torch.abs(init_electric_field) ** 2

plt.figure()
plt.imshow(gpu_to_numpy(init_intensity), cmap='turbo')
plt.title('Initial Simulated Camera Image')
plt.colorbar(label='Intensity (a.u.)')


slm_power = slm_camera_model.constant_field.amplitude.abs() ** 2
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
plt.imshow(gpu_to_numpy(target_top_hat), cmap='turbo')
plt.title('Target Top Hat Pattern')
plt.colorbar(label='Intensity (a.u.)')


# %% Setting up the phase retrieval module
phase_retriever = CGPhaseRetriever(
    slm_camera_model=slm_camera_model,
    target=target_top_hat,
    signal_region=signal_region,
    init_slm_phase=init_slm_phase,
    device=device,
)

# %% Phase retrieval
phase = phase_retriever.retrieve_phase(50, method="cg")

# %% Plotting the results
electric_field = phase_retriever.slm_camera_model()
intensity_out = torch.abs(electric_field) ** 2
phase_out = torch.angle(electric_field)

plt.figure()
plt.imshow(gpu_to_numpy(phase % (2 * torch.pi)), cmap="magma")
plt.title('Final Retrieved Phase')
plt.colorbar(label='Phase [radians]')

plt.figure()
plt.imshow(gpu_to_numpy(intensity_out), cmap='turbo')
plt.title('Final Retrieved Intensity')
plt.colorbar(label='Intensity [a.u.]')

 # %% Vortex detection
vortex_annihilator = VortexAnnihilator(phase_retriever)
vortex_annihilator.annihilate_vortices(
    target_intensity_threshold=0.05,
    max_iterations=5,
    cg_iterations=20
)

# %% Refine SLM phase after vortex annihilation
phase = phase_retriever.retrieve_phase(100, method="l-bfgs")

# %% Plotting the results
electric_field = phase_retriever.slm_camera_model()
intensity_out = torch.abs(electric_field) ** 2
phase_out = torch.angle(electric_field)

plt.figure()
plt.imshow(gpu_to_numpy(phase % (2 * torch.pi)), cmap="magma")
plt.title('Final Retrieved Phase')
plt.colorbar(label='Phase [radians]')

plt.figure()
plt.imshow(gpu_to_numpy(intensity_out), cmap='turbo')
plt.title('Final Retrieved Intensity')
plt.colorbar(label='Intensity [a.u.]')
# %%
