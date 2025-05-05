# %% Imports
import matplotlib.pyplot as plt

from hologradpy.holography.phase_retrieval import CGPhaseRetrieval
from hologradpy.torch_modules.utils.optics_utils import (
    gaussian_beam_intensity,
    lens_phase,
    rect_mask,
)
from hologradpy.torch_modules.utils.fourier_utils import get_spatial_grid
from hologradpy.torch_modules.utils.tensor_utils import (
    check_device,
    gpu_to_numpy,
)
from hologradpy.torch_modules.optical_systems import SlmFftAffine

from slmsuite.hardware.slms.slm import SLM
from slmsuite.hardware.cameras.simulated import SimulatedCamera as Camera

import torch
import torch.nn as nn

# %% Set up the SLM and camera devices
device = check_device(verbose=True)

slm = SLM(
    resolution=(1280, 1024),
    wav_um=0.670,
    pitch_um=12.5,
)

camera = Camera(
    slm=slm,
    resolution=(1440, 1024),
    pitch_um=3.75,
)

# %% Set up the SLM and camera modules
beam_radius = 4e-3 # beam radius in mm
focal_length = 250e-3 # focal length in mm
padded_resolution = (2048, 2048) # padded resolution for the FFT

slm_grid = get_spatial_grid(
    torch.tensor(slm.shape),
    torch.tensor(slm.pitch_um * 1e-6),
    device=device
)

slm_intensity = gaussian_beam_intensity(*slm_grid, beam_radius=beam_radius)
slm_field = slm_intensity + 0j

plt.figure()
plt.imshow(gpu_to_numpy(slm_intensity), cmap='turbo')
plt.title('SLM Intensity Pattern')
plt.colorbar(label='Intensity [a.u.]')

# TODO: Adapt the previous initial phase guess function.
init_slm_phase = lens_phase(
    *slm_grid,
    focal_length=5,
    wavenumber=2 * torch.pi / (slm.wav_um * 1e-6),
).to(torch.float32)

slm_camera_module = SlmFftAffine(
    slm_device=slm,
    camera_device=camera,
    focal_length=focal_length,
    constant_field_slm=slm_field,
    init_slm_phase=init_slm_phase,
    padded_resolution=padded_resolution,
    device=device,
)

# %% Plot initial simulated output
init_electric_field = slm_camera_module()
init_intensity = torch.abs(init_electric_field) ** 2

plt.figure()
plt.imshow(gpu_to_numpy(init_intensity), cmap='turbo')
plt.title('Initial Simulated Camera Image')
plt.colorbar(label='Intensity (a.u.)')

# %% Setting up the target potential and signal region
camera_grid = slm_camera_module.affine_transform.get_spatial_grid_output()

top_hat_width = 200e-6
top_hat_height = 200e-6
target_top_hat = rect_mask(*camera_grid, top_hat_width, top_hat_height) + 0.0
signal_region = torch.ones_like(camera_grid[0])

plt.figure()
plt.imshow(gpu_to_numpy(target_top_hat), cmap='turbo')
plt.title('Target Top Hat Pattern')
plt.colorbar(label='Intensity (a.u.)')

# %% Setting up the phase retrieval module
phase_retrieval = CGPhaseRetrieval(
    model=slm_camera_module,
    target=target_top_hat,
    signal_region=signal_region,
    init_slm_phase=init_slm_phase,
)

phase = phase_retrieval.retrieve_phase(100)
intensity_out = torch.abs(phase_retrieval.model()) ** 2

# %% Plotting the results
plt.figure()
plt.imshow(gpu_to_numpy(phase % (2 * torch.pi)), cmap='magma')
plt.title('Retrieved Phase Pattern')
plt.colorbar(label='Phase [radians]')

plt.figure()
plt.imshow(gpu_to_numpy(intensity_out), cmap='turbo')
plt.title('Optimized Simluated Camera Image')
plt.colorbar(label='Intensity [a.u.]')

# %%
