# %% Imports
import matplotlib.pyplot as plt
import torch

from hologradpy.hardware import SimulatedSLMTorch, SimulatedCameraTorch

from hologradpy.calibration.camera_mapping import CheckerboardMapper

from hologradpy.propagation.optical_systems import SLMFFT, SLMFFTAffine

from hologradpy.propagation.utils.optics_utils import gaussian_beam_intensity
from hologradpy.propagation.utils.tensor_utils import (
    check_device, gpu_to_numpy
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

camera.set_exposure(0.001)
test_image = camera.get_image()

# %%
slm_camera_model = SLMFFT(
    slm.virtual_slm,
    focal_length=0.25,
    constant_field_slm=torch.tensor(gaussian_beam, dtype=torch.complex64),
    device=device,
    padded_resolution=(2048, 2048),
)

# %% Camera mapping
camera_mapper = CheckerboardMapper(
    slm=slm,
    camera=camera,
    slm_camera_model=slm_camera_model,
    device=device,
)

camera_mapping = camera_mapper.map_camera(
    number_of_squares=(7, 9),
    square_size=16,
    number_of_cg_iterations=50,
    checkerboard_center="top-left",
)

camera_image = camera_mapping.camera_images[0]
simulated_image = camera_mapping.simulated_images[0]
slm_phase = slm_camera_model.virtual_slm.phase.detach().cpu().numpy()

# %% Plotting results
# TODO: Tidy up plotting
plt.figure()
plt.imshow(camera_image, cmap='turbo')
plt.plot(
    camera_mapping.detected_points[:, 0],
    camera_mapping.detected_points[:, 1],
    "wx"
)
plt.title("Captured Camera Image with Detected Corners")

plt.figure()
plt.imshow(simulated_image, cmap='turbo')
plt.plot(
    camera_mapping.calculated_points[:, 0],
    camera_mapping.calculated_points[:, 1],
    "wx"
)
plt.title("Simulated Camera Image with Detected Corners")

plt.figure()
plt.imshow(slm_phase, cmap='magma')
plt.colorbar()
plt.title("SLM Phase")
# %%
