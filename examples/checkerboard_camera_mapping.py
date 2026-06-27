# %% Imports
import matplotlib.pyplot as plt
import torch

from hologradpy.hardware.torch_slm import SimulatedSLMTorch
from hologradpy.hardware.torch_camera import SimulatedCameraTorch

from hologradpy.calibration.camera_mapping import CheckerboardMapper

from hologradpy.propagation.optical_systems import SLMFFT, SLMFFTAffine
from hologradpy.propagation.complex_amplitude import (
    ComplexAmplitude, FieldGeometry
)

from hologradpy.propagation.utils.optics_utils import gaussian_beam_intensity
from hologradpy.propagation.utils.tensor_utils import get_device

device = get_device(verbose=True)
data_path = "data/"
# %matplotlib qt5

# %% Initializing simulated SLM and camera
slm_geometry = FieldGeometry(
    resolution=(1024, 1280),
    pixel_size=torch.tensor([12.5e-6, 12.5e-6], device=device),
    wavelength=torch.tensor(0.630e-6, device=device),
)

slm = SimulatedSLMTorch(input_geometry=slm_geometry, bitdepth=8)

gaussian_intensity = gaussian_beam_intensity(
    *slm.get_spatial_grid(), beam_radius=5e-3,
)
gaussian_beam = ComplexAmplitude(
    gaussian_intensity.sqrt() + 0j, 
    wavelength=slm_geometry.wavelength, 
    pixel_size=slm_geometry.pixel_size
)

simulated_camera_model = SLMFFTAffine(
    input_geometry=slm_geometry,
    virtual_slm=slm.virtual_slm,
    camera_resolution=(960, 1440),
    camera_pixel_size=(3.75e-6, 3.75e-6),
    focal_length=0.25,
    constant_field_slm=gaussian_beam,
    padded_resolution=(2048, 2048),
    camera_angle=0,
    camera_shift=(0, 0),
)

camera = SimulatedCameraTorch(simulated_camera_model)

camera.set_exposure(0.001)
test_image = camera.get_image()

plt.figure()
plt.imshow(test_image, cmap='turbo')
plt.title('Initial Simulated Camera Image')

# %%
slm_camera_model = SLMFFT(
    input_geometry=slm_geometry,
    focal_length=0.25,
    constant_field_slm=gaussian_beam,
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

# %% Saving results
camera_mapping.save(data_path + "camera_mapping.pkl")
slm_camera_model.save(data_path + "slm_camera_model.pkl")
simulated_camera_model.save(data_path + "simulated_camera_model.pkl")

# %% Plotting results
camera_image = camera_mapping.camera_images[0]
simulated_image = camera_mapping.simulated_images[0]
slm_phase = slm_camera_model.virtual_slm.phase.detach().cpu().numpy()

# TODO: Tidy up plotting
print("Transformation matrix:")
print(camera_mapping.transform)
print("Inverse transformation matrix:")
print(camera_mapping.inverse_transform)

plt.figure()
plt.imshow(camera_image, cmap='turbo')
plt.plot(
    camera_mapping.detected_points[:, 0],
    camera_mapping.detected_points[:, 1],
    "wx",
    label="detected corners",
)
plt.plot(
    camera_mapping.zeroth_order_position[1],
    camera_mapping.zeroth_order_position[0],
    "r+",
    label="zeroth order position",
)
plt.plot(
    camera.shape[1] // 2,
    camera.shape[0] // 2,
    "w*",
    label="camera sensor center",
)
plt.legend()

plt.title("Camera Image with Detected Corners")

# %%
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
