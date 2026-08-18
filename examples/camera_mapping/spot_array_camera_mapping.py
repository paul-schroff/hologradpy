# %% Imports
import matplotlib.pyplot as plt
import numpy as np
import torch

from hologradpy.hardware import (
    SimulatedSLMTorch,
    SimulatedCameraTorch,
    open_camera,
    open_slm,
)

from hologradpy.calibration.camera_mapping import (
    CoarseMapper,
    SpotArrayMapper,
    CameraMapperVisualizer,
)

from hologradpy.optics.systems import SLMFFT, SLMCZT
from hologradpy.optics.modules.slm_fields import PixelwiseSLMField
from hologradpy.optics.modules.virtual_slms import VirtualSLM
from hologradpy.optics.complex_amplitude import ComplexAmplitude, FieldGeometry

from hologradpy.profiles.amplitude import gaussian_beam_intensity
from hologradpy.profiles.zernike import Zernike
from hologradpy.utils import get_device, gpu_to_numpy

device = get_device(verbose=True)
data_path = "../data/"
# %matplotlib qt5

# %% Initializing simulated SLM and camera
slm_geometry = FieldGeometry(
    resolution=(1024, 1280),
    pixel_size=torch.tensor([12.5e-6, 12.5e-6], device=device),
    wavelength=torch.tensor(0.630e-6, device=device),
)

slm = open_slm(SimulatedSLMTorch, input_geometry=slm_geometry, bitdepth=8)

gaussian_intensity = gaussian_beam_intensity(
    *slm.get_spatial_grid(device),
    beam_radius=5e-3,
)

zernike = Zernike(
    slm_geometry.resolution,
    unit_disk_mode="fill",
    number_of_radial_orders=10,
    device=device,
)
coefficients = torch.rand(zernike.number_of_zernikes, device=device) * 1
zernike_phase = zernike.get_phase(coefficients)

plt.figure()
plt.imshow(zernike_phase.cpu(), cmap="magma")
plt.colorbar()
plt.title("Injected Zernike Aberrations")

aberrated_beam = ComplexAmplitude(
    gaussian_intensity.sqrt() * torch.exp(1j * zernike_phase),
    wavelength=slm_geometry.wavelength,
    pixel_size=slm_geometry.pixel_size,
    power=1e-3,
)
gaussian_beam = ComplexAmplitude(
    gaussian_intensity.sqrt() + 0j,
    wavelength=slm_geometry.wavelength,
    pixel_size=slm_geometry.pixel_size,
    power=1e-3,
)

simulated_camera_model = SLMCZT(
    input_geometry=slm_geometry,
    virtual_slm=slm.virtual_slm,
    camera_resolution=(960, 1440),
    camera_pixel_size=(3.45e-6, 3.45e-6),
    focal_length=0.25,
    slm_field=PixelwiseSLMField(aberrated_beam),
    camera_angle=10,
    camera_shift=(70e-6, -35e-6),         # (x, y) metres in the focal plane
)

camera = open_camera(
    SimulatedCameraTorch,
    slm_camera_model=simulated_camera_model,
    quantum_efficiency=0.01,
    full_well_capacity=11e3,
    noise_level=4.0,
    nd_filter_optical_density=3,
    bitdepth=10,
)

camera.set_exposure(40e-6)
test_image = camera.get_image()

plt.figure()
plt.imshow(test_image, cmap="turbo")
plt.title("Initial Simulated Camera Image")

# %%
slm_camera_model = SLMFFT(
    input_geometry=slm_geometry,
    virtual_slm=VirtualSLM(phase_scaling=1.0),
    slm_field=PixelwiseSLMField(gaussian_beam),
    focal_length=0.25,
    padded_resolution=(2048, 2048),
)

# %% Coarse camera mapping
coarse_mapper = CoarseMapper(
    slm=slm,
    camera=camera,
    slm_camera_model=slm_camera_model,
)
coarse_mapping = coarse_mapper.map_camera()

print(f"Camera rotation : {coarse_mapping.rotation_degrees:.2f} deg")
print(f"Camera mirrored : {coarse_mapping.is_mirrored}")
print(f"Camera scales   : {coarse_mapping.scales}")
print(f"Zeroth order at (y, x): {coarse_mapping.zeroth_order_position} px")

# %% Spot array camera mapping
camera_mapper = SpotArrayMapper(
    slm=slm,
    camera=camera,
    slm_camera_model=slm_camera_model,
)

camera_mapping = camera_mapper.map_camera(
    number_of_spots=40,
    seed=0,
    coarse_mapping=coarse_mapping,
)

# %% Saving results
camera_mapping.save(data_path + "spot_array_mapping.asdf")
slm_camera_model.save(data_path + "slm_camera_model.pt")
simulated_camera_model.save(data_path + "simulated_camera_model.pt")

# %% Results
print("Transformation matrix:")
print(camera_mapping.transform)
print("Inverse transformation matrix:")
print(camera_mapping.inverse_transform)
print(
    "Average focal-spot waist: "
    f"{camera_mapping.spot_fit.waist * 1e6:.2f} +/- "
    f"{camera_mapping.spot_fit.waist_uncertainty * 1e6:.2f} um"
)

# %% Overview figure (camera / simulated images, reprojection, per-spot waists)
figure = CameraMapperVisualizer(camera_mapping).render()

# %% SLM phase used for the array
slm_phase = gpu_to_numpy(slm_camera_model.virtual_slm.get_phase())
plt.figure()
plt.imshow(slm_phase, cmap="magma")
plt.colorbar()
plt.title("SLM Phase (spot-array hologram)")

# %% Detected vs zeroth order on the camera image
detected = np.asarray(camera_mapping.detected_points)
plt.figure()
plt.imshow(camera_mapping.visualization_data.camera_image, cmap="turbo")
plt.plot(detected[:, 0], detected[:, 1], "wx", label="detected spots")
plt.plot(
    *camera_mapping.zeroth_order_xy,
    "r+",
    label="zeroth order position",
)
plt.legend()
plt.title("Camera Image with Detected Spots")

plt.show()
# %%
