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
    CameraMapperVisualizer,
    CoarseMapperVisualizer,
)

from hologradpy.optics.systems import SLMFFT, SLMFFTAffine
from hologradpy.optics.modules.diagonal_elements import StaticSLMField
from hologradpy.optics.modules.virtual_slms import VirtualSLM
from hologradpy.optics.complex_amplitude import ComplexAmplitude, FieldGeometry

from hologradpy.profiles.amplitude import gaussian_beam_intensity
from hologradpy.profiles.zernike import Zernike
from hologradpy.utils import get_device

device = get_device(verbose=True)
data_path = "../data/"
# %matplotlib qt5

# %% Setting up SLM and Camera
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

# Adding abberrations to the simulated beam
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

simulated_camera_model = SLMFFTAffine(
    input_geometry=slm_geometry,
    virtual_slm=slm.virtual_slm,
    camera_resolution=(960, 1440),
    camera_pixel_size=(3.45e-6, 3.45e-6),
    focal_length=0.25,
    static_slm_field=StaticSLMField(aberrated_beam),
    padded_resolution=(2048, 2048),
    camera_angle=15,
    camera_shift=(900, 300),
)

camera = open_camera(
    SimulatedCameraTorch,
    slm_camera_model=simulated_camera_model,
    exposure_time=100e-3,
    quantum_efficiency=0.01,
    full_well_capacity=11e3,
    noise_level=4.0,
    nd_filter_optical_density=3,
    bitdepth=10,
    background_scatter_power=1e-4,
    background_scatter_grain_radius=20e-6,
)

test_image = camera.get_image()

plt.figure()
plt.imshow(test_image, cmap="turbo")
plt.title("Initial Camera Image")
plt.colorbar()

# %% Ideal hologram model: the plane the camera is registered against
slm_camera_model = SLMFFT(
    input_geometry=slm_geometry,
    virtual_slm=VirtualSLM(phase_scaling=1.0),
    static_slm_field=StaticSLMField(gaussian_beam),
    focal_length=0.25,
    padded_resolution=(2048, 2048),
)

# %% Running the corse mapping
coarse_mapper = CoarseMapper(
    slm=slm,
    camera=camera,
    slm_camera_model=slm_camera_model,
)
coarse_mapping = coarse_mapper.map_camera()

print(f"Camera rotation: {coarse_mapping.rotation_degrees:.2f} deg")
print(f"Camera mirrored: {coarse_mapping.is_mirrored}")
print(f"Camera scales: {coarse_mapping.scales}")
print(f"Reprojection RMS: {coarse_mapping.reprojection_rms:.3f} px")
print(
    "Zeroth order (y, x): "
    f"({coarse_mapping.zeroth_order_position[0]:.0f}, "
    f"{coarse_mapping.zeroth_order_position[1]:.0f}) px "
    "(extrapolated, off the sensor)"
)

# %% Saving results
coarse_mapping.save(data_path + "coarse_mapping.pkl")

# %% Plotting
figure = CoarseMapperVisualizer(coarse_mapping.visualization_data).render()
figure = CameraMapperVisualizer(coarse_mapping).render()

detected = np.asarray(coarse_mapping.detected_points)
plt.figure()
plt.imshow(coarse_mapping.camera_images[0], cmap="turbo")
plt.plot(detected[:, 0], detected[:, 1], "wx", label="probe spots")
plt.legend()
plt.title("Probe Spots on the Camera")

