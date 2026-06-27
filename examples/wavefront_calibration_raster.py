# %% Imports
import matplotlib.pyplot as plt
import torch

from hologradpy.hardware.slm_simulated import SimulatedSLMTorch
from hologradpy.hardware.camera_simulated import SimulatedCameraTorch

from hologradpy.propagation.complex_amplitude import (
    ComplexAmplitude,
    FieldGeometry,
)

from hologradpy.calibration import (
    RasterCalibrator,
    get_diffraction_spot_position
)

from hologradpy.propagation.optical_systems import SLMFFTAffine
from hologradpy.propagation.diagonal_elements import StaticSLMField

from hologradpy.propagation.amplitude_profiles import gaussian_beam_intensity
from hologradpy.propagation.zernike import Zernike
from hologradpy.utils import get_device, gpu_to_numpy

device = get_device(verbose=True)

# %%
%matplotlib qt
# %% Initializing simulated SLM and camera
slm_geometry = FieldGeometry(
    resolution=(1024, 1280),
    pixel_size=torch.tensor([12.5e-6, 12.5e-6], device=device),
    wavelength=torch.tensor(0.630e-6, device=device),
)

slm = SimulatedSLMTorch(input_geometry=slm_geometry, bitdepth=8)

gaussian_intensity = gaussian_beam_intensity(
    *slm.get_spatial_grid(device),
    beam_radius=5e-3,
)

zernike = Zernike(
    slm_geometry.resolution, 
    unit_disk_mode="fill",
    number_of_radial_orders=10,
    device=device
)
coefficients = torch.rand(zernike.number_of_zernikes, device=device) * 1
zernike_phase = zernike.get_phase(coefficients)

plt.figure()
plt.imshow(zernike_phase.cpu(), cmap="magma")
plt.colorbar()

plt.figure()
plt.imshow(gaussian_intensity.cpu(), cmap="turbo")
plt.colorbar()

gaussian_beam = ComplexAmplitude(
    gaussian_intensity.sqrt() * torch.exp(1j * zernike_phase),
    wavelength=slm_geometry.wavelength,
    pixel_size=slm_geometry.pixel_size,
)

simulated_camera_model = SLMFFTAffine(
    input_geometry=slm_geometry,
    virtual_slm=slm.virtual_slm,
    camera_resolution=(960, 1440),
    camera_pixel_size=(3.75e-6, 3.75e-6),
    focal_length=0.25,
    static_slm_field=StaticSLMField(gaussian_beam),
    padded_resolution=(2048, 2048),
    camera_angle=0,
    camera_shift=(0, 0),
)

camera = SimulatedCameraTorch(simulated_camera_model)

camera.set_exposure(0.001)
test_image = camera.get_image()

plt.figure()
plt.imshow(test_image, cmap="turbo")
plt.title("Initial Simulated Camera Image")


# %%
(spot_position_x, spot_position_y), focal_spot_radius, calibration_image = (
    get_diffraction_spot_position(
        slm, camera, linear_phase_tilt=(500e-6, 500e-6), focal_length=0.25,
    )
)

plt.figure()
plt.imshow(calibration_image, cmap='turbo')
plt.colorbar()
plt.plot(spot_position_x, spot_position_y, 'ro', markersize=5)

plt.figure()
plt.imshow(slm.display, cmap='magma')
plt.colorbar()


# %% Initializing the calibrator
calibrator = RasterCalibrator(slm, camera, focal_length=0.25)

# %% Calibrate intesity
camera.set_woi(None)
intensity, camera_images = calibrator.measure_intensity(
    number_of_superpixels_x=20,
    number_of_superpixels_y=16,
    superpixel_width=64,
    superpixel_height=64,
    linear_phase_tilt=(500e-6, 500e-6),
    camera_roi_size=(50, 50),
    verbose=True
)
# %%

plt.figure()
plt.imshow(intensity, cmap='turbo')
plt.colorbar()

plt.figure()
plt.imshow(camera_images[3, ...], cmap='turbo')
plt.colorbar()

plt.figure()
plt.imshow(slm.display, cmap='magma')
plt.colorbar()

# %%
phase, camera_images, fitted_images = (
    calibrator.measure_phase(
        number_of_superpixels_x=20,
        number_of_superpixels_y=16,
        superpixel_width=32,
        superpixel_height=32,
        linear_phase_tilt=(500e-6, 500e-6),
        camera_roi_size=(100, 100),
        verbose=True,
        measured_intensity=intensity,
    )
)

# %%
plt.figure()
plt.imshow(slm.display, cmap="magma")
plt.colorbar()

plt.figure()
plt.imshow(phase, cmap="magma")
plt.colorbar()

#%%
plt.figure()
plt.imshow(camera_images[14, ...], cmap="turbo")
plt.colorbar()

# %%
import matplotlib.animation as animation

fig, ax = plt.subplots(1, 2)
ims = []
for i in range(camera_images.shape[0]):
    im = ax[0].imshow(camera_images[i, ...], animated=True)
    im1 = ax[1].imshow(fitted_images[i, ...], animated=True)
    text = ax[1].text(1, 1, f"Image {i+1}/{camera_images.shape[0]}",
                   color='white', fontsize=12, ha='left', va='top')
    if i == 0:
        ax[0].imshow(camera_images[i, ...])  # show an initial one first
    ims.append([im, im1, text])

ani = animation.ArtistAnimation(
    fig,
    ims,
    interval=200,
    blit=True,
    repeat_delay=500
)

plt.show()

# %%
ground_truth = gpu_to_numpy(zernike_phase.cpu())

plt.figure()
plt.imshow(ground_truth, cmap="magma")
plt.colorbar()
plt.title("Ground Truth Phase")

difference = ground_truth + phase
plt.figure()
plt.imshow(difference, cmap="bwr")
plt.colorbar()
plt.title("Difference between Measured Phase and Ground Truth")

plt.figure()
plt.imshow(-phase, cmap="magma")
plt.colorbar()
plt.title("Measured Phase")

# %%
