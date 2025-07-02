# %% Imports
import matplotlib.pyplot as plt
import numpy as np
import torch

from hologradpy.hardware import SimulatedSLMTorch, SimulatedCameraTorch

from hologradpy.calibration import (
    RasterCalibrator,
    get_diffraction_spot_position
)

from hologradpy.torch_modules.optical_systems import SLMFFTAffine

from hologradpy.torch_modules.utils.optics_utils import (
    gaussian_beam_intensity,
    linear_phase
)
from hologradpy.torch_modules.utils.tensor_utils import (
    check_device,
    gpu_to_numpy
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
    "padded_resolution": (2048, 2048)
}

camera = SimulatedCameraTorch(
    slm,
    resolution=(1440, 960),
    pitch_um=(3.75, 3.75),
    slm_camera_model_cls=SLMFFTAffine,
    slm_camera_model_args=slm_fft_affine_args,
)

# camera.set_woi((450, 100, 450, 100))  # Set the region of interest

# slm.write(np.random.rand(*slm.shape) * 2 * np.pi)

camera.set_exposure(0.001)
test_image = camera.get_image()
# plt.figure()
# plt.imshow(test_image)
# plt.colorbar()

# %%
(spot_position_x, spot_position_y), calibration_image = (
        get_diffraction_spot_position(
        slm,
        camera,
        (500e-6, 500e-6),
        focal_length=0.25,
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
calibrator = RasterCalibrator(
    slm=slm,
    camera=camera,
    focal_length=0.25,
)

# %% Calibrate intesity
camera.set_woi(None)
intensity, camera_images = calibrator.measure_intensity(
    number_of_superpixels_x=10,
    number_of_superpixels_y=8,
    superpixel_width=128,
    superpixel_height=128,
    linear_phase_tilt=(500e-6, 500e-6),
    camera_roi_size=(150, 150),
    verbose=True
)
# %%

plt.figure()
plt.imshow(intensity, cmap='turbo')
plt.colorbar()

plt.figure()
plt.imshow(camera_images[2, ...], cmap='turbo')
plt.colorbar()

plt.figure()
plt.imshow(slm.display, cmap='magma')
plt.colorbar()

# %%
phase, camera_images, fitted_images = calibrator.measure_phase(
    number_of_superpixels_x=40,
    number_of_superpixels_y=32,
    superpixel_width=32,
    superpixel_height=32,
    linear_phase_tilt=(500e-6, 500e-6),
    camera_roi_size=(100, 100),
    verbose=True,
    measured_intensity=intensity,
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
%matplotlib qt
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
