# %% Imports
import matplotlib.pyplot as plt
import numpy as np
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

from hologradpy.propagation.optical_systems import SLMCZT
from hologradpy.propagation.diagonal_elements import StaticSLMField

from hologradpy.propagation.amplitude_profiles import gaussian_beam_intensity
from hologradpy.propagation.phase_profiles import linear_phase
from hologradpy.propagation.zernike import Zernike
from hologradpy.utils import get_device, gpu_to_numpy, pad_from_roi

device = get_device(verbose=True)

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
plt.show()

plt.figure()
plt.imshow(gaussian_intensity.cpu(), cmap="turbo")
plt.colorbar()

gaussian_beam = ComplexAmplitude(
    gaussian_intensity.sqrt() * torch.exp(1j * zernike_phase),
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
    static_slm_field=StaticSLMField(gaussian_beam),
    camera_angle=0,
    camera_shift=(0, 0),
    power_normalized=True,
)

camera = SimulatedCameraTorch(
    simulated_camera_model,
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
plt.colorbar()


# %%
(
    (spot_position_x, spot_position_y),
    focal_spot_radius,
    calibration_image,
    calibration_roi,
) = get_diffraction_spot_position(
    slm, camera, linear_phase_tilt=(500e-6, 500e-6), focal_length=0.25,
    units="pixels",
)

# Pad the cropped spot image back to the full sensor so the detected pixel
# position lines up with the image.
calibration_image = pad_from_roi(calibration_image, calibration_roi, camera.shape)

plt.figure()
plt.imshow(calibration_image, cmap='turbo')
plt.colorbar()
plt.plot(spot_position_x, spot_position_y, 'wx', markersize=5)

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

# %% Drift-injection demo: simulate beam pointing drift and show it is tracked.
# The simulator has no pointing parameter, so wrap slm.set_phase to add a global
# tilt that grows each frame (a common-mode shift of every diffraction spot).
grid_x_np, grid_y_np = [gpu_to_numpy(g) for g in slm.get_spatial_grid(device)]
wavenumber = 2 * np.pi / (slm.wav_um * 1e-6)
drift_step = (0.2e-6, -0.1e-6)  # (x, y) metres of camera shift added per display

drift_state = {"n": 0}
original_set_phase = slm.set_phase

def drifting_set_phase(phase, *args, **kwargs):
    tilt = linear_phase(
        grid_x_np,
        grid_y_np,
        drift_state["n"] * drift_step[0],
        drift_state["n"] * drift_step[1],
        tilt_units="metres",
        focal_length=0.25,
        wavenumber=wavenumber,
    )
    drift_state["n"] += 1
    return original_set_phase(phase + tilt, *args, **kwargs)


slm.set_phase = drifting_set_phase
phase, _, _ = calibrator.measure_phase(
    number_of_superpixels_x=20,
    number_of_superpixels_y=16,
    superpixel_width=32,
    superpixel_height=32,
    linear_phase_tilt=(500e-6, 500e-6),
    measured_intensity=intensity,
    compensate_pointing=True,
    lattice_phase_tilt=(-800e-6, -800e-6),
    verbose=True,
)
slm.set_phase = original_set_phase  # restore

# Compare the tracked (lattice-measured) shift against the injected drift. The
# injected line is anchored to the first measured point to absorb the constant
# baseline offset.
frame_index = np.arange(len(calibrator.lattice_shift_x))
injected_x = frame_index * drift_step[0] + calibrator.lattice_shift_x[0]
injected_y = frame_index * drift_step[1] + calibrator.lattice_shift_y[0]

fig, (ax_track, ax_resid) = plt.subplots(
    2, 1, sharex=True, figsize=(7, 6),
    gridspec_kw={"height_ratios": [2, 1]},
)

# Tracked shift with 1-sigma fit error bars vs the injected drift.
line_x = ax_track.errorbar(
    frame_index, calibrator.lattice_shift_x * 1e6,
    yerr=calibrator.lattice_shift_x_err * 1e6, capsize=2, label="tracked x",
)
ax_track.plot(
    frame_index, injected_x * 1e6, "--",
    color=line_x[0].get_color(), label="injected x",
)
line_y = ax_track.errorbar(
    frame_index, calibrator.lattice_shift_y * 1e6,
    yerr=calibrator.lattice_shift_y_err * 1e6, capsize=2, label="tracked y",
)
ax_track.plot(
    frame_index, injected_y * 1e6, "--",
    color=line_y[0].get_color(), label="injected y",
)
ax_track.set_ylabel("Pointing drift [um]")
ax_track.set_title("Optical lattice tracks injected pointing drift")
ax_track.legend()

# Residual: tracked minus injected, with the same error bars.
ax_resid.errorbar(
    frame_index, (calibrator.lattice_shift_x - injected_x) * 1e6,
    yerr=calibrator.lattice_shift_x_err * 1e6, capsize=2, label="x",
)
ax_resid.errorbar(
    frame_index, (calibrator.lattice_shift_y - injected_y) * 1e6,
    yerr=calibrator.lattice_shift_y_err * 1e6, capsize=2, label="y",
)
ax_resid.axhline(0.0, color="gray", linewidth=0.8)
ax_resid.set_xlabel("Superpixel index")
ax_resid.set_ylabel("Residual [um]")
ax_resid.legend()
fig.tight_layout()

# %% Compare the (drift-compensated) detected phase to the ground truth.
ground_truth = gpu_to_numpy(zernike_phase)
detected_phase = -phase  # measure_phase returns the opposite-sign phase

difference = ground_truth - detected_phase

# Shared scale for the two phase maps; symmetric scale centred on zero for the
# difference so seismic's white maps to no error.
phase_min = min(ground_truth.min(), detected_phase.min())
phase_max = max(ground_truth.max(), detected_phase.max())
difference_limit = np.abs(difference).max()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

image_ground_truth = axes[0].imshow(
    ground_truth, cmap="magma", vmin=phase_min, vmax=phase_max
)
axes[0].set_title("Ground truth phase")
fig.colorbar(image_ground_truth, ax=axes[0])

image_detected = axes[1].imshow(
    detected_phase, cmap="magma", vmin=phase_min, vmax=phase_max
)
axes[1].set_title("Detected phase")
fig.colorbar(image_detected, ax=axes[1])

image_difference = axes[2].imshow(
    difference, cmap="seismic", vmin=-difference_limit, vmax=difference_limit
)
axes[2].set_title("Difference (ground truth - detected)")
fig.colorbar(image_difference, ax=axes[2])

fig.tight_layout()

# %%
