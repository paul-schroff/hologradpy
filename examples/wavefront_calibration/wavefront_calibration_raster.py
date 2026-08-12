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

from hologradpy.optics.complex_amplitude import (
    ComplexAmplitude,
    FieldGeometry,
)

from hologradpy.calibration import (
    RasterCalibrator,
    RasterCalibratorVisualizer,
    get_diffraction_spot_position,
)
from hologradpy.visualizer import GridCell, PlotBuilder, PlotLayout

from hologradpy.optics.systems import SLMCZT
from hologradpy.optics.modules.slm_fields import PixelwiseSLMField
from hologradpy.optics.modules.hardware_models.pointing_instability import PointingInstability

from hologradpy.profiles.amplitude import gaussian_beam_intensity
from hologradpy.profiles.zernike import Zernike
from hologradpy.analysis.fitting import remove_tilt
from hologradpy.utils import get_device, gpu_to_numpy

device = get_device(verbose=True)

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
coefficients = torch.rand(zernike.number_of_zernikes, device=device) * 0.2
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
    slm_field=PixelwiseSLMField(gaussian_beam),
    camera_angle=10,
    camera_shift=(50, -20),
    pointing_focal_shift_std=0.25e-6,
)

# Grab PointingInstability internally so we can read its per-frame tilt below.
pointing_instability = simulated_camera_model.get(PointingInstability)

# The simulated camera adds a fluctuating laser power (a PowerInstability drawing
# ~ N(1, power_std) each frame); measure_intensity(normalize_power=True) divides it
# out with a reference spot.
camera = open_camera(
    SimulatedCameraTorch,
    slm_camera_model=simulated_camera_model,
    quantum_efficiency=0.01,
    full_well_capacity=11e3,
    noise_level=4.0,
    nd_filter_optical_density=3,
    bitdepth=10,
    power_std=0.05,
    power_seed=0,
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
    slm,
    camera,
    linear_phase_tilt=(500e-6, 500e-6),
    focal_length=0.25,
    units="pixels",
)

# Pad the cropped spot image back to the full sensor so the detected pixel
# position lines up with the image.
calibration_image = calibration_roi.pad(calibration_image, camera.shape)

plt.figure()
plt.imshow(calibration_image, cmap="turbo")
plt.colorbar()
plt.plot(spot_position_x, spot_position_y, "wx", markersize=5)

plt.figure()
plt.imshow(slm.display, cmap="magma")
plt.colorbar()


# %% Initializing the calibrator
calibrator = RasterCalibrator(slm, camera, focal_length=0.25)

# %% Calibrate intensity, correcting the fluctuating laser power
camera.set_roi(None)
intensity, camera_images = calibrator.measure_intensity(
    number_of_superpixels_x=20,
    number_of_superpixels_y=16,
    superpixel_width=64,
    superpixel_height=64,
    normalize_power=True,
    verbose=True,
)
# %%

plt.figure()
plt.imshow(intensity, cmap="turbo")
plt.colorbar()

plt.figure()
plt.imshow(camera_images[3, ...], cmap="turbo")
plt.colorbar()

plt.figure()
plt.imshow(slm.display, cmap="magma")
plt.colorbar()

# The reference spot tracks the injected laser-power drift that normalization removed.
plt.figure()
plt.plot(calibrator.power_reference)
plt.xlabel("Superpixel index")
plt.ylabel("Reference spot power [counts]")
plt.title("Injected laser-power drift tracked by the reference spot")

# %% Pointing-noise demo: PointingInstability injects the jitter, the lattice tracks it.
# Record the displayed SLM phase per superpixel so the scan can be visualized.
with pointing_instability.record_samples():
    phase, _, _ = calibrator.measure_phase(
        number_of_superpixels_x=20,
        number_of_superpixels_y=16,
        superpixel_width=32,
        superpixel_height=32,
        measured_intensity=intensity,
        compensate_pointing=True,
        verbose=True,
        record_displayed_phases=True,
    )

data = calibrator.visualization_data
visualizer = RasterCalibratorVisualizer(data)

visualizer.save_gif("wavefront_calibration_raster.gif", max_frames=128, dpi=50, fps=4)

number_of_superpixels = len(data.lattice_shift_x)
angles = pointing_instability.angle_history.cpu().numpy()  # (n, 2): [angle_x, angle_y]
angle_x = angles[:, 0]
angle_y = angles[:, 1]
baseline = -(number_of_superpixels + 1)

focal_length = simulated_camera_model.fourier_lens.focal_length
injected_x = focal_length * (angle_x[-number_of_superpixels:] - angle_x[baseline])
injected_y = focal_length * (angle_y[-number_of_superpixels:] - angle_y[baseline])

drift_figure = visualizer.plot_drift_tracking(injected_x, injected_y)

# %% Compare the (drift-compensated) detected phase to the ground truth.
# Compare only over the illuminated aperture (the beam's 1/e^2 intensity disk) and
# remove piston and tilt from each phase first.
beam_intensity = gpu_to_numpy(gaussian_intensity)
aperture_mask = beam_intensity >= beam_intensity.max() * np.exp(-2)

ground_truth = remove_tilt(gpu_to_numpy(zernike_phase), mask=aperture_mask)
# measure_phase returns the opposite-sign phase.
detected_phase = remove_tilt(-phase, mask=aperture_mask)
difference = ground_truth - detected_phase
difference = difference - np.mean(difference[aperture_mask])

rmse = np.sqrt(np.mean(difference[aperture_mask] ** 2))
wavelength_error = 1 / (rmse / 2 / np.pi)
print(f"Wavefront residual RMSE over the 1/e^2 aperture: {rmse:.4f} rad")
print(f"Wavefront residual RMSE over the 1/e^2 aperture: lambda/{wavelength_error:.2f}")

# Blank everything outside the aperture so the plots show only the compared region.
ground_truth = np.where(aperture_mask, ground_truth, np.nan)
detected_phase = np.where(aperture_mask, detected_phase, np.nan)
difference = np.where(aperture_mask, difference, np.nan)

phase_min = float(np.nanmin([ground_truth, detected_phase]))
phase_max = float(np.nanmax([ground_truth, detected_phase]))
difference_limit = float(np.nanmax(np.abs(difference)))
aspect_ratio = ground_truth.shape[0] / ground_truth.shape[1]

comparison_layout = PlotLayout(column_width=4.0)
comparison_layout.add_row([
    GridCell("ground_truth", aspect=aspect_ratio, colorbar=True),
    GridCell("detected", aspect=aspect_ratio, colorbar=True),
    GridCell("difference", aspect=aspect_ratio, colorbar=True),
])
comparison_figure = (
    PlotBuilder(comparison_layout)
    .draw_image(
        "ground_truth",
        ground_truth,
        cmap="magma",
        vmin=phase_min,
        vmax=phase_max,
        title="Ground truth phase (piston/tilt removed)",
    )
    .draw_image(
        "detected",
        detected_phase,
        cmap="magma",
        vmin=phase_min,
        vmax=phase_max,
        title="Detected phase (piston/tilt removed)",
    )
    .draw_image(
        "difference",
        difference,
        cmap="seismic",
        vmin=-difference_limit,
        vmax=difference_limit,
        title=f"Difference (RMSE = {rmse:.3f} rad)",
    )
    .build()
)

# %%
