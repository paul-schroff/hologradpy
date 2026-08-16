# %% Imports
import matplotlib.pyplot as plt
import torch

from hologradpy.hardware import (
    SimulatedCameraTorch,
    SimulatedSLMTorch,
    open_camera,
    open_slm,
)
from hologradpy.holography.camera_feedback import SimpleFeedbackCorrector
from hologradpy.optics.complex_amplitude import ComplexAmplitude, FieldGeometry
from hologradpy.optics.modules.slm_fields import PixelwiseSLMField
from hologradpy.optics.modules.virtual_slms import VirtualSLM
from hologradpy.optics.systems import SLMCZT
from hologradpy.profiles.amplitude import gaussian_beam_intensity, gaussian_blur
from hologradpy.profiles.masks import rectangular_mask
from hologradpy.profiles.phase import lens_phase, linear_phase
from hologradpy.grids import get_spatial_grid
from hologradpy.profiles.zernike import Zernike
from hologradpy.utils import get_device, gpu_to_numpy

SEED = 1
FOCAL_LENGTH = 250e-3
BEAM_RADIUS = 5e-3

device = get_device(verbose=True)

# %% Simulated SLM
input_geometry = FieldGeometry(
    resolution=(1024, 1280),
    pixel_size=torch.tensor([12.5e-6, 12.5e-6], device=device),
    wavelength=torch.tensor(1039e-9, device=device),
)

slm = open_slm(SimulatedSLMTorch, input_geometry=input_geometry, bitdepth=8)

# %% Setting up the simulated camera
gaussian_amplitude = gaussian_beam_intensity(
    *slm.get_spatial_grid(device=device),
    beam_radius=BEAM_RADIUS,
).sqrt()

# The aberration the feedback has to absorb. The retriever never sees it.
zernike = Zernike(
    input_geometry.resolution,
    unit_disk_mode="fill",
    number_of_radial_orders=4,
    device=device,
)

coefficient_generator = torch.Generator().manual_seed(SEED)
zernike_coefficients = torch.rand(
    zernike.number_of_zernikes, generator=coefficient_generator
)
injected_phase = zernike.get_phase(zernike_coefficients.to(device))

aberrated_beam = ComplexAmplitude(
    gaussian_amplitude * torch.exp(1j * injected_phase),
    wavelength=input_geometry.wavelength,
    pixel_size=input_geometry.pixel_size,
    power=1e-3,
)

CAMERA_RESOLUTION = (900, 1440)
CAMERA_PIXEL_SIZE = (3.45e-6, 3.45e-6)
PADDED_RESOLUTION = (int(1.2 * 900), int(1.2 * 1440))

# Shares the SLM's virtual SLM, so whatever the feedback displays is what this sees.
simulated_camera_model = SLMCZT(
    input_geometry=input_geometry,
    virtual_slm=slm.virtual_slm,
    camera_resolution=CAMERA_RESOLUTION,
    camera_pixel_size=CAMERA_PIXEL_SIZE,
    focal_length=FOCAL_LENGTH,
    slm_field=PixelwiseSLMField(aberrated_beam),
    padded_resolution=PADDED_RESOLUTION,
    camera_angle=5.0,
    camera_shift=(50.0, 120.0),
    pointing_focal_shift_std=0.5e-6,
)

camera = open_camera(
    SimulatedCameraTorch,
    slm_camera_model=simulated_camera_model,
    bitdepth=12,
    nd_filter_optical_density=6,
    noise_level=4,
    power_std=0.05,
    power_seed=0,
)

plt.figure()
plt.imshow(gpu_to_numpy(injected_phase), cmap="magma")
plt.colorbar(label="Phase [rad]")
plt.title("Injected aberration (unknown to the model)")

# %% The model the retriever optimises against, aberration free
clean_beam = ComplexAmplitude(
    gaussian_amplitude + 0j,
    wavelength=input_geometry.wavelength,
    pixel_size=input_geometry.pixel_size,
    power=1e-3,
)

slm_camera_model = SLMCZT(
    input_geometry=input_geometry,
    virtual_slm=VirtualSLM.from_slm(slm),
    camera_resolution=tuple(camera.resolution),
    camera_pixel_size=tuple(float(pitch) for pitch in camera.pixel_size),
    focal_length=FOCAL_LENGTH,
    slm_field=PixelwiseSLMField(clean_beam),
    padded_resolution=PADDED_RESOLUTION,
)

# %% Setting up the target potential and signal region
slm_camera_model()

# (x, y) metres in the Nyquist plane, measured from the zeroth order. Placed off the
# zeroth order so the undiffracted spot does not sit in the middle of the potential.
TARGET_POSITION = (600e-6, -300e-6)

init_slm_phase = (
    lens_phase(
        *slm.get_spatial_grid(device=device),
        focal_length=2,
        wavenumber=2 * torch.pi / slm.wavelength,
    )
    + linear_phase(
        *slm.get_spatial_grid(device=device),
        tilt_x=TARGET_POSITION[0],
        tilt_y=TARGET_POSITION[1],
        wavenumber=2 * torch.pi / slm.wavelength,
        focal_length=FOCAL_LENGTH,
    )
).to(torch.float32)

PATCH_RESOLUTION = (240, 480)
patch_grid = get_spatial_grid(PATCH_RESOLUTION, CAMERA_PIXEL_SIZE)

top_hat_width = 500e-6
top_hat_height = 200e-6
target_top_hat = gaussian_blur(
    rectangular_mask(
        *patch_grid, top_hat_width, top_hat_height, 0e-6, 0e-6
    ).float(),
    beam_radius=4,
)
signal_region = rectangular_mask(
    *patch_grid, 2 * top_hat_width, 2 * top_hat_height, shift_x=0, shift_y=0
)

# %% Camera feedback
feedback = SimpleFeedbackCorrector(
    slm=slm,
    camera=camera,
    slm_camera_model=slm_camera_model,
    target=target_top_hat,
    signal_region=signal_region,
    target_position=TARGET_POSITION,
    init_slm_phase=init_slm_phase,
)

# %% Checking the placement of the target and the initial guess on the camera
placement = feedback.placement_data()
placement_figure = placement.visualizer().render()

# %% Running the feedback loop
data = feedback.run(
    retriever_iterations=[50, 30, 20, 20, 20, 20],
    gain=0.7,
    averages=5,
    blur=0.0,
    retrieve_options={"method": "cg"},
    verbose=True,
)

# %% Plotting results
visualizer = data.visualizer()
figure = visualizer.render()

# %%
