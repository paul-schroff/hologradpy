"""
Gradient-based phase retrieval
==============================

Finds the SLM phase that forms a target light potential, by minimizing a cost function
based on the output of a model that simulates the propagation of light from the SLM to
the camera.
"""

# %% Imports

from hologradpy.holography.phase_retrieval import PixelwisePhaseRetriever
from hologradpy.profiles.phase import gaussian_phase_guess
from hologradpy.profiles.amplitude import (
    gaussian_beam_intensity,
    gaussian_blur,
)
from hologradpy.profiles.masks import rectangular_mask
from hologradpy.utils import (
    get_device,
    gpu_to_numpy,
)

from hologradpy.optics.complex_amplitude import (
    ComplexAmplitude,
    FieldGeometry,
)
from hologradpy.optics.systems import SLMFFT
from hologradpy.optics.modules.slm_fields import PixelwiseSLMField
from hologradpy.optics.modules.virtual_slms import VirtualSLM

from hologradpy.hardware import SimulatedSLMTorch, open_slm
from hologradpy.visualizer import (
    INTENSITY_CMAP,
    GridCell,
    PlotBuilder,
    PlotLayout,
)

import torch

# %%
# Setting up the the SLM and camera devices
# -----------------------------------------
device = get_device(verbose=True)

slm_geometry = FieldGeometry(
    resolution=(1024, 1280),
    pixel_size=torch.tensor([12.5e-6, 12.5e-6], device=device),
    wavelength=torch.tensor(0.670e-6, device=device),
)

slm = open_slm(SimulatedSLMTorch, input_geometry=slm_geometry, bitdepth=8)

# %% 
# Setting up models of the SLM and the camera
# -------------------------------------------
beam_radius = 4e-3  # beam radius in mm
focal_length = 500e-3  # focal length in mm
padded_resolution = (2048, 2048)  # padded resolution for the FFT

slm_grid = slm_geometry.get_spatial_grid()
slm_intensity = gaussian_beam_intensity(*slm_grid, beam_radius=beam_radius)
slm_field = ComplexAmplitude.from_geometry(
    slm_geometry, data=slm_intensity.sqrt() + 0j
)

slm_camera_model = SLMFFT(
    input_geometry=slm_field.geometry,
    virtual_slm=VirtualSLM(phase_scaling=1.0),
    slm_field=PixelwiseSLMField(slm_field),
    focal_length=focal_length,
    padded_resolution=padded_resolution,
)
slm_camera_model()

# %% 
# Setting up the target potential and signal region
# -------------------------------------------------
camera_grid = slm_camera_model[-1].get_spatial_grid_output()

top_hat_width = 1500e-6
top_hat_height = 700e-6
target_top_hat = rectangular_mask(
    *camera_grid,
    top_hat_width,
    top_hat_height,
    0e-6,
    0e-6,
)

target_top_hat = gaussian_blur(target_top_hat.float(), beam_radius=4)

signal_region = rectangular_mask(
    *camera_grid,
    2 * top_hat_width,
    2 * top_hat_height,
    shift_x=0,
    shift_y=0,
)


# %%
# Defining the SLM phase guess
# ----------------------------
#
# A Gaussian filling the signal region.
init_slm_phase = gaussian_phase_guess(
    *slm_grid,
    input_beam_radius=(beam_radius, beam_radius),
    output_beam_radius=(top_hat_width, top_hat_height),
    focal_length=focal_length,
    wavenumber=2 * torch.pi / slm.wavelength,
).to(torch.float32)
slm_camera_model.virtual_slm.set_phase(init_slm_phase)

# %% 
# Initial simulated output
# ------------------------
init_electric_field = slm_camera_model()
init_intensity = init_electric_field.intensity

slm_power = slm_camera_model.slm_field.amplitude**2
image_power = init_intensity.sum()

print(f"SLM Power: {slm_power.sum().item()}")
print(f"Image Power: {image_power.item()}")

# %% 
# Sanity checking the target geometry and the output of the initial guess
# -----------------------------------------------------------------------
def _aspect(image) -> float:
    """Height over width, which is what GridCell wants."""
    return image.shape[0] / image.shape[1]


setup_layout = PlotLayout(column_width=3.6, margins=(1.0, 0.15, 0.5, 0.5))
setup_layout.add_row(
    [
        GridCell("slm", aspect=_aspect(slm_intensity), colorbar=True),
        GridCell("initial", aspect=_aspect(init_intensity), colorbar=True),
        GridCell("target", aspect=_aspect(target_top_hat), colorbar=True),
    ]
)
(
    PlotBuilder(setup_layout)
    .draw_image("slm", gpu_to_numpy(slm_intensity), cmap=INTENSITY_CMAP,
                title="SLM illumination")
    .draw_image("initial", gpu_to_numpy(init_intensity), cmap=INTENSITY_CMAP,
                title="initial camera image")
    .draw_image("target", gpu_to_numpy(target_top_hat), cmap=INTENSITY_CMAP,
                title="target")
    .build()
)

# %% 
# Setting up the phase retrieval module
# -------------------------------------
phase_retriever = PixelwisePhaseRetriever(
    slm_camera_model=slm_camera_model,
    target=target_top_hat,
    signal_region=signal_region,
    init_slm_phase=init_slm_phase,
)

# %% 
# Running phase retrieval
# -----------------------
retrieval = phase_retriever.retrieve(20, method="cg", name="conjugate gradient")

# %% 
# Plotting the results
# --------------------
figure = retrieval.visualizer().render()
print({name: f"{values[-1]:.4g}" for name, values in retrieval.metrics.items()})
