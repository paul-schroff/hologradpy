"""
Gradient-based phase retrieval
==============================

Finds the SLM phase that forms a target light potential, by minimizing a cost function
based on the output of a model that simulates the propagation of light from the SLM to
the camera.
"""

# sphinx_gallery_thumbnail_number = 2

# %% Imports
from hologradpy.holography.phase_retrieval import PixelwisePhaseRetriever
from hologradpy.profiles.phase import gaussian_phase_guess
from hologradpy.profiles.amplitude import (
    gaussian_beam_intensity,
    gaussian_blur,
)
from hologradpy.analysis.error_metrics import normalize
from hologradpy.analysis.unwrapping import wrap
from hologradpy.profiles.masks import rectangular_mask
from hologradpy.roi import ROI
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
    DIFFERENCE_CMAP,
    INTENSITY_CMAP,
    PHASE_CMAP,
    GridCell,
    PlotBuilder,
    PlotLayout,
    image_grid,
)

import time

import torch

# %%
# Setting up the SLM
# ------------------
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

frame = ROI.detect(signal_region.to(torch.float32), threshold=0.5, pad=0)


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
# Sanity checking the target geometry and the initial guess
# ---------------------------------------------------------
def _aspect(image) -> float:
    """Height over width, which is what GridCell wants."""
    return image.shape[0] / image.shape[1]


# The camera panels are cropped to the signal region.
initial_frame = frame.crop(gpu_to_numpy(init_intensity))
target_frame = frame.crop(gpu_to_numpy(target_top_hat))

setup_layout = PlotLayout(column_width=3.6, margins=(1.0, 0.15, 0.5, 0.5))
setup_layout.add_row(
    [
        GridCell("slm", aspect=_aspect(slm_intensity), colorbar=True),
        GridCell("initial", aspect=_aspect(initial_frame), colorbar=True),
        GridCell("target", aspect=_aspect(target_frame), colorbar=True),
    ]
)
figure = (
    PlotBuilder(setup_layout)
    .draw_image("slm", gpu_to_numpy(slm_intensity), cmap=INTENSITY_CMAP,
                title="SLM illumination")
    .draw_image("initial", initial_frame, cmap=INTENSITY_CMAP,
                title="initial camera image")
    .draw_image("target", target_frame, cmap=INTENSITY_CMAP,
                title="target")
    .build()
)

# %%
# Comparing conjugate gradient with L-BFGS
# ----------------------------------------
#
# Both searches start from the same guess against the same cost. Conjugate gradient
# steps are cheaper than L-BFGS steps, so it is given more iterations.
#
# The cost is recorded once per objective evaluation, and a line search evaluates the
# objective more than once per iteration.
SEARCHES = (
    ("cg", "conjugate gradient", 250),
    ("l-bfgs", "L-BFGS", 100),
)


def timed(cost, stamps):
    """The cost, noting the clock each time it is evaluated."""

    def evaluate(field=None, target=None):
        value = cost(field, target)
        stamps.append(time.perf_counter())
        return value

    return evaluate


results = []
for method, label, iterations in SEARCHES:
    phase_retriever = PixelwisePhaseRetriever(
        slm_camera_model=slm_camera_model,
        target=target_top_hat,
        signal_region=signal_region,
        init_slm_phase=init_slm_phase,
    )
    stamps = []
    phase_retriever.set_loss_function(timed(phase_retriever.loss_function, stamps))
    retrieval = phase_retriever.retrieve(iterations, method=method, name=label)
    seconds = [stamp - stamps[0] for stamp in stamps]
    intensity = frame.crop(gpu_to_numpy(slm_camera_model().intensity.squeeze()))
    results.append((retrieval, seconds, intensity))

for retrieval, seconds, _ in results:
    print(
        f"  {retrieval.name:<20}"
        f"rmse {retrieval.metrics['rmse'][-1]:.4f}   "
        f"{len(retrieval.loss_history):4d} evaluations   "
        f"{seconds[-1]:5.1f} s"
    )

# %%
# Convergence
# -----------
cost_layout = PlotLayout(column_width=4.4, margins=(0.62, 0.12, 0.28, 0.45))
cost_layout.add_row([GridCell("cost", aspect=0.652)])
figure = (
    PlotBuilder(cost_layout)
    .draw_line(
        "cost",
        [
            {
                "x": seconds,
                "y": list(retrieval.loss_history),
                "label": retrieval.name,
            }
            for retrieval, seconds, _ in results
        ],
        xlabel="time [s]",
        ylabel="cost",
        yscale="log",
        title="convergence",
        legend=True,
    )
    .build()
)

# %%
# Results
# --------
region_frame = frame.crop(gpu_to_numpy(signal_region))
scaled_target = normalize(target_frame, region_frame)
errors = [
    normalize(intensity, region_frame) - scaled_target for _, _, intensity in results
]

peak = max(float(intensity.max()) for _, _, intensity in results)
limit = max(float(abs(error).max()) for error in errors)

figure = image_grid(
    [
        [results[0][2], results[1][2]],
        [errors[0], errors[1]],
    ],
    titles=[
        results[0][0].name,
        results[1][0].name,
        f"{results[0][0].name} - target",
        f"{results[1][0].name} - target",
    ],
    cmap=[INTENSITY_CMAP, INTENSITY_CMAP, DIFFERENCE_CMAP, DIFFERENCE_CMAP],
    vmin=[0.0, 0.0, -limit, -limit],
    vmax=[peak, peak, limit, limit],
    colorbar_label=["intensity [a. u.]", "intensity [a. u.]", "error", "error"],
    column_width=3.6,
).build()

# %%
# Optimized SLM phase patterns
# ----------------------------
patterns = [wrap(retrieval.phase) for retrieval, _, _ in results]

figure = image_grid(
    patterns,
    titles=[results[0][0].name, results[1][0].name],
    cmap=PHASE_CMAP,
    vmin=-torch.pi,
    vmax=torch.pi,
    colorbar_label="phase [rad]",
    merge_colorbars=True,
    column_width=3.6,
).build()

# %%
