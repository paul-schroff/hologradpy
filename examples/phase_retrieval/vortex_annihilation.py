"""
Vortex detection and annihilation
=================================

Optical vortices can form when optimising more intricate target patterns, or when the
initial phase guess is chosen poorly. They appear as dark "holes" in the intensity, and
cause the optimisation to stagnate since a global phase winding is needed to remove
them.

This example demonstrates the vortex detection and annihilation scheme used in

    P. Schroff, A. La Rooij, E. Haller and S. Kuhr, *Accurate holographic light
    potentials using pixel crosstalk modelling*, Sci. Rep. **13**, 3252 (2023),
    `doi:10.1038/s41598-023-30296-6 <https://doi.org/10.1038/s41598-023-30296-6>`_.

While making the simulated potential more accurate, this vortex annihilation scheme
reduces the efficiency of the potential and introduces high spatial frequency to the SLM
phase pattern which degrades the accuracy when displayed on a real SLM due to pixel
crosstalk. This should be the last resort, finding a better initial phase guess will
typically generate better results.
"""

# %% Imports
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import hologradpy
from hologradpy.analysis.error_metrics import (
    DEFAULT_INTENSITY_METRICS,
    efficiency_metric,
)
from hologradpy.hardware import SimulatedSLMTorch, open_slm
from hologradpy.holography.phase_retrieval import GradientPhaseRetriever
from hologradpy.holography.vortices import VortexAnnihilator
from hologradpy.optics.complex_amplitude import ComplexAmplitude, FieldGeometry
from hologradpy.optics.modules.slm_fields import PixelwiseSLMField
from hologradpy.optics.modules.virtual_slms import VirtualSLM
from hologradpy.optics.systems import SLMFFT
from hologradpy.profiles.amplitude import gaussian_beam_intensity
from hologradpy.profiles.phase import lens_phase
from hologradpy.utils import get_device, gpu_to_numpy, to_canvas
from hologradpy.visualizer import (
    INTENSITY_CMAP,
    GridCell,
    PlotBuilder,
    PlotLayout,
    region_bounding_box,
)

device = get_device(verbose=True)

# %% The SLM and the beam on it
slm_geometry = FieldGeometry(
    resolution=(1024, 1280),
    pixel_size=torch.tensor([12.5e-6, 12.5e-6], device=device),
    wavelength=torch.tensor(670e-9, device=device),
)

slm = open_slm(SimulatedSLMTorch, input_geometry=slm_geometry, bitdepth=8)

beam_radius = 4e-3
focal_length = 250e-3
padded_resolution = (2048, 2048)

slm_grid = slm_geometry.get_spatial_grid()
slm_intensity = gaussian_beam_intensity(*slm_grid, beam_radius=beam_radius)
slm_field = ComplexAmplitude.from_geometry(
    slm_geometry, data=slm_intensity.sqrt() + 0j
)

# Quadratic initial phase guess.
init_slm_phase = lens_phase(
    *slm_grid,
    focal_length=0.35,
    wavenumber=2 * torch.pi / slm.wavelength,
).to(torch.float32)

slm_camera_model = SLMFFT(
    input_geometry=slm_field.geometry,
    virtual_slm=VirtualSLM(phase_scaling=1.0, init_phase=init_slm_phase),
    slm_field=PixelwiseSLMField(slm_field),
    focal_length=focal_length,
    padded_resolution=padded_resolution,
)

# %% Natural image target
slm_camera_model()
output_resolution = tuple(int(size) for size in slm_camera_model[-1].resolution_out)

targets = Path(hologradpy.__file__).parents[1] / "targets"
duke = Image.open(targets / "duke_v2.jpg").convert("L")
duke_array = np.asarray(duke, dtype=np.float64)
duke_array /= duke_array.max()

target_intensity = torch.as_tensor(
    to_canvas(duke_array, output_resolution), dtype=torch.float32, device=device
)

# The dark margin left around the image, in pixels.
SIGNAL_MARGIN = 40
lit = np.ones(np.array(duke_array.shape) + 2 * SIGNAL_MARGIN, dtype=bool)
signal_region = torch.as_tensor(
    to_canvas(lit, output_resolution), dtype=torch.bool, device=device
)

# %% Retrieve, with no vortex handling at all
phase_retriever = GradientPhaseRetriever(
    slm_camera_model=slm_camera_model,
    target=target_intensity,
    signal_region=signal_region,
    init_slm_phase=init_slm_phase,
)

# Adding an efficiency metric the the default rmse and psnr
metrics = DEFAULT_INTENSITY_METRICS + (
    efficiency_metric(
        slm_camera_model.incident_power(),
        slm_camera_model.output_pixel_area(),
    ),
)

before = phase_retriever.retrieve(
    60, method="l-bfgs", name="before annihilation", metrics=metrics
)
print({name: f"{values[-1]:.4g}" for name, values in before.metrics.items()})

figure = before.visualizer().render()

# %% Annihilate the vortices
# Each round detects the vortices, multiplies in a field of the opposite charge to
# cancel their winding, propagates that back to the SLM, and retrieves again.
annihilator = VortexAnnihilator(phase_retriever)
annihilation = annihilator.annihilate_vortices(
    target_intensity_threshold=0.1, max_iterations=10, cg_iterations=50
)

print(f"vortices per round: {annihilation.counts}")
print(f"converged: {annihilation.converged}")

figure = annihilation.visualizer().render()

# %% Reoptimize after the vortices have been removed
after = phase_retriever.retrieve(
    300, method="l-bfgs", name="after annihilation", metrics=metrics
)
print({name: f"{values[-1]:.4g}" for name, values in after.metrics.items()})

figure = after.visualizer().render()

# %% Before and after vortex removal
crop = region_bounding_box(gpu_to_numpy(signal_region))

before_intensity = gpu_to_numpy(
    torch.as_tensor(annihilation.initial_intensity)
)[crop]
after_intensity = gpu_to_numpy(torch.as_tensor(annihilation.final_intensity))[crop]
target_shown = gpu_to_numpy(target_intensity)[crop]
aspect = before_intensity.shape[0] / before_intensity.shape[1]

comparison = PlotLayout(column_width=3.6, margins=(1.0, 0.15, 0.5, 0.5))
comparison.add_row(
    [
        GridCell("target", aspect=aspect, colorbar=True),
        GridCell("before", aspect=aspect, colorbar=True),
        GridCell("after", aspect=aspect, colorbar=True),
    ]
)
(
    PlotBuilder(comparison)
    .draw_image("target", target_shown, cmap=INTENSITY_CMAP, title="target")
    .draw_image("before", before_intensity, cmap=INTENSITY_CMAP,
                title=f"before, {len(annihilation.initial_positions)} vortices")
    .draw_image("after", after_intensity, cmap=INTENSITY_CMAP,
                title=f"after, {len(annihilation.final_positions)} vortices")
    .build()
)

plt.show()

# %%
