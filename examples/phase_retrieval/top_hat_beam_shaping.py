"""
Top-hat beam shaping for Rydberg addressing
===========================================

Shaping a Gaussian Rydberg addressing beam into a 1D top-hat profile to achieve uniform
Rabi frequencies across our neutral atom array.

Optimising for the intensity only results in a high conversion efficiency, however, the
phase across the resulting top hat is curved, making it sensitive to aberrations
caused by for example the vacuum window. Another effect of the curved phase is that
intensity of the top hat will become less uniform as it propagates, making it less
suitable for large arrays.

At the cost of efficiency, constraining the phase to be flat across the top hat improves
its robustness to aberrations and leads to a uniform potential even out of focus. The
cost function that constrains both intensity and phase, 
:class:`~hologradpy.loss_functions.LossFidelity`, is equation 5 of Bowman et al.,
`Opt. Express 25, 11692 (2017) <https://doi.org/10.1364/OE.25.011692>`_.
"""

# %% Imports
import matplotlib.pyplot as plt
import numpy as np
import torch

from hologradpy.hardware import SimulatedSLMTorch, open_slm
from hologradpy.holography.phase_retrieval import GradientPhaseRetriever
from hologradpy.loss_functions import (
    LossAbsoluteIntensityMSE,
    LossFidelity,
)
from hologradpy.optics.complex_amplitude import ComplexAmplitude, FieldGeometry
from hologradpy.optics.modules.slm_fields import PixelwiseSLMField
from hologradpy.optics.modules.virtual_slms import VirtualSLM
from hologradpy.optics.systems import SLMCZT
from hologradpy.profiles.amplitude import (
    gaussian_beam_intensity,
    get_focal_spot_radius,
    top_hat_1D,
)
from hologradpy.utils import get_device, gpu_to_numpy
from hologradpy.visualizer import (
    INTENSITY_CMAP,
    PHASE_CMAP,
    GridCell,
    PlotBuilder,
    PlotLayout,
)

device = get_device(verbose=True)

WAVELENGTH = 1039e-9
BEAM_RADIUS = 3.5e-3
FOCAL_LENGTH = 300e-3
CAMERA_PIXEL_SIZE = (3.45e-6, 3.45e-6)
CAMERA_RESOLUTION = (128, 128)

TOPHAT_HEIGHT = 200e-6

# The diffraction-limited spot size in the Fourier plane
FOCAL_SPOT_WAIST = get_focal_spot_radius(
    beam_radius=BEAM_RADIUS, wavelength=WAVELENGTH, focal_length=FOCAL_LENGTH
)

# The region the cost is evaluated over
SIGNAL_HALF_HEIGHT = TOPHAT_HEIGHT / 2 + 2 * FOCAL_SPOT_WAIST
SIGNAL_HALF_WIDTH = 2 * FOCAL_SPOT_WAIST

NUMBER_OF_ITERATIONS = 100

# The share of the incident power the absolute cost is asked to put in the shape.
WANTED_EFFICIENCY = 0.95

# %% The SLM and the beam on it
slm_geometry = FieldGeometry(
    resolution=(1024, 1280),
    pixel_size=torch.tensor([12.5e-6, 12.5e-6], device=device),
    wavelength=torch.tensor(WAVELENGTH, device=device),
)

slm = open_slm(SimulatedSLMTorch, input_geometry=slm_geometry, bitdepth=8)

slm_x, slm_y = slm_geometry.get_spatial_grid()
slm_intensity = gaussian_beam_intensity(slm_x, slm_y, beam_radius=BEAM_RADIUS)
slm_field = ComplexAmplitude.from_geometry(
    slm_geometry, data=slm_intensity.sqrt() + 0j
)

# A cylindrical lens as the starting guess
wavenumber = 2 * torch.pi / slm.wavelength
cylinder_focal_length = BEAM_RADIUS * FOCAL_LENGTH / (TOPHAT_HEIGHT / 2)
init_slm_phase = (
    -wavenumber * slm_y**2 / (2 * cylinder_focal_length)
).to(torch.float32)

# A chirp-z lens sets the camera pitch directly, so the top hat can be sampled at the
# real sensor's pixel size
slm_camera_model = SLMCZT(
    input_geometry=slm_field.geometry,
    virtual_slm=VirtualSLM(phase_scaling=1.0, init_phase=init_slm_phase),
    slm_field=PixelwiseSLMField(slm_field),
    camera_resolution=CAMERA_RESOLUTION,
    camera_pixel_size=CAMERA_PIXEL_SIZE,
    focal_length=FOCAL_LENGTH,
    padded_resolution=(2048, 2048),
)

# The model builds itself lazily, so run it once before asking for its geometry.
slm_camera_model()
camera_x, camera_y = slm_camera_model[-1].get_spatial_grid_output()

# %% 1D top hat target with diffraction-limited shoulders and width
target_intensity = top_hat_1D(
    camera_x,
    camera_y,
    plateau_width=TOPHAT_HEIGHT,
    shoulder_radius=FOCAL_SPOT_WAIST,
    beam_radius=FOCAL_SPOT_WAIST,
    axis="y",
).to(torch.float32)
target_intensity = target_intensity / target_intensity.max()

# A flat phase over the top hat. This is the part an intensity-only cost leaves free.
target_phase = torch.zeros_like(target_intensity)

signal_region = (camera_y.abs() <= SIGNAL_HALF_HEIGHT) & (
    camera_x.abs() <= SIGNAL_HALF_WIDTH
)

# Where there is actually light, which is where a phase reading means anything.
lit = gpu_to_numpy(target_intensity > 0.05)

print(
    f"top hat: {TOPHAT_HEIGHT * 1e6:.0f} um tall, "
    f"{FOCAL_SPOT_WAIST * 1e6:.1f} um waist"
)
print(
    f"the waist is the focal spot of a {2 * BEAM_RADIUS * 1e3:.0f} mm beam through "
    f"{FOCAL_LENGTH * 1e3:.0f} mm, so the plateau is "
    f"{TOPHAT_HEIGHT / FOCAL_SPOT_WAIST:.0f} spots long"
)
print(
    f"sampled at {CAMERA_PIXEL_SIZE[0] * 1e6:.2f} um per pixel, so the waist spans "
    f"{FOCAL_SPOT_WAIST / CAMERA_PIXEL_SIZE[0]:.1f} pixels"
)

# %% How much light there is to lose
incident_power = slm_camera_model.incident_power()
pixel_area = slm_camera_model.output_pixel_area()

slm_pitch = float(slm_geometry.pixel_size.flatten()[0])
order_half_extent = WAVELENGTH * FOCAL_LENGTH / (2 * slm_pitch)
window_half_extent = CAMERA_RESOLUTION[0] * CAMERA_PIXEL_SIZE[0] / 2
print(
    f"the SLM can address +-{order_half_extent * 1e3:.1f} mm, "
    f"the camera sees +-{window_half_extent * 1e6:.0f} um of it"
)

print(
    "focal-plane sampling margin: "
    f"{tuple(round(v, 2) for v in slm_camera_model.focal_plane_sampling_margin())}"
)


def efficiency_of(record) -> float:
    """The share of the incident power a retrieval left inside the window."""
    intensity = np.asarray(record.visualization_data.retrieved_intensity)
    return float(intensity.sum()) * pixel_area / incident_power


# %% Top hat optimized with an intensity-only cost
absolute_target = (
    target_intensity
    / target_intensity.sum()
    * (WANTED_EFFICIENCY * incident_power / pixel_area)
)
print(f"\nasking for {WANTED_EFFICIENCY:.0%} of the incident power in the shape")

phase_retriever = GradientPhaseRetriever(
    slm_camera_model=slm_camera_model,
    target=absolute_target,
    signal_region=signal_region,
    init_slm_phase=init_slm_phase,
)
# A factory rather than a bare cost, so the choice survives the target being set again,
# as a camera-feedback loop does on every iteration.
phase_retriever.set_loss_factory(
    lambda target, mask: LossAbsoluteIntensityMSE(target, mask)
)

absolute = phase_retriever.retrieve(
    NUMBER_OF_ITERATIONS, method="l-bfgs", name="absolute intensity"
)

print(
    "absolute intensity cost:",
    {name: f"{values[-1]:.4g}" for name, values in absolute.metrics.items()},
)

absolute.visualizer().render()

# %% Optimize again, this time constraining the phase as well
slm_camera_model.virtual_slm.set_phase(init_slm_phase)
phase_retriever.set_loss_factory(None)
phase_retriever.set_target(target_intensity)
phase_retriever.set_loss_function(
    LossFidelity(
        target_intensity=target_intensity,
        target_phase=target_phase,
        signal_mask=signal_region,
    )
)

fidelity = phase_retriever.retrieve(
    NUMBER_OF_ITERATIONS, method="l-bfgs", name="fidelity"
)

print(
    "fidelity cost :",
    {name: f"{values[-1]:.4g}" for name, values in fidelity.metrics.items()},
)

fidelity.visualizer().render()

# %% Compare the two
target = gpu_to_numpy(target_intensity)
results = {}
for label, record in (("absolute", absolute), ("fidelity", fidelity)):
    intensity = np.asarray(record.visualization_data.retrieved_intensity)
    phase = np.asarray(record.visualization_data.retrieved_phase)

    intensity = intensity / intensity.max()
    rmse = float(np.sqrt(np.mean((intensity - target)[lit] ** 2)))

    resultant = np.abs(np.exp(1j * phase[lit]).mean())
    spread = float(np.sqrt(-2 * np.log(resultant)))

    masked_phase = phase.copy()
    masked_phase[~lit] = float("nan")
    results[label] = (intensity, masked_phase, rmse, spread)

    print(
        f"{label:9s}: phase spread {spread:.3f} rad, rmse {rmse:.4f}, "
        f"{efficiency_of(record):.1%} of the light kept"
    )

target_phase_over_trap = gpu_to_numpy(target_phase).copy()
target_phase_over_trap[~lit] = float("nan")
phase_limit = float(
    np.nanmax(np.abs(np.stack([results[label][1] for label in results])))
)

aspect = target.shape[0] / target.shape[1]
layout = PlotLayout(column_width=3.6, margins=(1.0, 0.15, 0.5, 0.5))
for row in ("intensity", "phase"):
    layout.add_row(
        [
            GridCell(f"target_{row}", aspect=aspect, colorbar=True),
            GridCell(f"absolute_{row}", aspect=aspect, colorbar=True),
            GridCell(f"fidelity_{row}", aspect=aspect, colorbar=True),
        ]
    )

builder = PlotBuilder(layout)
builder.draw_image(
    "target_intensity", target, cmap=INTENSITY_CMAP, vmin=0.0, vmax=1.0, title="target"
)
builder.draw_image(
    "target_phase", target_phase_over_trap, cmap=PHASE_CMAP,
    vmin=-phase_limit, vmax=phase_limit, title="target phase (flat)",
)
for cell, label in (("absolute", "absolute"), ("fidelity", "fidelity")):
    intensity, masked_phase, rmse, spread = results[label]
    builder.draw_image(
        f"{cell}_intensity", intensity, cmap=INTENSITY_CMAP, vmin=0.0, vmax=1.0,
        title=f"{label} (rmse {rmse:.3f})",
    )
    builder.draw_image(
        f"{cell}_phase", masked_phase, cmap=PHASE_CMAP,
        vmin=-phase_limit, vmax=phase_limit,
        title=f"phase, {label} (std {spread:.2f} rad)",
    )
builder.build()

plt.show()

# %%

