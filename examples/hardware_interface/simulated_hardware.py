"""
Simulated hardware
==================

Building a simulated SLM and camera that behave like the real device, so an algorithm
can be developed and tested without needing access to the physical hardware.

Each stage models its own experimental effects: the beam carries the illumination and
the aberrations, the SLM quantizes and blurs the phase it is given, and the camera turns
whatever arrives into counts. Every one of those effects is optional and off by default,
and each is described below where it is switched on.
"""

# %% Imports
from pathlib import Path

import numpy as np
import torch

from hologradpy.hardware import (
    SimulatedCameraTorch,
    SimulatedSLMTorch,
    open_camera,
    open_slm,
)
from hologradpy.optics.complex_amplitude import ComplexAmplitude, FieldGeometry
from hologradpy.optics.modules.slm_fields import PixelwiseSLMField
from hologradpy.optics.modules.virtual_slms import VirtualSLM
from hologradpy.optics.systems import SLMCZT, load_optical_system
from hologradpy.profiles.amplitude import gaussian_beam_intensity
from hologradpy.profiles.phase import gaussian_phase_guess
from hologradpy.profiles.zernike import Zernike
from hologradpy.roi import ROI
from hologradpy.utils import get_device, gpu_to_numpy
from hologradpy.visualizer import (
    INTENSITY_CMAP,
    GridCell,
    PlotBuilder,
    PlotLayout,
    image_grid,
)

device = get_device(verbose=True)

FOCAL_LENGTH = 0.25
SEED = 0
# Beam pointing jitter, as the resulting wander of the focal spot. Roughly half a
# camera pixel, which is enough to see between frames without smearing the image.
POINTING_FOCAL_SHIFT = 2e-6
CAMERA_RESOLUTION = (960, 1440)
CAMERA_PIXEL_SIZE = (3.45e-6, 3.45e-6)

torch.manual_seed(SEED)


def aspect(image) -> float:
    """Height over width, which is what :class:`GridCell` wants."""
    return image.shape[0] / image.shape[1]


# %% 
# Simulating an SLM
# -----------------
#
# ``SimulatedSLMTorch`` needs a ``FieldGeometry`` and ``bitdepth``. It quantizes a
# desired phase to gray levels as a real device would. Other effects like crosstalk
# between neighbouring liquid-crystal pixels, comes further down.
input_geometry = FieldGeometry(
    resolution=(1024, 1280),
    pixel_size=torch.tensor([12.5e-6, 12.5e-6], device=device),
    wavelength=torch.tensor(1039e-9, device=device),
)

slm = open_slm(SimulatedSLMTorch, input_geometry=input_geometry, bitdepth=8)
print(
    f"SLM: {slm.resolution} pixels, {slm.bitdepth}-bit, {slm.wavelength * 1e9:.0f} nm"
)

# %% 
# Defining the incident laser beam
# --------------------------------
#
# The ``ComplexAmplitude`` incident onto the SLM, carrying a certain optical power in
# watts.
#
# Here, we define a Gaussian intensity profile, modelling phase aberrations using a
# :class:`~hologradpy.profiles.zernike.Zernike` expansion.
BEAM_RADIUS = 3.5e-3

beam_amplitude = gaussian_beam_intensity(
    *slm.get_spatial_grid(device=device), beam_radius=BEAM_RADIUS
).sqrt()

# Selecting astigmatism, coma and spherical Zernike polynomials.
ABERRATIONS = {5: 1.5, 8: -0.8, 12: 0.6}

zernike = Zernike(
    resolution=input_geometry.resolution,
    indices=list(ABERRATIONS),
    device=device,
)
beam_phase = zernike.get_phase(
    torch.tensor(list(ABERRATIONS.values()), dtype=beam_amplitude.dtype, device=device)
)
print(f"wavefront error: {float(beam_phase.std()):.2f} rad rms")

beam = ComplexAmplitude(
    beam_amplitude * torch.exp(1j * beam_phase),
    wavelength=input_geometry.wavelength,
    pixel_size=input_geometry.pixel_size,
    power=1e-3,
)


# %% 
# An ideal camera
# ---------------
#
# The optical model carries the SLM, the beam and a Fourier lens. ``camera_angle`` and
# ``camera_shift`` mount the sensor off axis, the way a real one never sits perfectly
# square to the optical axis.
#
# This camera is noiseless and unquantized. The only thing it does to the light is
# attenuate it with a neutral-density filter, applied before any read noise.
#
# Every camera takes ownership of the model it is given, so each one below gets its own.
# These settings are the same for all of them.
optics_settings = dict(
    input_geometry=input_geometry,
    camera_resolution=CAMERA_RESOLUTION,
    camera_pixel_size=CAMERA_PIXEL_SIZE,
    focal_length=FOCAL_LENGTH,
    camera_angle=5.0,
    camera_shift=(-300e-6, 100e-6),
    # padded_resolution=(480, 768),
)

ideal_camera = open_camera(
    SimulatedCameraTorch,
    # The SLM's own virtual SLM, so displaying a phase on the SLM shows up here.
    slm_camera_model=SLMCZT(
        **optics_settings,
        virtual_slm=slm.virtual_slm,
        slm_field=PixelwiseSLMField(beam),
    ),
    bitdepth=12,
    nd_filter_optical_density=6,
    add_noise=False,
    quantize=False,
)

# %% 
# A realistic camera
# ------------------
#
# The same model, with the effects seen in the lab switched on:
#
# - Read noise, an additive floor drawn from a Poisson distribution of variance
#   ``noise_level ** 2``. It does not depend on how much light arrived.
# - Saturation, the clamp at the full-well capacity. How quickly it is reached depends
#   on the quantum efficiency and the gain.
# - Quantization of the counts to the camera's bit depth.
# - Intensity fluctuations, redrawing the laser power for every frame.
# - Stray light, a static laser-speckle background scattered across the sensor.
# - Pointing instability, which shifts the focal plane frame to frame.
#
# Power fluctuations are configured on the camera, since they are drawn afresh for every
# frame it captures. Beam pointing is configured on the optical model instead: the
# jitter is a phase tilt in the SLM plane.
realistic_camera = open_camera(
    SimulatedCameraTorch,
    slm_camera_model=SLMCZT(
        **optics_settings,
        virtual_slm=slm.virtual_slm,
        slm_field=PixelwiseSLMField(beam),
        # Beam pointing jitter, as the wander of the focal spot in metres.
        pointing_focal_shift_std=POINTING_FOCAL_SHIFT,
        pointing_seed=SEED,
    ),
    bitdepth=12,
    nd_filter_optical_density=6,
    # Sensor: shot and read noise, a finite well, and quantization to whole counts.
    noise_level=4,
    full_well_capacity=1e4,
    quantum_efficiency=0.5,
    # A laser whose power drifts a few percent from frame to frame.
    power_std=0.03,
    power_seed=SEED,
    # Stray light scattered across the whole sensor.
    background_scatter_power=2e-9,
    background_scatter_seed=SEED,
)

# %% 
# Capture from both
# -----------------
#
# autoexpose finds an exposure that fills the well without saturating, exactly as it
# would on a real camera.
for camera in (ideal_camera, realistic_camera):
    camera.autoexpose(set_fraction=0.5)

ideal_image = ideal_camera.get_image()
realistic_image = realistic_camera.get_image()

print(f"ideal exposure     : {ideal_camera.get_exposure() * 1e6:.1f} us")
print(f"realistic exposure : {realistic_camera.get_exposure() * 1e6:.1f} us")
print(f"peak counts        : {ideal_image.max():.0f} vs {realistic_image.max():.0f}")

layout = PlotLayout(column_width=3.6, margins=(1.0, 0.15, 0.5, 0.5))
layout.add_row([
    GridCell("ideal", aspect=aspect(ideal_image), colorbar=True),
    GridCell("realistic", aspect=aspect(realistic_image), colorbar=True),
])
plot = (
    PlotBuilder(layout)
    .draw_image("ideal", ideal_image, cmap=INTENSITY_CMAP, title="noiseless")
    .draw_image(
        "realistic",
        realistic_image,
        cmap=INTENSITY_CMAP,
        title="noise, drift and stray light",
    )
    .build()
)

# %% 
# Frame to frame variation
# ------------------------
# The ideal camera repeats exactly. The realistic one does not, which is what a feedback
# loop or a calibration fit has to deal with.
ideal_repeat = ideal_camera.get_image()
realistic_repeat = realistic_camera.get_image()

print(
    f"ideal frames differ by     : {abs(ideal_repeat - ideal_image).max():.0f} counts"
)
print(
    f"realistic frames differ by : "
    f"{abs(realistic_repeat.astype(float) - realistic_image).max():.0f} counts"
)

# %% 
# Pixel crosstalk
# ===============
#
# Fringing fields between neighbouring liquid-crystal pixels, modelled on a sub-pixel
# grid on the SLM.
#
# .. tip::
#    This is memory intensive, since every plane after the SLM grows by an upscale
#    factor, and runs best on a GPU.
#
# SLM phase pattern is a combination of a linear and a quadratic phase, generating a
# certain spot size on the sensor, away from the optical axis. This pattern contains
# many 0 - $2 \pi$ phase jumps, which makes the effect of pixel crosstalk visisble on
# the camera.
SIGNAL_SHIFT = (1.5e-3, -1e-3)
CAMERA_SPOT_RADIUS = 1.0e-3

slm_phase = gaussian_phase_guess(
    *slm.get_spatial_grid(device=device),
    input_beam_radius=(BEAM_RADIUS, BEAM_RADIUS),
    output_beam_radius=(CAMERA_SPOT_RADIUS, CAMERA_SPOT_RADIUS),
    wavenumber=2 * np.pi / float(input_geometry.wavelength),
    focal_length=FOCAL_LENGTH,
    output_beam_shift=SIGNAL_SHIFT,
)

figure = image_grid(
    gpu_to_numpy(slm_phase) % (2 * np.pi),
    titles="SLM phase pattern",
    cmap="magma",
    colorbar_label="phase [rad]",
    column_width=6,
).build()

# %%
# Opening two simulated cameras, with and without pixel crosstalk.
#
sensor_settings = dict(
    bitdepth=12,
    nd_filter_optical_density=6,
    add_noise=False,
    quantize=True,
)

no_crosstalk_camera = open_camera(
    SimulatedCameraTorch,
    slm_camera_model=SLMCZT(
        **optics_settings,
        virtual_slm=VirtualSLM.from_slm(slm, init_phase=slm_phase),
        slm_field=PixelwiseSLMField(beam),
    ),
    **sensor_settings,
)

crosstalk_camera = open_camera(
    SimulatedCameraTorch,
    slm_camera_model=SLMCZT(
        **optics_settings,
        virtual_slm=VirtualSLM.from_slm(slm, init_phase=slm_phase),
        slm_field=PixelwiseSLMField(beam),
    ),
    **sensor_settings,
    crosstalk_upscale_factor=3,
    crosstalk_extent=3,
    crosstalk_order=2,
    crosstalk_width=1,
)

no_crosstalk_camera.autoexpose(set_fraction=0.99)
crosstalk_camera.set_exposure(no_crosstalk_camera.get_exposure())

crosstalk_image = crosstalk_camera.get_image()
no_crosstalk_image = no_crosstalk_camera.get_image()

# %% 
# The pixel crosstalk kernel used in the model
# --------------------------------------------
#
kernel = crosstalk_camera.static_crosstalk_kernel

figure = image_grid(
    kernel,
    titles="Pixel crosstalk kernel",
    cmap=INTENSITY_CMAP,
    colorbar_label="weight",
    column_width=3.2,
).build()

# %% 
# Comparing the two simulated camera outputs
# ------------------------------------------
#
# The full sensor beside the signal region, a row per camera. Detecting the region on
# the crosstalk-free frame crops both to the same window, showing the artefacts caused
# by pixel crosstalk.
signal_roi = ROI.detect(no_crosstalk_image, threshold=0.5)

figure = image_grid(
    [
        [no_crosstalk_image, signal_roi.crop(no_crosstalk_image)],
        [crosstalk_image, signal_roi.crop(crosstalk_image)],
    ],
    titles=[
        "No crosstalk",
        "No crosstalk, zoom",
        "With crosstalk",
        "With crosstalk, zoom",
    ],
    cmap=INTENSITY_CMAP,
    shared_scale=True,
    merge_colorbars=True,
    colorbar_label="camera counts [ADU]",
    column_width=3.4,
).build()

# %% 
# Saving the optical model
# ------------------------
data_directory = Path("../data")
data_directory.mkdir(parents=True, exist_ok=True)

bench_path = data_directory / "simulated_bench.pt"
SLMCZT(
    **optics_settings,
    virtual_slm=VirtualSLM.from_slm(slm),
    slm_field=PixelwiseSLMField(beam),
).save(str(bench_path))
print(f"saved the optics: {bench_path.stat().st_size / 1e6:.1f} MB")

# load_optical_system reopens it without needing to know which system class wrote it
reloaded_model = load_optical_system(bench_path, map_location=device)
reloaded_camera = open_camera(
    SimulatedCameraTorch,
    slm_camera_model=reloaded_model,
    bitdepth=12,
    nd_filter_optical_density=6,
    noise_level=4,
    full_well_capacity=1e4,
    quantum_efficiency=0.5,
)
reloaded_camera.set_exposure(realistic_camera.get_exposure())

print(f"reloaded a {type(reloaded_model).__name__}")
print(f"same camera geometry: {tuple(reloaded_camera.resolution)}")
same_beam = np.allclose(
    reloaded_camera.static_slm_field, realistic_camera.static_slm_field
)
print(f"beam preserved: {same_beam}")

# %% 
# Saving the whole simulated camera
# ---------------------------------
camera_path = data_directory / "realistic_camera.pt"
realistic_camera.save(camera_path)
print(f"saved the camera: {camera_path.stat().st_size / 1e6:.1f} MB")

restored_camera = SimulatedCameraTorch.load(camera_path, map_location=device)

print(f"exposure preserved: {restored_camera.get_exposure() * 1e6:.1f} us")
speckle = "background.background"
same_speckle = torch.equal(
    restored_camera.slm_camera_model.state_dict()[speckle],
    realistic_camera.slm_camera_model.state_dict()[speckle],
)
print(f"stray light preserved: {same_speckle}")

# A frame off the restored camera is the same kind of frame, noise and all.
restored_image = restored_camera.get_image()
print(f"peak counts: {restored_image.max():.0f} vs {realistic_image.max():.0f}")
