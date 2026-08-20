"""
Simulated hardware
==================

Building a simulated SLM and camera that behave like the real device, so an algorithm
can be developed and tested without needing access to the physical hardware.

The laser beam illuminating the SLM models:

- Its intensity profile across the SLM, a Gaussian here.
- Its absolute optical power, in watts. That is what makes the camera counts and any
  efficiency below physical numbers rather than relative ones.
- Aberrations, carried as the phase of the field. Any wavefront can be put there, a
  :class:`~hologradpy.profiles.zernike.Zernike` expansion being the usual choice. The
  beam below carries astigmatism, coma and spherical aberration.
- Intensity fluctuations, which scale the power of every frame.
- Pointing instability, which shifts the focal plane frame to frame. Available on the
  model, though not switched on below.

A simulated SLM models:

- Quantisation of the phase to the SLM's bit depth.
- Pixel crosstalk, the fringing field between neighbouring liquid-crystal pixels, which
  blurs the phase pattern and reduces efficiency.

A simulated camera models:

- Read noise, an additive floor drawn from a Poisson distribution of variance
  ``noise_level ** 2``. It does not depend on how much light arrived.
- Saturation, the clamp at the full-well capacity. How quickly it is reached depends on
  the quantum efficiency and the gain.
- Quantisation of the counts to the camera's bit depth.
- Attenuation by a neutral-density filter, applied to the signal before the read noise.
- Stray light, a static laser-speckle background scattered across the sensor.

The beam's fluctuations and pointing are configured on the camera rather than on the
beam itself, since they are drawn afresh for every frame it captures.
"""

# %% Imports
from pathlib import Path

import matplotlib.pyplot as plt
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
from hologradpy.profiles.zernike import Zernike
from hologradpy.utils import get_device
from hologradpy.visualizer import (
    INTENSITY_CMAP,
    GridCell,
    PlotBuilder,
    PlotLayout,
)

device = get_device(verbose=True)

FOCAL_LENGTH = 0.1
SEED = 0
CAMERA_RESOLUTION = (400, 640)
CAMERA_PIXEL_SIZE = (3.45e-6, 3.45e-6)

torch.manual_seed(SEED)


def aspect(image) -> float:
    """Height over width, which is what :class:`GridCell` wants."""
    return image.shape[0] / image.shape[1]


# %% The SLM
# A simulated SLM needs only its geometry and bit depth. It quantizes a desired phase
# to gray levels exactly as a real device would.
input_geometry = FieldGeometry(
    resolution=(256, 320),
    pixel_size=torch.tensor([12.5e-6, 12.5e-6], device=device),
    wavelength=torch.tensor(1039e-9, device=device),
)

slm = open_slm(SimulatedSLMTorch, input_geometry=input_geometry, bitdepth=8)
print(
    f"SLM: {slm.resolution} pixels, {slm.bitdepth}-bit, {slm.wavelength * 1e9:.0f} nm"
)

# %% The beam illuminating it
# Whatever field you put here is the ground truth a wavefront calibration would try to
# recover, so it carries a wavefront as well as an intensity profile.
beam_amplitude = gaussian_beam_intensity(
    *slm.get_spatial_grid(device=device), beam_radius=1.2e-3
).sqrt()

# Astigmatism, coma and spherical: the low-order aberrations a real beam path is left
# with once tilt and defocus have been aligned out. Piston and tilt are left off on
# purpose, since they move the pattern rather than degrade it. The indices are ANSI and
# the coefficients are in radians of wavefront error.
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


# %% An ideal camera
# The optical model carries the SLM, the beam and a Fourier lens. ``camera_angle`` and
# ``camera_shift`` mount the sensor off axis, the way a real one never sits perfectly
# square to the optical axis.
def build_model(virtual_slm=None) -> SLMCZT:
    """A fresh model of the setup, since a camera takes ownership of the one it gets.

    Args:
        virtual_slm: The SLM stage to build on. Defaults to the one the simulated SLM
            owns, so displaying a phase on the SLM shows up on the camera.
    """
    return SLMCZT(
        input_geometry=input_geometry,
        virtual_slm=slm.virtual_slm if virtual_slm is None else virtual_slm,
        camera_resolution=CAMERA_RESOLUTION,
        camera_pixel_size=CAMERA_PIXEL_SIZE,
        focal_length=FOCAL_LENGTH,
        slm_field=PixelwiseSLMField(beam),
        camera_angle=5.0,
        camera_shift=(-300e-6, 100e-6),
        padded_resolution=(480, 768),
    )


ideal_camera = open_camera(
    SimulatedCameraTorch,
    slm_camera_model=build_model(),
    bitdepth=12,
    nd_filter_optical_density=6,
    add_noise=False,
    quantize=False,
)

# %% A realistic camera
# The same model, with the imperfections a real sensor and a real laser bring. Every
# one of these is optional and off by default.
realistic_camera = open_camera(
    SimulatedCameraTorch,
    slm_camera_model=build_model(),
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

# %% Capture from both
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
(
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

# %% Frame to frame variation
# The ideal camera repeats exactly. The realistic one does not, which is what a
# feedback loop or a calibration fit has to cope with.
ideal_repeat = ideal_camera.get_image()
realistic_repeat = realistic_camera.get_image()

print(
    f"ideal frames differ by     : {abs(ideal_repeat - ideal_image).max():.0f} counts"
)
print(
    f"realistic frames differ by : "
    f"{abs(realistic_repeat.astype(float) - realistic_image).max():.0f} counts"
)

# %% Pixel crosstalk
# Fringing fields between neighbouring liquid-crystal pixels, modelled on a sub-pixel
# grid on the SLM. This is memory intensive, since every plane after the SLM grows by 
# an upscale factor, and runs best on a GPU.
crosstalk_camera = open_camera(
    SimulatedCameraTorch,
    slm_camera_model=build_model(VirtualSLM.from_slm(slm)),
    bitdepth=12,
    nd_filter_optical_density=6,
    add_noise=False,
    quantize=True,
    crosstalk_upscale_factor=3,
    crosstalk_extent=3,
    crosstalk_order=2,
    crosstalk_width=1,
)
crosstalk_camera.set_exposure(ideal_camera.get_exposure())
crosstalk_image = crosstalk_camera.get_image()

# The kernel the camera was built with, which a calibration would try to recover.
kernel = crosstalk_camera.static_crosstalk_kernel

kernel_layout = PlotLayout(column_width=3.6, margins=(1.0, 0.15, 0.5, 0.5))
kernel_layout.add_row([
    GridCell("without", aspect=aspect(ideal_image), colorbar=True),
    GridCell("with", aspect=aspect(crosstalk_image), colorbar=True),
    GridCell("kernel", aspect="equal", colorbar=True),
])
(
    PlotBuilder(kernel_layout)
    .draw_image("without", ideal_image, cmap=INTENSITY_CMAP, title="no crosstalk")
    .draw_image("with", crosstalk_image, cmap=INTENSITY_CMAP, title="with crosstalk")
    .draw_image(
        "kernel",
        kernel,
        cmap=INTENSITY_CMAP,
        title="fringing-field kernel",
        interpolation="nearest",
    )
    .build()
)

# %% Saving just the optical model
data_directory = Path("../data")
data_directory.mkdir(parents=True, exist_ok=True)

bench_path = data_directory / "simulated_bench.pt"
build_model(VirtualSLM.from_slm(slm)).save(str(bench_path))
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

# %% Saving the whole simulated camera
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

# %% The ground truth a calibration would recover
print("static_slm_field       :", ideal_camera.static_slm_field.shape)
print("static_crosstalk_kernel:", kernel.shape)
