"""Using the HoloGradPy hardware interface.

HoloGradPy talks to cameras and SLMs through a small native interface, so the rest of
the library never has to know which physical device is attached. This script walks
through the whole surface on a simulated device.

There are three entry points, all re-exported from ``hologradpy.hardware``.

    open_camera / open_slm
        Construct a device and return it ready to use, in one call. This is the
        recommended way to obtain a device.

    as_camera / as_slm
        Adapt a device you have already constructed. Idempotent, so it is safe to call
        on a device that is already native.

    register_camera_backend / register_slm_backend
        Give a driver class a short name, so open_camera / open_slm can build it by
        name, for example ``open_camera("thorcam", serial=...)``.

Two conventions hold for every native device, whatever the underlying driver.

    Geometry is (y, x) = (height, width). A per axis quantity such as pixel_size is
    ordered (y, x), and regions of interest are (row, col). This matches array indexing
    and is the reverse of slmsuite, which uses (x, y).

    Units are SI. pixel_size and wavelength are in metres, exposure is in seconds.
"""

# %% Imports
import matplotlib.pyplot as plt
import torch

from hologradpy.hardware import (
    Camera,
    SLM,
    ROI,
    SimulatedSLMTorch,
    SimulatedCameraTorch,
    open_camera,
    open_slm,
    as_camera,
    as_slm,
    register_slm_backend,
)

from hologradpy.optics.complex_amplitude import ComplexAmplitude, FieldGeometry
from hologradpy.optics.systems import SLMFFTAffine
from hologradpy.optics.modules.slm_fields import PixelwiseSLMField
from hologradpy.profiles.amplitude import gaussian_beam_intensity
from hologradpy.utils import get_device

device = get_device(verbose=True)

# %% 1. Open a device with the factory
# The simulated devices need an optical model. Most code in this section is setting up
# the optical model. On real hardware you would go straight to 
# open_camera(YourDriver, ...) without any of it.
slm_geometry = FieldGeometry(
    resolution=(1024, 1280),
    pixel_size=torch.tensor([12.5e-6, 12.5e-6], device=device),
    wavelength=torch.tensor(0.630e-6, device=device),
)

slm = open_slm(SimulatedSLMTorch, input_geometry=slm_geometry, bitdepth=8)

gaussian_intensity = gaussian_beam_intensity(
    *slm.get_spatial_grid(device), beam_radius=5e-3
)
beam = ComplexAmplitude(
    gaussian_intensity.sqrt() + 0j,
    wavelength=slm_geometry.wavelength,
    pixel_size=slm_geometry.pixel_size,
    power=1e-3,
)

camera_model = SLMFFTAffine(
    input_geometry=slm_geometry,
    virtual_slm=slm.virtual_slm,
    camera_resolution=(960, 1440),
    camera_pixel_size=(3.75e-6, 3.75e-6),
    focal_length=0.25,
    slm_field=PixelwiseSLMField(beam),
    padded_resolution=(2048, 2048),
    camera_angle=0,
    camera_shift=(0, 0),
    power_normalized=True,
)

camera = open_camera(
    SimulatedCameraTorch,
    slm_camera_model=camera_model,
    nd_filter_optical_density=5.0,
    quantum_efficiency=0.01,
)

# %% 2. Read geometry and units through the native interface
# These properties read the same way for a simulated device and for real hardware,
# always in (y, x) order and SI units.
print("SLM")
print("resolution (h, w):", slm.resolution)
print("pixel_size (y, x):", slm.pixel_size, "m")
print("wavelength:", slm.wavelength, "m")

print("Camera")
print("resolution (h, w):", camera.resolution)
print("pixel_size (y, x):", camera.pixel_size, "m")
print("adu_levels:", camera.adu_levels)
print("max pixel value:", camera.adu_levels - 1)
print("exposure_bounds:", camera.exposure_bounds, "s")

# Both subclass the native Camera / SLM base classes, which is how the library accepts
# any conforming device.
print("camera is a native Camera:", isinstance(camera, Camera))
print("slm is a native SLM:", isinstance(slm, SLM))

# %% 3. Set the exposure and capture a frame
# camera.autoexpose(set_fraction=0.5) is also available. It picks an exposure that
# fills the sensor to a target fraction. Here we set the exposure by hand.
camera.set_exposure(100e-6)
print("exposure now:", camera.get_exposure(), "s")

frame = camera.get_image()
print("frame shape (h, w):", frame.shape, "dtype:", frame.dtype)

plt.figure()
plt.imshow(frame, cmap="turbo")
plt.title("Full frame")
plt.colorbar()

# %% 4. Regions of interest with ROI
# ROI is a frozen (top_row, left_column, height, width) value object in native (row,
# col) pixels. Build one centred on a point and hand it to set_roi. get_image then
# returns only that window.
center = (camera.resolution[0] // 2, camera.resolution[1] // 2)  # (row, col)
window = ROI.centered(center=center, size=(256, 256))
camera.set_roi(window)
print("current roi:", camera.roi)

cropped = camera.get_image()
print("cropped frame shape (h, w):", cropped.shape)

# The same ROI slices a full frame directly, so you can crop an array you already hold
# without touching the camera.
patch = frame[window.rows, window.columns]
print("patch from the stored full frame:", patch.shape)

plt.figure()
plt.imshow(cropped, cmap="turbo")
plt.title("Region of interest")
plt.colorbar()

# Passing None resets the camera to the full sensor.
camera.set_roi(None)
print("roi after reset:", camera.roi)

# %% 5. Normalize a device you already built
# If you construct a device yourself, as_slm / as_camera return it ready to use. This is
# exactly what open_slm / open_camera call internally. The simulated devices implement
# the native interface directly, so they are passed through unchanged. A real slmsuite
# driver is wrapped in an adapter here instead (see section 7).
raw_slm = SimulatedSLMTorch(input_geometry=slm_geometry, bitdepth=8)
print("raw device is a native SLM:", isinstance(raw_slm, SLM))

native = as_slm(raw_slm)
print("as_slm passes a native device through:", native is raw_slm)

# An already native device is returned unchanged. Calibrators rely on this to accept 
# either form.
print("camera as is:", as_camera(camera) is camera)

# %% 6. Register a backend and open it by name
# Name a driver once, then open it by that name anywhere. The symmetric
# register_camera_backend lets open_camera("mycam", ...) build a camera.
register_slm_backend("simulated", SimulatedSLMTorch)
named_slm = open_slm("simulated", input_geometry=slm_geometry, bitdepth=8)
print("opened by name is a native SLM:", isinstance(named_slm, SLM))

# %% 7. The same code with real hardware
# Nothing above is specific to simulation. With a real slmsuite driver you would
# write one of the following, and every native call shown here works the same.
#
#     from slmsuite.hardware.cameras.thorlabs import ThorCam
#
#     camera = open_camera(ThorCam, serial="12345")
#
# or, if you already hold a driver instance,
#
#     camera = as_camera(ThorCam(serial="12345"))
#
# or register it once and open it by name,
#
#     register_camera_backend("thorcam", ThorCam)
#     camera = open_camera("thorcam", serial="12345")
#
# Every slmsuite driver already has a short name (thorlabs, basler, hamamatsu, ...).
# Enable them all in one opt-in call, then open by name without importing the driver
# yourself. Each vendor SDK is imported lazily, only when its backend is opened.
#
#     from hologradpy.hardware import register_slmsuite_backends
#
#     register_slmsuite_backends()
#     camera = open_camera("thorlabs", serial="12345")

plt.show()
