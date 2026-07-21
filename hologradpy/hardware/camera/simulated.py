from __future__ import annotations
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray

from ...optics.systems import SLMFourierLensModel
from ...optics.modules.hardware_models import (
    CameraSensor,
    BackgroundScatter,
    PowerInstability,
)
from ...optics.modules.diagonal_elements import StaticSLMField
from ...utils import gpu_to_numpy
from ...roi import ROI
from .abstract import Camera, get_orientation_transformation


class SimulatedCameraTorch(Camera):
    """A native HoloGradPy camera backed by a differentiable optical model.

    Implements the :class:`~hologradpy.hardware.camera.Camera` interface directly (no
    slmsuite base). A :class:`CameraSensor` built from the sensor keyword arguments is
    appended as the terminal module of ``slm_camera_model``, so a frame is the model
    evaluated to digital pixel values (ADU). The ``rot`` / ``fliplr`` / ``flipud``
    orientation is applied to each captured frame via :attr:`transform`, and the
    :class:`~hologradpy.roi.ROI` crops the reoriented frame.
    """

    def __init__(
        self,
        slm_camera_model: SLMFourierLensModel,
        bitdepth: int = 8,
        name: str = "SimulatedCameraTorch",
        exposure_bounds: tuple[float, float] | None = (0.0, 1.0),
        rot: float | str = "0",
        fliplr: bool = False,
        flipud: bool = False,
        quantum_efficiency: float = 1.0,
        full_well_capacity: float = 1e4,
        exposure_time: float = 1e-3,
        gain: float = 1.0,
        noise_level: float = 0.0,
        nd_filter_optical_density: float = 0.0,
        add_noise: bool = True,
        quantize: bool = True,
        background_scatter_power: float | None = None,
        background_scatter_grain_radius: float = 5e-6,
        background_scatter_seed: int | None = None,
        power_std: float | None = None,
        power_seed: int | None = None,
    ) -> None:
        """Initialize a simulated camera with a given SLM camera model.

        A :class:`CameraSensor` is constructed from the sensor keyword arguments
        and appended as the terminal module of ``slm_camera_model``, so the model
        emits digital pixel values (ADU) from photons to electrons to ADU with the
        quantum efficiency, exposure, gain, full-well saturation, read noise and
        bit depth, instead of a bare ``|E|^2``. The camera exposure
        (``set_exposure`` / ``autoexpose``) drives the sensor's exposure.

        ``exposure_bounds`` defaults to ``(0.0, 1.0)`` so the simulated camera
        advertises a 1 s maximum integration like real hardware. Callers such as
        ``CoarseMapper`` read this ceiling instead of hardcoding it.

        When ``background_scatter_power`` is given, a static laser-speckle
        stray-light background of that total power [W] (grain
        ``background_scatter_grain_radius`` [m], reproducible via
        ``background_scatter_seed``) is generated and inserted as a
        :class:`BackgroundScatter` module immediately before the sensor, so it is
        added before the ND filter.

        When ``power_std`` is given, a :class:`PowerInstability` (a fluctuating laser
        that scales the field power by a factor drawn ~ N(1, power_std) each frame) is
        inserted just after the model's ``StaticSLMField``, reproducible via
        ``power_seed``.

        ``rot`` (``"90"`` / ``"180"`` / ``"270"`` or the ``numpy.rot90`` code) with
        ``fliplr`` / ``flipud`` orient the captured frame the way a real camera would
        be mounted, matching the slmsuite orientation convention.
        """
        # Camera geometry comes from the last *optical* module (the Fourier
        # lens / affine), not the sensor, whose output geometry mirrors its input.
        output_module = slm_camera_model[-1]
        if isinstance(output_module, CameraSensor):
            output_module = slm_camera_model[-2]

        pixel_size_out = output_module.pixel_size_out
        if pixel_size_out is None:
            pixel_size_out = getattr(output_module, "_pixel_size_out_init", None)
        if pixel_size_out is None:
            raise ValueError(
                "Could not infer camera pixel size from the final optical "
                "module. Set output pixel size in the model configuration."
            )
        # Pixel pitch (y, x) in SI, taken directly from the model (no unit round-trip).
        self._pixel_size = (
            torch.as_tensor(pixel_size_out).detach().cpu().numpy().astype(np.float64)
        )

        # Raw sensor shape (rows, cols). The displayed shape swaps for 90/270
        # rotation, matching the slmsuite Camera.shape / default_shape convention.
        raw_shape = tuple(int(size) for size in output_module.resolution_out)
        if rot in ("90", 1, "270", 3):
            displayed_shape = (raw_shape[1], raw_shape[0])
        else:
            displayed_shape = raw_shape
        self._resolution: tuple[int, int] = displayed_shape
        self.default_shape: tuple[int, int] = displayed_shape
        self.shape: tuple[int, int] = displayed_shape
        self.transform = get_orientation_transformation(rot, fliplr, flipud)

        self.name = str(name)
        self.bitdepth = int(bitdepth)
        self._adu_levels = 2 ** self.bitdepth
        self._exposure_bounds = (
            (float(np.min(exposure_bounds)), float(np.max(exposure_bounds)))
            if exposure_bounds is not None
            else None
        )
        self.exposure_s: float = 1.0  # Default to 1 s like a real simulated camera.
        self._roi = ROI(0, 0, displayed_shape[0], displayed_shape[1])

        self.slm_camera_model: SLMFourierLensModel = slm_camera_model

        # A fluctuating laser: scale the SLM-plane field power by a fresh N(1,
        # power_std) factor each frame, inserted right after the static beam.
        if power_std is not None:
            self.power_instability = PowerInstability(power_std, seed=power_seed)
            slm_camera_model.insert_after(
                StaticSLMField, "power_instability", self.power_instability
            )

        # Build the sensor and append it as the terminal module (reusing one the
        # model may already carry), so the model emits digital pixel values.
        if isinstance(slm_camera_model[-1], CameraSensor):
            if background_scatter_power is not None:
                raise ValueError(
                    "background_scatter_power cannot be added when "
                    "slm_camera_model already terminates in a CameraSensor; "
                    "build the model without a pre-attached sensor."
                )
            self.sensor = slm_camera_model[-1]
        else:
            # A static laser-speckle stray-light background is added just before
            # the sensor (so it passes through the ND filter, which lives in
            # CameraSensor).
            if background_scatter_power is not None:
                self.background_scatter = BackgroundScatter(
                    background_scatter_power,
                    grain_radius=background_scatter_grain_radius,
                    seed=background_scatter_seed,
                )
                slm_camera_model.add("background", self.background_scatter)
            self.sensor = CameraSensor(
                quantum_efficiency=quantum_efficiency,
                full_well_capacity=full_well_capacity,
                exposure_time=exposure_time,
                gain=gain,
                noise_level=noise_level,
                nd_filter_optical_density=nd_filter_optical_density,
                bitdepth=bitdepth,
                add_noise=add_noise,
                quantize=quantize,
            )
            slm_camera_model.add("sensor", self.sensor)

    # Native geometry / exposure surface

    @property
    def pixel_size(self) -> NDArray[np.float64]:
        """Pixel pitch ``(y, x)`` in metres."""
        return self._pixel_size

    @property
    def resolution(self) -> tuple[int, int]:
        """Displayed resolution ``(height, width)`` in pixels (post orientation)."""
        return self._resolution

    @property
    def adu_levels(self) -> int:
        """Number of digital levels (``2 ** bitdepth``)."""
        return self._adu_levels

    @property
    def exposure_bounds(self) -> tuple[float, float] | None:
        """The ``(min, max)`` exposure time in seconds, or ``None`` if unbounded."""
        return self._exposure_bounds

    @property
    def roi(self) -> ROI:
        """The current region of interest, in displayed ``(row, col)`` coordinates."""
        return self._roi

    def set_roi(self, roi: ROI | None) -> None:
        """Set the region of interest (``None`` resets to the full frame)."""
        if roi is None:
            self._roi = ROI(0, 0, self._resolution[0], self._resolution[1])
        else:
            self._roi = roi

    def get_exposure(self) -> float:
        """The current exposure time in seconds."""
        return float(self.exposure_s)

    def set_exposure(self, exposure_s: float) -> None:
        """Set the exposure time in seconds."""
        self.exposure_s = float(exposure_s)

    # Capture

    def _capture_frame(self) -> torch.Tensor:
        """One raw sensor frame as a tensor (full frame, before orientation and ROI
        crop).

        Drives the sensor exposure from the camera exposure so ``set_exposure`` /
        ``autoexpose`` take effect.
        """
        self.sensor.exposure_time = float(self.exposure_s)
        return self.slm_camera_model()  # CameraSensor returns pixel values

    def get_image(
        self,
        exposure_s: float | None = None,
        averaging: int = 1,
        backend: Literal["numpy", "torch"] = "numpy",
    ) -> NDArray | torch.Tensor:
        """Capture a frame as a ``(height, width)`` array of digital counts.

        ``exposure_s`` sets the exposure first when given. ``averaging`` sums that many
        fresh frames (integer sum, not mean), promoting to float to avoid overflow, as
        in slmsuite. The frame is then reoriented via :attr:`transform` and cropped to
        :attr:`roi`.

        ``backend="torch"`` runs the whole pipeline (averaging, orientation, ROI crop)
        on the model's tensors and returns a live tensor on their device, avoiding the
        CPU transfer of the default numpy path.
        """
        if backend not in ("numpy", "torch"):
            raise ValueError("Backend must be either 'numpy' or 'torch'.")
        if exposure_s is not None:
            self.set_exposure(exposure_s)

        averaging = max(1, int(averaging))
        if backend == "numpy":
            if averaging > 1:
                frames = [
                    gpu_to_numpy(self._capture_frame()) for _ in range(averaging)
                ]
                image = np.sum(np.stack(frames).astype(np.float64), axis=0)
            else:
                image = gpu_to_numpy(self._capture_frame())
        else:
            if averaging > 1:
                frames = [self._capture_frame() for _ in range(averaging)]
                image = torch.stack(frames).to(torch.float64).sum(dim=0)
            else:
                image = self._capture_frame()

        image = self.transform(image)
        return self._roi.crop(image)

    def close(self) -> None:
        torch.cuda.empty_cache()
