from __future__ import annotations
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray

from ...optics.systems import SLMFourierLensModel, with_pixel_crosstalk
from ...optics.modules.hardware_models import (
    CameraSensor,
    BackgroundScatter,
    PowerInstability,
)
from ...optics.modules.pixel_crosstalk import (
    ConvolutionalCrosstalk,
    SuperGaussianCrosstalk,
)
from ...optics.modules.slm_fields import PixelwiseSLMField
from ...utils import gpu_to_numpy
from ...roi import ROI
from .abstract import Camera, CameraOrientation


class SimulatedCameraTorch(Camera):
    """A native HoloGradPy camera backed by a differentiable optical model.

    Implements the :class:`~hologradpy.hardware.camera.Camera` interface directly (no
    slmsuite base). A :class:`CameraSensor` built from the sensor keyword arguments is
    appended as the terminal module of ``slm_camera_model``, so a frame is the model
    evaluated to digital pixel values (ADU). The
    :class:`~hologradpy.hardware.camera.CameraOrientation` is applied to each captured
    frame via :attr:`transform`, and the :class:`~hologradpy.roi.ROI` crops the
    reoriented frame.
    """

    def __init__(
        self,
        slm_camera_model: SLMFourierLensModel,
        name: str = "SimulatedCameraTorch",
        exposure_bounds: tuple[float, float] | None = (0.0, 1.0),
        orientation: CameraOrientation = CameraOrientation(),
        background_scatter_power: float | None = None,
        background_scatter_grain_radius: float = 5e-6,
        background_scatter_seed: int | None = None,
        power_std: float | None = None,
        power_seed: int | None = None,
        crosstalk_upscale_factor: int | None = None,
        crosstalk_order: float = 2.0,
        crosstalk_width: float = 1.0,
        crosstalk_extent: int = 3,
        **sensor_kwargs,
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
        inserted just after the model's ``PixelwiseSLMField``, reproducible via
        ``power_seed``.

        When ``crosstalk_upscale_factor`` is given, the fringing field between
        neighbouring liquid-crystal pixels is modeled with a
        :class:`~hologradpy.optics.modules.pixel_crosstalk.SuperGaussianCrosstalk` of
        that many sub-pixels per SLM pixel, ``crosstalk_order`` (``q``),
        ``crosstalk_width`` (``sigma``, in cycles per SLM pixel) and
        ``crosstalk_extent`` (the reach, in SLM pixels). This is quite memory intensive,
        and works best on a GPU.

        ``orientation`` mounts the sensor the way a real camera would be, matching the
        slmsuite orientation convention. :meth:`set_orientation` remounts it later.
        """
        if crosstalk_upscale_factor is not None:
            slm_camera_model = with_pixel_crosstalk(
                slm_camera_model,
                SuperGaussianCrosstalk(
                    upscale_factor=crosstalk_upscale_factor,
                    extent=crosstalk_extent,
                    order=crosstalk_order,
                    width=crosstalk_width,
                ),
            )

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

        # Raw sensor shape (rows, cols), before the orientation is applied.
        self._raw_shape: tuple[int, int] = tuple(
            int(size) for size in output_module.resolution_out
        )
        self.set_orientation(orientation)

        self.name = str(name)
        self._exposure_bounds = (
            (float(np.min(exposure_bounds)), float(np.max(exposure_bounds)))
            if exposure_bounds is not None
            else None
        )
        self.exposure_s: float = 1.0  # Default to 1 s like a real simulated camera.

        self.slm_camera_model: SLMFourierLensModel = slm_camera_model

        # A fluctuating laser: scale the SLM-plane field power by a fresh N(1,
        # power_std) factor each frame, inserted right after the static beam.
        if power_std is not None:
            self.power_instability = PowerInstability(power_std, seed=power_seed)
            slm_camera_model.insert_after(
                PixelwiseSLMField, "power_instability", self.power_instability
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
            if sensor_kwargs:
                raise ValueError(
                    "slm_camera_model already terminates in a CameraSensor, so "
                    f"{sorted(sensor_kwargs)} cannot be applied. Configure that "
                    "sensor, or build the model without one."
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
            self.sensor = CameraSensor(**sensor_kwargs)
            slm_camera_model.add("sensor", self.sensor)

    # Native geometry / exposure surface

    @property
    def pixel_size(self) -> NDArray[np.float64]:
        """Pixel pitch ``(y, x)`` in metres."""
        return self._pixel_size

    @property
    def bitdepth(self) -> int:
        """Bits per pixel, from the sensor that produces the counts."""
        return self.sensor.bitdepth

    @property
    def max_pixel_value(self) -> int:
        """The sensor's own ceiling, the value the counts are scaled to.

        Read through rather than stored, so a camera cannot report a ceiling the frames
        it returns disagree with.
        """
        return self.sensor.max_pixel_value

    @property
    def exposure_bounds(self) -> tuple[float, float] | None:
        """The ``(min, max)`` exposure time in seconds, or ``None`` if unbounded."""
        return self._exposure_bounds

    @property
    def roi(self) -> ROI:
        """The current region of interest, in displayed ``(row, col)`` coordinates."""
        return self._roi

    @property
    def static_slm_field(self) -> NDArray | None:
        """The SLM-plane complex field this simulated Camera was built with.

        The ground truth a calibration is trying to recover, which only a simulated
        camera can answer.

        Read through the field module's ``get_wavefront``, so it is not limited to a
        literal ``PixelwiseSLMField``. ``None`` if the model carries no field yet, which
        is the case before its lazy modules have been built.
        """
        field = getattr(self.slm_camera_model, "slm_field", None)
        if field is None:
            return None
        try:
            wavefront = field.get_wavefront()
        except (AttributeError, RuntimeError):
            return None
        return wavefront.detach().cpu().numpy()

    @property
    def static_crosstalk_kernel(self) -> NDArray | None:
        """The fringing-field kernel this simulated Camera was built with."""
        crosstalk = getattr(
            self.slm_camera_model.virtual_slm, "pixel_crosstalk", None
        )
        if not isinstance(crosstalk, ConvolutionalCrosstalk):
            return None
        return crosstalk.kernel().detach().cpu().numpy()

    def set_orientation(self, orientation: CameraOrientation) -> None:
        """Remount the sensor, reorienting every frame from here on.

        A quarter turn swaps the displayed shape, so the region of interest resets to
        the whole frame rather than keeping a crop expressed in the old one.
        """
        raw_shape = self._raw_shape
        displayed_shape = (
            (raw_shape[1], raw_shape[0]) if orientation.swaps_axes() else raw_shape
        )
        # shape / default_shape follow the slmsuite convention of naming the displayed
        # frame rather than the sensor.
        self._resolution: tuple[int, int] = displayed_shape
        self.default_shape: tuple[int, int] = displayed_shape
        self.shape: tuple[int, int] = displayed_shape
        self.transform = orientation.transformation()
        self.set_roi(None)

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
