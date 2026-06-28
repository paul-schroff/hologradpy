from __future__ import annotations
from typing import Literal

from copy import deepcopy

import numpy as np
import torch

from slmsuite.hardware.cameras.camera import Camera

from ..propagation.optical_systems import SLMFourierLensModel
from ..propagation.camera_sensor import CameraSensor
from ..utils import gpu_to_numpy, crop_to_roi


class SimulatedCameraTorch(Camera):
    def __init__(
        self,
        slm_camera_model: SLMFourierLensModel,
        bitdepth: int = 8,
        name: str = "SimulatedCameraTorch",
        averaging: int = 1,
        capture_attempts: int = 1,
        hdr: bool = False,
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
    ) -> None:
        """Initialize a simulated camera with a given SLM camera model.

        A :class:`CameraSensor` is constructed from the sensor keyword arguments
        and appended as the terminal module of ``slm_camera_model``, so the model
        emits digital pixel values (ADU) -- photons -> electrons -> ADU with the
        quantum efficiency, exposure, gain, full-well saturation, read noise and
        bit depth -- instead of a bare ``|E|^2``. The camera exposure
        (``set_exposure`` / ``autoexposure``) drives the sensor's exposure.
        """
        # Camera geometry comes from the last *optical* module (the Fourier
        # lens / affine), not the sensor, whose output geometry mirrors its input.
        output_module = slm_camera_model[-1]
        if isinstance(output_module, CameraSensor):
            output_module = slm_camera_model[-2]
        camera_resolution = output_module.resolution_out[::-1]

        pixel_size_out = output_module.pixel_size_out
        if pixel_size_out is None:
            pixel_size_out = getattr(output_module, "_pixel_size_out_init", None)

        if pixel_size_out is None:
            raise ValueError(
                "Could not infer camera pixel size from the final optical "
                "module. Set output pixel size in the model configuration."
            )

        pixel_size_out = torch.as_tensor(pixel_size_out)
        camera_pixel_size_um = tuple((pixel_size_out * 1e6).tolist())

        super().__init__(
            resolution=camera_resolution,
            bitdepth=bitdepth,
            pitch_um=camera_pixel_size_um,
            name=name,
            averaging=averaging,
            capture_attempts=capture_attempts,
            hdr=hdr,
            rot=rot,
            fliplr=fliplr,
            flipud=flipud,
        )

        self.slm_camera_model: SLMFourierLensModel = slm_camera_model

        # Build the sensor and append it as the terminal module (reusing one the
        # model may already carry), so the model emits digital pixel values.
        if isinstance(slm_camera_model[-1], CameraSensor):
            self.sensor = slm_camera_model[-1]
        else:
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

        self.woi = (0, self.shape[1], 0, self.shape[0])  # (x, width, y, height)

    def _get_exposure_hw(self):
        return self.exposure_s

    def _set_exposure_hw(self, exposure_s: float) -> None:
        """Set the exposure time for the virtual camera."""
        self.exposure_s = exposure_s

    def set_woi(self, woi: tuple[int, int, int, int] | None = None) -> None:
        """Set the region of interest (WOI) for the camera."""
        if woi is None:
            woi = (0, self.shape[1], 0, self.shape[0])
        self.woi = woi

    def close(self) -> None:
        torch.cuda.empty_cache()

    def autoexposure(self, *args, **kwargs):
        # TODO: Ideally, self.autoexposure should work with self.woi, this is
        # just a temporary workaround.
        stored_woi = deepcopy(self.woi)
        self.set_woi(None)
        output = super().autoexposure(*args, **kwargs)
        self.set_woi(stored_woi)
        return output

    def _get_image_hw(
        self,
        timeout_s: float | None = None,
        backend: Literal["numpy", "torch"] = "numpy",
    ) -> torch.Tensor:
        """Get an image from the camera hardware (CameraSensor pixel values)."""
        # Drive the sensor exposure from the camera exposure so set_exposure /
        # autoexposure work; fall back to the sensor's own exposure_time if the
        # camera exposure has not been set yet.
        exposure = getattr(self, "exposure_s", None)
        if exposure is not None:
            self.sensor.exposure_time = float(exposure)

        image = self.slm_camera_model()  # CameraSensor returns pixel values

        roi = (
            self.woi[2],
            self.woi[2] + self.woi[3],
            self.woi[0],
            self.woi[0] + self.woi[1],
        )
        image = crop_to_roi(image, roi)

        if backend == "numpy":
            return gpu_to_numpy(image)  # .astype(self.dtype)
        elif backend == "torch":
            return image
        else:
            raise ValueError("Backend must be either 'numpy' or 'torch'.")

    def _get_images_hw(
        self,
        image_count: int,
        timeout_s: float,
        out=None,
        backend: Literal["numpy", "torch"] = "numpy",
    ) -> torch.Tensor:
        """Get multiple images from the camera hardware."""
        images = []
        for _ in range(image_count):
            image = self._get_image_hw(timeout_s, backend)
            images.append(image)

        if backend == "numpy":
            return np.stack(images)
        elif backend == "torch":
            return torch.stack(images)
        else:
            raise ValueError("Backend must be either 'numpy' or 'torch'.")
