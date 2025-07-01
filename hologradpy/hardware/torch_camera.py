from typing import Type, Literal

import numpy as np
import torch
import copy
from slmsuite.hardware.cameras.camera import Camera

from . import SimulatedSLMTorch

from ..torch_modules.optical_systems import SLMCameraModel, SLMFFTAffine
from ..torch_modules.utils.tensor_utils import gpu_to_numpy, crop_to_roi


class SimulatedCameraTorch(Camera):
    def __init__(
        self,
        slm: SimulatedSLMTorch,
        resolution: tuple[int, int],
        pitch_um: tuple[float, float],
        slm_camera_model_cls: Type[SLMCameraModel] = SLMFFTAffine,
        slm_camera_model_args: dict = {},
        bitdepth: int = 8,
        name: str = "SimulatedCameraTorch",
        averaging: int = 1,
        capture_attempts: int = 1,
        hdr: bool = False,
        rot: float | str = "0",
        fliplr: bool = False,
        flipud: bool = False
    ) -> None:
        """Initialize a simulated camera with a given SLM."""
        super().__init__(
            resolution=resolution,
            bitdepth=bitdepth,
            pitch_um=pitch_um,
            name=name,
            averaging=averaging,
            capture_attempts=capture_attempts,
            hdr=hdr,
            rot=rot,
            fliplr=fliplr,
            flipud=flipud
        )

        self.slm_camera_model: SLMCameraModel = slm_camera_model_cls(
            virtual_slm=slm.virtual_slm,
            camera=self,  # At risk of causing circular references.
            **slm_camera_model_args,
        )

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
        stored_woi = copy.deepcopy(self.woi)
        self.set_woi(None)
        output = super().autoexposure(*args, **kwargs)
        self.set_woi(stored_woi)
        return output

    def _get_image_hw(
        self,
        timeout_s: float | None = None,
        backend: Literal["numpy", "torch"] = "numpy",
    ) -> torch.Tensor:
        """Get an image from the camera hardware."""

        intensity = self.slm_camera_model().abs() ** 2
        intensity = crop_to_roi(intensity, self.woi)

        if backend == "numpy":
            return gpu_to_numpy(intensity).astype(self.dtype)
        elif backend == "torch":
            return intensity
        else:
            raise ValueError("Backend must be either 'numpy' or 'torch'.")
    
    def _get_images_hw(
        self,
        image_count: int,
        timeout_s: float,
        out = None,
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
    
    
        