from __future__ import annotations

import torch

from .abstract import FitSettings, SpeckleCalibrator

from ..visualizer import PSFSpeckleVisualizationData

from .....loss_functions import MaskedIntensityMSE

from ....spot_detection import capture_focal_spot

from .....optics.modules.slm_fields import (
    PSFSLMField,
    kernel_size_from_waist,
    waist_from_camera_mapping,
)


class PSFSpeckleCalibrator(SpeckleCalibrator):
    """Recover the SLM-plane field through a compact camera-plane point spread function.

    Fits a small kernel instead of the whole SLM plane, so it has far fewer parameters
    and is band limited by construction.
    """

    slm_field_type = PSFSLMField
    visualization_data_type = PSFSpeckleVisualizationData

    def _build_slm_field(self) -> PSFSLMField:
        """A kernel sized and seeded from the measured focal spot."""
        camera_pixel_size = tuple(float(pitch) for pitch in self.camera.pixel_size)
        # Needed before the capture, which crops to it.
        kernel_size = kernel_size_from_waist(
            waist_from_camera_mapping(self.camera_mapping), camera_pixel_size[1]
        )

        return PSFSLMField.from_camera_mapping(
            self.camera_mapping,
            focal_length=self.focal_length,
            camera_pixel_size=camera_pixel_size,
            kernel_size=kernel_size,
            init_psf_kernel=torch.as_tensor(
                capture_focal_spot(
                    self.slm,
                    self.camera,
                    self.camera_mapping,
                    self.focal_length,
                    kernel_size,
                ),
                dtype=torch.float32,
            ),
        )

    def _fit_settings(self, mask: torch.Tensor) -> FitSettings:
        # Band limited by construction, being a compact kernel, so it cannot fit the
        # speckle with a rough solution and needs no smoothness loss.
        return FitSettings(loss=MaskedIntensityMSE(mask), learning_rate=3e-2)

    def _visualization_extras(self) -> dict:
        """The fitted kernel this parameterisation optimised."""
        kernel = self.slm_camera_model.slm_field.get_psf_kernel()
        return {"psf_kernel": kernel.detach().cpu().numpy()}
