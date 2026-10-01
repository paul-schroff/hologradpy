from __future__ import annotations

import os

import torch

from .abstract import WavefrontSpeckleCalibrator
from ....camera_mapping import CameraMapping
from .....hardware import Camera, SLM
from .....optics import SLMFourierLensModel
from ....speckle.calibrator import FitSettings

from ..visualizer import PSFSpeckleVisualizationData

from .....loss_functions import MaskedIntensityMSE

from ....spot_detection import capture_focal_spot

from .....optics.modules.slm_fields import (
    PSFSLMField,
    kernel_size_from_waist,
    waist_from_camera_mapping,
)


class PSFSpeckleCalibrator(WavefrontSpeckleCalibrator):
    """Recover the SLM-plane field through a compact camera-plane point spread function.

    Fits a small kernel instead of the whole SLM plane, so it has far fewer parameters
    and is band limited by construction.
    """

    slm_field_type = PSFSLMField
    visualization_data_type = PSFSpeckleVisualizationData

    def __init__(
        self,
        slm: SLM,
        camera: Camera,
        slm_camera_model: SLMFourierLensModel,
        dataset_path: str | os.PathLike,
        camera_mapping: CameraMapping | None = None,
        number_of_random_patterns: int = 10,
        zeroth_order_mask_waists: float = 4.0,
        psf_kernel_waists: float = 10.0,
    ) -> None:
        """
        Args:
            slm: The SLM to drive.
            camera: The camera watching its focal plane.
            slm_camera_model: The differentiable model of this setup.
            dataset_path: The dataset file, holding the captured samples and their
                description.
            camera_mapping: The camera mapping to calibrate the model's affine
                transform from. If None, a
                :class:`~hologradpy.calibration.camera_mapping.CoarseMapper` measures
                one with the SLM and camera.
            number_of_random_patterns: How many speckle patterns to capture.
            zeroth_order_mask_waists: Radius of the disc around the zeroth order that
                is left out of the fit, in fitted focal-spot waists. Defaults to 4.
            psf_kernel_waists: Side of the fitted PSF kernel, in fitted focal-spot
                waists (:func:`~hologradpy.optics.modules.slm_fields.kernel_size_from_waist`).
                A kernel much wider than the spot fits noise around it, and one too
                narrow clips the aberrated wings. Ignored when the model already holds
                a :class:`PSFSLMField`, whose kernel is kept. Defaults to 10.

        Raises:
            ValueError: ``psf_kernel_waists`` is not positive.
        """
        if psf_kernel_waists <= 0:
            raise ValueError(
                f"psf_kernel_waists is a size, so it must be positive, got "
                f"{psf_kernel_waists}."
            )
        # Set before the base constructor, which builds the kernel.
        self.psf_kernel_waists: float = float(psf_kernel_waists)
        super().__init__(
            slm=slm,
            camera=camera,
            slm_camera_model=slm_camera_model,
            dataset_path=dataset_path,
            camera_mapping=camera_mapping,
            number_of_random_patterns=number_of_random_patterns,
            zeroth_order_mask_waists=zeroth_order_mask_waists,
        )

    def _build_slm_field(self) -> PSFSLMField:
        """A kernel sized and seeded from the measured focal spot."""
        camera_pixel_size = tuple(float(pitch) for pitch in self.camera.pixel_size)
        # Needed before the capture, which crops to it.
        kernel_size = kernel_size_from_waist(
            waist_from_camera_mapping(self.camera_mapping),
            camera_pixel_size[1],
            extent_in_waists=self.psf_kernel_waists,
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
        """The fitted kernel this parameterization optimized."""
        kernel = self.slm_camera_model.slm_field.get_psf_kernel()
        return {"psf_kernel": kernel.detach().cpu().numpy()}
