from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar, Sequence

import numpy as np
import torch
from numpy.typing import NDArray

from ....hardware.camera import Background
from ....loss_functions import MaskedIntensityMSE
from ....optics.modules.pixel_crosstalk import ConvolutionalCrosstalk, PixelCrosstalk

from ...speckle.calibrator import SpeckleCalibrator
from ...model_fit import ModelFitter

from .crosstalk_fitter import CrosstalkFitter
from .records import PixelCrosstalkCalibrationData
from .visualizer import CrosstalkVisualizationData


class CrosstalkSpeckleCalibrator(SpeckleCalibrator):
    """Recover the pixel-crosstalk kernel from captured speckle.

    The SLM-plane beam is held fixed, so calibrate the wavefront first. The fringing
    field acts on the difference between neighboring pixels, so the patterns are drawn
    one pixel at a time rather than band limited.

    The model's SLM stage carries the crosstalk model to fit, so the model is built
    with a :class:`~hologradpy.optics.modules.virtual_slms.VirtualSLM` that has one,
    such as ``VirtualSLM.from_slm(slm, pixel_crosstalk=...)``. The focal-plane
    partial affine keeps the values calibrated from the camera mapping.
    """

    fitter_type: ClassVar[type[CrosstalkFitter]] = CrosstalkFitter
    visualization_data_type: ClassVar[type[CrosstalkVisualizationData]] = (
        CrosstalkVisualizationData
    )

    # Adam step size for the crosstalk parameters.
    learning_rate: float = 1e-2

    def _prepare_model(self) -> None:
        """Turn on the crosstalk parameters this calibrator fits."""
        virtual_slm = self.slm_camera_model.virtual_slm
        if virtual_slm.pixel_crosstalk is None:
            raise ValueError(
                "This calibration fits a pixel-crosstalk model and the SLM stage "
                "carries none. Build the model with a VirtualSLM that carries the "
                "crosstalk model to fit."
            )

        for parameter in self.slm_camera_model.slm_field.parameters():
            parameter.requires_grad_(False)
        for parameter in virtual_slm.pixel_crosstalk.parameters():
            parameter.requires_grad_(True)

    @property
    def pixel_crosstalk(self) -> PixelCrosstalk:
        """The crosstalk model being fitted."""
        return self.slm_camera_model.virtual_slm.pixel_crosstalk

    def _fit(
        self,
        fitter: ModelFitter,
        subset_indices: Sequence[int] | None,
        verbose: bool,
        number_of_epochs: int = 50,
        batch_size: int = 5,
    ) -> list[float]:
        """Adam steps on the masked intensity cost at :attr:`learning_rate`, and the
        mean loss of each epoch.

        Args:
            fitter: The fitter of the dataset.
            subset_indices: Which dataset samples to fit against, or None for all.
            verbose: Show a progress bar.
            number_of_epochs: Passes over the dataset.
            batch_size: Phase patterns per step.
        """
        return self._fit_by_adam(
            fitter,
            MaskedIntensityMSE(fitter.frames.region.to(fitter.frames.dtype)),
            self.learning_rate,
            subset_indices,
            verbose,
            number_of_epochs,
            batch_size,
        )

    def _visualization_extras(self) -> dict:
        """The upscale factor of the crosstalk model, which every panel draws on."""
        return {"upscale_factor": self.pixel_crosstalk.upscale_factor}

    def recovered_kernel(self) -> NDArray:
        """The fitted fringing-field kernel, on the sub-pixel grid."""
        return _effective_kernel(self.pixel_crosstalk)

    def _injected_kernel(self) -> NDArray | None:
        """The kernel the model was built with."""
        return getattr(self.camera, "static_crosstalk_kernel", None)

    def _fitted_parameters(self) -> dict:
        values = {}
        for name, parameter in self.pixel_crosstalk.named_parameters():
            array = parameter.detach().cpu().numpy()
            values[name] = float(array) if array.ndim == 0 else array
        return values

    def _residual_metrics(self, kernel: NDArray) -> dict:
        """How close the recovered kernel came, when the model injected one."""
        injected = self._injected_kernel()
        if injected is None:
            return {}

        injected = np.asarray(injected, dtype=float)
        if injected.shape != kernel.shape:
            return {}

        recovered = kernel / kernel.sum()
        injected = injected / injected.sum()
        difference = recovered - injected
        return {
            "kernel_rms_error": float(np.sqrt(np.mean(difference**2))),
            "kernel_relative_rms_error": float(
                np.sqrt(np.mean(difference**2)) / np.sqrt(np.mean(injected**2))
            ),
        }

    def generate_crosstalk_calibration(self) -> PixelCrosstalkCalibrationData:
        """The recovered kernel, as a saveable calibration."""
        crosstalk = self.pixel_crosstalk
        kernel = self.recovered_kernel()

        return PixelCrosstalkCalibrationData(
            timestamp=datetime.now(),
            name="speckle",
            model=type(crosstalk).__name__,
            upscale_factor=crosstalk.upscale_factor,
            extent=crosstalk.extent,
            kernel=kernel,
            parameters=self._fitted_parameters(),
            metadata={
                "focal_length": self.focal_length,
                "number_of_random_patterns": self.number_of_random_patterns,
                "learning_rate": self.learning_rate,
                "background_level": self.fitter.frames.background_level,
                **self._residual_metrics(kernel),
            },
            visualization_data=self._build_visualization_data(
                kernel=kernel, injected_kernel=self._injected_kernel()
            ),
        )

    def calibrate(
        self,
        subset_indices: Sequence[int] | None = None,
        seed: int | None = None,
        verbose: bool = True,
        background: Background | None = None,
        **fit_settings: Any,
    ) -> PixelCrosstalkCalibrationData:
        """Capture a dataset, fit the model to it, and return the kernel.

        The patterns are drawn per pixel and fill the whole addressable area, and the
        region of interest is the sensor minus the zeroth order. A dataset captured
        once can be refitted several times with :meth:`fit`.

        Args:
            subset_indices: Fit only these patterns of the dataset. Defaults to all.
            seed: Seed for the pattern noise.
            verbose: Print the loss as the fit runs.
            background: The counts subtracted from every frame before the fit, in any
                form :meth:`fit` takes. None subtracts nothing.
            **fit_settings: The settings of the Adam fit (:meth:`_fit`).

        Returns:
            PixelCrosstalkCalibrationData: The fitted kernel and its parameters.
        """
        capture_data = self.dataset_generator.generate_dataset(
            seed=seed, pattern="uniform"
        )

        self.fit(
            capture_data=capture_data,
            background=background,
            subset_indices=subset_indices,
            verbose=verbose,
            **fit_settings,
        )

        return self.generate_crosstalk_calibration()


def _effective_kernel(crosstalk: PixelCrosstalk) -> NDArray:
    """The fringing-field kernel of ``crosstalk``, ``(extent * P, extent * P)``."""
    if isinstance(crosstalk, ConvolutionalCrosstalk):
        return crosstalk.kernel().detach().cpu().numpy()

    extent = crosstalk.extent
    parameter = next(crosstalk.parameters(), None)
    device = None if parameter is None else parameter.device

    impulse = torch.zeros(extent, extent, device=device)
    impulse[extent // 2, extent // 2] = 1.0
    with torch.no_grad():
        response = crosstalk(impulse)
    return response.detach().cpu().numpy()
