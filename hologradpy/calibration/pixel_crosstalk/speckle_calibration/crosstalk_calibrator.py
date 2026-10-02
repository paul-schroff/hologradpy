from __future__ import annotations

from datetime import datetime
from typing import ClassVar, Sequence

import numpy as np
import torch
from numpy.typing import NDArray

from ....datasets import CaptureStore
from ....hardware.camera import Background
from ....loss_functions import MaskedIntensityMSE
from ....optics.modules.pixel_crosstalk import ConvolutionalCrosstalk, PixelCrosstalk

from ...speckle import SpeckleCaptureData
from ...speckle.calibrator import FitSettings, SpeckleCalibrator

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

    def _fit_settings(self, mask: torch.Tensor) -> FitSettings:
        return FitSettings(
            loss=MaskedIntensityMSE(mask), learning_rate=self.learning_rate
        )

    def fit_crosstalk(self, *args, **kwargs) -> list[float]:
        """Fit the crosstalk model to a captured dataset. The crosstalk name for
        :meth:`~hologradpy.calibration.speckle.calibrator.SpeckleCalibrator.fit`.
        """
        return self.fit(*args, **kwargs)

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

    def dataset_visualization_data(
        self,
        capture_data: SpeckleCaptureData | None = None,
        sample_index: int = 0,
    ) -> CrosstalkVisualizationData:
        """The captured dataset before any fitting.

        Args:
            capture_data: The dataset to show. Defaults to the one from the last
                :meth:`fit_crosstalk`, so a fitted calibrator needs no argument.
            sample_index: Which captured pattern to show.

        Returns:
            CrosstalkVisualizationData: Carrying only what a dataset holds.

        Raises:
            RuntimeError: If no dataset has been captured or loaded yet.
        """
        if capture_data is not None:
            self.capture_data = capture_data

        if self.capture_data is None:
            raise RuntimeError(
                "No dataset to show. Capture one with "
                "calibrator.dataset_generator.generate_dataset(...) and pass it as "
                "capture_data."
            )

        with CaptureStore.open(self.dataset_path) as store:
            sample = store.read(sample_index)
        return self.visualization_data_type(
            camera_image=sample["camera_image"],
            roi_mask=self.capture_data.roi_mask,
            upscale_factor=self.pixel_crosstalk.upscale_factor,
            slm_pattern=sample["slm_levels"],
        )

    def _build_visualization_data(
        self, kernel: NDArray
    ) -> CrosstalkVisualizationData | None:
        """Collect the panels' inputs, or None if anything needed is missing."""
        try:
            measured_roi, predicted_roi = self.fitter.measured_and_predicted_roi(0)
            with CaptureStore.open(self.dataset_path) as store:
                sample = store.read(0)
            return self.visualization_data_type(
                camera_image=sample["camera_image"],
                roi_mask=self.capture_data.roi_mask,
                upscale_factor=self.pixel_crosstalk.upscale_factor,
                slm_pattern=sample["slm_levels"],
                measured_roi=measured_roi,
                predicted_roi=predicted_roi,
                kernel=kernel,
                injected_kernel=self._injected_kernel(),
                loss_history=list(self.loss_history),
                loss_component_history={
                    label: list(values)
                    for label, values in self.loss_component_history.items()
                },
            )
        except (RuntimeError, OSError, KeyError, IndexError) as error:
            print(f"Could not collect visualization data ({error}).")
            return None

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
                "background_level": self.fitter.background_level,
                **self._residual_metrics(kernel),
            },
            visualization_data=self._build_visualization_data(kernel),
        )

    def calibrate(
        self,
        speckle_pattern_extent: tuple[float, float] | None = None,
        number_of_epochs: int = 50,
        batch_size: int = 5,
        subset_indices: Sequence[int] | None = None,
        seed: int | None = None,
        verbose: bool = True,
        background: Background | None = None,
    ) -> PixelCrosstalkCalibrationData:
        """Capture a dataset, fit the model to it, and return the kernel.

        To capture once and refit several times, call the two phases yourself:
        ``dataset_generator.generate_dataset(...)`` followed by :meth:`fit_crosstalk`.

        Args:
            speckle_pattern_extent: Full width ``(y, x)`` of the speckle in the image
                plane, in metres. When None, the patterns are drawn per pixel and
                fill the whole addressable area. The region of interest is then the
                sensor minus the zeroth order. With an extent, the patterns are band
                limited, so the fit loses most of its signal.
            number_of_epochs: Passes over the dataset.
            batch_size: Patterns per optimizer step.
            subset_indices: Fit only these patterns of the dataset. Defaults to all.
            seed: Seed for the pattern noise.
            verbose: Print the loss as the fit runs.
            background: The counts subtracted from every frame before the fit. It is
                one level for every pixel, a whole-sensor frame, or a function of the
                exposure in seconds returning either. A function is evaluated at the
                recorded exposure of the dataset. None, the default, subtracts nothing.

        Returns:
            PixelCrosstalkCalibrationData: The fitted kernel and its parameters.
        """
        # A previous fit holds the file mapped, which blocks writing over it.
        self.release_dataset()

        capture_data = self.dataset_generator.generate_dataset(
            speckle_pattern_extent, seed=seed, pattern="uniform"
        )

        self.fit_crosstalk(
            number_of_epochs=number_of_epochs,
            batch_size=batch_size,
            subset_indices=subset_indices,
            verbose=verbose,
            capture_data=capture_data,
            background=background,
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
