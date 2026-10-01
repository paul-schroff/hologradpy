from __future__ import annotations

from typing import Callable, ClassVar, Sequence

import os
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

import torch

from ..abstract import CalibratorBase
from ..camera_mapping import CameraMapping, CoarseMapper

from ...hardware import Camera, SLM
from ...hardware.camera import Background
from ...optics import SLMFourierLensModel

from .dataset_generator import DatasetGenerator
from .fitter import SpeckleFitter
from .records import SpeckleCaptureData


@dataclass(frozen=True)
class FitSettings:
    """The cost and the step size a parameterization is fitted with."""

    #: The cost, taking ``(predicted_field, camera_image)``.
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    #: Adam step size. The parameterization sets the gradient scale, so it also sets
    #: this.
    learning_rate: float


class SpeckleCalibrator(CalibratorBase):
    """Fit a model to the speckle a set of random SLM phase patterns makes.

    First capture a dataset, then step some part of the model until the simulated
    speckle matches the measured speckle.
    """

    #: The fitter that names which parameters move.
    fitter_type: ClassVar[type[SpeckleFitter]] = SpeckleFitter

    def __init__(
        self,
        slm: SLM,
        camera: Camera,
        slm_camera_model: SLMFourierLensModel,
        dataset_path: str | os.PathLike,
        camera_mapping: CameraMapping | None = None,
        number_of_random_patterns: int = 10,
        zeroth_order_mask_waists: float = 4.0,
    ) -> None:
        """
        Args:
            slm: The SLM to drive.
            camera: The camera watching its focal plane.
            slm_camera_model: The differentiable model of this setup.
            dataset_path: The dataset file, holding the captured samples and their
                description.
            camera_mapping: The camera mapping to calibrate the model's affine
                transform from. It also places the region of interest, and is checked
                against ``camera``. If None, a
                :class:`~hologradpy.calibration.camera_mapping.CoarseMapper` measures
                one with the SLM and camera.
            number_of_random_patterns: How many speckle patterns to capture.
            zeroth_order_mask_waists: Radius of the disc around the zeroth order that
                is left out of the fit, in fitted focal-spot waists. Passed to
                :class:`DatasetGenerator`. Defaults to 4.
        """
        super().__init__(slm, camera, slm_camera_model.device)

        self.dataset_path: Path = Path(dataset_path)
        self.number_of_random_patterns: int = number_of_random_patterns

        self.slm_camera_model: SLMFourierLensModel = slm_camera_model
        self.focal_length: float = slm_camera_model.focal_length

        if camera_mapping is None:
            camera_mapping = self._map_camera()
        else:
            camera_mapping.check_camera(self.camera)
        self.camera_mapping: CameraMapping = camera_mapping

        self._prepare_model()
        # One run builds the modules that make their parameters lazily
        self.slm_camera_model()
        self.slm_camera_model.calibrate_from_mapping(camera_mapping)

        self.dataset_generator: DatasetGenerator = DatasetGenerator(
            slm=self.slm,
            camera=self.camera,
            camera_mapping=camera_mapping,
            focal_length=self.focal_length,
            dataset_path=self.dataset_path,
            number_of_random_patterns=self.number_of_random_patterns,
            zeroth_order_mask_waists=zeroth_order_mask_waists,
        )

        self.capture_data: SpeckleCaptureData | None = None
        self.fitter: SpeckleFitter | None = None
        self.loss_history: list[float] = []
        self.loss_component_history: dict[str, list[float]] = {}

    def release_dataset(self) -> None:
        """Let go of the dataset file, so the same path can be captured to again."""
        if self.fitter is not None:
            self.fitter.close()

    def _map_camera(self) -> CameraMapping:
        """Map the camera when no mapping was supplied, using :class:`CoarseMapper`."""
        print("No camera mapping supplied. Running a coarse mapping.")
        return CoarseMapper(self.slm, self.camera, self.slm_camera_model).map_camera()

    def _prepare_model(self) -> None:
        """Put the model into the shape this calibration fits."""

    @abstractmethod
    def _fit_settings(self, mask: torch.Tensor) -> FitSettings:
        """The cost and step size this parameterization is fitted with.

        Args:
            mask: The region of interest, cropped to its bounding box and already in the
                model's dtype and on its device.
        """

    def _visualization_extras(self) -> dict:
        """Payload fields beyond the shared panels. Most parameterizations add none."""
        return {}

    def fit(
        self,
        number_of_epochs: int = 100,
        batch_size: int = 5,
        subset_indices: Sequence[int] | None = None,
        verbose: bool = True,
        capture_data: SpeckleCaptureData | None = None,
        background: Background | None = None,
    ) -> list[float]:
        """Fit the model to a captured dataset.

        Only the optimization runs here, so a dataset can be captured once and refitted
        several times, with different settings or more epochs. Capture one with
        :meth:`~hologradpy.calibration.speckle.DatasetGenerator.generate_dataset` on
        :attr:`dataset_generator`.

        Args:
            number_of_epochs: Passes over the dataset.
            batch_size: Phase patterns per iteration.
            subset_indices: Which dataset samples to fit against. Defaults to all.
            verbose: Show a progress bar.
            capture_data: The dataset to fit. Defaults to the one from the last call, so
                a refit needs only the settings that changed.
            background: The counts subtracted from every frame before it is compared
                with the model. It is one level for every pixel, a whole-sensor frame,
                or a function of the exposure in seconds returning either. A function
                is evaluated at the recorded exposure of the dataset. None, the
                default, subtracts nothing.

        Returns:
            list[float]: The mean loss of each epoch, also kept on :attr:`loss_history`.
        """
        if capture_data is not None:
            self.capture_data = capture_data

        if self.capture_data is None:
            raise RuntimeError(
                "No dataset to fit. Capture one with "
                "calibrator.dataset_generator.generate_dataset(...) and pass it as "
                "capture_data, or call calibrate() to do both."
            )

        self.fitter = self.fitter_type(
            capture_data=self.capture_data,
            slm_camera_model=self.slm_camera_model,
            dataset_path=self.dataset_path,
            background=background,
        )
        # The loss takes the cropped region mask of the fitter. The fitter reads the
        # loss and the learning rate only once fit() runs.
        settings = self._fit_settings(self.fitter.roi_mask)
        self.fitter.loss = settings.loss
        self.fitter.learning_rate = settings.learning_rate

        self.loss_history = self.fitter.fit(
            number_of_epochs=number_of_epochs,
            batch_size=batch_size,
            subset_indices=subset_indices,
            verbose=verbose,
        )
        self.loss_component_history = self.fitter.component_history
        return self.loss_history
