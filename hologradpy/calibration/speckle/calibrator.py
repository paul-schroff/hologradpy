from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar, Sequence

import os
from pathlib import Path

from ..abstract import CalibratorBase
from ..camera_mapping import CameraMapping, SpotArrayMapper

from ...datasets import CaptureStore, HologramFrames
from ...hardware import Camera, SLM
from ...hardware.camera import Background
from ...loss_functions import LossFunction
from ...optics import SLMFourierLensModel
from ...visualizer import VisualizationData

from .dataset_generator import DatasetGenerator
from ..model_fit import ModelFitter
from .records import SpeckleCaptureData


class SpeckleCalibrator(CalibratorBase):
    """Fit a model to the speckle a set of random SLM phase patterns makes.

    First capture a dataset, then step some part of the model until the simulated
    speckle matches the measured speckle.
    """

    #: The fitter that names which parameters move.
    fitter_type: ClassVar[type[ModelFitter]] = ModelFitter
    #: The payload the panels of this calibration draw.
    visualization_data_type: ClassVar[type[VisualizationData]]

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
            camera: The camera imaging its focal plane.
            slm_camera_model: The differentiable model of this setup.
            dataset_path: The dataset file, holding the captured samples and their
                description.
            camera_mapping: The camera mapping to calibrate the model's affine
                transform from. It also places the region of interest, and is checked
                against ``camera``. If None, a
                :class:`~hologradpy.calibration.camera_mapping.SpotArrayMapper`
                measures one with the SLM and camera.
            number_of_random_patterns: How many speckle patterns to capture.
            zeroth_order_mask_waists: Radius of the disc around the zeroth order that
                is left out of the fit, in fitted focal-spot waists. Passed to
                :class:`DatasetGenerator`.
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
        self.slm_camera_model.build()
        self.slm_camera_model.calibrate_from_mapping(camera_mapping)

        self.dataset_generator: DatasetGenerator = DatasetGenerator(
            slm=self.slm,
            camera=self.camera,
            camera_mapping=camera_mapping,
            slm_camera_model=self.slm_camera_model,
            dataset_path=self.dataset_path,
            number_of_random_patterns=self.number_of_random_patterns,
            zeroth_order_mask_waists=zeroth_order_mask_waists,
        )

        self.capture_data: SpeckleCaptureData | None = None
        self.fitter: ModelFitter | None = None
        self.loss_history: list[float] = []
        self.loss_component_history: dict[str, list[float]] = {}

    def _map_camera(self) -> CameraMapping:
        """Map the camera when no mapping was supplied, using :class:`SpotArrayMapper`.

        The mapper displays all its spots in one hologram, so a pointing jitter of the
        beam moves them together. The jitter changes only the translation of the
        mapping, which the tilt of the fitted field takes up.
        """
        print("No camera mapping supplied. Running a spot array mapping.")
        return SpotArrayMapper(
            self.slm, self.camera, self.slm_camera_model
        ).map_camera()

    def _prepare_model(self) -> None:
        """Put the model into the shape this calibration fits."""

    @abstractmethod
    def _fit(
        self,
        fitter: ModelFitter,
        subset_indices: Sequence[int] | None,
        verbose: bool,
        **fit_settings: Any,
    ) -> list[float]:
        """Run the fit of this parameterization on ``fitter``, and return its cost
        history.

        Args:
            fitter: The fitter of the dataset.
            subset_indices: Which dataset samples to fit against, or None for all.
            verbose: Show a progress bar.
            **fit_settings: The settings of the parameterization's fit.
        """

    @staticmethod
    def _fit_by_adam(
        fitter: ModelFitter,
        loss: LossFunction,
        learning_rate: float,
        subset_indices: Sequence[int] | None,
        verbose: bool,
        number_of_epochs: int,
        batch_size: int,
    ) -> list[float]:
        """Run Adam steps of ``fitter`` on ``loss`` at ``learning_rate``
        (:meth:`~hologradpy.calibration.model_fit.ModelFitter.fit`), and return the
        mean loss of each epoch.
        """
        fitter.loss = loss
        fitter.learning_rate = learning_rate
        return fitter.fit(
            number_of_epochs=number_of_epochs,
            batch_size=batch_size,
            subset_indices=subset_indices,
            verbose=verbose,
        )

    def _visualization_extras(self) -> dict:
        """Payload fields beyond the shared panels, added to the payloads of the dataset
        and of the fit. Most parameterizations add none.
        """
        return {}

    def fit(
        self,
        capture_data: SpeckleCaptureData | None = None,
        background: Background | None = None,
        subset_indices: Sequence[int] | None = None,
        verbose: bool = True,
        **fit_settings: Any,
    ) -> list[float]:
        """Fit the model to a captured dataset, by the fit of the parameterization
        (:meth:`_fit`).

        Only the optimization runs here, so a dataset can be captured once and refitted
        several times, with different settings or more steps. Capture one with
        :meth:`~hologradpy.calibration.speckle.DatasetGenerator.generate_dataset` on
        :attr:`dataset_generator`.

        Args:
            capture_data: The dataset to fit. Defaults to the one from the last call, so
                a refit needs only the settings that changed.
            background: The counts subtracted from every frame before it is compared
                with the model. It is one level for every pixel, a whole-sensor frame,
                or a function of the equivalent exposure in seconds returning either.
                A function is evaluated at the equivalent exposure of the dataset
                (:attr:`~hologradpy.hardware.camera.CameraData.equivalent_exposure`).
                None subtracts nothing.
            subset_indices: Which dataset samples to fit against. Defaults to all.
            verbose: Show a progress bar.
            **fit_settings: The settings of the parameterization's fit, such as
                ``number_of_epochs`` and ``batch_size`` of an Adam fit, or ``steps``
                and ``chunk_size`` of a Levenberg-Marquardt fit.

        Returns:
            list[float]: The cost history of the fit, also kept on
            :attr:`loss_history`. The fitter, of :attr:`fitter_type`, is kept on
            :attr:`fitter`, and the dataset it fits on the fitter's ``frames``
            (:meth:`~hologradpy.datasets.HologramFrames.from_capture`).

        Raises:
            RuntimeError: No dataset has been captured or passed.
        """
        if capture_data is not None:
            self.capture_data = capture_data

        if self.capture_data is None:
            raise RuntimeError(
                "No dataset to fit. Capture one with "
                "calibrator.dataset_generator.generate_dataset(...) and pass it as "
                "capture_data, or call calibrate() to do both."
            )

        frames = HologramFrames.from_capture(
            self.dataset_path,
            self.slm_camera_model,
            self.capture_data.roi_mask,
            camera_mapping=self.capture_data.camera_mapping,
            background=background,
            equivalent_exposure=self.capture_data.camera_data.equivalent_exposure,
        )
        self.fitter = self.fitter_type(self.slm_camera_model, frames)
        self.loss_history = self._fit(
            self.fitter, subset_indices, verbose, **fit_settings
        )
        self.loss_component_history = self.fitter.component_history
        return self.loss_history

    def dataset_visualization_data(
        self,
        capture_data: SpeckleCaptureData | None = None,
        sample_index: int = 0,
    ) -> VisualizationData:
        """The captured dataset before any fitting.

        Args:
            capture_data: The dataset to show. Defaults to the one from the last
                :meth:`fit`, so a fitted calibrator needs no argument.
            sample_index: Which captured pattern to show.

        Returns:
            VisualizationData: A :attr:`visualization_data_type` carrying only what a
            dataset holds.

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
            slm_pattern=sample["slm_levels"],
            **self._visualization_extras(),
        )

    def _build_visualization_data(self, **fit_fields: Any) -> VisualizationData | None:
        """The panels' inputs after a fit, or None if anything needed is missing.

        Visualization must never be the reason a calibration fails, so a problem here
        gives no figure and keeps the result.

        Args:
            **fit_fields: The payload fields of what the parameterization recovered.
        """
        try:
            measured_roi, predicted_roi = self.fitter.measured_and_predicted_roi(0)
            with CaptureStore.open(self.dataset_path) as store:
                sample = store.read(0)
            return self.visualization_data_type(
                camera_image=sample["camera_image"],
                roi_mask=self.capture_data.roi_mask,
                slm_pattern=sample["slm_levels"],
                measured_roi=measured_roi,
                predicted_roi=predicted_roi,
                loss_history=list(self.loss_history),
                loss_component_history={
                    label: list(values)
                    for label, values in self.loss_component_history.items()
                },
                **self._visualization_extras(),
                **fit_fields,
            )
        except (RuntimeError, OSError, KeyError, IndexError) as error:
            print(f"Could not collect visualization data ({error}).")
            return None
