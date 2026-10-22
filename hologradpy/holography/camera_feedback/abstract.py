from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .visualizer import CameraFeedbackVisualizer, TargetPlacementData

from ..phase_retrieval import (
    GradientPhaseRetriever,
    PixelwisePhaseRetriever,
    PhaseRetrievalData,
)

from ...analysis.fourier_filter import fourier_filter, intensity_band_mask
from ...analysis.error_metrics import IntensityMetric, normalize
from ...analysis.fitting import TranslationFit, fit_translation
from ...calibration.camera_mapping import CameraMapping, SpotArrayMapper
from ...fourier_transforms import translate_intensity, translate_real
from ...hardware import Camera, SLM, as_camera, as_slm
from ...loss_functions import LossAbsoluteIntensityMSE
from ...optics.systems import SLMFourierLensModel
from ...profiles.amplitude import gaussian_blur
from ...roi import ROI
from ...serialization import SaveableRecord, record_type
from ...utils import ProgressBar, as_image, gpu_to_numpy


@record_type("camera_feedback")
@dataclass
class CameraFeedbackData(SaveableRecord):
    """One completed camera feedback run."""

    timestamp: datetime
    name: str
    target: NDArray
    signal_region: NDArray
    corrected_targets: list[NDArray] = field(default_factory=list)
    measured_images: list[NDArray] = field(default_factory=list)
    background_images: list[NDArray] | None = None
    final_camera_image: NDArray | None = None
    initial_guess: NDArray | None = None
    camera_mapping: CameraMapping | None = None
    retrievals: list[PhaseRetrievalData] = field(default_factory=list)
    metrics: dict[str, list[float]] = field(default_factory=dict)
    lower_is_better: dict[str, bool] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @property
    def signal_roi(self) -> ROI:
        """Bounding box of the signal region, which the per-iteration images are cropped
        to.
        """
        return ROI.detect(
            np.asarray(self.signal_region).astype(bool), threshold=0.0, pad=0
        )

    @property
    def number_of_iterations(self) -> int:
        return len(self.measured_images)

    def full_corrected_target(self, iteration: int = -1) -> NDArray:
        """A corrected target back on the full sensor grid."""
        return self.signal_roi.pad(self.corrected_targets[iteration], self.target.shape)

    def full_measured_image(self, iteration: int = -1) -> NDArray:
        """A measured frame back on the full sensor grid, zero outside the region."""
        return self.signal_roi.pad(self.measured_images[iteration], self.target.shape)

    def visualizer(self, iteration: int = -1) -> CameraFeedbackVisualizer:
        """The visualizer that renders this run, showing ``iteration`` (last by
        default).
        """
        return CameraFeedbackVisualizer(self, iteration)


@dataclass(frozen=True)
class SignalWindow:
    """The bounding box of the signal region in a run, with the region, the target and
    the band of the optics on it. The region and the target are also held on the whole
    sensor.
    """

    #: The bounding box of the signal region, in sensor pixels.
    roi: ROI
    #: The signal region on the whole sensor, where overexposure is checked.
    sensor_region: NDArray[np.bool_]
    #: The target on the whole sensor, divided by the mean of its bright pixels.
    sensor_target: Tensor
    #: True inside the signal region, on the bounding box.
    region: Tensor
    #: The target on the bounding box.
    target: Tensor
    #: The band of the optics on the bounding box
    #: (:func:`~hologradpy.analysis.fourier_filter.intensity_band_mask`).
    band_mask: Tensor


class FeedbackCorrectorBase(ABC):
    """Correct a light potential against the camera, driving a phase retriever.

    Needs a :class:`~hologradpy.holography.phase_retrieval.GradientPhaseRetriever`,
    since each round gives the retriever a camera-corrected target and a signal region.
    Builds a :class:`~hologradpy.holography.phase_retrieval.PixelwisePhaseRetriever` by
    default.

    During a run, the model computes only the bounding box of the signal region, the
    target's patch with its margin. The retriever compares the intensity over the whole
    box with the target in absolute units
    (:class:`~hologradpy.loss_functions.LossAbsoluteIntensityMSE`). The target is scaled
    to hold a set fraction of the power that enters the Fourier lens,
    ``target_efficiency``. Below 1, the retriever can send the remaining light outside
    the box.

    The frames of every measurement are registered before they are averaged. The
    translation of each frame is fitted with a pattern, the target or the model's
    prediction. Each frame is then moved by its own translation, so the pointing jitter
    between the frames is removed from the average.
    """

    def __init__(
        self,
        slm: SLM,
        camera: Camera,
        target: torch.Tensor | None = None,
        signal_region: torch.Tensor | None = None,
        target_position: tuple[float, float] = (0.0, 0.0),
        slm_camera_model: SLMFourierLensModel | None = None,
        init_slm_phase: torch.Tensor | None = None,
        phase_retriever: GradientPhaseRetriever | None = None,
        camera_mapping: CameraMapping | None = None,
        loss_scale: float | None = None,
        target_efficiency: float = 1.0,
    ) -> None:
        """
        Args:
            slm: The SLM displaying the hologram, or a driver that
                :func:`~hologradpy.hardware.as_native.as_slm` wraps.
            camera: The camera watching the light potential, or a driver that
                :func:`~hologradpy.hardware.as_native.as_camera` wraps.
            target: The target intensity to produce, on the camera grid.
            signal_region: Where the target is optimized.
            target_position: Where to place the target, as ``(x, y)`` metres in the
                focal plane from the zeroth order. The mapping applies the camera's turn
                and zoom, so this is a focal-plane position and not a sensor position.
                :meth:`~.CameraMapping.tilt_to_sensor_center` gives the focal-plane
                position of the sensor's centre pixel.
            slm_camera_model: The model simulating the propagation of light from the SLM
                to the camera. Required unless ``phase_retriever`` is given. The model
                is then taken from the retriever.
            init_slm_phase: Starting phase for the retriever.
            phase_retriever: An existing phase retriever. One is built when None.
            camera_mapping: The map from camera pixels to the model's output pixels,
                checked against ``camera``. The model is calibrated from it before the
                loop starts. When None,
                :meth:`apply_camera_mapping` measures one with a
                :class:`~hologradpy.calibration.camera_mapping.SpotArrayMapper`.
            loss_scale: Slope of the cost function of the default retriever.
            target_efficiency: The fraction of the power entering the Fourier lens
                held by the target in the absolute intensity cost, above 0 and at
                most 1.
        """
        self.slm = as_slm(slm)
        self.camera = as_camera(camera)
        if camera_mapping is not None:
            camera_mapping.check_camera(self.camera)
        self.camera_mapping = camera_mapping
        self.target_position = target_position
        if not 0.0 < target_efficiency <= 1.0:
            raise ValueError(
                f"The target efficiency {target_efficiency} has to lie above 0 and at "
                "most 1, as it is a fraction of the power entering the Fourier lens."
            )
        self.target_efficiency = float(target_efficiency)

        if target is None:
            target = getattr(phase_retriever, "target", None)
        if target is None:
            raise ValueError(
                "Camera feedback needs a target intensity. Pass one, or give it a "
                "phase retriever that already has one."
            )
        self.target_patch = target

        if signal_region is None:
            signal_region = getattr(phase_retriever, "signal_region", None)
        if signal_region is None:
            signal_region = torch.ones_like(target)
        if signal_region.shape != target.shape:
            raise ValueError(
                f"The signal region is {tuple(signal_region.shape)} but the target is "
                f"{tuple(target.shape)}. Both are placed together, so they have to be "
                "the same shape."
            )
        self.signal_region_patch = signal_region

        if phase_retriever is None:
            phase_retriever = self._build_retriever(
                slm_camera_model, init_slm_phase, loss_scale
            )
        self.phase_retriever = phase_retriever

        self._target = target
        self._corrected_target = target
        self.signal_region = signal_region
        if self.camera_mapping is not None:
            self.place_target()

    def _build_retriever(
        self,
        slm_camera_model: SLMFourierLensModel | None,
        init_slm_phase: torch.Tensor | None,
        loss_scale: float | None,
    ) -> PixelwisePhaseRetriever:
        """The default search, so the usual case needs no retriever built by hand."""
        if slm_camera_model is None:
            raise ValueError(
                "Camera feedback needs either a slm_camera_model to build its own "
                "phase retriever from, or a phase_retriever to drive."
            )

        options = {} if loss_scale is None else {"loss_scale": loss_scale}
        return PixelwisePhaseRetriever(
            slm_camera_model=slm_camera_model,
            init_slm_phase=init_slm_phase,
            **options,
        )

    def target_center_pixels(self) -> tuple[float, float]:
        """Where :attr:`target_position` lands on the sensor, as ``(column, row)``.

        ``target_position`` is ``(x, y)`` metres in the focal plane, measured from the
        zeroth order, and :meth:`~.CameraMapping.image_plane_to_sensor` carries it onto
        the sensor. On an unturned sensor, x increases with the column index and y with
        decreasing row index, so positive y is upward on a displayed image. On a turned
        sensor, a request of ``(+500e-6, 0)`` is half a millimetre along the x axis of
        the focal plane, turned on the sensor by the mapping's
        :attr:`~.CameraMapping.camera_angle`.
        """
        column, row = self._mapping.image_plane_to_sensor(
            np.asarray([self.target_position])
        )[0]
        return float(column), float(row)

    def _addressable_corners_pixels(self) -> np.ndarray:
        """The four corners of the addressable region, as sensor ``(column, row)``."""
        half_x, half_y = self.slm_camera_model.addressable_half_extent()
        corners = np.array(
            [
                [-half_x, -half_y],
                [half_x, -half_y],
                [half_x, half_y],
                [-half_x, half_y],
            ]
        )
        return self._mapping.image_plane_to_sensor(corners)

    def place_target(self) -> None:
        """Build the full-frame target from the patch and hand it to the retriever.

        Called by :meth:`run` after the mapping is known. The patch is pasted with its
        axes along the sensor's columns and rows, centred on
        :meth:`target_center_pixels`. On a turned camera the target is therefore turned
        in the focal plane by minus the mapping's :attr:`~.CameraMapping.camera_angle`.
        """
        device = self.slm_camera_model.device

        center = self.target_center_pixels()
        self._target = self._paste(self.target_patch.to(device), center)
        # A region is whole samples either way, so it is not worth a transform and
        # the ringing that comes with one.
        self.signal_region = self._paste(
            self.signal_region_patch.to(device), center, subsample=False
        )

        self._warn_if_zeroth_order_inside()

        self._corrected_target = self._target
        self.phase_retriever.set_target(self._target, self.signal_region)

    def _warn_if_zeroth_order_inside(self) -> None:
        """Warn when the undiffracted spot sits inside the signal region. Only warn if
        the signal region is not the entire sensor.
        """
        region = self.signal_region
        if bool(region.all()):
            return

        column, row = self._mapping.zeroth_order_position
        height, width = (int(size) for size in region.shape[-2:])
        pixel_row, pixel_column = int(round(row)), int(round(column))

        # Off the sensor
        if not (0 <= pixel_row < height and 0 <= pixel_column < width):
            return
        if not bool(region[pixel_row, pixel_column]):
            return

        warnings.warn(
            f"The zeroth order at (column {column:.0f}, row {row:.0f}) lies inside "
            "the signal region.",
            stacklevel=3,
        )

    def _paste(
        self,
        patch: torch.Tensor,
        center: tuple[float, float],
        subsample: bool = True,
    ) -> torch.Tensor:
        """A frame of the whole sensor with ``patch`` centred on the pixel ``center``,
        ``(column, row)``.

        The centre is generally not a whole pixel. The whole-sample part of the centre
        places the patch, and the remainder translates it by
        :func:`~hologradpy.fourier_transforms.fft_translate`. This translation is a
        phase ramp in the Fourier domain and needs no resampling kernel.
        """
        center_column, center_row = center
        height, width = (int(size) for size in self.camera.sensor_resolution)
        placed = ROI.centered(
            (round(center_column), round(center_row)), tuple(patch.shape[-2:])
        )
        try:
            on_sensor = placed.trimmed_to((height, width))
        except ValueError as error:
            raise ValueError(
                f"The target lands entirely off the sensor: its center is at "
                f"(column {center_column:.0f}, row {center_row:.0f}) on a "
                f"{height} by {width} frame. Check target_position, which is (x, y) "
                "metres from the zeroth order."
            ) from error

        frame = torch.zeros(
            (height, width), dtype=patch.dtype, device=patch.device
        )
        frame[on_sensor.rows, on_sensor.columns] = on_sensor.relative_to(placed).crop(
            patch
        )

        residual = (
            center_column - round(center_column),
            center_row - round(center_row),
        )
        if subsample and any(residual):
            frame = translate_intensity(frame, residual)
        return frame

    @property
    def _mapping(self) -> CameraMapping:
        """The mapping every sensor-plane coordinate here is measured against.

        Raises:
            ValueError: None has been passed or measured yet.
        """
        if self.camera_mapping is None:
            raise ValueError(
                "Camera feedback needs a camera mapping: where the target sits on the "
                "sensor is measured from the zeroth order, which only the mapping "
                "knows. Pass camera_mapping, or call apply_camera_mapping() to measure "
                "one."
            )
        return self.camera_mapping

    @property
    def slm_camera_model(self) -> SLMFourierLensModel:
        return self.phase_retriever.slm_camera_model

    @property
    def target(self) -> torch.Tensor:
        return self._target

    @property
    def corrected_target(self) -> torch.Tensor:
        """The corrected target last given to the retriever."""
        return self._corrected_target

    def update_target(self, target: torch.Tensor) -> None:
        """Give the retriever ``target``, on the bounding box of the signal region while
        the model computes only that box, and on the whole sensor otherwise.
        """
        self._corrected_target = target
        roi = self.slm_camera_model.output_roi
        region = (
            self.signal_region
            if roi is None
            else torch.ones_like(roi.crop(self.signal_region))
        )
        self.phase_retriever.set_target(target, region)

    @contextmanager
    def _restrict_to_signal_roi(self) -> Generator[ROI, None, None]:
        """Run the model and the retriever on the bounding box of the signal region for
        the duration of a ``with`` block.

        The model computes only the box
        (:meth:`~hologradpy.optics.systems.abstract.OpticalSystem.restrict_output_to`).
        The retriever is given the target on the box, the whole box as its signal
        region, and the absolute intensity cost (:meth:`_absolute_intensity_cost`). The
        whole plane, the retriever's cost and the last target on the whole sensor are
        put back on exit, also after an exception.

        Yields:
            ROI: The bounding box, in sensor pixels.
        """
        roi = self._signal_roi
        retriever = self.phase_retriever
        held_loss_factory = retriever.loss_factory
        sensor_shape = tuple(int(size) for size in self.signal_region.shape[-2:])
        with self.slm_camera_model.restrict_output_to(roi):
            try:
                retriever.set_target(
                    roi.crop(self._target),
                    torch.ones_like(roi.crop(self.signal_region)),
                )
                retriever.set_loss_factory(self._absolute_intensity_cost)
                yield roi
            finally:
                retriever.set_loss_factory(held_loss_factory)
                corrected = self._corrected_target
                if tuple(corrected.shape[-2:]) != sensor_shape:
                    corrected = roi.pad(corrected, sensor_shape)
                self._corrected_target = corrected
                retriever.set_target(corrected, self.signal_region)

    def _absolute_intensity_cost(
        self, target: torch.Tensor, signal_region: torch.Tensor
    ) -> LossAbsoluteIntensityMSE:
        """The absolute intensity cost of ``target`` over ``signal_region``, with the
        target scaled to hold ``target_efficiency`` of the power entering the Fourier
        lens.
        """
        model = self.slm_camera_model
        weighted = target * signal_region
        power_per_unit = (
            self.target_efficiency
            * model.incident_power()
            / (model.output_pixel_area() * float(weighted.sum()))
        )
        scaled = target * power_per_unit
        total = float((scaled * signal_region).sum())
        return LossAbsoluteIntensityMSE(
            scaled, signal_region, scale=self.phase_retriever.loss_scale / total**2
        )

    def apply_camera_mapping(self, verbose: bool = True) -> CameraMapping:
        """Calibrate the focal-plane partial affine of the model from a camera mapping.

        The measured frame does not line up with the predicted one when the camera is
        rotated or displaced relative to the model. The loop then attempts to correct a
        misaligned image. The given ``camera_mapping`` is used. When there is none, a
        mapping is measured with a
        :class:`~hologradpy.calibration.camera_mapping.SpotArrayMapper`.

        The mapping is kept for :meth:`target_center_pixels`. It describes the focal
        plane as the camera shows it. The mappers measure against the
        model without its partial affine, so the mapping is the same whatever values
        the affine held before. The calibration follows from the mapping alone, so a
        repeated call sets the same values.

        Returns:
            CameraMapping: The mapping applied.
        """
        if self.camera_mapping is None:
            if verbose:
                print("No camera mapping supplied. Measuring one with a spot array.")
            virtual_slm = self.slm_camera_model.virtual_slm
            held_phase = virtual_slm.get_phase().detach().clone()
            try:
                self.camera_mapping = SpotArrayMapper(
                    self.slm, self.camera, self.slm_camera_model
                ).map_camera()
            finally:
                virtual_slm.set_phase(held_phase)
                self.slm.set_phase(gpu_to_numpy(held_phase))

        self.camera_mapping.check_camera(self.camera)
        self.slm_camera_model.calibrate_from_mapping(self.camera_mapping)
        return self.camera_mapping

    def placement_data(self) -> TargetPlacementData:
        """Where the target will sit on the sensor, as data for the visualizer.

        Returns:
            TargetPlacementData: The placed target and the geometry around it.
        """
        self.apply_camera_mapping(verbose=False)
        self.place_target()

        center_column, center_row = self.target_center_pixels()
        patch_height, patch_width = (
            int(size) for size in self.target_patch.shape[-2:]
        )

        half_x, half_y = self.slm_camera_model.addressable_half_extent()
        corners_optical = self._mapping.sensor_to_image_plane(
            np.array(
                [
                    [center_column - patch_width / 2, center_row - patch_height / 2],
                    [center_column + patch_width / 2, center_row - patch_height / 2],
                    [center_column - patch_width / 2, center_row + patch_height / 2],
                    [center_column + patch_width / 2, center_row + patch_height / 2],
                ]
            )
        )
        overshoot = max(
            float(np.abs(corners_optical[:, 0]).max() - half_x),
            float(np.abs(corners_optical[:, 1]).max() - half_y),
        )

        return TargetPlacementData(
            target=gpu_to_numpy(self._target),
            signal_region=gpu_to_numpy(self.signal_region),
            zeroth_order=self._mapping.zeroth_order_position,
            target_center=(center_column, center_row),
            patch_shape=(patch_height, patch_width),
            addressable_corners=self._addressable_corners_pixels(),
            target_position=self.target_position,
            overshoot=overshoot,
        )

    @property
    def _signal_roi(self) -> ROI:
        """Bounding box of the signal region."""
        return ROI.detect(
            gpu_to_numpy(self.signal_region).astype(bool), threshold=0.0, pad=0
        )

    def _autoexpose(self, sensor_region: NDArray[np.bool_]) -> None:
        """Autoexpose on the signal region."""
        signal_roi = self._signal_roi
        self.camera.autoexpose(
            roi=signal_roi, mask=signal_roi.crop(sensor_region), raise_on_rail=False
        )

    def _prepare_run(self, verbose: bool) -> SignalWindow:
        """Check the grids, calibrate the model from the camera mapping, place the
        target, and return the window of the run.
        """
        self._check_grids_match()
        self.apply_camera_mapping(verbose=verbose)
        self.place_target()

        device = self.target.device
        dtype = self.target.dtype
        sensor_region = self.signal_region.to(device=device).bool()
        sensor_target = normalize(self.target.to(dtype), sensor_region.to(dtype))
        signal_roi = self._signal_roi
        region = signal_roi.crop(sensor_region)
        band_mask = intensity_band_mask(
            self.slm_camera_model, self._mapping, tuple(region.shape)
        )
        return SignalWindow(
            roi=signal_roi,
            sensor_region=gpu_to_numpy(sensor_region).astype(bool),
            sensor_target=sensor_target,
            region=region,
            target=signal_roi.crop(sensor_target),
            band_mask=band_mask.to(device),
        )

    @contextmanager
    def _retrieval_loop(
        self, iterations: int, retriever_steps: Sequence[int], verbose: bool
    ) -> Generator[tuple[ProgressBar, ProgressBar], None, None]:
        """The context of a loop of retrievals and measurements, which yields a
        progress bar over the iterations and one over each retrieval.

        The whole sensor is read out, since the signal region and its bounding box are
        in sensor pixels. The camera's exposure, gain and region of interest are put
        back when the loop ends. The model and the retriever run on the bounding box of
        the signal region (:meth:`_restrict_to_signal_roi`).
        """
        retriever_bar = ProgressBar(
            total=max(retriever_steps),
            description="Phase retrieval",
            verbose=verbose,
            position=1,
            leave=False,
        )
        with (
            self.camera.preserve_exposure_gain_and_roi(full_sensor=True),
            ProgressBar(
                total=iterations,
                description="Camera feedback",
                verbose=verbose,
                position=0,
            ) as bar,
            retriever_bar,
            self._restrict_to_signal_roi(),
        ):
            yield bar, retriever_bar

    def _retrieve_and_display(
        self,
        steps: int,
        iteration: int,
        metrics: Sequence[IntensityMetric],
        progress_bar: ProgressBar,
        **options: Any,
    ) -> PhaseRetrievalData:
        """Retrieve the next hologram, warm started from the last one, and display it.

        Args:
            steps: Optimizer iterations of the retrieval.
            iteration: The feedback iteration, from zero, which names the retrieval.
            metrics: How the retrieval compares its prediction with its target.
            progress_bar: The bar the retrieval reports to.
            **options: Further keyword arguments of the retriever's ``retrieve``.

        Returns:
            PhaseRetrievalData: The retrieval.
        """
        retrieval = self.phase_retriever.retrieve(
            steps,
            name=f"feedback iteration {iteration + 1}",
            metrics=metrics,
            progress_bar=progress_bar,
            **options,
        )
        self.slm.set_phase(retrieval.phase)
        return retrieval

    def _expose(
        self,
        autoexpose: bool,
        exposure: float | None,
        sensor_region: NDArray[np.bool_],
    ) -> tuple[float, float]:
        """Autoexpose on the signal region, or set ``exposure`` while ``autoexpose`` is
        off. Return the camera's exposure in seconds and its gain in dB.
        """
        if autoexpose:
            self._autoexpose(sensor_region)
        elif exposure is not None:
            self.camera.set_exposure(exposure)
        return float(self.camera.get_exposure()), float(self.camera.get_gain())

    def _normalized_prediction(self, region: Tensor) -> Tensor:
        """The model's intensity of the hologram it holds, divided by the mean of its
        bright pixels in the region, on the target's device and in its dtype.
        """
        with torch.no_grad():
            intensity = as_image(self.slm_camera_model().intensity)
        intensity = intensity.to(dtype=self.target.dtype, device=self.target.device)
        return normalize(intensity, region.to(intensity.dtype))

    @staticmethod
    def _place_target_on(
        frame: Tensor, target: Tensor, region: Tensor
    ) -> tuple[Tensor, tuple[float, float]]:
        """Move ``target`` onto the pattern of ``frame`` by its fitted translation
        (:func:`~hologradpy.analysis.fitting.fit_translation`), so a comparison of the
        two holds the shape alone. Return the moved target and the translation,
        ``(columns, rows)`` in sensor pixels.
        """
        offset = fit_translation(frame, target, region).shift
        return translate_real(target, offset), offset

    @staticmethod
    def _warn_overexposed(description: str) -> None:
        """Warn that the measurements of ``description`` are overexposed inside the
        signal region.
        """
        warnings.warn(
            f"The measurements of {description} are overexposed inside the signal "
            "region. An overexposed pixel reads below the light it receives, which the "
            "correction takes for too little light. Autoexpose, or lower the exposure.",
            stacklevel=3,
        )

    def _record(
        self,
        name: str,
        target: NDArray,
        signal_region: NDArray,
        metrics: Sequence[IntensityMetric],
        history: dict[str, list[float]],
        **fields: Any,
    ) -> CameraFeedbackData:
        """The record of a run, named ``name`` or after the corrector, with the history
        of its metrics and the camera mapping it ran with.
        """
        return CameraFeedbackData(
            timestamp=datetime.now(),
            name=name or type(self).__name__,
            target=target,
            signal_region=signal_region,
            camera_mapping=self._mapping.lean(),
            metrics=history,
            lower_is_better={metric.name: metric.lower_is_better for metric in metrics},
            **fields,
        )

    def _aligned_frame(
        self,
        averages: int,
        pattern: Tensor,
        region: Tensor,
        sensor_region: NDArray[np.bool_],
        destination: tuple[float, float] | None = None,
        background: NDArray | float | None = None,
    ) -> tuple[Tensor, list[TranslationFit], bool, NDArray]:
        """Capture ``averages`` frames at the camera's exposure, move them to one
        translation from ``pattern``, and return their mean.

        Each frame is cropped to the bounding box of the signal region, and its
        translation is fitted with ``pattern``
        (:func:`~hologradpy.analysis.fitting.fit_translation`). Every frame is then
        moved by ``destination`` less its own translation, so its pattern sits at
        ``destination`` from ``pattern``. Without a destination, the mean stays where
        the pattern landed. A destination of ``(0.0, 0.0)`` moves every frame onto
        ``pattern``. The mean is divided by the mean of its bright pixels in the
        region, so its level does not depend on the exposure.

        Args:
            averages: The number of frames.
            pattern: The intensity every frame is fitted with, the target or the
                model's prediction, on the bounding box of the signal region.
            region: True inside the signal region, on the bounding box, where the fits
                are evaluated.
            sensor_region: The signal region on the whole sensor, for the overexposure
                check.
            destination: The translation from ``pattern`` every frame is moved to, in
                sensor pixels along ``(columns, rows)``, or None for the mean of the
                frames' translations.
            background: The counts subtracted from every frame before its fit, one
                level or a frame of the whole sensor, or None to subtract nothing.

        Returns:
            tuple[Tensor, list[TranslationFit], bool, NDArray]: The normalized mean
            frame on the bounding box, the fit of every frame, whether any frame was
            overexposed inside the region, and the mean of the frames as captured, on
            the whole sensor.
        """
        signal_roi = self._signal_roi
        captured_total: NDArray | None = None
        raws: list[Tensor] = []
        fits: list[TranslationFit] = []
        overexposed = False
        frame_count = max(1, int(averages))
        for _ in range(frame_count):
            captured = np.asarray(
                self.camera.get_image(mask=sensor_region), dtype=np.float64
            )
            overexposed = overexposed or bool(self.camera.overexposed)
            captured_total = (
                captured.copy() if captured_total is None else captured_total + captured
            )
            signal = captured
            if background is not None:
                signal = captured - np.broadcast_to(background, captured.shape)
            raw = torch.as_tensor(
                signal_roi.crop(signal), dtype=pattern.dtype, device=pattern.device
            )
            fits.append(fit_translation(raw, pattern, region))
            raws.append(raw)

        if destination is None:
            destination = tuple(np.mean([fit.shift for fit in fits], axis=0).tolist())
        moved = [
            translate_real(
                raw, (destination[0] - fit.shift[0], destination[1] - fit.shift[1])
            )
            for raw, fit in zip(raws, fits)
        ]
        mean = torch.stack(moved).mean(dim=0)
        return (
            normalize(mean, region.to(mean.dtype)),
            fits,
            overexposed,
            captured_total / frame_count,
        )

    def _check_grids_match(self) -> None:
        """The measured frame and the model's prediction have to be the same picture.

        The feedback measures the whole sensor, so the model's output is compared with
        the camera's :attr:`~hologradpy.hardware.camera.Camera.sensor_resolution`.
        """
        model = self.slm_camera_model
        model_resolution = tuple(model[-1].resolution_out)
        sensor_resolution = tuple(int(size) for size in self.camera.sensor_resolution)
        if model_resolution != sensor_resolution:
            raise ValueError(
                f"The model's output is {model_resolution} but the camera's sensor is "
                f"{sensor_resolution}. Camera feedback compares the two directly, so "
                "build the model with camera_resolution taken from "
                "camera.sensor_resolution and camera_pixel_size from "
                "camera.pixel_size."
            )

    @staticmethod
    def _retriever_steps(retriever_iterations: Sequence[int]) -> list[int]:
        """The per-iteration optimizer budgets, validated."""
        if isinstance(retriever_iterations, (int, np.integer)):
            raise TypeError(
                "retriever_iterations is one entry per feedback iteration, not a "
                f"single count. Pass [{retriever_iterations}] * n to run n iterations "
                f"of {retriever_iterations} optimizer steps each."
            )

        steps = [int(value) for value in retriever_iterations]
        if not steps:
            raise ValueError(
                "retriever_iterations is empty, so there are no feedback iterations "
                "to run."
            )
        return steps

    @staticmethod
    def _per_iteration(
        value: float | Sequence[float], iterations: int, name: str
    ) -> list:
        """One value per iteration, from either a scalar or a sequence."""
        if np.isscalar(value):
            return [value] * iterations

        values = list(value)
        if len(values) != iterations:
            raise ValueError(
                f"{name} has {len(values)} entries but there are {iterations} feedback "
                "iterations. Pass one value per iteration, or a single value for all."
            )
        return values

    @abstractmethod
    def run(self) -> CameraFeedbackData:
        pass


def corrected_target_intensity(
    prediction: Tensor,
    frame: Tensor,
    target: Tensor,
    region: Tensor,
    gain: float,
    band_mask: Tensor | None = None,
    blur: float = 0.0,
) -> Tensor:
    """The intensity the next retrieval is given.

    ``prediction`` plus ``gain`` times the measured error, ``target - frame``, in the
    signal region. With ``band_mask``, the error is first Fourier-filtered to the band
    of the optics. The result is clipped at zero, set to zero outside the signal region,
    and blurred by ``blur``. With ``prediction`` equal to the last retrieval's target,
    this is van Bijnen's Eq. (4.44) with ``gain`` as ``beta``.

    Args:
        prediction: The model's intensity of the displayed hologram, in the target's
            units, or the last retrieval's target.
        frame: The measured intensity, in the target's units.
        target: The target intensity.
        region: True inside the signal region.
        gain: The fraction of the error corrected.
        band_mask: The mask of
            :func:`~hologradpy.analysis.fourier_filter.intensity_band_mask`, or None to
            keep every frequency of the error.
        blur: The radius in pixels of the Gaussian blur of the result
            (:func:`~hologradpy.profiles.amplitude.gaussian_blur`), or zero for none.

    Returns:
        Tensor: The corrected target intensity.
    """
    error = (target - frame) * region
    if band_mask is not None:
        error = fourier_filter(error, band_mask)
    corrected = torch.clamp(prediction + gain * error, min=0.0) * region
    if blur > 0:
        corrected = gaussian_blur(corrected.float(), blur).to(corrected.dtype)
    return corrected
