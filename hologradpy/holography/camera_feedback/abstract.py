from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import torch
from numpy.typing import NDArray

from .visualizer import CameraFeedbackVisualizer, TargetPlacementData

from ..phase_retrieval import (
    GradientPhaseRetriever,
    PhaseRetrievalData,
    PhaseRetrieverBase,
)

from ...calibration.camera_mapping import CameraMapping, SpotArrayMapper
from ...fourier_transforms import translate_intensity
from ...hardware import Camera, SLM
from ...optics.systems import SLMFourierLensModel
from ...roi import ROI
from ...serialization import SaveableRecord, record_type
from ...utils import gpu_to_numpy


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


class FeedbackCorrectorBase(ABC):
    """Correct a light potential against the camera, driving a phase retriever.

    Works with any :class:`~hologradpy.holography.phase_retrieval.PhaseRetrieverBase`
    that implements set_target() and run(). Uses 
    :class:`~hologradpy.holography.phase_retrieval.GradientPhaseRetriever` by default.
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
        phase_retriever: PhaseRetrieverBase | None = None,
        camera_mapping: CameraMapping | None = None,
        loss_scale: float | None = None,
    ) -> None:
        """
        Args:
            slm: The SLM displaying the hologram.
            camera: The camera watching the light potential.
            target: The target intensity to produce, on the camera grid.
            signal_region: Where the target is optimized.
            target_position: Where to place the target, as ``(x, y)`` metres in
                the Nyquist plane measured from the zeroth order. The camera's
                rotation and scale are applied, so this is a position in the
                image plane and not in sensor pixels.
            slm_camera_model: The model simulating the propagation of light from the SLM
                to the camera. Required unless ``phase_retriever`` is given, in which
                case it is taken from there.
            init_slm_phase: Starting phase for the retriever.
            phase_retriever: Use an existing phase retriever instead of building one.
            camera_mapping: How the camera sits relative to the model, used to seed the
                model's registration before the loop starts. Measured with a
                :class:`~hologradpy.calibration.camera_mapping.SpotArrayMapper` when not
                given.
            loss_scale: Slope of the cost function used by the retriever this builds.
        """
        self.slm = slm
        self.camera = camera
        self.camera_mapping = camera_mapping
        self.target_position = target_position
        self._registered = False

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
    ) -> GradientPhaseRetriever:
        """The default search, so the usual case needs no retriever built by hand."""
        if slm_camera_model is None:
            raise ValueError(
                "Camera feedback needs either a slm_camera_model to build its own "
                "phase retriever from, or a phase_retriever to drive."
            )

        options = {} if loss_scale is None else {"loss_scale": loss_scale}
        return GradientPhaseRetriever(
            slm_camera_model=slm_camera_model,
            init_slm_phase=init_slm_phase,
            **options,
        )

    def zeroth_order_pixels(self) -> tuple[float, float]:
        """Where the undiffracted spot sits on the sensor, as ``(row, column)``."""
        row, column = self._mapping.zeroth_order_position
        return float(row), float(column)

    def target_center_pixels(self) -> tuple[float, float]:
        """Where :attr:`target_position` lands on the sensor, as ``(row, column)``.

        ``target_position`` is ``(x, y)`` metres in the Nyquist plane, measured from the
        zeroth order. The sign convention follows the rest of the package: x runs along
        the sensor columns and y along the rows, both increasing with pixel index, so
        positive y is downward on a displayed image.

        The camera's rotation and scale are applied, so a request of ``(+500e-6, 0)``
        means half a millimetre along the *optical* x axis, wherever that falls on a
        rotated sensor, rather than half a millimetre along the sensor's own columns.
        """
        row, column = self._camera_pixels_from_optical_metres(
            np.asarray([self.target_position])
        )[0]
        return float(row), float(column)

    # TODO: Check how the camera returns the pixel pitch
    def _camera_pitch(self) -> tuple[float, float]:
        """Camera pixel pitch as ``(y, x)`` floats."""
        pitch = np.asarray(self.camera.pixel_size, dtype=float).reshape(-1)
        return float(pitch[0]), float(pitch[1])

    def _camera_pixels_from_optical_metres(self, points: np.ndarray) -> np.ndarray:
        """Optical-plane ``(x, y)`` metres from the zeroth order to sensor
        ``(row, column)``.
        """
        pitch_y, pitch_x = self._camera_pitch()
        zeroth_row, zeroth_column = self.zeroth_order_pixels()

        linear = self._mapping.partial_affine.inverse().linear
        points = np.atleast_2d(np.asarray(points, dtype=float))
        scaled = np.stack([points[:, 0] / pitch_x, points[:, 1] / pitch_y], axis=1)
        columns, rows = (linear @ scaled.T)

        return np.stack([rows + zeroth_row, columns + zeroth_column], axis=1)

    def _optical_metres_from_pixels(self, points: np.ndarray) -> np.ndarray:
        """Sensor ``(row, column)`` to optical-plane ``(x, y)`` metres from the zeroth
        order, the inverse of :meth:`_camera_pixels_from_optical_metres`.
        """
        pitch_y, pitch_x = self._camera_pitch()
        zeroth_row, zeroth_column = self.zeroth_order_pixels()

        linear = self._mapping.partial_affine.linear
        points = np.atleast_2d(np.asarray(points, dtype=float))
        offsets = np.stack(
            [points[:, 1] - zeroth_column, points[:, 0] - zeroth_row], axis=1
        )
        x_pixels, y_pixels = (linear @ offsets.T)
        return np.stack([x_pixels * pitch_x, y_pixels * pitch_y], axis=1)

    def _addressable_corners_pixels(self) -> np.ndarray:
        """The four corners of the addressable region, as sensor ``(row, column)``."""
        half_x, half_y = self.slm_camera_model.addressable_half_extent()
        corners = np.array(
            [
                [-half_x, -half_y],
                [half_x, -half_y],
                [half_x, half_y],
                [-half_x, half_y],
            ]
        )
        return self._camera_pixels_from_optical_metres(corners)

    def place_target(self) -> None:
        """Build the full-frame target from the patch and hand it to the retriever.

        Called by :meth:`run` after the mapping is known. The patch is *not* rotated to
        match the camera. On a rotated camera, the potential is tilted by the same angle
        in the optical plane, matching the camera's rotation.
        """
        device = self.slm_camera_model.device

        center_row, center_column = self.target_center_pixels()
        self._target = self._paste(
            self.target_patch.to(device), center_row, center_column
        )
        # A region is whole samples either way, so it is not worth a transform and
        # the ringing that comes with one.
        self.signal_region = self._paste(
            self.signal_region_patch.to(device),
            center_row,
            center_column,
            subsample=False,
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

        row, column = self.zeroth_order_pixels()
        height, width = (int(size) for size in region.shape[-2:])
        pixel_row, pixel_column = int(round(row)), int(round(column))

        # Off the sensor
        if not (0 <= pixel_row < height and 0 <= pixel_column < width):
            return
        if not bool(region[pixel_row, pixel_column]):
            return

        warnings.warn(
            f"The zeroth order at (row {row:.0f}, column {column:.0f}) lies inside "
            "the signal region.",
            stacklevel=3,
        )

    def _paste(
        self,
        patch: torch.Tensor,
        center_row: float,
        center_column: float,
        subsample: bool = True,
    ) -> torch.Tensor:
        """A sensor-sized frame with ``patch`` centered on the given pixel.

        The center is generally not a whole pixel, so the the whole-sample part places 
        the patch, and the remainder translates it by
        :func:`~hologradpy.fourier_transforms.fft_translate`, a phase ramp in the
        Fourier domain that needs no resampling kernel.
        """
        height, width = (int(size) for size in self.camera.resolution)
        patch_height, patch_width = (int(size) for size in patch.shape[-2:])

        top = int(round(center_row)) - patch_height // 2
        left = int(round(center_column)) - patch_width // 2

        frame = torch.zeros(
            (height, width), dtype=patch.dtype, device=patch.device
        )
        frame_top, frame_left = max(top, 0), max(left, 0)
        frame_bottom = min(top + patch_height, height)
        frame_right = min(left + patch_width, width)

        if frame_bottom <= frame_top or frame_right <= frame_left:
            raise ValueError(
                f"The target lands entirely off the sensor: its center is at "
                f"(row {center_row:.0f}, column {center_column:.0f}) on a "
                f"{height} by {width} frame. Check target_position, which is (x, y) "
                "metres from the zeroth order."
            )

        frame[frame_top:frame_bottom, frame_left:frame_right] = patch[
            frame_top - top : frame_bottom - top,
            frame_left - left : frame_right - left,
        ]

        residual = (
            center_row - round(center_row),
            center_column - round(center_column),
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
                "knows. Pass camera_mapping, or call register() to measure one."
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
        self._corrected_target = target
        self.phase_retriever.set_target(target, self.signal_region)

    def register(self, verbose: bool = True) -> CameraMapping:
        """Seed the model's registration from a camera mapping.

        A camera that is rotated or displaced relative to the model produces a
        measured frame that does not line up with the predicted one, and the loop then
        attempts to correct a misregistered image. Uses the mapping it was given, or 
        measures one with a
        :class:`~hologradpy.calibration.camera_mapping.SpotArrayMapper`.

        The mapping is kept as the description of how the camera sits relative to the
        optical plane, which :meth:`target_center_pixels` needs. It must be measured
        against the *unregistered* model since once registered, the model's output plane
        should be aligned with the sensor.

        Returns:
            CameraMapping: The mapping applied.
        """
        if self._registered:
            return self.camera_mapping

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

        self.slm_camera_model.calibrate_from_mapping(self.camera_mapping)
        self._registered = True
        return self.camera_mapping

    def placement_data(self) -> TargetPlacementData:
        """Where the target will sit on the sensor, as data for the visualizer.

        Returns:
            TargetPlacementData: The placed target and the geometry around it.
        """
        self.register(verbose=False)
        self.place_target()

        center_row, center_column = self.target_center_pixels()
        patch_height, patch_width = (
            int(size) for size in self.target_patch.shape[-2:]
        )

        half_x, half_y = self.slm_camera_model.addressable_half_extent()
        corners_optical = self._optical_metres_from_pixels(
            np.array(
                [
                    [center_row - patch_height / 2, center_column - patch_width / 2],
                    [center_row - patch_height / 2, center_column + patch_width / 2],
                    [center_row + patch_height / 2, center_column - patch_width / 2],
                    [center_row + patch_height / 2, center_column + patch_width / 2],
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
            zeroth_order=self.zeroth_order_pixels(),
            target_center=(center_row, center_column),
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

    def _check_grids_match(self) -> None:
        """The measured frame and the model's prediction have to be the same picture."""
        model = self.slm_camera_model
        model_resolution = tuple(model[-1].resolution_out)
        camera_resolution = tuple(self.camera.resolution)
        if model_resolution != camera_resolution:
            raise ValueError(
                f"The model's output is {model_resolution} but the camera is "
                f"{camera_resolution}. Camera feedback compares the two directly, so "
                "build the model with camera_resolution and camera_pixel_size taken "
                "from the camera."
            )

    @abstractmethod
    def run(self) -> CameraFeedbackData:
        pass
