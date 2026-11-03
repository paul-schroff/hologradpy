"""Camera feedback on the intensities and the positions of the spots of an array."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter, map_coordinates

from .abstract import FeedbackCorrectorBase

from ..phase_retrieval import (
    GradientPhaseRetriever,
    PhaseRetrievalData,
    PhaseRetrieverBase,
    WeightedGerchbergSaxtonPhaseRetriever,
)

from ...calibration.camera_mapping import CameraMapping
from ...calibration.spot_detection import smallest_spacing, subpixel_peak
from ...hardware import Camera, SLM, as_camera, as_slm
from ...optics.systems import SLMFourierLensModel
from ...profiles.masks import disc_mask
from ...roi import ROI
from ...serialization import SaveableRecord, record_type
from ...utils import as_image, gpu_to_numpy

# Upper bound of the spot window's width in waists.
WINDOW_WAISTS = 3.0

# A spot below this fraction of the mean target weight is dark.
_DARK_SPOT_FRACTION = 1e-2


@dataclass(frozen=True)
class SpotIntensities:
    """The intensities of the spots of an array, measured on a camera
    (:func:`measure_spot_intensities`).
    """

    #: The ``(N,)`` peak of each spot, smoothed over a pixel, less the background, in
    #: counts per second.
    intensities: NDArray
    #: The ``(height, width)`` image of the whole sensor in counts, averaged over the
    #: ``averages`` frames.
    frame: NDArray
    #: The ``(N, 2)`` ``(column, row)`` sensor pixels of the spots' maxima.
    sensor_positions: NDArray
    #: The background of a pixel in counts, the median of the frame outside the
    #: windows.
    background: float
    #: The equivalent exposure in seconds
    #: (:meth:`~hologradpy.hardware.camera.abstract.Camera.get_equivalent_exposure`).
    exposure: float
    #: Whether a pixel in a window was overexposed.
    overexposed: bool


def measure_spot_intensities(
    camera: Camera,
    camera_mapping: CameraMapping,
    positions: ArrayLike,
    *,
    averages: int = 1,
    exposure: float | None = None,
) -> SpotIntensities:
    """Measure the intensity of each spot of an array on a camera.

    The camera mapping places each spot at a site on the sensor
    (:meth:`~.CameraMapping.image_plane_to_sensor`). The frame, less the background, is
    smoothed over a pixel, with a Gaussian of sigma 1 px. A spot's intensity is the
    smoothed frame's maximum in its window, refined between the pixels, per second of
    equivalent exposure. Averaging frames lowers its noise.

    Each site has a window of ``WINDOW_WAISTS`` mapped waists
    (:attr:`~.CameraMapping.spot_fit`), or of half the smallest spacing if that is
    smaller. The camera exposes on the windows, and the background is the median
    outside them.

    The whole sensor is read out. The exposure, the gain and the region of interest are
    put back afterwards
    (:meth:`~hologradpy.hardware.camera.abstract.Camera.preserve_exposure_gain_and_roi`).

    Args:
        camera: The camera, or a driver that
            :func:`~hologradpy.hardware.as_native.as_camera` wraps.
        camera_mapping: The mapping of the camera, measured with its current mounting.
        positions: ``(N, 2)`` focal-plane positions of the spots, ``(x, y)`` in metres
            from the zeroth order.
        averages: The number of frames to average.
        exposure: The exposure in seconds. If None, the camera autoexposes on the
            windows.

    Returns:
        SpotIntensities: The intensities, the frame and the maxima.

    Raises:
        ValueError: A window runs off the sensor. A spot finds no maximum of its own
            inside its window, or holds no light above the background.
    """
    camera = as_camera(camera)
    camera_mapping.check_camera(camera)
    positions = _checked_positions(positions)
    sites = np.asarray(
        camera_mapping.image_plane_to_sensor(positions), dtype=np.float64
    )
    # The mapping fits the waist on the camera, in metres of the sensor.
    waist = abs(float(camera_mapping.spot_fit.waist)) / float(
        np.min(camera.pixel_size)
    )
    reach = min(WINDOW_WAISTS * waist, smallest_spacing(sites) / 2)
    shape = tuple(int(size) for size in camera.sensor_resolution)
    windows = _windows(shape, sites, reach)
    inside = np.zeros(shape, dtype=bool)
    for region, disc in windows:
        inside[region.rows, region.columns] |= disc

    with camera.preserve_exposure_gain_and_roi(full_sensor=True):
        if exposure is None:
            camera.autoexpose(mask=inside, raise_on_rail=False)
        else:
            camera.set_exposure(exposure)
        frame = camera.get_image(averaging=averages, mask=inside)
        frame = np.asarray(frame, dtype=np.float64) / max(1, int(averages))
        overexposed = bool(camera.overexposed)
        equivalent_exposure = float(camera.get_equivalent_exposure())

    background = float(np.median(frame[~inside]))
    signal = frame - background
    smooth = gaussian_filter(signal, 1.0)
    peaks = [subpixel_peak(region.crop(smooth), disc) for region, disc in windows]
    lost = [index for index, peak in enumerate(peaks) if peak is None]
    if lost:
        raise ValueError(
            f"Spots {lost} find no maximum of their own inside their windows. The "
            "spots may be dark, or the camera mapping out of date."
        )
    peaks = np.array(peaks)
    intensities = peaks[:, 0] / equivalent_exposure
    dark = intensities <= 0
    if dark.any():
        raise ValueError(
            f"Spots {np.flatnonzero(dark).tolist()} hold no light above the "
            "background. The camera mapping may be out of date."
        )
    if overexposed:
        warnings.warn(
            "A pixel in a spot's window was overexposed, so the brightest spots read "
            "low.",
            stacklevel=2,
        )
    # The peaks are in the pixels of their regions.
    corners = [(region.left_column, region.top_row) for region, _ in windows]
    return SpotIntensities(
        intensities=intensities,
        frame=frame,
        sensor_positions=peaks[:, 1:] + corners,
        background=background,
        exposure=equivalent_exposure,
        overexposed=overexposed,
    )


@record_type("spot_array_correction")
@dataclass
class SpotArrayCorrectionData(SaveableRecord):
    """One completed run of a :class:`SpotArrayCorrector`.

    Each list holds one entry per feedback iteration. Entry ``i`` belongs to the
    hologram of iteration ``i``.
    """

    timestamp: datetime
    name: str
    #: The class name of the phase retriever.
    retriever: str
    #: The ``(N, 2)`` focal-plane positions of the spots, ``(x, y)`` in metres from the
    #: zeroth order.
    positions: NDArray
    #: The ``(N,)`` requested relative intensities, with a mean of 1.
    requested_weights: NDArray
    #: The weights of the spots in the retriever's target, with a mean of 1, when it
    #: computed each hologram.
    adjusted_weights: list[NDArray] = field(default_factory=list)
    #: The ``(N, 2)`` focal-plane positions the retriever aimed the spots at, in metres.
    adjusted_positions: list[NDArray] = field(default_factory=list)
    #: The intensities measured on the camera (:attr:`SpotIntensities.intensities`).
    measured_intensities: list[NDArray] = field(default_factory=list)
    #: The ``(N, 2)`` focal-plane positions of the spots' maxima on the camera, in
    #: metres.
    measured_positions: list[NDArray] = field(default_factory=list)
    #: The model's intensity at each spot, relative to their mean.
    predicted_intensities: list[NDArray] = field(default_factory=list)
    retrievals: list[PhaseRetrievalData] = field(default_factory=list)
    #: The equivalent exposures in seconds.
    exposures: list[float] = field(default_factory=list)
    #: Whether a pixel in a window was overexposed.
    overexposed: list[bool] = field(default_factory=list)
    #: The ``(height, width)`` image of the whole sensor from the last measurement, in
    #: counts (:attr:`SpotIntensities.frame`).
    final_camera_image: NDArray | None = None
    camera_mapping: CameraMapping | None = None

    @property
    def number_of_iterations(self) -> int:
        return len(self.measured_intensities)

    @property
    def non_uniformity(self) -> list[float]:
        """The rms deviation of measured over requested intensities from 1, per
        iteration, with both relative to their mean.
        """
        requested = np.asarray(self.requested_weights, dtype=np.float64)
        requested = requested / requested.mean()
        deviations = []
        for measured in self.measured_intensities:
            measured = np.asarray(measured, dtype=np.float64)
            ratio = measured / measured.mean() / requested
            deviations.append(float(np.sqrt(np.mean((ratio - 1) ** 2))))
        return deviations

    @property
    def position_error(self) -> list[float]:
        """The rms distance of the measured positions from :attr:`positions`, in metres,
        per iteration.
        """
        return [
            float(np.sqrt(np.mean(np.sum((measured - self.positions) ** 2, axis=1))))
            for measured in map(np.asarray, self.measured_positions)
        ]


class SpotArrayCorrector:
    """Correct a spot array's hologram for the intensities and the positions of its
    spots on a camera.

    Each iteration runs the phase retriever, displays the hologram and measures the
    spots (:func:`measure_spot_intensities`). Between holograms, each spot's weight is
    multiplied by its requested over its measured intensity, both relative to their
    mean. Each spot's target also moves by minus its position error: where its maximum
    lies, less its position. The shapes of the spots are not fed back. Each hologram
    starts from the last.

    - A weighted Gerchberg-Saxton retriever
      (:class:`~hologradpy.holography.phase_retrieval.weighted_gerchberg_saxton.WeightedGerchbergSaxtonPhaseRetriever`)
      holds the positions and the requested intensities. Its ``target_intensities``
      become the adjusted weights and its ``target_positions`` the adjusted positions.
      It holds its spots' focal-plane phases (``focal_phase_iterations`` of 1).
    - A gradient retriever
      (:class:`~hologradpy.holography.phase_retrieval.abstract.GradientPhaseRetriever`),
      such as the pixelwise one, takes the positions as an argument. The requested
      intensities are its target's values at the positions. Its target is multiplied
      by a map that equals each spot's factor at its position, and resampled so that
      each spot moves. The target keeps its total power.
    """

    def __init__(
        self,
        slm: SLM,
        camera: Camera,
        phase_retriever: PhaseRetrieverBase,
        camera_mapping: CameraMapping,
        positions: ArrayLike | None = None,
    ) -> None:
        """
        Args:
            slm: The SLM, or a driver that
                :func:`~hologradpy.hardware.as_native.as_slm` wraps.
            camera: The camera, or a driver that
                :func:`~hologradpy.hardware.as_native.as_camera` wraps.
            phase_retriever: The retriever that computes the holograms, with its target
                set.
            camera_mapping: The mapping of the camera, which places the spots on the
                sensor.
            positions: ``(N, 2)`` focal-plane positions of the spots, ``(x, y)`` in
                metres from the zeroth order. Defaults to the ``target_positions`` of a
                weighted Gerchberg-Saxton retriever.

        Raises:
            TypeError: The retriever is neither weighted Gerchberg-Saxton nor a
                gradient retriever.
            ValueError: A gradient retriever has no positions or no target. The target
                is dark at a spot. The positions are not ``(N, 2)``. The mapping was
                measured with another camera.
        """
        if isinstance(phase_retriever, WeightedGerchbergSaxtonPhaseRetriever):
            if positions is None:
                positions = phase_retriever.target_positions
        elif isinstance(phase_retriever, GradientPhaseRetriever):
            if positions is None:
                raise ValueError(
                    "The target of a gradient retriever is an image, so the positions "
                    "of its spots have to be passed."
                )
            if phase_retriever.target is None:
                raise ValueError("The phase retriever has no target.")
        else:
            raise TypeError(
                "A SpotArrayCorrector drives a weighted Gerchberg-Saxton or a gradient "
                f"retriever, got a {type(phase_retriever).__name__}."
            )
        self.slm = as_slm(slm)
        self.camera = as_camera(camera)
        camera_mapping.check_camera(self.camera)
        self.camera_mapping = camera_mapping
        self.phase_retriever = phase_retriever
        self.positions = _checked_positions(positions)
        # Where the retriever aims the spots, which moves by the position corrections.
        self._targets = self.positions.copy()
        model = self.slm_camera_model
        model.build()
        # The output pixels' focal-plane positions, for an image target. The grid is
        # affine in the pixel indices, so its origin and its column and row steps carry
        # metres to fractional pixels.
        self._grid: tuple[torch.Tensor, torch.Tensor] | None = None
        if isinstance(phase_retriever, GradientPhaseRetriever):
            x, y = model.focal_plane_positions(*model[-1].get_spatial_grid_output())
            self._grid = (x, y)
            self._grid_origin = np.array([float(x[0, 0]), float(y[0, 0])])
            self._grid_steps = np.array(
                [
                    [float(x[0, 1] - x[0, 0]), float(x[1, 0] - x[0, 0])],
                    [float(y[0, 1] - y[0, 0]), float(y[1, 0] - y[0, 0])],
                ]
            )
        requested = self._target_weights()
        dark = requested < _DARK_SPOT_FRACTION
        if dark.any():
            raise ValueError(
                f"The target is dark at spots {np.flatnonzero(dark).tolist()}."
            )
        #: The ``(N,)`` requested relative intensities, with a mean of 1.
        self.requested_weights = requested

    @property
    def slm_camera_model(self) -> SLMFourierLensModel:
        """The retriever's model."""
        return self.phase_retriever.slm_camera_model

    def run(
        self,
        retriever_iterations: Sequence[int] = (50,) * 5,
        averages: int = 1,
        exposure: float | None = None,
        retrieve_options: dict[str, Any] | None = None,
        name: str = "",
        verbose: bool = True,
        correct_positions: bool = True,
    ) -> SpotArrayCorrectionData:
        """Run the feedback loop.

        Args:
            retriever_iterations: The iterations of the phase retriever for each
                hologram. Its length is the number of feedback iterations.
            averages: The number of frames averaged per measurement.
            exposure: The exposure in seconds. If None, the camera autoexposes on the
                spots' windows at every measurement.
            retrieve_options: Further keyword arguments of the retriever's
                ``retrieve``, such as ``{"method": "cg"}``.
            name: The name of the run.
            verbose: Show the retrievers' progress and each iteration's errors.
            correct_positions: Move each spot's target by minus its position error.
                Switch it off if the camera's own optics distort the image.

        Returns:
            SpotArrayCorrectionData: The holograms, the weights, the positions and the
            measurements.
        """
        steps = FeedbackCorrectorBase._retriever_steps(retriever_iterations)
        options = dict(retrieve_options or {})
        history: dict[str, list] = {
            "adjusted_weights": [],
            "adjusted_positions": [],
            "measured_intensities": [],
            "measured_positions": [],
            "predicted_intensities": [],
            "retrievals": [],
            "exposures": [],
            "overexposed": [],
        }
        for iteration, iterations in enumerate(steps):
            history["adjusted_weights"].append(self._target_weights())
            history["adjusted_positions"].append(self._targets.copy())
            retrieval = self.phase_retriever.retrieve(
                iterations,
                name=f"spot array feedback {iteration + 1}",
                verbose=verbose,
                **options,
            )
            self._start_next_from(retrieval.phase)
            self.slm.set_phase(retrieval.phase)
            measured = measure_spot_intensities(
                self.camera,
                self.camera_mapping,
                self.positions,
                averages=averages,
                exposure=exposure,
            )
            found = self.camera_mapping.sensor_to_image_plane(measured.sensor_positions)
            history["measured_intensities"].append(measured.intensities)
            history["measured_positions"].append(found)
            history["predicted_intensities"].append(self._predicted_intensities())
            history["retrievals"].append(retrieval.lean())
            history["exposures"].append(measured.exposure)
            history["overexposed"].append(measured.overexposed)
            if verbose:
                ratio = (
                    measured.intensities
                    / measured.intensities.mean()
                    / self.requested_weights
                )
                offset = np.sqrt(np.mean(np.sum((found - self.positions) ** 2, axis=1)))
                print(
                    f"Feedback iteration {iteration + 1}: the spots are "
                    f"{100 * np.sqrt(np.mean((ratio - 1) ** 2)):.2f} % rms off the "
                    f"requested intensities and {offset * 1e6:.2f} um rms off their "
                    "positions."
                )
            # The next hologram corrects what this one measured.
            if iteration < len(steps) - 1:
                self._scale_weights(
                    _correction_factors(self.requested_weights, measured.intensities)
                )
                if correct_positions:
                    self._move_spots(found - self.positions)

        return SpotArrayCorrectionData(
            timestamp=datetime.now(),
            name=name,
            retriever=type(self.phase_retriever).__name__,
            positions=self.positions,
            requested_weights=self.requested_weights,
            final_camera_image=measured.frame,
            camera_mapping=self.camera_mapping.lean(),
            **history,
        )

    def _target_weights(self) -> NDArray:
        """The weights of the spots in the retriever's target, with a mean of 1."""
        retriever = self.phase_retriever
        if self._grid is None:
            weights = np.asarray(retriever.target_intensities, dtype=np.float64)
        else:
            weights = self._target_at_positions()
        mean = weights.mean()
        return weights / mean if mean > 0 else np.zeros_like(weights)

    def _target_at_positions(self) -> NDArray:
        """The target's value where it aims each spot, from a cubic spline between the
        pixels.
        """
        image = gpu_to_numpy(as_image(self.phase_retriever.target)).astype(np.float64)
        columns, rows = np.linalg.solve(
            self._grid_steps, (self._targets - self._grid_origin).T
        )
        return map_coordinates(image, [rows, columns])

    def _scale_weights(self, factors: NDArray) -> None:
        """Multiply the weight of each spot in the retriever's target by its factor.

        A gradient retriever's target is multiplied by
        :meth:`_interpolate_between_spots` and keeps its total power.
        """
        retriever = self.phase_retriever
        if self._grid is None:
            weights = np.asarray(retriever.target_intensities) * factors
            retriever.target_intensities = weights / weights.mean()
            return
        target = retriever.target
        factor_map = self._interpolate_between_spots(factors).to(target)
        scaled = target * factor_map
        retriever.set_target(scaled * (target.sum() / scaled.sum()))

    def _move_spots(self, errors: NDArray) -> None:
        """Move each spot's target by minus its position error, ``(N, 2)`` in metres.

        - If the retriever is weighted Gerchberg-Saxton, shift its target positions.
        - Otherwise, resample the target with a cubic spline, along the errors
          interpolated between the spots. The target keeps its total power.
        """
        retriever = self.phase_retriever
        if self._grid is not None:
            target = retriever.target
            image = gpu_to_numpy(as_image(target)).astype(np.float64)
            fields = [
                gpu_to_numpy(self._interpolate_between_spots(error))
                for error in errors.T
            ]
            # Sampling the target at p + E(p) moves each spot by minus its error.
            columns, rows = np.tensordot(np.linalg.inv(self._grid_steps), fields, 1)
            pixels = np.indices(image.shape, dtype=np.float64)
            moved = map_coordinates(image, pixels + [rows, columns], mode="constant")
            moved = moved * (image.sum() / moved.sum())
            moved = torch.as_tensor(moved, dtype=target.dtype, device=target.device)
            retriever.set_target(moved.reshape(target.shape))
        self._targets = self._targets - errors
        if isinstance(retriever, WeightedGerchbergSaxtonPhaseRetriever):
            retriever.target_positions = self._targets

    def _interpolate_between_spots(self, values: NDArray) -> torch.Tensor:
        """A map that equals each spot's value where the target aims it and changes
        smoothly between the spots.

        The map is ``sum_k v_k b_k / sum_k b_k``, with ``b_k`` a Gaussian around the
        spot ``k`` whose sigma is a quarter of the smallest spacing. Each spot keeps
        its own value to within ``exp(-8)``.
        """
        x, y = self._grid
        # 2 sigma^2 of b_k.
        width = smallest_spacing(self._targets) ** 2 / 8
        # Exponents relative to the nearest spot keep the sums from underflowing.
        nearest = torch.full_like(x, float("inf"))
        for spot_x, spot_y in self._targets:
            nearest = torch.minimum(nearest, (x - spot_x) ** 2 + (y - spot_y) ** 2)
        numerator = torch.zeros_like(x)
        denominator = torch.zeros_like(x)
        for (spot_x, spot_y), value in zip(self._targets, values):
            squared = (x - spot_x) ** 2 + (y - spot_y) ** 2
            weight = torch.exp(-(squared - nearest) / width)
            numerator += float(value) * weight
            denominator += weight
        return numerator / denominator

    def _start_next_from(self, phase: ArrayLike) -> None:
        """Start the next hologram from ``phase``.

        - If the retriever is weighted Gerchberg-Saxton, it starts from ``phase`` and
          holds the focal-plane phases of its spots.
        - Otherwise, the model already holds ``phase``, and the retriever starts from
          it.
        """
        retriever = self.phase_retriever
        if isinstance(retriever, WeightedGerchbergSaxtonPhaseRetriever):
            retriever.init_slm_phase = torch.as_tensor(
                phase, device=self.slm_camera_model.device
            )
            retriever.focal_phase_iterations = 1

    def _predicted_intensities(self) -> NDArray:
        """The model's intensity where the retriever aims each spot, relative to their
        mean.
        """
        intensities = gpu_to_numpy(
            self.slm_camera_model.focal_field(self._targets).abs() ** 2
        )
        return intensities / intensities.mean()


def _checked_positions(positions: ArrayLike) -> NDArray:
    """``positions`` as an ``(N, 2)`` float array.

    Raises:
        ValueError: The positions are not ``(N, 2)`` with N at least 1, or not finite.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2 or len(positions) == 0:
        raise ValueError(
            f"The positions are an (N, 2) array of (x, y), got shape {positions.shape}."
        )
    if not np.isfinite(positions).all():
        raise ValueError("The positions are not all finite.")
    return positions


def _correction_factors(requested: NDArray, measured: NDArray) -> NDArray:
    """The factor of each spot's weight: its requested over its measured intensity,
    each relative to their mean.
    """
    return (requested / requested.mean()) / (measured / measured.mean())


def _windows(
    shape: tuple[int, int], sites: NDArray, radius: float
) -> list[tuple[ROI, NDArray]]:
    """The disc of ``radius`` around each site, as the region around it and the disc on
    that region.

    Raises:
        ValueError: A disc runs off the sensor.
    """
    windows = []
    for column, row in sites:
        region = ROI.from_bounds(
            np.floor(row - radius),
            np.ceil(row + radius) + 1,
            np.floor(column - radius),
            np.ceil(column + radius) + 1,
        )
        disc = disc_mask(
            (region.height, region.width),
            (column - region.left_column, row - region.top_row),
            radius,
        )
        windows.append((region, disc))
    off_sensor = [
        index
        for index, (region, _) in enumerate(windows)
        if not region.lies_inside(shape)
    ]
    if off_sensor:
        raise ValueError(
            f"The windows of spots {off_sensor}, {radius:.1f} px in radius, run off "
            "the sensor."
        )
    return windows
