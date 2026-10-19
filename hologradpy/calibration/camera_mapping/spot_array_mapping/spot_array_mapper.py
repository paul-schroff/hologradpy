from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from ....geometry import AffineTransform

from ....hardware import Camera, SLM

from ....optics.systems import SLMFourierLensModel
from ....grids import metres_to_pixel, pixel_to_metres, plane_center
from ....fourier_optics import get_focal_spot_radius
from ....analysis.uncertainties import standard_deviation
from ....profiles.gaussian import Gaussian, GaussianFit
from ....utils import as_image, gpu_to_numpy
from ....roi import ROI
from ....holography.phase_retrieval import LinearSuperpositionPhaseRetriever
from ....profiles.masks import disc_mask

from ...spot_detection import (
    DETECTION_HALF_WIDTH_RADII,
    background_noise,
    zeroth_order_mask_radius,
)
from ..coarse_mapping.coarse_mapper import CoarseMapper

from ..abstract import CameraMapper
from ..mapping import CameraMapping, FocalSpotFit
from ..visualizer import CameraMappingVisualizationData

# Minimum correspondences for an affine fit
_MIN_AFFINE_POINTS = 3

# Target-match tolerance as a fraction of the minimum spot separation
_MATCH_TOLERANCE_FRACTION = 0.5

# Floor for the per-spot Gaussian fit-window side, in pixels
_MIN_ROI_SIZE_PX = 8

# Measured focal-spot radius capped at this multiple of the diffraction limit
_MAX_SPOT_RADIUS_FACTOR = 5.0

# Spot-spacing bounds, in focal-spot radii
_MIN_SEPARATION_RADII = 3.0
_MAX_SEPARATION_RADII = 6.0

# Fraction of the SLM addressable half-extent used for reachable targets
_ADDRESSABLE_MARGIN = 0.9


@dataclass
class _SpotDetections:
    """The spots fitted by :meth:`SpotArrayMapper._detect_and_fit`, as parallel lists
    of ``(column, row)`` pixel positions and Gaussian fits. The peaks with a rejected
    fit are listed separately.
    """

    points: list[tuple[float, float]]
    fits: list[GaussianFit]
    rejected_peaks: list[tuple[float, float]]


class SpotArrayMapper(CameraMapper):
    """Map the model's output coordinates to the camera sensor using a random spot
    array, similar to slmsuite's
    :meth:`~slmsuite.hardware.cameraslms.FourierSLM.fourier_calibrate`.

    A linear-superposition hologram (:class:`LinearSuperpositionPhaseRetriever`)
    diffracts ``number_of_spots`` focal spots to random positions in the focal plane,
    at least ``minimum_separation`` apart. The targets are sampled to fill the sensor
    inside a detection-window border. They are placed through the coarse mapping, so
    the array lands on the sensor whatever its rotation, mirroring or offset.

    The spots are detected in the camera image and in the model's output, and each is
    fitted with a 2D Gaussian. An affine transform is fitted between the two sets.
    """

    mapping_name = "spot_array"

    def __init__(
        self,
        slm: SLM,
        camera: Camera,
        slm_camera_model: SLMFourierLensModel,
    ) -> None:
        super().__init__(slm, camera, slm_camera_model)
        self.device = slm_camera_model.device

    def map_camera(
        self,
        number_of_spots: int = 50,
        minimum_separation: float | None = None,
        roi_size: int | None = None,
        exposure_time: float | None = None,
        snr_threshold: float = 5.0,
        seed: int | None = None,
        randomize_phases: bool = True,
        coarse_mapping: CameraMapping | None = None,
    ) -> CameraMapping:
        """Calibrate the camera from a random spot array.

        The mapping is measured with the model's focal-plane partial affine at identity
        (:meth:`~hologradpy.optics.systems.SLMFourierLensModel.bypass_partial_affine`).
        The geometry of the simulated image is read from the model's output layer. The
        whole sensor is read out while the camera is mapped, and its exposure, gain and
        region of interest are put back afterwards
        (:meth:`~hologradpy.hardware.camera.Camera.preserve_exposure_gain_and_roi`).

        Args:
            number_of_spots: Number of focal spots in the array.
            minimum_separation: Minimum distance between any two spots in metres.
                Defaults to a few focal-spot radii.
            roi_size: Side length in camera pixels of the per-spot fit window.
                Defaults to about ``minimum_separation`` in camera pixels.
            exposure_time: Camera exposure in seconds for the array capture. If None,
                an exposure loop targets the camera's dynamic range on the
                zeroth-order-masked image.
            snr_threshold: Spots are detected and accepted only above this multiple of
                the background noise sigma, so the transform excludes any fit that
                converged on empty background.
            seed: Seed for the random spot positions and phases.
            randomize_phases: Give each spot a random phase to suppress ghost orders in
                the superposition hologram.
            coarse_mapping: Result of :meth:`CoarseMapper.map_camera`. If None, a
                :class:`CoarseMapper` is built and run internally. A spot may land on
                the zeroth order, which is masked out only during detection.

        Returns:
            CameraMapping: The mapping with the affine transform, the per-spot Gaussian
            fits and the uncertainty-weighted average waist.
        """
        with (
            self.camera.preserve_exposure_gain_and_roi(full_sensor=True),
            self.slm_camera_model.bypass_partial_affine(),
        ):
            pitch = np.asarray(self.camera.pixel_size, dtype=float)  # (y, x) metres
            camera_shape = tuple(self.camera.resolution)  # (height, width)

            if coarse_mapping is None:
                coarse_mapping = CoarseMapper(
                    self.slm, self.camera, self.slm_camera_model
                ).map_camera()
            else:
                coarse_mapping.check_camera(self.camera)

            focal_spot_radius = self._focal_spot_radius(coarse_mapping)

            # The disc around the zeroth order that the detection leaves out.
            zeroth_mask = coarse_mapping.zeroth_order_mask(
                camera_shape, zeroth_order_mask_radius(focal_spot_radius, pitch)
            )

            generator = torch.Generator(device=self.device)
            if seed is not None:
                generator.manual_seed(seed)
            target_positions, minimum_separation = self._sample_targets(
                coarse_mapping,
                number_of_spots,
                minimum_separation,
                focal_spot_radius,
                generator,
            )
            if roi_size is None:
                roi_size = max(
                    int(round(minimum_separation / pitch.min())), _MIN_ROI_SIZE_PX
                )

            if randomize_phases:
                target_phases = 2 * torch.pi * torch.rand(
                    len(target_positions), generator=generator, device=self.device
                )
            else:
                target_phases = torch.zeros(len(target_positions), device=self.device)

            camera_image, simulated_image = self._capture_array(
                target_positions, target_phases, exposure_time, zeroth_mask
            )
            masked_image = camera_image * (~zeroth_mask)

            # Detect + fit the spots in the camera image (brightest first).
            camera_spots = self._detect_and_fit(
                masked_image, pitch, roi_size, focal_spot_radius, snr_threshold,
                number_of_peaks=number_of_spots,
            )
            if len(camera_spots.points) < _MIN_AFFINE_POINTS:
                raise RuntimeError(
                    f"Only {len(camera_spots.points)} spots were detected and fitted "
                    "on the camera. Need at least 3 for an affine transform. Increase "
                    "exposure / number_of_spots, or check the array is on the sensor."
                )
            simulated_spots = self._detect_and_fit_simulated(
                simulated_image,
                focal_spot_radius,
                minimum_separation,
                snr_threshold,
                number_of_spots,
            )

            camera_indices, simulated_indices = self._pair_spots(
                camera_spots,
                simulated_spots,
                gpu_to_numpy(target_positions).astype(np.float64),
                coarse_mapping,
                minimum_separation,
                simulated_image.shape,
            )

            affine, inliers = AffineTransform.fit_robust(
                [camera_spots.points[i] for i in camera_indices],
                [simulated_spots.points[i] for i in simulated_indices],
            )
            if inliers.sum() < _MIN_AFFINE_POINTS:
                raise RuntimeError(
                    f"Only {int(inliers.sum())} of {len(camera_indices)} "
                    "matched spots are consistent with an affine transform. Need at "
                    "least 3."
                )
            camera_inliers = [i for i, keep in zip(camera_indices, inliers) if keep]
            simulated_inliers = [
                i for i, keep in zip(simulated_indices, inliers) if keep
            ]
            fits = [camera_spots.fits[i] for i in camera_inliers]
            waists = [fit.beam.mean_radius for fit in fits]
            waist_variances = [
                standard_deviation(fit, lambda fit: fit.beam.mean_radius) ** 2
                for fit in fits
            ]
            average_waist, average_waist_uncertainty = self._weighted_average(
                np.asarray(waists), np.asarray(waist_variances)
            )

            # Camera detections that did not make it into the transform, kept so none
            # disappears from a plot without explanation.
            excluded_points = list(camera_spots.rejected_peaks)
            excluded_points.extend(
                point
                for index, point in enumerate(camera_spots.points)
                if index not in camera_inliers
            )

            return self._record(
                affine,
                [camera_spots.points[i] for i in camera_inliers],
                [simulated_spots.points[i] for i in simulated_inliers],
                FocalSpotFit(
                    waist=average_waist,
                    waist_uncertainty=average_waist_uncertainty,
                    fits=fits,
                ),
                CameraMappingVisualizationData(
                    camera_image=masked_image,
                    simulated_image=simulated_image,
                    zeroth_order_mask=zeroth_mask,
                ),
                excluded_points=excluded_points,
            )

    def _focal_spot_radius(self, coarse_mapping: CameraMapping) -> float:
        """The waist of the coarse mapping, held between the diffraction limit of the
        SLM aperture and ``_MAX_SPOT_RADIUS_FACTOR`` times it.
        """
        diffraction_limit = get_focal_spot_radius(
            beam_radius=0.5 * min(self.slm.aperture_extent),
            wavelength=self.slm.wavelength,
            focal_length=self.slm_camera_model.focal_length,
        )
        return float(
            np.clip(
                abs(coarse_mapping.spot_fit.waist),
                diffraction_limit,
                _MAX_SPOT_RADIUS_FACTOR * diffraction_limit,
            )
        )

    def _sample_targets(
        self,
        coarse_mapping: CameraMapping,
        number_of_spots: int,
        minimum_separation: float | None,
        focal_spot_radius: float,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, float]:
        """Random focal-plane targets with spots that fill the sensor inside a
        detection-window border.

        The targets are sampled on the sensor and carried into the focal plane through
        the full affine of ``coarse_mapping``, so the array fills the sensor of a
        mapping with unequal scales or a shear. Those beyond ``_ADDRESSABLE_MARGIN`` of
        the field the SLM addresses are dropped.

        Args:
            coarse_mapping: The coarse mapping that places the targets on the sensor.
            number_of_spots: The number of targets to sample.
            minimum_separation: The smallest distance between two targets in metres,
                or None for a separation that packs the targets into the sensor.
            focal_spot_radius: The focal-spot radius in metres.
            generator: The random generator of the positions.

        Returns:
            tuple[torch.Tensor, float]: The ``(N, 2)`` target positions ``(x, y)`` in
            metres, and the minimum separation in metres.

        Raises:
            ValueError: The sensor leaves no room inside the detection-window border.
            RuntimeError: Fewer than 4 targets lie inside the addressable field.
        """
        camera_shape = tuple(self.camera.resolution)  # (height, width)

        # Camera pixels per focal-plane metre from the coarse mapping.
        camera_scale = float(
            np.sqrt(abs(np.linalg.det(coarse_mapping.sensor_pixels_per_metre)))
        )

        border_pixels = DETECTION_HALF_WIDTH_RADII * focal_spot_radius * camera_scale
        box_pixels = (
            camera_shape[1] - 2.0 * border_pixels,
            camera_shape[0] - 2.0 * border_pixels,
        )
        if box_pixels[0] <= 0 or box_pixels[1] <= 0:
            raise ValueError(
                "The sensor is smaller than twice the detection-window border. "
                "There is no room for the spot array."
            )

        if minimum_separation is None:
            box_metres = (box_pixels[0] / camera_scale, box_pixels[1] / camera_scale)
            # Hex-packing separation (cell area sqrt(3)/2 * d^2), with headroom so
            # rejection sampling can still place every spot.
            feasible = 0.7 * float(
                np.sqrt(
                    2.0
                    * box_metres[0]
                    * box_metres[1]
                    / (np.sqrt(3.0) * number_of_spots)
                )
            )
            minimum_separation = min(
                _MAX_SEPARATION_RADII * focal_spot_radius,
                max(feasible, _MIN_SEPARATION_RADII * focal_spot_radius),
            )

        # Centred camera-pixel samples -> sensor centre -> focal-plane metres.
        sampled_pixels = self._sample_positions(
            number_of_spots,
            box_pixels,
            minimum_separation * camera_scale,
            generator,
        )
        sampled_pixels = sampled_pixels + torch.tensor(
            [camera_shape[1] / 2.0, camera_shape[0] / 2.0],
            device=self.device, dtype=sampled_pixels.dtype,
        )
        pixel_size_out, resolution_out = self.slm_camera_model.output_plane
        model_pixels = coarse_mapping.affine.transform_points(
            sampled_pixels.cpu().numpy()
        )
        metres = np.column_stack(
            pixel_to_metres(
                (model_pixels[:, 0], model_pixels[:, 1]), pixel_size_out, resolution_out
            )
        )
        addressable = self.slm_camera_model.addressable_half_extent()
        reachable = (
            np.abs(metres[:, 0]) <= _ADDRESSABLE_MARGIN * addressable[0]
        ) & (np.abs(metres[:, 1]) <= _ADDRESSABLE_MARGIN * addressable[1])
        metres = metres[reachable]
        # One target more than the affine minimum, to fit robustly.
        if len(metres) < 4:
            raise RuntimeError(
                "Fewer than 4 spot targets fall within the SLM's addressable "
                "range. The sensor may lie largely outside it."
            )
        return (
            torch.tensor(metres, device=self.device, dtype=torch.float64),
            minimum_separation,
        )

    def _capture_array(
        self,
        target_positions: torch.Tensor,
        target_phases: torch.Tensor,
        exposure_time: float | None,
        zeroth_order_mask: NDArray[np.bool_],
    ) -> tuple[NDArray, NDArray]:
        """Render the superposition hologram of the targets in the model, then display
        it on the SLM and capture it.

        Args:
            target_positions: The ``(N, 2)`` target positions ``(x, y)`` in metres.
            target_phases: The phase of each spot.
            exposure_time: The exposure in seconds, or None to expose for the array
                with the zeroth order masked (:meth:`_expose_for_array`).
            zeroth_order_mask: True over the disc around the zeroth order.

        Returns:
            tuple[NDArray, NDArray]: The camera frame and the model's image.
        """
        slm_phase = LinearSuperpositionPhaseRetriever(
            self.slm_camera_model, target_positions, target_phases=target_phases
        ).retrieve_phase()

        # Simulated image (for the record / visualizer).
        self.slm_camera_model.virtual_slm.set_phase(slm_phase)
        simulated_image = gpu_to_numpy(as_image(self.slm_camera_model().intensity))

        self.slm.set_phase(gpu_to_numpy(slm_phase))
        if exposure_time is None:
            self._expose_for_array(zeroth_order_mask)
        else:
            self.camera.set_exposure(exposure_time)
        return self.camera.get_image(), simulated_image

    def _detect_and_fit_simulated(
        self,
        simulated_image: NDArray,
        focal_spot_radius: float,
        minimum_separation: float,
        snr_threshold: float,
        number_of_spots: int,
    ) -> _SpotDetections:
        """Detect and fit the spots of the model's image, with the zeroth order at the
        centre of the plane masked out (:meth:`_detect_and_fit`).

        Raises:
            RuntimeError: Fewer spots than an affine transform needs were fitted.
        """
        pixel_size_out, resolution_out = self.slm_camera_model.output_plane
        # The pitch is (y, x), as the camera's is.
        simulated_pitch = np.asarray(pixel_size_out, dtype=float)
        simulated_zod = plane_center(resolution_out)
        # Simulated DC is a clean point, so a few focal-spot radii suffice.
        simulated_mask_radius = max(
            int(round(3.0 * focal_spot_radius / simulated_pitch.min())), 3
        )
        simulated_masked = simulated_image * (
            ~disc_mask(
                tuple(simulated_image.shape), simulated_zod, simulated_mask_radius
            )
        )
        simulated_roi_size = max(
            int(round(minimum_separation / simulated_pitch.min())), _MIN_ROI_SIZE_PX
        )
        simulated_spots = self._detect_and_fit(
            simulated_masked, simulated_pitch, simulated_roi_size,
            focal_spot_radius, snr_threshold, number_of_peaks=number_of_spots,
        )
        if len(simulated_spots.points) < _MIN_AFFINE_POINTS:
            raise RuntimeError(
                f"Only {len(simulated_spots.points)} spots were detected and "
                "fitted in the simulated image. Need at least 3 for an affine "
                "transform."
            )
        return simulated_spots

    def _pair_spots(
        self,
        camera_spots: _SpotDetections,
        simulated_spots: _SpotDetections,
        targets: NDArray,
        coarse_mapping: CameraMapping,
        minimum_separation: float,
        simulated_shape: tuple[int, int],
    ) -> tuple[list[int], list[int]]:
        """The indices of the camera and model spots that match the same targets.

        The model spots are matched to where the targets lie in the model's image, and
        the camera spots to where the inverse of the full affine of ``coarse_mapping``
        carries those pixels onto the sensor (:meth:`_match_targets`). A target matched
        in both gives a pair.

        Args:
            camera_spots: The spots fitted on the camera.
            simulated_spots: The spots fitted in the model's image.
            targets: The ``(N, 2)`` target positions ``(x, y)`` in metres.
            coarse_mapping: The coarse mapping that placed the targets.
            minimum_separation: The smallest distance between two targets in metres,
                which sets the match tolerance.
            simulated_shape: The ``(height, width)`` of the model's image.

        Returns:
            tuple[list[int], list[int]]: The indices into the camera spots and into the
            model spots, one pair for each target matched in both.

        Raises:
            RuntimeError: Fewer targets than an affine transform needs were matched in
                both.
        """
        pixel_size_out, _ = self.slm_camera_model.output_plane
        pitch = np.asarray(self.camera.pixel_size, dtype=float)
        target_pixels = np.column_stack(
            metres_to_pixel(
                (targets[:, 0], targets[:, 1]), pixel_size_out, simulated_shape
            )
        )
        camera_indices, camera_targets = self._match_targets(
            coarse_mapping.affine.inverse().transform_points(target_pixels),
            targets,
            np.asarray(camera_spots.points, dtype=np.float64),
            tolerance=(
                _MATCH_TOLERANCE_FRACTION * minimum_separation / float(pitch.min())
            ),
        )
        simulated_indices, simulated_targets = self._match_targets(
            target_pixels,
            targets,
            np.asarray(simulated_spots.points, dtype=np.float64),
            tolerance=(
                _MATCH_TOLERANCE_FRACTION
                * minimum_separation
                / float(np.min(pixel_size_out))
            ),
        )
        camera_by_target = dict(zip(camera_targets, camera_indices))
        simulated_by_target = dict(zip(simulated_targets, simulated_indices))
        common_targets = sorted(set(camera_by_target) & set(simulated_by_target))
        if len(common_targets) < _MIN_AFFINE_POINTS:
            raise RuntimeError(
                f"Only {len(common_targets)} spots could be matched to targets in "
                "both the camera and the simulated image. Need at least 3 for an "
                "affine transform."
            )

        return (
            [camera_by_target[target] for target in common_targets],
            [simulated_by_target[target] for target in common_targets],
        )

    def _sample_positions(
        self,
        number_of_spots: int,
        extent: tuple[float, float],
        minimum_separation: float,
        generator: torch.Generator,
        max_attempts_per_spot: int = 2000,
    ) -> torch.Tensor:
        """Rejection-sample ``number_of_spots`` points uniformly in the box centered at
        the origin, each at least ``minimum_separation`` from the rest.
        """
        width, height = extent
        box = torch.tensor([width, height], device=self.device)
        positions = torch.empty((0, 2), device=self.device)
        max_attempts = max_attempts_per_spot * number_of_spots
        for _ in range(max_attempts):
            if positions.shape[0] >= number_of_spots:
                break
            candidate = (
                torch.rand(2, generator=generator, device=self.device) - 0.5
            ) * box
            if positions.shape[0] == 0 or torch.min(
                torch.linalg.norm(positions - candidate, dim=1)
            ) >= minimum_separation:
                positions = torch.cat([positions, candidate.unsqueeze(0)], dim=0)
        if positions.shape[0] < number_of_spots:
            raise ValueError(
                f"Could not place {number_of_spots} spots with minimum separation "
                f"{minimum_separation:.3e} m in extent {extent}. Reduce "
                "number_of_spots / minimum_separation, or enlarge extent."
            )
        return positions

    def _expose_for_array(
        self,
        zeroth_order_mask: NDArray[np.bool_],
        relative_target_brightness: float = 0.8,
        tolerance: float = 0.1,
    ) -> None:
        """Set the exposure so the brightest array spot reaches
        ``relative_target_brightness`` of the camera's dynamic range, with the
        zeroth order masked out.

        A rectangular ``autoexpose`` window cannot be used here: any window guaranteed
        to contain the (arbitrarily oriented) array also contains the much brighter
        zeroth order. The boolean ``mask`` drops the zeroth order so the peak tracks the
        array, and ``raise_on_rail`` is disabled so a dim array settles at the exposure
        bound rather than raising.
        """
        self.camera.autoexpose(
            set_fraction=relative_target_brightness,
            tolerance=tolerance,
            mask=~zeroth_order_mask,
            raise_on_rail=False,
        )

    @classmethod
    def _detect_peaks(
        cls,
        image: NDArray,
        number_of_peaks: int,
        blank_radius: int,
        threshold: float,
    ) -> list[tuple[float, float]]:
        """Iteratively pick the brightest pixel and blank a disc around it, so each spot
        yields exactly one peak. Stops once the remaining maximum falls to the
        background ``threshold``. Returns ``(column, row)`` pixel positions, brightest
        first.
        """
        working = np.array(image, dtype=float, copy=True)
        peaks: list[tuple[float, float]] = []
        for _ in range(number_of_peaks):
            row, column = np.unravel_index(int(np.argmax(working)), working.shape)
            if working[row, column] <= threshold:
                break
            peaks.append((float(column), float(row)))
            blank = disc_mask(working.shape, (column, row), blank_radius)
            working[blank] = 0.0
        return peaks

    def _detect_and_fit(
        self,
        image: NDArray,
        pitch: NDArray,
        roi_size: int,
        radius_guess: float,
        snr_threshold: float,
        number_of_peaks: int,
    ) -> _SpotDetections:
        """Detect up to ``number_of_peaks`` bright spots in ``image`` and fit a 2D
        Gaussian to each. Returns the fitted ``(column, row)`` pixel positions, the
        Gaussian fits, and the peaks with a rejected fit.
        """
        noise_std = background_noise(image)
        peaks = self._detect_peaks(
            image,
            number_of_peaks=number_of_peaks,
            blank_radius=max(roi_size // 2, 4),
            threshold=snr_threshold * noise_std,
        )

        points: list[tuple[float, float]] = []
        fits: list[GaussianFit] = []
        rejected_peaks: list[tuple[float, float]] = []
        for peak in peaks:
            fit = self._fit_spot(
                image, peak, roi_size, radius_guess, pitch, snr_threshold * noise_std
            )
            if fit is None:
                rejected_peaks.append(peak)
                continue
            beam = fit.beam
            points.append(
                metres_to_pixel((beam.shift_x, beam.shift_y), pitch, image.shape)
            )
            fits.append(fit)
        return _SpotDetections(points, fits, rejected_peaks)

    @staticmethod
    def _match_targets(
        predicted: NDArray, targets: NDArray, detected: NDArray, tolerance: float
    ) -> tuple[NDArray, NDArray]:
        """Pair each target with a detected spot, starting from the ``predicted``
        landing positions of the targets.

        The pairing is refined twice, since a rigid starting guess captures neither
        shear nor a field-dependent warp. Aberrations displace each spot by its local
        wavefront tilt. In each refinement, an affine is fitted to the current pairs and
        every target is reassigned, so spots that only miss the rigid guess still match.

        Args:
            predicted: Where each target is expected to land, in the pixels of
                ``detected``, shape (N, 2).
            targets: Target positions in focal-plane metres, shape (N, 2).
            detected: Detected spot centres in pixels, shape (M, 2).
            tolerance: Match radius in pixels. Half the minimum spot separation makes
                the assignment unambiguous.

        Returns:
            tuple[NDArray, NDArray]: The ``(detected_indices, target_indices)`` of the
            matched pairs.
        """
        mapped = np.asarray(predicted, dtype=np.float64)
        cost = np.linalg.norm(mapped[:, None, :] - detected[None, :, :], axis=-1)
        target_indices, detected_indices = linear_sum_assignment(cost)
        matched = cost[target_indices, detected_indices] < tolerance

        for _ in range(2):
            if matched.sum() < 3:
                break
            matching_affine = AffineTransform.fit(
                targets[target_indices[matched]],
                detected[detected_indices[matched]],
            )
            mapped = matching_affine.transform_points(targets)
            cost = np.linalg.norm(mapped[:, None, :] - detected[None, :, :], axis=-1)
            target_indices, detected_indices = linear_sum_assignment(cost)
            matched = cost[target_indices, detected_indices] < tolerance

        return detected_indices[matched], target_indices[matched]

    def _fit_spot(
        self,
        image: NDArray,
        peak: tuple[float, float],
        roi_size: int,
        radius_guess: float,
        pitch: NDArray,
        amplitude_threshold: float,
    ) -> GaussianFit | None:
        """Fit a 2D Gaussian to the ROI around a detected peak.

        Returns the fit, or None when the fit fails or is poor. An amplitude at or below
        ``amplitude_threshold`` counts as background, not a spot.
        """
        half = roi_size // 2
        roi = ROI.centered(
            (round(peak[0]), round(peak[1])), (2 * half, 2 * half)
        ).trimmed_to(image.shape)

        # Blur only seeds the centroid guess. Scale it to the spot size (in camera
        # pixels), not the large default that would flatten a small per-spot ROI.
        blur_sigma = max(float(radius_guess / pitch.min()), 1.0)
        try:
            fit = Gaussian(image.shape, list(pitch), roi).fit(
                roi.crop(image), beam_radius_guess=radius_guess, blur_sigma=blur_sigma
            )
            beam = fit.beam
        except (RuntimeError, ValueError):
            return None

        if (
            not np.isfinite(fit.covariance).all()
            or not np.isfinite(beam.mean_radius)
            or beam.intensity <= amplitude_threshold
        ):
            return None
        # Reject if the fitted center wandered outside the ROI (spurious fit).
        detected = metres_to_pixel((beam.shift_x, beam.shift_y), pitch, image.shape)
        off_x = abs(detected[0] - peak[0])
        off_y = abs(detected[1] - peak[1])
        if off_x > half or off_y > half:
            return None
        return fit

    @staticmethod
    def _weighted_average(
        values: NDArray, variances: NDArray
    ) -> tuple[float, float]:
        """Inverse-variance (uncertainty) weighted mean and its uncertainty."""
        weights = 1.0 / np.clip(variances, a_min=np.finfo(float).tiny, a_max=None)
        mean = float(np.sum(weights * values) / np.sum(weights))
        uncertainty = float(np.sqrt(1.0 / np.sum(weights)))
        return mean, uncertainty
