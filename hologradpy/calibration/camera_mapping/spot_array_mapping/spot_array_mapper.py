from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from ....geometry import AffineTransform

from ....hardware import Camera, SLM

from ....propagation.optical_systems import SLMFourierLensModel
from ....propagation.fourier import get_spatial_grid, metres_to_pixel
from ....propagation.amplitude_profiles import get_focal_spot_radius
from ....analysis.fitting import fit_gaussian_beam_intensity
from ....utils import gpu_to_numpy
from ....roi import ROI
from ....holography.phase_retrieval import LinearSuperpositionPhaseRetriever

from ...spot_detection import (
    _WINDOW_SPOT_RADII,
    background_noise,
    disc_mask,
)
from ..coarse_mapping.coarse_mapper import CoarseMapper

from ..abstract import CameraMapper, CameraMapping

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
    """Fitted spots from :meth:`SpotArrayMapper._detect_and_fit`: pixel positions with 
    their per-spot Gaussian fit parameters and covariances (parallel lists), and the 
    peaks whose fit was rejected."""

    points: list[tuple[float, float]]
    fit_parameters: list[NDArray]
    fit_covariances: list[NDArray]
    rejected_peaks: list[tuple[float, float]]


class SpotArrayMapper(CameraMapper):
    """Map the model's output coordinates to the camera sensor using a random spot 
    array. Similar to slmsuite's
    :meth:`~slmsuite.hardware.cameraslms.FourierSLM.fourier_calibrate`.

    A linear-superposition hologram (:class:`LinearSuperpositionPhaseRetriever`)
    diffracts ``number_of_spots`` focal spots to random positions (separated by at 
    least ``minimum_separation``) in the focal plane. The spots are
    ``minimum_separation``). The targets are sampled to fill the sensor via the coarse 
    transform (built internally if not supplied), so the array lands on the actual 
    sensor whatever its rotation / flip / offset.

    The spots are detected directly in the camera image and the model's output. The 
    spots are fitted with 2D Gaussians registered with an affine transform.
    """

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

        Args:
            number_of_spots: Number of focal spots in the array.
            minimum_separation: Minimum distance between any two spots in metres.
                Defaults to a few focal-spot radii.
            roi_size: Side length in camera pixels of the per-spot fit window.
                Defaults to ~ ``minimum_separation`` in camera pixels.
            exposure_time: Camera exposure in seconds for the array capture. If None 
                (default), an exposure loop targets the camera's dynamic range on the
                zeroth-order-masked image.
            snr_threshold: Spots are detected and accepted only above this multiple of 
                the background noise sigma, so a fit that converged on empty background
                is rejected instead of entering the transform.
            seed: Seed for the random spot positions / phases (reproducibility).
            randomize_phases: Give each spot a random phase to suppress ghost orders in
                the superposition hologram.
            coarse_mapping: Result of :meth:`CoarseMapper.map_camera`. If None
                (default), a :class:`CoarseMapper` is built and run internally. The
                array targets are sampled to fill the sensor (inset by a
                detection-window border) and mapped back through the coarse transform,
                so every spot lands on the actual (rotated, flipped, shifted or
                off-axis) sensor. A spot may land on the zeroth order (it is masked out
                only during detection).

        Returns:
            CameraMapping with the affine transform, the per-spot Gaussian fits and the
            uncertainty-weighted average waist.
        """
        # TODO: Fix inconsistend x/y, h/w conventions.
        lens = self.slm_camera_model.fourier_lens
        pixel_size_out = lens.pixel_size_out.tolist()[0]    # (y, x) metres
        resolution_out = tuple(lens.resolution_out)         # (height, width)
        focal_length = float(lens.focal_length)
        pitch = np.asarray(self.camera.pixel_size, dtype=float)  # (y, x) metres
        camera_shape = tuple(self.camera.resolution)        # (height, width)

        if coarse_mapping is None:
            coarse_mapping = CoarseMapper(
                self.slm, self.camera, self.slm_camera_model
            ).map_camera()

        aperture_radius = 0.5 * min(
            self.slm.resolution[i] * self.slm.pixel_size[i] for i in range(2)
        )
        diffraction_limit = get_focal_spot_radius(
            beam_radius=aperture_radius,
            wavelength=self.slm.wavelength,
            focal_length=focal_length,
        )
        focal_spot_radius = float(
            np.clip(
                abs(coarse_mapping.focal_spot_radius),
                diffraction_limit,
                _MAX_SPOT_RADIUS_FACTOR * diffraction_limit,
            )
        )

        # Zeroth-order pixel (stored as (y, x)) and its detection mask (radius
        # clamped to [8 px, one sixth of the sensor]).
        zeroth_pixel = (
            float(coarse_mapping.zeroth_order_position[1]),
            float(coarse_mapping.zeroth_order_position[0]),
        )
        mask_radius = int(
            np.clip(
                _WINDOW_SPOT_RADII * focal_spot_radius / pitch.min(),
                8,
                min(camera_shape) // 6,
            )
        )
        zeroth_mask = disc_mask(camera_shape, zeroth_pixel, mask_radius)

        # Camera pixels per focal-plane metre from the coarse transform.
        transform = np.asarray(coarse_mapping.transform, dtype=np.float64)
        inverse = np.asarray(coarse_mapping.inverse_transform, dtype=np.float64)
        scale_linear = inverse[:, :2] @ np.diag(
            [1.0 / pixel_size_out[1], 1.0 / pixel_size_out[0]]
        )
        camera_scale = float(np.sqrt(abs(np.linalg.det(scale_linear))))

        border_pixels = _WINDOW_SPOT_RADII * focal_spot_radius * camera_scale
        box_pixels = (
            camera_shape[1] - 2.0 * border_pixels,
            camera_shape[0] - 2.0 * border_pixels,
        )
        if box_pixels[0] <= 0 or box_pixels[1] <= 0:
            raise ValueError(
                "The sensor is smaller than twice the detection-window border. There is"
                " no room for the spot array."
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
        if roi_size is None:
            roi_size = max(
                int(round(minimum_separation / pitch.min())), _MIN_ROI_SIZE_PX
            )

        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)
        # Centred camera-pixel samples -> sensor centre -> focal-plane metres.
        sampled_pixels = self._sample_positions(
            number_of_spots, box_pixels, minimum_separation * camera_scale, generator
        )
        sampled_pixels = sampled_pixels + torch.tensor(
            [camera_shape[1] / 2.0, camera_shape[0] / 2.0],
            device=self.device, dtype=sampled_pixels.dtype,
        )
        model_pixels = AffineTransform.from_matrix(transform).transform_points(
            sampled_pixels.cpu().numpy()
        )
        metres = np.column_stack(
            [
                (model_pixels[:, 0] - resolution_out[1] / 2) * pixel_size_out[1],
                (model_pixels[:, 1] - resolution_out[0] / 2) * pixel_size_out[0],
            ]
        )
        # Drop targets the SLM cannot reach (beyond its Nyquist deflection).
        addressable = self.slm_camera_model.addressable_half_extent()
        reachable = (np.abs(metres[:, 0]) <= _ADDRESSABLE_MARGIN * addressable[0]) & (
            np.abs(metres[:, 1]) <= _ADDRESSABLE_MARGIN * addressable[1]
        )
        metres = metres[reachable]
        # Need a small margin above the affine minimum to fit robustly.
        if len(metres) < 4:
            raise RuntimeError(
                "Fewer than 4 spot targets fall within the SLM's addressable "
                "range; the sensor may lie largely outside it."
            )
        target_positions = torch.tensor(
            metres, device=self.device, dtype=torch.float64
        )

        if randomize_phases:
            target_phases = 2 * torch.pi * torch.rand(
                len(metres), generator=generator, device=self.device
            )
        else:
            target_phases = torch.zeros(len(metres), device=self.device)

        # Superposition hologram for the spot array.
        slm_phase = LinearSuperpositionPhaseRetriever(
            self.slm_camera_model, target_positions, target_phases=target_phases
        ).retrieve_phase()

        # Simulated image (for the record / visualizer).
        self.slm_camera_model.virtual_slm.set_phase(slm_phase)
        simulated_image = gpu_to_numpy(self.slm_camera_model().intensity)

        # Display the array and expose for it.
        self.slm.set_phase(gpu_to_numpy(slm_phase))
        if exposure_time is None:
            self._expose_for_array(zeroth_mask)
        else:
            self.camera.set_exposure(exposure_time)
        camera_image = self.camera.get_image()
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

        # Repeat for the simulated image.
        simulated_pitch = np.asarray([pixel_size_out[1], pixel_size_out[0]])
        simulated_zod = (resolution_out[1] // 2, resolution_out[0] // 2)
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
                f"Only {len(simulated_spots.points)} spots were detected and fitted in "
                "the simulated image; need at least 3 for an affine transform."
            )

        targets = gpu_to_numpy(target_positions).astype(np.float64)
        camera_indices, camera_targets = self._match_targets(
            np.asarray(camera_spots.points, dtype=np.float64),
            targets,
            expected_scale=camera_scale,
            tolerance=(
                _MATCH_TOLERANCE_FRACTION * minimum_separation / float(pitch.min())
            ),
        )
        simulated_indices, simulated_targets = self._match_targets(
            np.asarray(simulated_spots.points, dtype=np.float64),
            targets,
            expected_scale=float(1.0 / simulated_pitch.mean()),
            tolerance=(
                _MATCH_TOLERANCE_FRACTION
                * minimum_separation
                / float(simulated_pitch.min())
            ),
        )
        camera_by_target = dict(zip(camera_targets, camera_indices))
        simulated_by_target = dict(zip(simulated_targets, simulated_indices))
        common_targets = sorted(set(camera_by_target) & set(simulated_by_target))
        if len(common_targets) < _MIN_AFFINE_POINTS:
            raise RuntimeError(
                f"Only {len(common_targets)} spots could be matched to targets in "
                "both the camera and the simulated image; need at least 3 for an "
                "affine transform."
            )

        detected_points = [
            camera_spots.points[camera_by_target[t]] for t in common_targets
        ]
        calculated_points = [
            simulated_spots.points[simulated_by_target[t]] for t in common_targets
        ]
        fit_parameters = [
            camera_spots.fit_parameters[camera_by_target[t]] for t in common_targets
        ]
        fit_covariances = [
            camera_spots.fit_covariances[camera_by_target[t]] for t in common_targets
        ]

        detected = np.asarray(detected_points, dtype=np.float64)
        calculated = np.asarray(calculated_points, dtype=np.float64)

        # Robust affine: least squares with iterative median-based outlier rejection.
        inliers = np.ones(len(detected), dtype=bool)
        affine = AffineTransform.fit(detected, calculated, robust=False)
        for _ in range(3):
            affine = AffineTransform.fit(
                detected[inliers], calculated[inliers], robust=False
            )
            residuals = np.linalg.norm(
                affine.transform_points(detected) - calculated, axis=1
            )
            threshold = max(5.0 * float(np.median(residuals[inliers])), 1.0)
            new_inliers = residuals <= threshold
            if np.array_equal(new_inliers, inliers):
                break
            inliers = new_inliers
        if inliers.sum() < _MIN_AFFINE_POINTS:
            raise RuntimeError(
                f"Only {int(inliers.sum())} of {len(detected_points)} matched "
                "spots are consistent with an affine transform. Need at least 3."
            )
        transform = affine.as_matrix(homogeneous=False)

        detected = detected[inliers]
        calculated = calculated[inliers]
        detected_points = [p for p, keep in zip(detected_points, inliers) if keep]
        calculated_points = [
            p for p, keep in zip(calculated_points, inliers) if keep
        ]
        fit_parameters = [p for p, keep in zip(fit_parameters, inliers) if keep]
        fit_covariances = [c for c, keep in zip(fit_covariances, inliers) if keep]
        waists = [float(popt[0]) for popt in fit_parameters]
        waist_variances = [float(pcov[0, 0]) for pcov in fit_covariances]

        # Camera detections that did not make it into the transform, kept for inspection
        # / visualization with the reason they were excluded.
        matched_camera = {camera_by_target[t] for t in common_targets}
        used_camera = {
            camera_by_target[t]
            for t, keep in zip(common_targets, inliers)
            if keep
        }
        excluded_points = list(camera_spots.rejected_peaks)
        excluded_reasons = ["fit rejected"] * len(camera_spots.rejected_peaks)
        for index, point in enumerate(camera_spots.points):
            if index in used_camera:
                continue
            excluded_points.append(point)
            excluded_reasons.append(
                "affine outlier" if index in matched_camera else "unmatched"
            )

        inverse_transform = (
            AffineTransform.from_matrix(transform).inverse().as_matrix(homogeneous=False)
        )
        reprojection_errors, reprojection_rms = self.calculate_reprojection_error(
            detected, calculated, transform
        )

        center = (resolution_out[0] // 2, resolution_out[1] // 2)
        zeroth_order_position = (
            inverse_transform[1, 0] * center[0]
            + inverse_transform[1, 1] * center[1]
            + inverse_transform[1, 2],
            inverse_transform[0, 0] * center[0]
            + inverse_transform[0, 1] * center[1]
            + inverse_transform[0, 2],
        )

        average_waist, average_waist_uncertainty = self._weighted_average(
            np.asarray(waists), np.asarray(waist_variances)
        )

        return CameraMapping(
            timestamp=datetime.now(),
            name="spot_array",
            transform=transform,
            inverse_transform=inverse_transform,
            detected_points=detected_points,
            calculated_points=calculated_points,
            camera_images=[masked_image],
            simulated_images=[simulated_image],
            zeroth_order_position=zeroth_order_position,
            focal_spot_radius=average_waist,
            reprojection_errors=reprojection_errors,
            reprojection_rms=reprojection_rms,
            spot_fit_parameters=fit_parameters,
            spot_fit_covariances=fit_covariances,
            average_waist=average_waist,
            average_waist_uncertainty=average_waist_uncertainty,
            zeroth_order_mask=zeroth_mask,
            excluded_points=excluded_points,
            excluded_reasons=excluded_reasons,
        )

    def _sample_positions(
        self,
        number_of_spots: int,
        extent: tuple[float, float],
        minimum_separation: float,
        generator: torch.Generator,
        max_attempts_per_spot: int = 2000,
    ) -> torch.Tensor:
        """Rejection-sample ``number_of_spots`` points uniformly in the box centred at 
        the origin, each at least ``minimum_separation`` from the rest."""
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
        background ``threshold``. Returns (x, y) pixel positions, brightest first."""
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
        Gaussian to each. Returns the fitted (x, y) pixel positions, the per-spot fit
        parameters / covariances (parallel lists), and the peaks whose fit was 
        rejected."""
        
        grid = get_spatial_grid(image.shape, list(pitch))
        noise_std = background_noise(image)
        peaks = self._detect_peaks(
            image,
            number_of_peaks=number_of_peaks,
            blank_radius=max(roi_size // 2, 4),
            threshold=snr_threshold * noise_std,
        )

        points: list[tuple[float, float]] = []
        fit_parameters: list[NDArray] = []
        fit_covariances: list[NDArray] = []
        rejected_peaks: list[tuple[float, float]] = []
        for peak in peaks:
            fit = self._fit_spot(
                image, grid, peak, roi_size, radius_guess, pitch,
                snr_threshold * noise_std,
            )
            if fit is None:
                rejected_peaks.append(peak)
                continue
            popt, pcov = fit
            points.append(
                metres_to_pixel((popt[1], popt[2]), pitch, image.shape)
            )
            fit_parameters.append(popt)
            fit_covariances.append(pcov)
        return _SpotDetections(
            points, fit_parameters, fit_covariances, rejected_peaks
        )

    @staticmethod
    def _similarity_from_pair(
        source: NDArray, destination: NDArray, chirality: int
    ) -> tuple[NDArray, NDArray]:
        """Similarity transform (isotropic scale + rotation + translation, with a
        reflection first when ``chirality`` is -1) mapping two source points onto two
        destination points. Returns ``(matrix, translation)`` such that
        ``mapped = points @ matrix.T + translation``."""
        flip = np.array([[1.0, 0.0], [0.0, float(chirality)]])
        source_vector = flip @ (source[1] - source[0])
        destination_vector = destination[1] - destination[0]
        scale = np.linalg.norm(destination_vector) / np.linalg.norm(source_vector)
        angle = np.arctan2(destination_vector[1], destination_vector[0]) - np.arctan2(
            source_vector[1], source_vector[0]
        )
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        matrix = scale * rotation @ flip
        translation = destination[0] - matrix @ source[0]
        return matrix, translation

    @classmethod
    def _match_targets(
        cls,
        detected: NDArray,
        targets: NDArray,
        expected_scale: float,
        tolerance: float,
        scale_tolerance: float = 0.3,
    ) -> tuple[NDArray, NDArray]:
        """Match detected camera peaks (pixels) to target positions (metres) without 
        assuming the camera orientation.

        Similarity hypotheses are generated from widely separated point pairs: the most
        separated detected pairs are tried against every target pair of compatible
        length (both orderings and both chiralities, so any rotation and a mirrored
        camera are handled). The hypothesis mapping the most targets onto detected
        points within ``tolerance`` pixels wins, and the final one-to-one assignment
        under it is solved with the Hungarian algorithm.

        Args:
            detected: Detected spot centres in camera pixels, shape (M, 2).
            targets: Target positions in focal-plane metres, shape (N, 2).
            expected_scale: Rough pixels-per-metre scale, used only to prune
                implausible target pairs (within ``scale_tolerance``).
            tolerance: Match radius in pixels; half the minimum spot separation
                makes the assignment unambiguous.
            scale_tolerance: Allowed fractional deviation from ``expected_scale``.

        Returns:
            ``(detected_indices, target_indices)`` of the matched pairs.
        """
        # The few most separated detected pairs anchor the hypotheses; more than one in
        # case the single most separated pair involves a spurious detection.
        separations = np.linalg.norm(
            detected[:, None, :] - detected[None, :, :], axis=-1
        )
        pair_rows, pair_columns = np.triu_indices(detected.shape[0], k=1)
        by_separation = np.argsort(separations[pair_rows, pair_columns])[::-1][:3]
        anchor_pairs = list(zip(pair_rows[by_separation], pair_columns[by_separation]))

        target_separations = np.linalg.norm(
            targets[:, None, :] - targets[None, :, :], axis=-1
        )
        target_rows, target_columns = np.triu_indices(targets.shape[0], k=1)

        best_transform = None
        best_inliers = 0
        for i, j in anchor_pairs:
            detected_pair = detected[[i, j]]
            length = separations[i, j]
            compatible = (
                np.abs(
                    target_separations[target_rows, target_columns] * expected_scale
                    - length
                )
                <= scale_tolerance * length
            )
            candidate_pairs = zip(target_rows[compatible], target_columns[compatible])
            for first, second in candidate_pairs:
                for a, b in ((first, second), (second, first)):
                    for chirality in (1, -1):
                        matrix, translation = cls._similarity_from_pair(
                            targets[[a, b]], detected_pair, chirality
                        )
                        mapped = targets @ matrix.T + translation
                        distances = np.linalg.norm(
                            mapped[:, None, :] - detected[None, :, :], axis=-1
                        )
                        inliers = int((distances.min(axis=1) < tolerance).sum())
                        if inliers > best_inliers:
                            best_inliers = inliers
                            best_transform = (matrix, translation)

        if best_transform is None or best_inliers < 3:
            raise RuntimeError(
                "Could not establish a correspondence between the detected spots "
                "and the targets. Check that the array is on the sensor and well "
                "exposed, or increase number_of_spots."
            )

        matrix, translation = best_transform
        mapped = targets @ matrix.T + translation
        cost = np.linalg.norm(mapped[:, None, :] - detected[None, :, :], axis=-1)
        target_indices, detected_indices = linear_sum_assignment(cost)
        matched = cost[target_indices, detected_indices] < tolerance

        # Refine: a similarity from one anchor pair captures neither shear nor a
        # field-dependent warp (e.g. aberrations displacing each spot by its local
        # wavefront tilt). Fit an affine to the current matches and re-assign, so spots
        # that only miss the rigid hypothesis still match.
        for _ in range(2):
            if matched.sum() < 3:
                break
            matching_affine = AffineTransform.fit(
                targets[target_indices[matched]],
                detected[detected_indices[matched]],
                robust=False,
            )
            mapped = matching_affine.transform_points(targets)
            cost = np.linalg.norm(mapped[:, None, :] - detected[None, :, :], axis=-1)
            target_indices, detected_indices = linear_sum_assignment(cost)
            matched = cost[target_indices, detected_indices] < tolerance

        return detected_indices[matched], target_indices[matched]

    def _fit_spot(
        self,
        image: NDArray,
        camera_grid: list[torch.Tensor],
        peak: tuple[float, float],
        roi_size: int,
        radius_guess: float,
        pitch: NDArray,
        amplitude_threshold: float,
    ) -> tuple[NDArray, NDArray] | None:
        """Fit a 2D Gaussian to the ROI around a detected peak. Return ``(popt, pcov)`` 
        or ``None`` if the fit is missing / poor (amplitude at or below 
        ``amplitude_threshold`` counts as background, not a spot)."""
        rows, cols = image.shape
        half = roi_size // 2
        top = int(np.clip(round(peak[1]) - half, 0, rows - 1))
        bottom = int(np.clip(round(peak[1]) + half, top + 1, rows))
        left = int(np.clip(round(peak[0]) - half, 0, cols - 1))
        right = int(np.clip(round(peak[0]) + half, left + 1, cols))
        roi = ROI.from_bounds(top, bottom, left, right)

        cropped_image = roi.crop(image)
        cropped_grid = [roi.crop(grid) for grid in camera_grid]
        # Blur only seeds the centroid guess. Scale it to the spot size (in camera
        # pixels), not the large default that would flatten a small per-spot ROI.
        blur_sigma = max(float(radius_guess / pitch.min()), 1.0)
        try:
            popt, pcov = fit_gaussian_beam_intensity(
                *cropped_grid,
                cropped_image,
                beam_radius_guess=radius_guess,
                blur_sigma=blur_sigma,
            )
        except (RuntimeError, ValueError):
            return None

        if (
            not np.isfinite(pcov).all()
            or popt[0] <= 0
            or popt[3] <= amplitude_threshold
        ):
            return None
        # Reject if the fitted centre wandered outside the ROI (spurious fit).
        detected = metres_to_pixel((popt[1], popt[2]), pitch, image.shape)
        off_x = abs(detected[0] - peak[0])
        off_y = abs(detected[1] - peak[1])
        if off_x > half or off_y > half:
            return None
        return popt, pcov

    @staticmethod
    def _weighted_average(
        values: NDArray, variances: NDArray
    ) -> tuple[float, float]:
        """Inverse-variance (uncertainty) weighted mean and its uncertainty."""
        weights = 1.0 / np.clip(variances, a_min=np.finfo(float).tiny, a_max=None)
        mean = float(np.sum(weights * values) / np.sum(weights))
        uncertainty = float(np.sqrt(1.0 / np.sum(weights)))
        return mean, uncertainty
