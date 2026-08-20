from __future__ import annotations

from datetime import datetime
import torch
import numpy as np
from numpy.typing import NDArray

from checkerboard import detect_checkerboard
from scipy.ndimage import gaussian_filter
from ....geometry import AffineTransform

from ....hardware import Camera, SLM
from ....roi import ROI

from ....profiles.amplitude import checkerboard
from ..coarse_mapping.coarse_mapper import CoarseMapper

from ....optics.systems import SLMFFT
from ....utils import gpu_to_numpy
from ....profiles.phase import analytic_phase_guess
from ....profiles.masks import rectangular_mask
from ....grids import get_spatial_grid, pixel_to_metres, plane_center

from ....holography.phase_retrieval import GradientPhaseRetriever
from ....holography.vortices.vortex_annihilator import VortexAnnihilator

from ..abstract import CameraMapper, CameraMapping
from ..mapping import FocalSpotFit, MappingFit
from ..visualizer import CameraMappingVisualizationData

DETECTOR_MARGIN = 20


class CheckerboardMapper(CameraMapper):
    def __init__(
        self,
        slm: SLM,
        camera: Camera,
        slm_camera_model: SLMFFT,
        verbose: bool = True,
    ) -> None:
        super().__init__(slm, camera, slm_camera_model)
        self.device = slm_camera_model.device
        self.verbose = verbose

    def map_camera(
        self,
        number_of_squares: tuple[int, int] = (7, 9),
        square_size: int | None = None,
        number_of_cg_iterations: int = 50,
        phase_guess: torch.Tensor | None = None,
        target_blur_width: float = 1.0,
        annihilate_vortices: bool = True,
        exposure_time: float | None = None,
        coarse_mapping: CameraMapping | None = None,
    ) -> CameraMapping:
        """Map the camera from a checkerboard target.
        
        .. warning::

            This mapper has a chicken-and-egg problem: it is sensitive to aberrations
            and only works with a calibrated wavefront. It is included for legacy
            reasons. :class:`SpotArrayMapper` is more robust, more accurate, and does
            not need a calibrated wavefront.

        An SLM phase for a checkerboard target is computed (CG phase retrieval) and
        displayed. The checkerboard corners are detected sub-pixel in both the captured
        and simulated images and fitted with an affine transform. The board is placed
        via a coarse mapping so it lands on the sensor, clears the zeroth order, and
        stays within a quarter of the SLM's Nyquist adressable extent.

        Args:
            number_of_squares: Squares in (rows, columns). Defaults to (7, 9).
            square_size: Square size in model-plane pixels. If None (default), it is
                chosen from the coarse mapping (<= 1/4 the Nyquist-rectangle width,
                ~half the sensor). The board is then placed automatically in the
                sensor quadrant opposite the zeroth order (or the sensor center if
                the zeroth order is off it).
            number_of_cg_iterations: CG phase-retrieval iterations. Defaults to 50.
            phase_guess: Initial SLM phase guess. If None, a quadratic guess.
            target_blur_width: Gaussian blur of the target. Defaults to 1.0.
            annihilate_vortices: Run vortex annihilation after CG retrieval. Defaults to
                True.
            exposure_time: Camera exposure in seconds. If None (default), the camera 
                autoexposes.
            coarse_mapping: Result of :meth:`CoarseMapper.map_camera`. Built and run
                internally when None (default).

        Returns:
            CameraMapping with the affine transform and corner correspondences.
        """
        number_of_corners = tuple([i - 1 for i in number_of_squares])

        if coarse_mapping is None:
            coarse_mapping = CoarseMapper(
                self.slm, self.camera, self.slm_camera_model
            ).map_camera()

        lens = self.slm_camera_model.fourier_lens
        simulation_pixel_size = lens.pixel_size_out.tolist()[0]  # (y, x) metres
        resolution_out = tuple(lens.resolution_out)  # (height, width)
        focal_length = float(lens.focal_length)
        pitch = self.camera.pixel_size[::-1]  # (x, y) for Cartesian geometry
        camera_shape = tuple(self.camera.resolution)  # (height, width)
        focal_spot_radius = float(abs(coarse_mapping.spot_fit.waist))

        # Choose the board center (model-pixel shift), the square size, and the
        # camera-pixel center the board lands on from the coarse transform.
        board_shift, square_size, center_camera = self._place_checkerboard(
            number_of_squares,
            square_size,
            coarse_mapping,
            focal_length,
            simulation_pixel_size,
            resolution_out,
            camera_shape,
        )

        # Generating checkerboard target and signal region (model plane).
        target = checkerboard(
            resolution_out,
            number_of_squares=number_of_squares,
            square_size=square_size,
            dark_square_brightness=0.0,
            light_square_brightness=1.0,
            shift_x=board_shift[0],
            shift_y=board_shift[1],
        )
        target = torch.tensor(
            gaussian_filter(target, target_blur_width),
            device=self.device,
            dtype=torch.float32,
        )
        signal_region_roi = ROI.detect(target, pad=square_size * 2)
        signal_region = torch.zeros_like(target, device=self.device) > 1
        signal_region[signal_region_roi.rows, signal_region_roi.columns] = True

        board_shift_metres = (
            board_shift[0] * simulation_pixel_size[1],
            board_shift[1] * simulation_pixel_size[0],
        )

        # ROI for corner detection: The model plane is centered on the board, and the 
        # camera plane is centered where the board lands.
        roi_width = simulation_pixel_size[1] * square_size * (number_of_squares[1] + 1)
        roi_height = simulation_pixel_size[0] * square_size * (number_of_squares[0] + 1)
        roi_mask_simulation = rectangular_mask(
            *lens.get_spatial_grid_output(),
            width=torch.tensor(roi_width),
            height=torch.tensor(roi_height),
            shift_x=board_shift_metres[0],
            shift_y=board_shift_metres[1],
        )

        inverse = np.asarray(coarse_mapping.inverse_transform, dtype=np.float64)
        magnification = float(np.sqrt(abs(np.linalg.det(inverse[:, :2]))))
        squares_plus = (number_of_squares[1] + 1, number_of_squares[0] + 1)
        center_metres = pixel_to_metres(
            center_camera, self.camera.pixel_size, camera_shape
        )
        camera_grid = get_spatial_grid(
            self.camera.resolution, self.camera.pixel_size, device=self.device
        )
        roi_mask_camera = rectangular_mask(
            *camera_grid,
            width=square_size * squares_plus[0] * magnification * pitch[0],
            height=square_size * squares_plus[1] * magnification * pitch[1],
            shift_x=center_metres[0],
            shift_y=center_metres[1],
        )

        # Calculating SLM phase guess to seed conjugate gradient minimization
        # if phase_guess is not provided
        if phase_guess is None:
            aspect_ratio = 1 / (1 + number_of_squares[1] / number_of_squares[0])
            curvature = (
                1.8e-6
                * square_size
                * (number_of_squares[0] ** 2 + number_of_squares[1] ** 2) ** 0.5
            )

            phase_guess = analytic_phase_guess(
                *self.slm_camera_model.virtual_slm.get_slm_grid(),
                tilt_x=board_shift_metres[0],
                tilt_y=board_shift_metres[1],
                curvature=curvature,
                aspect_ratio=aspect_ratio,
                focal_length=self.slm_camera_model.fourier_lens.focal_length,
                wavenumber=self.slm_camera_model.init_field.wavenumber,
                tilt_units="metres",
                curvature_units="radians_per_pixel_squared",
            )

        # Performing phase retrieval to find SLM phase pattern for checkerboard
        # target
        phase_retriever = GradientPhaseRetriever(
            self.slm_camera_model,
            target=target,
            signal_region=signal_region,
            init_slm_phase=phase_guess,
        )

        slm_phase = phase_retriever.retrieve_phase(number_of_cg_iterations)

        # Optional vortex removal
        if annihilate_vortices:
            vortex_annihilator = VortexAnnihilator(phase_retriever)
            vortex_annihilator.annihilate_vortices(
                target_intensity_threshold=0.2,
                max_iterations=5,
                cg_iterations=20,
            )

            slm_phase = phase_retriever.retrieve_phase(number_of_cg_iterations)

        simulated_camera_image = gpu_to_numpy(self.slm_camera_model().intensity)

        self.slm.set_phase(gpu_to_numpy(slm_phase))
        if exposure_time is not None:
            self.camera.set_exposure(exposure_time)
        else:
            # Autoexpose on a region around the board (center_camera is (x, y)).
            board_roi = ROI.centered(
                (center_camera[1], center_camera[0]),  # (row, col)
                (
                    square_size * squares_plus[1] * magnification,  # height
                    square_size * squares_plus[0] * magnification,  # width
                ),
            )

            self.camera.autoexpose(
                set_fraction=0.95,
                roi=board_roi,
                raise_on_rail=False,
                verbose=self.verbose,
            )

        # Capturing camera image
        averaged_camera_image = self.capture_phase_shifted_image(
            gpu_to_numpy(slm_phase), number_of_shifts=10
        )

        # Detecting checkerboard corners in the captured and simulated images
        detected_corners, detected_score = self.detect_in_region(
            averaged_camera_image,
            gpu_to_numpy(roi_mask_camera),
            number_of_corners=number_of_corners,
        )

        calculated_corners, calculated_score = self.detect_in_region(
            simulated_camera_image,
            gpu_to_numpy(roi_mask_simulation),
            number_of_corners=number_of_corners,
        )

        # A failed detection returns a scalar rather than an (N, 2) array.
        number_expected = number_of_corners[0] * number_of_corners[1]
        for label, corners in (
            ("camera", detected_corners),
            ("simulated", calculated_corners),
        ):
            if np.asarray(corners).shape != (number_expected, 2):
                raise RuntimeError(
                    f"The checkerboard could not be detected in the {label} image "
                    f"({number_expected} inner corners expected). The board is likely "
                    "too dim or low contrast. Increase the exposure or square_size, "
                    "improve the wavefront calibration, or use the more robust "
                    "SpotArrayMapper."
                )

        # Fitting affine transformation to detected and calculated corners
        affine = AffineTransform.fit(detected_corners, calculated_corners)
        transform = affine.as_matrix(homogeneous=False)
        reprojection_errors, reprojection_rms = self.calculate_reprojection_error(
            detected_corners, calculated_corners, transform
        )

        zeroth_order_position = CameraMapping.zeroth_order_from(
            affine, self.slm_camera_model.fourier_lens.resolution_out
        )

        # Generating and returning CameraMapping dataclass.
        return CameraMapping(
            timestamp=datetime.now(),
            name="checkerboard",
            transform=transform,
            detected_points=detected_corners,
            calculated_points=calculated_corners,
            zeroth_order_position=zeroth_order_position,
            spot_fit=FocalSpotFit(waist=focal_spot_radius),
            fit=MappingFit(
                reprojection_errors=reprojection_errors,
                reprojection_rms=reprojection_rms,
            ),
            visualization_data=CameraMappingVisualizationData(
                camera_image=averaged_camera_image,
                simulated_image=simulated_camera_image,
            ),
        )

    def _place_checkerboard(
        self,
        number_of_squares: tuple[int, int],
        square_size: int | None,
        coarse_mapping: CameraMapping,
        focal_length: float,
        pixel_size_out: list[float],
        resolution_out: tuple[int, int],
        camera_shape: tuple[int, int],
    ) -> tuple[tuple[int, int], int, tuple[float, float]]:
        """Choose the checkerboard's model-pixel center shift, square size, and the
        camera-pixel center it lands on, from the coarse transform.

        ``square_size`` (model px) defaults so the board width is <= 1/4 of the
        Nyquist-rectangle width and spans ~half the sensor. The board is placed in
        the sensor quadrant opposite the zeroth order (clear of it), or at the
        sensor center when the zeroth order is off the sensor.
        """
        rows, columns = number_of_squares
        transform = np.asarray(coarse_mapping.transform, dtype=np.float64)
        inverse = np.asarray(coarse_mapping.inverse_transform, dtype=np.float64)
        magnification = float(np.sqrt(abs(np.linalg.det(inverse[:, :2]))))
        height, width = camera_shape
        model_center = np.array(plane_center(resolution_out), dtype=float)
        addressable = self.slm_camera_model.addressable_half_extent()  # (x, y) m

        if square_size is None:
            # <= 1/4 of the Nyquist-rectangle width per axis, and ~half the sensor.
            cap_x = 0.5 * addressable[0] / (columns * pixel_size_out[1])
            cap_y = 0.5 * addressable[1] / (rows * pixel_size_out[0])
            cap_sensor = 0.5 * min(camera_shape) / (max(rows, columns) * magnification)
            square_size = int(max(4, min(cap_x, cap_y, cap_sensor)))

        board_radius = 0.5 * magnification * float(
            np.hypot(columns * square_size, rows * square_size)
        ) + square_size * magnification
        low = np.array([board_radius, board_radius])
        high = np.array([width - 1 - board_radius, height - 1 - board_radius])
        if (low > high).any():
            raise ValueError(
                "The checkerboard does not fit on the sensor. Reduce number_of_squares "
                "or square_size."
            )

        zeroth = np.array(coarse_mapping.zeroth_order_xy)
        dc_on_sensor = 0.0 <= zeroth[0] < width and 0.0 <= zeroth[1] < height

        if dc_on_sensor:
            # The sensor corner opposite the zeroth order.
            center_camera = np.array(
                [
                    high[0] if zeroth[0] < width / 2 else low[0],
                    high[1] if zeroth[1] < height / 2 else low[1]
                ]
            )
        else:
            center_camera = np.array([(width - 1) / 2.0, (height - 1) / 2.0])

        if dc_on_sensor and np.linalg.norm(center_camera - zeroth) < board_radius:
            raise ValueError(
                "The checkerboard cannot be placed clear of the zeroth order on "
                "this sensor. Reduce number_of_squares / square_size."
            )
        model_point = AffineTransform.from_matrix(transform).transform_points(
            center_camera
        )[0]
        shift = (
            int(round(model_point[0] - model_center[0])),
            int(round(model_point[1] - model_center[1])),
        )
        return shift, square_size, (float(center_camera[0]), float(center_camera[1]))

    def capture_phase_shifted_image(
        self,
        slm_phase: NDArray,
        number_of_shifts: int = 10,
    ) -> NDArray:
        """Displays a phase pattern on the SLM with multiple phase shifts and
        captures and averages the resulting camera images to reduce fringes
        caused by pixel crosstalk on the SLM.

        Args:
            slm_phase: SLM phase pattern to display.
            number_of_shifts: Number of phase shifts to apply. Defaults to 10.

        Returns:
            NDArray: Averaged camera image.
        """
        averaged_camera_image = np.zeros(self.camera.resolution)
        for i in range(number_of_shifts):
            shifted_slm_phase = slm_phase + i * 2 * np.pi / number_of_shifts
            self.slm.set_phase(shifted_slm_phase)
            averaged_camera_image += self.camera.get_image() / number_of_shifts
        return averaged_camera_image

    @classmethod
    def detect_in_region(
        cls,
        image: NDArray,
        mask: NDArray,
        number_of_corners: tuple[int, int],
        number_of_attempts: int = 3,
    ) -> tuple[NDArray, float]:
        """Detect the checkerboard within ``mask``, and return its corners in the
        coordinates of the whole ``image``.

        Args:
            image: The whole frame, from the camera or from the model.
            mask: True over the region the board was placed in.
            number_of_corners: Inner corners as (rows, columns).
            number_of_attempts: Detection attempts with increasing blur.

        Returns:
            tuple[NDArray, float]: Corner coordinates in ``image`` and the detection
                score. A failed detection returns a scalar rather than corners.
        """
        region = ROI.detect(mask.astype(float), threshold=0.5, pad=DETECTOR_MARGIN)
        corners, score = cls.detect_checkerboard(
            region.crop(image * mask),
            number_of_corners=number_of_corners,
            number_of_attempts=number_of_attempts,
        )
        corners = np.asarray(corners)
        if corners.ndim == 2:
            corners = corners + (region.left_column, region.top_row)
        return corners, score

    @staticmethod
    def detect_checkerboard(
        image: NDArray,
        number_of_corners: tuple[int, int],
        number_of_attempts: int = 3,
    ) -> tuple[NDArray, float]:
        """Detects the corners of a checkerboard pattern in an image with
        sub-pixel precision. If the checkerboard cannot be detected, the image
        will be blurred with increasing kernel sizes until detection is
        successful or the maximum number of attempts is reached.

        Args:
            image: 2D array containing the image to detect the checkerboard
                pattern.
            number_of_corners: Number of corners in the checkerboard pattern
                (rows, columns).
            number_of_attempts: Number of attempts to detect the checkerboard with
                increasing blur.

        Returns:
            tuple[NDArray, float]: Detected corner coordinates and the
                detection score.
        """
        for i in range(number_of_attempts):
            kernel_width = i
            blurred_image = gaussian_filter(image, kernel_width)
            blurred_image_normalized = blurred_image / blurred_image.max() * 255

            corners, score = detect_checkerboard(
                blurred_image_normalized, (number_of_corners[1], number_of_corners[0])
            )
            corners = np.squeeze(corners)

            if corners.ndim == 0:
                print("No checkerboard could be detected.")
            else:
                print("Checkerboard detected.")
                break
        return corners, score
