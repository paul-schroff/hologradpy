from typing import Literal
from datetime import datetime
import torch
import numpy as np
from numpy.typing import NDArray

from checkerboard import detect_checkerboard
from scipy.ndimage import gaussian_filter
from cv2 import estimateAffine2D, invertAffineTransform

from slmsuite.hardware.slms.slm import SLM
from slmsuite.hardware.cameras.camera import Camera

from .utils import get_diffraction_spot_position

from ...analysis.functions import checkerboard

from ...propagation.optical_systems import SLMFFT
from ...propagation.utils.tensor_utils import gpu_to_numpy, find_roi
from ...propagation.utils.optics_utils import (
    rectangular_mask,
    analytic_phase_guess,
)
from ...propagation.utils.fourier_utils import get_spatial_grid

from ...holography.phase_retrieval import CGPhaseRetriever
from ...holography.vortices.vortex_annihilator import VortexAnnihilator

from . import CameraMapper, CameraMapping

CheckerboardCenterType = (
    Literal["top-left", "top-right", "bottom-left", "bottom-right"]
    | tuple[float, float]
)


class CheckerboardMapper(CameraMapper):
    def __init__(
        self,
        slm: SLM,
        camera: Camera,
        slm_camera_model: SLMFFT,
        device: torch.device = "cpu",
    ) -> None:
        super().__init__(slm, camera, slm_camera_model)
        self.device = device

    def map_camera(
        self,
        number_of_squares: tuple[int, int] = (7, 9),
        square_size: int = 32,
        checkerboard_center: CheckerboardCenterType = "top-left",
        number_of_cg_iterations: int = 50,
        phase_guess: torch.Tensor | None = None,
        target_blur_width: float = 1.0,
        annihilate_vortices: bool = True,
    ) -> CameraMapping:
        """This function performs the camera mapping to obtain the coordinate
        transform between the camera pixels and the pixels of the simulated
        camera image. An SLM phase pattern is calculated for a checkerboard-
        shaped target using CG minimization and displayed on the SLM. The
        corners of the checkerboard in the resulting camera image are detected
        with sub-pixel-precision and fitted to the corners of the checkerboard
        in the simulated image using an affine transformation.

        Args:
            number_of_squares (tuple[int, int], optional): Number of squares
                in the checkerboard pattern in (rows, columns). Defaults to
                (7, 9).
            square_size (int, optional): Size of each square in the
                checkerboard pattern. Defaults to 32.
            checkerboard_center (CheckerboardCenterType, optional): Center of
                the checkerboard pattern. Options are "top-left", "top-right",
                "bottom-left", "bottom-right", relative to the zeroth-order.
                Can be specified as (x, y) coordinates in units of Fourier
                pixels.
            number_of_cg_iterations (int, optional): Number of iterations to
                run the conjugate gradient minimization for phase retrieval.
                Defaults to 50.
            phase_guess (torch.Tensor | None, optional): Initial guess for the
                SLM phase pattern. If None, a quadratic guess will be used
                based on the size of the checkerboard. Defaults to None.
            target_blur_width (float, optional): Standard deviation of the
                Gaussian blur applied to the target checkerboard pattern.
                Defaults to 1.0.
            annihilate_vortices (bool, optional): Whether to perform vortex
                annihilation after the initial phase retrieval. This makes
                the phase retrieval more robust to poor initial phase guesses.
                Defaults to True.

        Returns:
            CameraMapping: Dataclass containing the camera mapping results.
        """

        number_of_corners = tuple([i - 1 for i in number_of_squares])

        if isinstance(checkerboard_center, str):
            checkerboard_shift_x = int(square_size * (number_of_squares[1] / 2 + 2))
            checkerboard_shift_y = int(square_size * (number_of_squares[0] / 2 + 2))
            if checkerboard_center == "top-left":
                checkerboard_center = (
                    -checkerboard_shift_x,
                    -checkerboard_shift_y,
                )
            elif checkerboard_center == "top-right":
                checkerboard_center = (
                    checkerboard_shift_x,
                    -checkerboard_shift_y,
                )
            elif checkerboard_center == "bottom-left":
                checkerboard_center = (
                    -checkerboard_shift_x,
                    checkerboard_shift_y,
                )
            elif checkerboard_center == "bottom-right":
                checkerboard_center = (
                    checkerboard_shift_x,
                    checkerboard_shift_y,
                )
            else:
                raise ValueError(f"Invalid checkerboard_center: {checkerboard_center}")
        elif isinstance(checkerboard_center, tuple):
            pass
        else:
            raise TypeError(
                "checkerboard_center must be a string or a tuple of floats."
            )

        # Generating checkerboard target and signal region
        target = checkerboard(
            self.slm_camera_model[-1].resolution_out,
            number_of_squares=number_of_squares,
            square_size=square_size,
            dark_square_brightness=0.0,
            light_square_brightness=1.0,
            shift_x=checkerboard_center[0],
            shift_y=checkerboard_center[1],
        )

        target = torch.tensor(
            gaussian_filter(target, target_blur_width),
            device=self.device,
            dtype=torch.float32,
        )

        signal_region_roi = find_roi(target, pad=square_size * 2)

        signal_region = torch.zeros_like(target, device=self.device) > 1
        signal_region[
            signal_region_roi[0] : signal_region_roi[1],
            signal_region_roi[2] : signal_region_roi[3],
        ] = True

        # Finding position of zeroth-order diffraction spot on camera
        simulation_pixel_size = self.slm_camera_model[-1].pixel_size_out.tolist()[0]

        checkerboard_center_meters = tuple(
            [checkerboard_center[i] * simulation_pixel_size[::-1][i] for i in range(2)]
        )

        (spot_position_x, spot_position_y), focal_spot_radius, _ = (
            get_diffraction_spot_position(
                self.slm,
                self.camera,
                linear_phase_tilt=checkerboard_center_meters,
                focal_length=self.slm_camera_model[-1].focal_length,
            )
        )

        # Defining region of interest in the simulated camera image
        roi_width = simulation_pixel_size[1] * square_size * (number_of_squares[1] + 1)
        roi_height = simulation_pixel_size[0] * square_size * (number_of_squares[0] + 1)

        roi_mask_simulation = rectangular_mask(
            *self.slm_camera_model[-1].get_spatial_grid_output(),
            width=torch.tensor(roi_width),
            height=torch.tensor(roi_height),
            shift_x=checkerboard_center_meters[0],
            shift_y=checkerboard_center_meters[1],
        )

        # Defining the region of interest in the camera image
        camera_grid = get_spatial_grid(
            self.camera.shape,
            [self.camera.pitch_um[i] * 1e-6 for i in range(2)],
            device=self.device,
        )
        roi_mask_camera = rectangular_mask(
            *camera_grid,
            width=roi_width,
            height=roi_height,
            shift_x=spot_position_x,
            shift_y=spot_position_y,
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
                *self.slm_camera_model.virtual_slm.get_spatial_grid_input(),
                tilt_x=checkerboard_center_meters[0],
                tilt_y=checkerboard_center_meters[1],
                curvature=curvature,
                aspect_ratio=aspect_ratio,
                focal_length=self.slm_camera_model.fourier_lens.focal_length,
                wavenumber=self.slm_camera_model.init_field.wavenumber,
                tilt_units="metres",
                curvature_units="radians_per_pixel_squared",
            )

        # Performing phase retrieval to find SLM phase pattern for checkerboard
        # target
        phase_retriever = CGPhaseRetriever(
            self.slm_camera_model,
            target=target,
            signal_region=signal_region,
            init_slm_phase=phase_guess,
            device=self.device,
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

        # Capturing camera image
        averaged_camera_image = self.capture_phase_shifted_image(
            gpu_to_numpy(slm_phase), number_of_shifts=10
        )

        # Detecting checkerboard corners in the captured and simulated images
        detected_corners, detected_score = self.detect_checkerboard(
            averaged_camera_image * gpu_to_numpy(roi_mask_camera),
            number_of_corners=number_of_corners,
            number_of_attempts=3,
        )

        calculated_corners, calculated_score = self.detect_checkerboard(
            simulated_camera_image * gpu_to_numpy(roi_mask_simulation),
            number_of_corners=number_of_corners,
            number_of_attempts=3,
        )

        # Fitting affine transformation to detected and calculated corners
        transform, _ = estimateAffine2D(detected_corners, calculated_corners)
        inverse_transform = invertAffineTransform(transform)

        center = (
            self.slm_camera_model[-1].resolution_out[0] // 2,
            self.slm_camera_model[-1].resolution_out[1] // 2,
        )

        zeroth_order_position = (
            inverse_transform[1, 0] * center[0]
            + inverse_transform[1, 1] * center[1]
            + inverse_transform[1, 2],
            inverse_transform[0, 0] * center[0]
            + inverse_transform[0, 1] * center[1]
            + inverse_transform[0, 2],
        )

        # Generating and returning CameraMapping dataclass.
        return CameraMapping(
            timestamp=datetime.now(),
            name="checkerboard",
            transform=transform,
            inverse_transform=inverse_transform,
            detected_points=detected_corners,
            calculated_points=calculated_corners,
            camera_images=[averaged_camera_image],
            simulated_images=[simulated_camera_image],
            zeroth_order_position=zeroth_order_position,
            focal_spot_radius=focal_spot_radius,
        )

    def capture_phase_shifted_image(
        self,
        slm_phase: NDArray,
        number_of_shifts: int = 10,
    ) -> NDArray:
        """Displays a phase pattern on the SLM with multiple phase shifts and
        captures and averages the resulting camera images to reduce fringes
        caused by pixel crosstalk on the SLM.

        Args:
            slm_phase (torch.Tensor): SLM phase pattern to display.
            number_of_shifts (int, optional): Number of phase shifts to apply.
                Defaults to 10.

        Returns:
            NDArray: Averaged camera image.
        """
        averaged_camera_image = np.zeros(self.camera.shape)
        for i in range(number_of_shifts):
            shifted_slm_phase = slm_phase + i * 2 * np.pi / number_of_shifts
            self.slm.set_phase(shifted_slm_phase)
            averaged_camera_image += self.camera.get_image() / number_of_shifts
        return averaged_camera_image

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
            image (NDArray): 2D array containing the image to detect the
                checkerboard pattern.
            number_of_corners (tuple[int, int]): Number of corners in the
                checkerboard pattern (rows, columns).
            number_of_attempts (int): Number of attempts to detect the
                checkerboard with increasing blur.

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
