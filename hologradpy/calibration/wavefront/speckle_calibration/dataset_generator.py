from __future__ import annotations
from typing import Dict, List, OrderedDict

from datetime import datetime
from copy import copy

import numpy as np
import torch
from numpy.typing import NDArray

from opensimplex import noise4array, random_seed

from .calibration_dataset import (
    DatasetDescriptor,
    TrainingSampleFilenames,
)

from slmsuite.hardware.slms.slm import SLM
from slmsuite.hardware.cameras.camera import Camera

from ..abstract import WavefrontCalibrationData

from ...camera_mapping import CameraMapping

from ....propagation.utils.optics_utils import circular_mask
from ....propagation.utils.tensor_utils import get_device, find_roi
from ....propagation.utils.fourier_utils import get_spatial_grid
from ....propagation.optical_systems import SLMFourierLensModel
from ....propagation.propagators import FourierLensFFT
from ....propagation.virtual_slms import VirtualSLM
from ....propagation.elements import ConstantSLMField, PartialAffineTransform


class DatasetGenerator:
    def __init__(
        self,
        slm: SLM,
        camera: Camera,
        camera_mapping: CameraMapping,
        focal_length: float,
        directory: str,
        number_of_random_patterns: int = 1,
        padded_resolution: tuple[int, int] = (2048, 2048),
        device: str = None,
    ) -> None:
        self.slm: SLM = slm
        self.camera: Camera = camera
        self.camera_mapping: CameraMapping = camera_mapping
        self.directory: str = directory
        self.number_of_random_patterns: int = number_of_random_patterns
        self.benchmark_calibration: WavefrontCalibrationData | None = None
        self.number_of_phase_patterns: int = self.number_of_random_patterns

        if device is None:
            device = get_device()

        self.number_of_masks: int = 0
        self.number_of_phase_patterns: int = self.number_of_random_patterns

        self.camera_background_image: NDArray[np.float_] = np.zeros(self.camera.shape)

        self.exposure_time: float = 0.0

        self.data_filenames: List[TrainingSampleFilenames] = []

        self.phase_pattern_type: str = "simplex"
        self.metadata: Dict[str, tuple[float, float] | float | None] = {
            "feature_size": None,
            "scale": None,
        }

        virtual_slm: VirtualSLM = VirtualSLM(self.slm)

        constant_field: ConstantSLMField = ConstantSLMField(
            init_field=torch.ones(self.slm.shape),
            pixel_pitch=virtual_slm.slm.pitch_um[0] * 1e-6,
        )

        fourier_lens: FourierLensFFT = FourierLensFFT(
            focal_length=focal_length,
            wavelength=virtual_slm.slm.wav_um * 1e-6,
            resolution_in=virtual_slm.slm.shape,
            pixel_pitch_in=virtual_slm.slm.pitch_um[0] * 1e-6,
            padded_resolution=padded_resolution,
            fft_kwargs={"norm": "ortho"},
        )

        affine_transform: PartialAffineTransform = PartialAffineTransform(
            resolution_in=fourier_lens.padded_resolution,
            resolution_out=self.camera.shape,
            pixel_size_in=fourier_lens.pixel_size_out,
            pixel_size_out=tuple(self.camera.pitch_um * 1e-6),
            device=device,
            verbose=False,
        )

        self.slm_camera_model: SLMFourierLensModel = SLMFourierLensModel(
            OrderedDict(
                [
                    ("virtual_slm", virtual_slm),
                    ("constant_field", constant_field),
                    ("fourier_lens", fourier_lens),
                    ("affine_transform", affine_transform),
                ]
            )
        )

        self.roi_mask: torch.Tensor[torch.bool] = torch.ones(
            self.camera.shape, dtype=torch.bool
        )

    def generate_phase_patterns(
        self,
        extent: tuple[float, float] | None = None,
        benchmark_calibration: WavefrontCalibrationData | None = None,
    ) -> None:
        """Generate a set of random phase patterns using Simplex noise."""
        if extent is None:
            extent = tuple(
                self.camera.shape[i] * self.camera.pitch_um[i] * 1e-6 for i in range(2)
            )

        self.metadata["simplex_extent"] = extent

        self.benchmark_calibration = benchmark_calibration
        if self.benchmark_calibration is None:
            benchmark_phase = np.zeros(self.slm.shape)
        else:
            # TODO: Fix WavefrontCalibrationData
            benchmark_phase = np.angle(self.benchmark_calibration.field)

        simplex_4d_norm = 1 / 0.86875

        feature_size = tuple(
            simplex_4d_norm
            * self.slm.wav_um
            * 1e-6
            * self.slm_camera_model.fourier_lens.focal_length
            / (1e-6 * self.slm.pitch_um[i] * extent[i])
            for i in range(2)
        )

        random = np.random.rand(2) * self.slm.shape[0] / 2 / feature_size[0]
        z = np.asarray(
            [
                random[0],
            ]
        )
        w = np.asarray(
            [
                random[1],
            ]
        )

        grating = np.zeros(self.slm.shape)
        grating[:, ::2] = np.pi

        for i in range(self.number_of_random_patterns):
            print(
                f"Generating phase pattern {i + 1} of {self.number_of_random_patterns}."
            )

            simplex_coords = tuple(
                (np.arange(self.slm.shape[i]) - self.slm.shape[i] / 2) / feature_size[i]
                for i in range(2)
            )[::-1]

            random_seed()
            simplex_noise = noise4array(*simplex_coords, z, w).squeeze()

            simplex_noise = simplex_noise * simplex_4d_norm * np.pi + np.pi

            simplex_noise = np.remainder(simplex_noise + benchmark_phase, 2 * np.pi)

            phase_filename = f"phase_pattern_{i}.npy"

            sample_filename = TrainingSampleFilenames(
                phase_pattern=phase_filename,
            )

            np.save(self.directory + phase_filename, np.copy(simplex_noise))
            self.data_filenames.append(sample_filename)

        # Generating the ROI mask based on the camera mapping and extent
        camera_grid = get_spatial_grid(self.camera.shape, self.camera.pitch_um * 1e-6)

        # TODO: Fix this
        shift_pixels = tuple(
            self.camera_mapping.zeroth_order_position[i] - self.camera.shape[i] // 2
            for i in range(2)
        )

        shift_y, shift_x = [
            shift_pixels[i] * self.camera.pitch_um[i] * 1e-6 for i in range(2)
        ]

        simplex_mask = circular_mask(
            *camera_grid,
            extent[0],
            shift_x=shift_x,
            shift_y=shift_y,
        )

        zeroth_order_mask = circular_mask(
            *camera_grid,
            4 * self.camera_mapping.focal_spot_radius,
            shift_x=shift_x,
            shift_y=shift_y,
        )

        self.roi_mask = simplex_mask * ~zeroth_order_mask

    def capture_camera_images(
        self, capture_background_image: bool = True
    ) -> DatasetDescriptor:
        for i, filename in enumerate(self.data_filenames):
            print(
                f"Displaying phase pattern {i + 1} of {self.number_of_phase_patterns}"
            )

            phase_pattern = np.load(self.directory + filename["phase_pattern"])

            self.slm.set_phase(phase_pattern, phase_correct=False)

            if i == 0:
                # Find ROI and reformat to window
                # (x_center, width, y_center, height)
                roi = find_roi(self.roi_mask, pad=0)

                width, height = roi[3] - roi[2], roi[1] - roi[0]
                window = (roi[2] + width // 2, width, roi[0] + height // 2, height)
                print(roi)
                print(window)

                self.exposure_time = self.camera.autoexposure(
                    set_fraction=0.7, window=window
                )

            camera_image_filename = f"camera_image_{i}.npy"

            print(f"Capturing camera image {i + 1} of {self.number_of_phase_patterns}")

            camera_image = self.camera.get_image()

            np.save(self.directory + camera_image_filename, np.copy(camera_image))

            self.data_filenames[i]["camera_image"] = copy(camera_image_filename)

        if capture_background_image:
            message = input(
                'Block the laser beam and enter "Y" to capture the '
                'background image. Enter "N" to skip capturing a '
                "background image: "
            )
            if message.lower() == "y":
                self.camera_background_image = self.camera.get_image()
            else:
                if message.lower() == "n":
                    print("Skipping background image capture.")
                else:
                    print("Invalid input. Skipping background image capture.")

                self.camera_background_image = np.zeros(self.camera.shape)

        return DatasetDescriptor(
            timestamp=datetime.now(),
            phase_pattern_type=self.phase_pattern_type,
            directory=self.directory,
            number_of_patterns=self.number_of_random_patterns,
            slm_data=self.slm.pickle(),
            camera_data=self.camera.pickle(),
            camera_mapping=self.camera_mapping,
            roi_mask=self.roi_mask,
            data_filenames=self.data_filenames,
            camera_background_image=self.camera_background_image,
            benchmark_calibration=self.benchmark_calibration,
            metadata=self.metadata,
        )
