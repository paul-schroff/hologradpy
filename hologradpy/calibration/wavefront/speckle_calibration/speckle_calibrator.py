from __future__ import annotations
from typing import Sequence, Literal
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from torch.utils.data import DataLoader

from .propagation_trainers import PropagationTrainer, SmoothWavefrontTrainer
from .dataset_generator import DatasetGenerator
from .calibration_dataset import DatasetDescriptor

from ..abstract import WavefrontCalibratorBase, WavefrontCalibrationData

from ...camera_mapping import CameraMapping

from ....propagation.utils.tensor_utils import get_device

from slmsuite.hardware.slms.slm import SLM
from slmsuite.hardware.cameras.camera import Camera


# TODO: Add automatic cuda device detection
class SpeckleCalibrator(WavefrontCalibratorBase):
    def __init__(
        self,
        slm: SLM,
        camera: Camera,
        camera_mapping: CameraMapping,
        focal_length: float,
        directory: str,
        number_of_random_patterns: int = 10,
        slm_masks: NDArray[np.bool_] | None = None,
    ):
        super().__init__(slm, camera)

        self.directory: str = directory
        self.number_of_random_patterns: int = number_of_random_patterns

        self.dataset_manager: DatasetGenerator = DatasetGenerator(
            slm=self.slm,
            camera=self.camera,
            camera_mapping=camera_mapping,
            focal_length=focal_length,
            directory=self.directory,
            number_of_random_patterns=self.number_of_random_patterns,
            slm_masks=slm_masks,
        )

        self.dataset_descriptor: DatasetDescriptor | None = None
        self.trainer: PropagationTrainer | None = None
        self.dataloader: DataLoader | None = None

    def generate_dataset(
        self,
        speckle_pattern_extent: tuple[float, float] = (1.5e-3, 1.5e-3),
        capture_background_image: bool = False,
        benchmark_calibration: WavefrontCalibrationData | None = None,
    ) -> None:
        self.dataset_manager.generate_phase_patterns(
            speckle_pattern_extent,
            benchmark_calibration=benchmark_calibration,
        )
        self.dataset_descriptor = self.dataset_manager.capture_camera_images(
            capture_background_image
        )
        self.dataset_descriptor.save(self.directory + "dataset_descriptor.pkl")

    def train_propagator(
        self,
        number_of_epochs: int = 100,
        batch_size: int = 5,
        subset_indices: Sequence[int] | None = None,
        trainer_type: Literal["smooth_wavefront"] = "smooth_wavefront",
        device: str | None = None,
    ) -> None:
        if trainer_type == "smooth_wavefront":
            self.trainer = SmoothWavefrontTrainer(
                dataset_descriptor=self.dataset_descriptor,
                load_path=self.directory,
                device=device,
            )
        else:
            raise ValueError(
                f"Unknown trainer type: {trainer_type}. "
                "Choose either 'psf' or 'smooth_wavefront'."
            )

        self.dataloader, optimizer = self.trainer.initialize_optimization(
            subset_indices=subset_indices,
            batch_size=batch_size,
            shuffle=True,
        )

        self.trainer.run_optimization(
            self.dataloader,
            optimizer,
            number_of_epochs=number_of_epochs,
        )

    def generate_slm_beam_calibration(self) -> WavefrontCalibrationData:
        field = self.trainer.get_wavefront().detach().cpu().numpy()

        intensity = np.abs(field) ** 2
        intensity /= np.max(intensity)
        beam_radius, shift_x, shift_y = self.fit_gaussian_beam(intensity)

        slm_mask = intensity > 0.005 * np.max(intensity)

        phase = np.angle(field)
        phase_no_tilt = self.trainer.remove_phase_tilt(phase, slm_mask)

        zernike_coefficients = self.fit_zernike(phase_no_tilt)

        amplitude = np.sqrt(intensity)
        calibrated_field = amplitude * np.exp(1j * phase_no_tilt)

        return WavefrontCalibrationData(
            timestamp=datetime.now(),
            name="speckle_psf",
            constant_slm_field=calibrated_field,
            mask=slm_mask,
            beam_waist_x=beam_radius,
            beam_waist_y=beam_radius,
            zernike_coefficients=zernike_coefficients,
        )

    def calibrate_wavefront(
        self,
        speckle_pattern_extent: tuple[float, float] = (1.5e-3, 1.5e-3),
        number_of_epochs: int = 50,
        batch_size: int = 5,
        subset_indices: Sequence[int] | None = None,
        capture_background_image: bool = False,
        trainer_type: Literal["smooth_wavefront"] = "smooth_wavefront",
        benchmark_calibration: WavefrontCalibrationData | None = None,
        device: str | None = None,
    ) -> WavefrontCalibrationData:
        if device is None:
            device = get_device()

        self.generate_dataset(
            speckle_pattern_extent=speckle_pattern_extent,
            capture_background_image=capture_background_image,
            benchmark_calibration=benchmark_calibration,
        )

        self.train_propagator(
            number_of_epochs=number_of_epochs,
            batch_size=batch_size,
            subset_indices=subset_indices,
            trainer_type=trainer_type,
            device=device,
        )

        return self.generate_slm_beam_calibration()

    def calibrate_intensity(self, *args, **kwargs) -> WavefrontCalibrationData:
        print(
            "PSFCalibrator simulateneously calibrates the intensity and "
            "phase at the SLM. Running calibrate_wavefront() instead."
        )
        return self.calibrate_wavefront(*args, **kwargs)

    def calibrate_phase(self, *args, **kwargs) -> WavefrontCalibrationData:
        print(
            "PSFCalibrator simulateneously calibrates the intensity and "
            "phase at the SLM. Running calibrate_wavefront() instead."
        )
        return self.calibrate_wavefront(*args, **kwargs)
