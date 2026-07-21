from __future__ import annotations
import pickle
from datetime import datetime
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

import torch
from torch import Tensor

from ...hardware import Camera, SLM, as_camera, as_slm

from ...propagation.complex_amplitude import ComplexAmplitude
from ...propagation.fourier import get_spatial_grid
from ...visualizer import VisualizationData

from ...analysis.fitting import fit_gaussian_beam_intensity


@dataclass
class WavefrontCalibrationData:
    timestamp: datetime
    name: str
    complex_amplitude: ComplexAmplitude
    metadata: dict
    visualization_data: VisualizationData | None = None

    def save(self, filename: str):
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename: str) -> WavefrontCalibrationData:
        with open(filename, "rb") as file:
            calibration_data: WavefrontCalibrationData = pickle.load(file)
        return calibration_data


class WavefrontCalibratorBase:
    """
    Class to calibrate the intensity and the phase at the SLM.
    """

    def __init__(self, slm: SLM, camera: Camera, device: torch.device = "cpu"):
        """
        Initialize the SLMWavefrontCalobrator.

        Args:
            slm (SLM): The SLM object.
            camera (Camera): The camera object.
            device (torch.device): Torch device for calculations.
            virtual_slm (VirtualSLM): Virtual SLM to be calibrated.
        """
        self.camera: Camera = as_camera(camera)
        self.slm: SLM = as_slm(slm)
        self.device: torch.device = device

    @property
    def spatial_grid_slm(self) -> tuple[Tensor, Tensor]:
        """Get the spatial grid coordinates for the SLM pixels.

        Args:
            slm (SLM): The SLM to get the spatial grid for.
            device (torch.device): The device to use for the spatial grid.

        Returns:
            tuple[Tensor, Tensor]: The x and y coordinates of the SLM pixels.
        """
        return get_spatial_grid(
            self.slm.resolution, self.slm.pixel_size, device=self.device
        )

    def calibrate(self) -> WavefrontCalibrationData:
        """
        Calibrate the SLM wavefront consisting of the amplitude and the phase
        at the SLM.
        Returns:
            WavefrontCalibrationData: The calibrated wavefront data.
        """
        raise NotImplementedError(
            "The calibrate method should be implemented in the derived class."
        )

    def fit_gaussian_beam(
        self,
        measured_intensity: NDArray[np.float_],
    ) -> tuple[float, float, float]:
        """Fit a Gaussian beam to the measured intensity.
        Args:
            measured_intensity (NDArray[np.float_]): The measured intensity
                from the camera.
        Returns:
            tuple[float, float, float]: The fitted beam radius and shifts in
                x and y.
        """
        beam_radius_guess = min(self.slm.resolution) * self.slm.pixel_size[1] / 2

        popt, _ = fit_gaussian_beam_intensity(
            *self.spatial_grid_slm, measured_intensity, beam_radius_guess, blur_sigma=10
        )

        beam_radius = popt[0]
        shift_x = popt[1]
        shift_y = popt[2]

        return beam_radius, shift_x, shift_y

    def fit_zernike(self, measured_phase: NDArray[np.float_]) -> NDArray[np.float_]:
        """Fit a Zernike polynomial to the measured phase.

        Args:
            measured_phase (NDArray[np.float_]): The measured phase from the
                camera.

        Returns:
            NDArray[np.float_]: The fitted Zernike coefficients.
        """
        # TODO: Implement the Zernike fitting method.
        raise NotImplementedError(
            "The fit_zernike method should be implemented in the derived class."
        )
        # TODO: Finish implementing this
