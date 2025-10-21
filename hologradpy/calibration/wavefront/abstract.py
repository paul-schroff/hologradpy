import os
import time
from datetime import datetime
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

import torch

from slmsuite.hardware.slms.slm import SLM
from slmsuite.hardware.cameras.camera import Camera

from ...propagation.elements import (
    ConstantSLMField,
)
from ...propagation.virtual_slms.abstract import VirtualSLM

from ...analysis.fitting import fit_gaussian_beam_intensity

@dataclass
class WavefrontCalibrationData:
    timestamp: datetime
    name: str
    constant_slm_field_state_dict: dict
    beam_waist_x: float
    beam_waist_y: float
    zernike_coefficients: NDArray[np.float_]
    

class WavefrontCalibratorBase:
    """
    Class to calibrate the intensity and the phase at the SLM.
    """
    def __init__(self, slm: SLM, camera: Camera, device: str = 'cpu'):
        """
        Initialize the SLMWavefrontCalobrator.

        Args:
            slm (SLM): The SLM object.
            camera (Camera): The camera object.
            device (torch.device): Torch device for calculations.
            virtual_slm (VirtualSLM): Virtual SLM to be calibrated.
        """
        self.camera: Camera = camera
        self.slm: SLM = slm
        self.device: torch.device = device
        self.virtual_slm: VirtualSLM = VirtualSLM(
            self.slm,
            device = self.device
        )    
    def calibrate(self) -> ConstantSLMField:
        """
        Calibrate the SLM wavefront consisting of the amplitude and the phase 
        at the SLM.
        Returns:
            ConstantSLMField: The calibrated electric field at the SLM.
        """
        raise NotImplementedError(
            "The calibrate method should be implemented in the derived class."
        )
    
    def fit_gaussian_beam(
        self,
        measured_intensity: NDArray[np.float_],
    ) -> tuple[float, float, float]:
        """ Fit a Gaussian beam to the measured intensity.
        Args:
            measured_intensity (NDArray[np.float_]): The measured intensity 
                from the camera.
        Returns:
            tuple[float, float, float]: The fitted beam radius and shifts in 
                x and y.
        """
        beam_radius_guess = (
            min(self.slm.shape) * self.slm.pitch_um[0] * 1e-6 / 2
        )
        
        popt, _ = fit_gaussian_beam_intensity(
            *self.virtual_slm.get_spatial_grid_input(),
            measured_intensity,
            beam_radius_guess,
            blur_sigma=10
        )

        beam_radius = popt[0]
        shift_x = popt[1]
        shift_y = popt[2]
        
        return beam_radius, shift_x, shift_y
    
    def fit_zernike(
        self,
        measured_phase: NDArray[np.float_]
    ) -> NDArray[np.float_]:
        """ Fit a Zernike polynomial to the measured phase.
        
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
    
    def save(self, slm_field: ConstantSLMField, save_path: str):
        """
        Save the SLM field to a file.

        Args:
            slm_field (ConstantSLMField): The SLM field to save.
            filename (str): The filename to save the SLM field to.
        """
        date_saved = time.strftime('%y-%m-%d_%H-%M-%S', time.localtime())
        path = save_path + date_saved + '_' + 'measure_slm_intensity'
        os.mkdir(path)
        # TODO: Finish implementing this
    
    def load(self, filename: str) -> ConstantSLMField:
        """
        Load the SLM field from a file.

        Args:
            filename (str): The filename to load the SLM field from.

        Returns:
            ConstantSLMField: The loaded SLM field.
        """
        raise NotImplementedError(
            "The load method should be implemented in the derived class."
        )
        # TODO: Finish implementing this