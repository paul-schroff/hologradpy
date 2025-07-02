import os
import time

import numpy as np
from numpy.typing import NDArray

from slmsuite.hardware.slms.slm import SLM
from slmsuite.hardware.cameras.camera import Camera

from ...torch_modules.elements import (
    VirtualSLM,
    ConstantSLMField,
)

from ...analysis.fitting import fit_gaussian_beam_intensity


class WavefrontCalibratorBase:
    """
    Class to calibrate the SLM wavefront.
    """
    def __init__(self, slm: SLM, camera: Camera, device: str = 'cpu'):
        """
        Initialize the SLMWavefrontCalobrator.

        Args:
            slm (SLM): The SLM object.
            camera (Camera): The camera object.
        """
        self.camera: Camera = camera
        self.slm : SLM = slm
        self.device = device
        self.virtual_slm: VirtualSLM = VirtualSLM(
            self.slm,
            device = self.device
        )    
    def calibrate(self) -> ConstantSLMField:
        """
        Calibrate the SLM wavefront.
        Returns:
            ConstantSLMField: The calibrated SLM field.
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