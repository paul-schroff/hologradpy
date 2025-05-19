import os
import time

from slmsuite.hardware.slms.slm import SLM
from slmsuite.hardware.cameras.camera import Camera

from ...torch_modules.elements import (
    SpatialLightModulator,
    ConstantSLMField,
)


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
        self.virtual_slm: SpatialLightModulator = SpatialLightModulator(
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
    
    def fit_gaussian_beam(self):
        # Fit Gaussian to measured intensity
        # extent_slm = (
        #     slm_disp_obj.slm_size[0] + aperture_width * slm_disp_obj.pitch
        # ) / 2
        # x_fit = np.linspace(-extent_slm, extent_slm, aperture_number)
        # x_fit, y_fit = np.meshgrid(x_fit, x_fit)
        # sig_x, sig_y = pms_obj.beam_diameter, pms_obj.beam_diameter
        # popt_slm, perr_slm = ft.fit_gaussian(
        #     superpixel_intensity,
        #     dx=0,
        #     dy=0,
        #     sig_x=sig_x,
        #     sig_y=sig_y,
        #     xy=[x_fit, y_fit]
        # )

        # i_fit_slm = pt.gaussian(
        #     slm_disp_obj.meshgrid_slm[0],
        #     slm_disp_obj.meshgrid_slm[1],
        #     *popt_slm
        # )
        raise NotImplementedError(
            "The fit_gaussian_beam method is not implemented yet."
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