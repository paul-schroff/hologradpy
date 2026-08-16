from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

import torch
from torch import Tensor

from ...hardware import Camera, SLM, as_camera, as_slm

from ...optics.complex_amplitude import ComplexAmplitude
from ...grids import get_spatial_grid
from ...serialization import SaveableRecord, record_type
from ...visualizer import VisualizationData

from ...analysis.fitting import fit_gaussian_beam_intensity


@record_type("wavefront_calibration")
@dataclass
class WavefrontCalibrationData(SaveableRecord):
    timestamp: datetime
    name: str
    complex_amplitude: ComplexAmplitude
    metadata: dict
    visualization_data: VisualizationData | None = None

    # save / load come from SaveableRecord.


class WavefrontCalibratorBase(ABC):
    """Base for calibrators that measure the amplitude and phase at the SLM.

    Concrete calibrators differ widely in how they take their measurements, so
    the only shared contract is :meth:`calibrate`: run the measurement and
    return a :class:`WavefrontCalibrationData`. Everything the two current
    implementations happen to share (device wrapping, the SLM spatial grid, the
    Gaussian-beam fit) lives here as concrete helpers.
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

    @abstractmethod
    def calibrate(self, *args, **kwargs) -> WavefrontCalibrationData:
        """Run the calibration and return the measured SLM-plane wavefront.

        The arguments are specific to each calibrator, since the measurement
        strategies have little in common. Only the return type is part of the
        contract.
        """

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

        # The fit runs on the CPU with numpy, so bring the coordinate grid over from the
        # (possibly CUDA) device.
        grid_x, grid_y = self.spatial_grid_slm
        popt, _ = fit_gaussian_beam_intensity(
            grid_x.cpu(),
            grid_y.cpu(),
            measured_intensity,
            beam_radius_guess,
            blur_sigma=10,
        )

        beam_radius = popt[0]
        shift_x = popt[1]
        shift_y = popt[2]

        return beam_radius, shift_x, shift_y
