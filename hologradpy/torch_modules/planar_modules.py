"""Modules which modify the electric field in a 2D plane
"""
from __future__ import annotations

import torch
import torch.nn as nn

from kornia.geometry.transform import get_affine_matrix2d
from kornia.geometry import warp_perspective

from .utils.optics_utils import lens_phase, circular_mask
from .utils.tensor_utils import unsqueeze_to
from .propagators import PropagatorBase

from slmsuite.hardware.slms.slm import SLM

class SpatialLightModulator(PropagatorBase):
    def __init__(
            self: SpatialLightModulator,
            slm_hardware: SLM,
            init_phase: torch.Tensor | None = None,
        ) -> None:
        
        super().__init__(
            slm_hardware.shape,
            (slm_hardware.pitch_um * 1e-6, slm_hardware.pitch_um * 1e-6)
        )
        
        if init_phase is None:
            init_phase = torch.zeros(
                slm_hardware.shape, dtype=self.dtype, device=self.device
            )

        # Phase parameter requires gradient only in inference mode
        self.phase = nn.Parameter(
            torch.tensor(init_phase, self.dtype, self.device),
            requires_grad = not self.training
        )
        
    def forward(
            self: SpatialLightModulator, 
            phase: torch.Tensor
        ) -> torch.Tensor:
        
        self.phase.data = phase
        phase = unsqueeze_to(self.phase, 3)
        
        return torch.exp(1j * phase).squeeze()


class ConstantSLMField(PropagatorBase):
    def __init__(
            self: ConstantSLMField,
            init_field: torch.Tensor[torch.complex],
            pixel_pitch: float,
        ) -> None:

        super().__init__(
            init_field.shape[-2:],
            (pixel_pitch, pixel_pitch)
        )
        self.phase = nn.Parameter(
            torch.tensor(init_field.angle(), self.dtype, self.device),
            requires_grad = self.training
        )
        
        self.amplitude = nn.Parameter(
            torch.tensor(init_field.abs(), self.dtype, self.device),
            requires_grad = self.training
        )

    def forward(
            self: ConstantSLMField,
            input_field: torch.Tensor = None
        ) -> torch.Tensor:
        input_field = unsqueeze_to(input_field, 3)
        amplitude = unsqueeze_to(self.amplitude, 3)
        phase = unsqueeze_to(self.phase, 3)
        
        return (input_field * amplitude * torch.exp(1j * phase)).squeeze()
    

class PartialAffineTransform(PropagatorBase):
    def __init__(
            self: PartialAffineTransform,
            resolution_in: tuple[int, int],
            pixel_pitch_in: float,
            resolution_out: tuple[int, int],  
            scale: tuple[float, float] = (1, 1),
            shift: tuple[float, float] = (0, 0),
            angle: float = 0.0,
            verbose: bool = True,
        ) -> None:
        self.scale = scale
        self.shift = shift
        self.angle = angle
        self._resolution_out = resolution_out
        self.verbose = verbose

        super().__init__(resolution_in, (pixel_pitch_in, pixel_pitch_in))

        self.scale = nn.Parameter(
            torch.tensor(scale, self.dtype, self.device),
            requires_grad = self.training
        )

        # Shift from the center of the in pixels
        self.shift = nn.Parameter(
            torch.tensor(shift, self.dtype, self.device),
            requires_grad = self.training
        )

        # Rotation angle in radians
        self.angle = nn.Parameter(
            torch.tensor(angle, self.dtype, self.device),
            requires_grad = self.training
        )

        self.center = torch.tensor([0.0, 0.0], device=self.device)
        self.affine_matrix = self.get_affine_matrix()
    
    @property
    def pixel_size_out(self: PartialAffineTransform) -> tuple[float, float]:
        return tuple(self.pixel_size_in[i] / self.scale[i] for i in range(2))
    
    @property
    def resolution_out(self: PartialAffineTransform) -> tuple[int, int]:
        return self._resolution_out
    
    def get_affine_matrix(self) -> torch.Tensor:
        return get_affine_matrix2d(self.shift.unsqueeze(0),
                                   self.center.unsqueeze(0),
                                   self.scale.unsqueeze(0),
                                   self.angle.unsqueeze(0))
    
    def forward(
            self: PartialAffineTransform,
            input_field: torch.Tensor
        ) -> torch.Tensor:
        """Applies partial affine transformation to input_field."""
        if self.verbose:
            print('Scale:', self.scale.data)
            print('Shift:', self.shift.data)
            print('Angle:', self.angle.data)

        input_field = unsqueeze_to(input_field, 4)
        self.affine_matrix = self.get_affine_matrix()

        # Kornia does not support complex numbers in warp_perspective(),
        # so we need to split the real and imaginary parts and then 
        # combine them again.
        output_real = warp_perspective(
            input_field.real,
            self.affine_matrix,
            self.resolution_out
        )
        output_imag = warp_perspective(
            input_field.imag,
            self.affine_matrix,
            self.resolution_out
        )
        return (output_real + 1j * output_imag).squeeze()


class SimpleLens(PropagatorBase):
    def __init__(
            self: SimpleLens,
            focal_length: float,
            aperture_radius: float,
            wavelength: float,
            resolution_in: tuple[int, int],
            pixel_pitch_in: float,
        ) -> None:
        self.focal_length = focal_length
        self.wavelength = wavelength
        self.aperture_radius = aperture_radius

        super().__init__(
            wavelength = wavelength,
            resolution_in = resolution_in,
            pixel_size_in = (pixel_pitch_in, pixel_pitch_in),
        )

        spatial_grid = self.get_spatial_grid_input()

        self.lens_phase = lens_phase(
            spatial_grid[1],
            spatial_grid[0],
            self.focal_length,
            self.wavenumber
        )

        self.lens_aperture = circular_mask(
            spatial_grid[1],
            spatial_grid[0],
            self.aperture_radius,
            shift_x = self.spatial_extent_in[0] / 2,
            shift_y = self.spatial_extent_in[1] / 2
        )
    
    @property
    def pixel_size_out(self: SimpleLens) -> tuple[float, float]:
        return self.pixel_size_in
    
    @property
    def resolution_out(self: SimpleLens) -> tuple[int, int]:
        return self.resolution_in
    
    def forward(self: SimpleLens, input_field: torch.Tensor) -> torch.Tensor:
        input_field = unsqueeze_to(input_field, 3)
        lens_phase = unsqueeze_to(self.lens_phase, 3)
        lens_aperture = unsqueeze_to(self.lens_aperture, 3)

        return input_field * lens_aperture * torch.exp(1j * lens_phase)

# TODO: Implement doublet lens module
# TODO: Implement affine transform module