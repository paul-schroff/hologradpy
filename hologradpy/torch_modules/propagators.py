from __future__ import annotations

import numpy as np

import torch
import torch.nn as nn
from torch._prims_common import (
    corresponding_real_dtype,
    corresponding_complex_dtype
)
from torchkbnufft import KbNufft

from .utils.tensor_utils import (
    pad_to_shape_2D,
    crop_to_shape_2D,
    unsqueeze_to,
)
from .utils.optics_utils import lens_phase, circular_mask
from .utils.fourier_utils import (
    get_spatial_grid,
    get_frequency_grid,
    fft_2d,
    ifft_2d
)

# %% Propagator base class
class PropagatorBase(nn.Module):
    dtype = torch.float32
    wavelength: float
    device: str = "cpu"

    def __init__(
            self: PropagatorBase,
            resolution_in: tuple[int, int],
            pixel_size_in: tuple[float, float],
        ) -> None:
        
        super().__init__()

        self.resolution_in = resolution_in
        self.pixel_size_in = pixel_size_in

        self.spatial_extent_in = tuple(
            self.resolution_in[i] * self.pixel_size_in[i]
            for i in range(2)
        )
    
    @property
    def wavenumber(self: PropagatorBase) -> float:
        return 2 * np.pi / self.wavelength
    
    @property
    def pixel_size_out(self: PropagatorBase) -> tuple[float, float]:
        return self.pixel_size_in
    
    @property
    def resolution_out(self: PropagatorBase) -> tuple[int, int]:
        return self.resolution_in
    
    @property
    def spatial_extent_out(self: PropagatorBase) -> tuple[float, float]:
        return tuple(
            self.resolution_out[i] * self.pixel_size_out[i]
            for i in range(2)
        )

    def get_spatial_grid_input(
            self: PropagatorBase
        ) -> tuple[torch.Tensor, torch.Tensor]:
        return get_spatial_grid(
            self.resolution_in,
            self.pixel_size_in,
            self.device
        )
    
    def get_spatial_grid_output(
            self: PropagatorBase
        ) -> tuple[torch.Tensor, torch.Tensor]:
        return get_spatial_grid(
            self.resolution_out,
            self.pixel_size_out,
            self.device
        )


# %% Torch modules modelling the propagation of light
class FourierLensFft(PropagatorBase):
    def __init__(
            self: FourierLensFft,
            focal_length: float,
            wavelength: float,
            resolution_in: tuple[int, int],
            pixel_pitch_in: float,
            padded_resolution: tuple[int, int] = None,
            fft_kwargs: dict = {},
        ) -> None:

        self.focal_length = focal_length
        self.wavelength = wavelength
        self.fft_kwargs = fft_kwargs

        if padded_resolution is None:
            padded_resolution = tuple(2 * resolution_in[i] for i in range(2))
        self.padded_resolution = padded_resolution

        super().__init__(
            resolution_in = resolution_in,
            pixel_size_in = (pixel_pitch_in, pixel_pitch_in),
        )

    @property
    def pixel_size_out(self: FourierLensFft) -> tuple[float, float]:
        return tuple(
            self.wavelength * self.focal_length / self.spatial_extent_in[i]
            for i in range(2)
        )
    
    @property
    def resolution_out(self: FourierLensFft) -> tuple[int, int]:
        return self.padded_resolution

    def forward(
            self: FourierLensFft,
            input_field: torch.Tensor
        ) -> torch.Tensor:
        e_in = pad_to_shape_2D(input_field, self.padded_resolution)
        return fft_2d(e_in, **self.fft_kwargs)
    

class FourierLensNufft(PropagatorBase):  
    def __init__(
            self: FourierLensNufft,
            focal_length: float,
            resolution_in: tuple[int, int],
            pixel_pitch_in: float,
            resolution_out: tuple[int, int],
            calculation_resolution: tuple[int, int] | None = None,
            scale: tuple[float, float] = (1, 1),
            shift: tuple[float, float] = (0, 0),
            angle: float = 0,
            nufft_kwargs: dict = {},
        ) -> None:
        self.focal_length = focal_length
        self.resolution_out = resolution_out

        if calculation_resolution is None:
            calculation_resolution = tuple(
                resolution_in[i] * 2 for i in range(2)
            )
        self.calculation_resolution = calculation_resolution

        # Scaling of the output pixel size relative to the pixel size 
        # assuming 2-fold zero padding of the input field.
        self.scale = scale

        # Shift from the center of the in pixels
        self.shift = shift

        # Rotation angle in radians
        self.angle = angle

        super().__init__(
            resolution_in = resolution_in,
            pixel_size_in = (pixel_pitch_in, pixel_pitch_in),
        )

        self.eps = torch.finfo(self.dtype).eps

        self.resolution_ratio = tuple(
            self.resolution_out[i] / self.calculation_resolution[i] 
            for i in range(2)
        )
    
        self.kbnufft = KbNufft(
            im_size = self.resolution_in,
            grid_size = self.calculation_resolution,
            device = self.device,
            dtype = corresponding_complex_dtype(self.dtype),
            **nufft_kwargs,
        )
        
        frequency_grid = get_frequency_grid(
            self.resolution_out,
            self.pixel_size_out,
            self.device,
        )
        
        # Flatten frequenciy grid
        self.frequencies_x = frequency_grid[0].flatten()
        self.frequencies_y = frequency_grid[1].flatten()

        self.frequencies_flattened = self.get_transformed_coordinates(
            self.scale,
            self.shift,
            self.angle
        )
    
    @property
    def pixel_size_out(self: FourierLensNufft) -> tuple[float, float]:
        return tuple(
            self.wavelength * self.focal_length / self.spatial_extent_in[i] /
            self.scale[i] for i in range(2)
        )
    
    @property
    def resolution_out(self: FourierLensNufft) -> tuple[int, int]:
        return self.resolution_out

    def get_transformed_coordinates(
            self: FourierLensNufft,
            scale: tuple[float, float],
            shift: tuple[float, float],
            angle: float
        ) -> torch.Tensor:
        # Conversion factor from spatial frequency to radians
        frequency_step_radians = (
            2 * torch.pi / self.resolution_out[i] * self.resolution_ratio[i] /
            scale[i] for i in range(2)
        )
        
        angle_sin = np.sin(angle)
        angle_cos = np.cos(angle)

        frequencies_x_transformed = (
            (self.frequencies_x * angle_cos - self.frequencies_y * angle_sin 
             + shift[1]) * frequency_step_radians[1]
        ) 
        frequencies_y_transformed = (
            (self.frequencies_x * angle_sin + self.frequencies_y * angle_cos
             + shift[0]) * frequency_step_radians[0]
        )
    
        return torch.stack(
            (frequencies_y_transformed, frequencies_x_transformed),
            axis=0
        )
        
    def forward(
            self: FourierLensNufft,
            input_field: torch.Tensor
        ) -> torch.Tensor:
        self.number_of_images = (
            input_field.shape[-3] if input_field.dim() > 2 else 1
        )
        input_field = unsqueeze_to(input_field, 3)
        
        output_field = self.kbnufft(
            unsqueeze_to(input_field, 4),
            unsqueeze_to(self.frequencies_flattened, 3)
        ).squeeze()
            
        return output_field.reshape((self.number_of_images,
                                     self.resolution_out[0],
                                     self.resolution_out[1])).squeeze()


class AngularSpectrumMethod(PropagatorBase):
    def __init__(
            self: AngularSpectrumMethod,
            propagation_distance: float,
            resolution_in: tuple[int, int],
            pixel_pitch_in: float,
            padded_resolution: tuple[int, int] | None = None,
            fft_kwargs: dict = {},
        ) -> None:
        self.propagation_distance = propagation_distance

        if padded_resolution is None:
            padded_resolution = tuple(
                2 * resolution_in[i] for i in range(2)
            )
        self.padded_resolution = padded_resolution

        self.fft_kwargs = fft_kwargs

        super().__init__(
            resolution_in = resolution_in,
            pixel_size_in = (pixel_pitch_in, pixel_pitch_in),
        )

        self.frequency_grid = get_frequency_grid(
            self.padded_resolution,
            self.pixel_size_in,
            device=self.device
        )

        self.phase_factor = self.get_phase_factor(
            self.propagation_distance
        ).to(corresponding_complex_dtype(self.dtype))
    
    @property
    def pixel_size_out(self: AngularSpectrumMethod) -> tuple[float, float]:
        return self.pixel_size_in

    @property
    def resolution_out(self: AngularSpectrumMethod) -> tuple[int, int]:
        return self.padded_resolution
        
    def get_phase_factor(
            self: AngularSpectrumMethod,
            propagation_distance: float
        ) -> torch.Tensor:
        return torch.exp(1j * propagation_distance *
                         torch.sqrt((self.wavenumber ** 2 -
                                     self.frequency_grid[0] ** 2 -
                                     self.frequency_grid[1] ** 2 + 0j)))

    def forward(
            self: AngularSpectrumMethod,
            input_field: torch.Tensor
        ) -> torch.Tensor:
        input_field = unsqueeze_to(input_field, 3)
        phase_factor = unsqueeze_to(self.phase_factor, 3)

        padded_field = pad_to_shape_2D(input_field, self.padded_resolution)
        angular_spectrum = fft_2d(padded_field, **self.fft_kwargs)
        return ifft_2d(angular_spectrum * phase_factor, **self.fft_kwargs)
    