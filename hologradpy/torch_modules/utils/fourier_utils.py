from __future__ import annotations
import os
from typing import List, Tuple, Literal
import torch
from torch import Tensor
import torch.nn as nn
from torch._prims_common import (corresponding_real_dtype,
                                 corresponding_complex_dtype)
from torch.fft import fftn, ifftn, fftshift, ifftshift

import torchkbnufft as tkbn

from .tensor_utils import unsqueeze_to

# %% Utility functions
def get_spatial_grid(resolution: tuple[int, int],
                     pixel_size: tuple[float, float],
                     device: str = 'cpu') -> Tuple[Tensor, Tensor]:
    resolution = torch.tensor(resolution, device=device)
    pixel_size = torch.tensor(pixel_size, device=device)
    
    spatial_extent = resolution * pixel_size

    coordinates_x = (torch.arange(-resolution[1] // 2,
                                  resolution[1] // 2,
                                  device=device) * 
                     spatial_extent[1] / resolution[1])

    coordinates_y = (torch.arange(-resolution[0] // 2,
                                  resolution[0] // 2,
                                  device=device) * 
                     spatial_extent[0] / resolution[0])
    return torch.meshgrid(coordinates_x, coordinates_y, indexing='xy')

def get_frequency_grid(resolution: tuple[int, int],
                       pixel_size: tuple[float, float],
                       device: str = 'cpu') -> Tuple[Tensor, Tensor]:
    resolution = torch.tensor(resolution, device=device)
    pixel_size = torch.tensor(pixel_size, device=device)

    frequency_extent = 2 * torch.pi / pixel_size

    frequencies_x = (torch.arange(-resolution[1] // 2,
                                  resolution[1] // 2,
                                  device=device) *
                     frequency_extent[1] / resolution[1])
    
    frequencies_y = (torch.arange(-resolution[0] // 2,
                                  resolution[0] // 2,
                                  device=device)
                     * frequency_extent[0] / resolution[0])
    
    return torch.meshgrid(frequencies_x, frequencies_y, indexing='xy')

# %% Pytorch FFT and IFFT wrappers
def fft_2d(input: Tensor,
           norm: Literal['backward', 'forward', 'ortho'] = 'backward',
           fft_shift: bool = True
           ) -> Tensor:
    if fft_shift is True:
        return torch.fft.fftshift(
            torch.fft.fftn(
                torch.fft.ifftshift(input, dim=(-1, -2)),
                norm=norm,
                dim=(-1, -2)),
            dim=(-1, -2))
    else:
        return torch.fft.fftn(input, norm=norm, dim=(-1, -2))
    
def ifft_2d(input: Tensor,
            norm: Literal['backward', 'forward', 'ortho'] = 'backward',
            fft_shift: bool = True
            ) -> Tensor:
    if fft_shift is True:
        return torch.fft.ifftshift(
            torch.fft.ifftn(
                torch.fft.fftshift(input, dim=(-1, -2)),
                norm=norm,
                dim=(-1, -2)),
            dim=(-1, -2))
    else:
        return torch.fft.ifftn(input, norm=norm, dim=(-1, -2))
    
# %% Base classes
class FourierBase(nn.Module):
    def __init__(self,
                 pixel_size: Tensor,
                 resolution: Tensor,
                 device: str | None = None):
        super().__init__()
        self.resolution = resolution
        self._pixel_size = pixel_size
        self._frequency_extent = 2 * torch.pi / self.pixel_size
        self.device = device

    @property
    def frequency_extent(self) -> Tensor:
        return self._frequency_extent
    
    @property
    def pixel_size(self) -> Tensor:
        return self._pixel_size
    
    @pixel_size.setter
    def pixel_size(self, value: Tensor):
        self._pixel_size = value
        self._frequency_extent = 2 * torch.pi / self.pixel_size

    @property
    def frequency_step(self):
        return self.frequency_extent / self.resolution

    @property
    def frequency_grid(self) -> Tuple[Tensor, Tensor]:
        return get_frequency_grid(self.resolution,
                                  self.pixel_size,
                                  device=self.device)
    
    @property
    def spatial_grid(self) -> Tuple[Tensor, Tensor]:
        return get_spatial_grid(self.resolution,
                                self.pixel_size,
                                device=self.device)
    
    def forward(self, input: Tensor) -> Tensor:
        raise NotImplementedError

# %% Fourier transform modules
class FastFourierTransform(FourierBase):
    def __init__(
            self: FastFourierTransform,
            pixel_size: Tensor,
            resolution: Tensor,
            device: str | None = None,
            fft_shift: bool = True,
            norm: Literal['backward', 'forward', 'ortho'] = 'backward'
        ) -> None:

        super().__init__(pixel_size, resolution, device)
        self.fft_shift: bool = fft_shift
        self.norm: Literal['backward', 'forward', 'ortho'] = norm
    
    def forward(self: FastFourierTransform, input: Tensor) -> Tensor:
        return fft_2d(input, norm=self.norm, fft_shift=self.fft_shift)


class InverseFastFourierTransform(FourierBase):
    def __init__(
            self: InverseFastFourierTransform,
            pixel_size: Tensor,
            resolution: Tensor,
            device: str | None = None,
            fft_shift: bool = True,
            norm: Literal['backward', 'forward', 'ortho'] = 'backward'
        ) -> None:

        super().__init__(pixel_size, resolution, device)
        self.fft_shift: bool = fft_shift
        self.norm: Literal['backward', 'forward', 'ortho'] = norm
    
    def forward(self: InverseFastFourierTransform, input: Tensor) -> Tensor:
        return ifft_2d(input, norm=self.norm, fft_shift=self.fft_shift)


class KbNufftZoom(FourierBase):
    def __init__(self: KbNufftZoom,
                 pixel_size: Tensor,
                 resolution: Tensor,
                 resolution_out: Tensor,
                 magnification: Tensor,
                #  shift: Tensor,
                 dtype: torch.dtype,
                 device: str | None = None,
                 grid_size: Tensor | None = None,
                 norm: str | None = 'ortho',
                 **nufft_kwargs):
        
        super().__init__(pixel_size, resolution, device)
        if grid_size is None:
            grid_size = resolution
        self.grid_size = grid_size
        self.norm = norm
        self.resolution_out = resolution_out

        self.kbnufft = tkbn.KbNufft(
            im_size = resolution.to(int).tolist(),
            grid_size = grid_size.to(int).tolist(),
            device = device,
            **nufft_kwargs
        ).to(device)

        resolution_ratio = self.resolution_out / resolution
        frequency_step_radians = (2 * torch.pi / self.resolution_out
                                  * resolution_ratio / magnification)
        
        frequencies_x = (torch.arange(-self.resolution_out[1] // 2,
                                      self.resolution_out[1] // 2,
                                      device=device) *
                         frequency_step_radians[1])
        frequencies_y = (torch.arange(-self.resolution_out[0] // 2,
                                      self.resolution_out[0] // 2,
                                      device=device) *
                         frequency_step_radians[0])
        
        frequency_grid = torch.meshgrid(frequencies_x,
                                        frequencies_y,
                                        indexing='xy')
        
        self.frequencies_flattened = torch.stack(
            (frequency_grid[1].flatten(), frequency_grid[0].flatten()),
            axis=0
        ).to(device, corresponding_real_dtype(dtype))

    def forward(self, input: Tensor) -> Tensor:
        number_of_images = input.shape[-3] if input.dim() > 2 else 1
        
        output = self.kbnufft(
            unsqueeze_to(input, 4),
            unsqueeze_to(self.frequencies_flattened, 3),
            norm=self.norm
        )
        
        return output.reshape(
            (number_of_images, self.resolution_out[0], self.resolution_out[1])
        ).squeeze()


class InverseKbNufftZoom(KbNufftZoom):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frequencies_flattened = self.frequencies_flattened.flip(-1)
    
    def forward(self, e_in: Tensor) -> Tensor:
        return super().forward(e_in)