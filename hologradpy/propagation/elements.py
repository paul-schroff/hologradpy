"""Modules that modify the electric field in a 2D plane"""

from __future__ import annotations

import torch
import torch.nn as nn

from kornia.geometry.transform import get_affine_matrix2d
from kornia.geometry import warp_perspective

from aotools.functions import zernikeArray
from aotools.functions.zernike import zernIndex

from .utils.optics_utils import lens_phase, circular_mask
from .utils.tensor_utils import (
    unsqueeze_to, pad_to_shape_2D, crop_to_shape_2D
)
from .propagators.abstract import PropagatorBase


class ConstantSLMField(PropagatorBase):
    def __init__(
        self: ConstantSLMField,
        init_field: torch.Tensor[torch.complex],
        pixel_pitch: float,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            init_field.shape[-2:],
            (pixel_pitch, pixel_pitch),
            device=device,
        )
        self.phase = nn.Parameter(
            torch.tensor(
                init_field.angle(), dtype=self.dtype, device=self.device
            ),
            requires_grad=self.training,
        )

        self.amplitude = nn.Parameter(
            torch.tensor(
                init_field.abs(), dtype=self.dtype, device=self.device
            ),
            requires_grad=self.training,
        )

    def forward(
        self: ConstantSLMField, input_field: torch.Tensor = None
    ) -> torch.Tensor:
        input_field = unsqueeze_to(input_field, 3)
        amplitude = unsqueeze_to(self.amplitude, 3)
        phase = unsqueeze_to(self.phase, 3)

        return (input_field * amplitude * torch.exp(1j * phase)).squeeze()


class PartialAffineTransform(PropagatorBase):
    def __init__(
        self: PartialAffineTransform,
        resolution_in: tuple[int, int],
        resolution_out: tuple[int, int],
        pixel_size_in: tuple[float, float],
        pixel_size_out: tuple[float, float],
        scale_factor: tuple[float, float] = (1, 1),
        shift: tuple[float, float] = (0, 0),
        angle: float = 0.0,
        rotation_center_shift: tuple[float, float] = (0, 0),
        verbose: bool = True,
        device: str = "cpu",
    ) -> None:
        super().__init__(resolution_in, pixel_size_in, device=device)
        
        self._resolution_out = resolution_out
        self._pixel_size_out = pixel_size_out

        # Scaling factor
        self.scale_factor = nn.Parameter(
            torch.tensor(scale_factor, dtype=self.dtype, device=self.device),
            requires_grad=False,
        )

        # Shift from the center in pixels
        self.shift = nn.Parameter(
            torch.tensor(shift, dtype=self.dtype, device=self.device),
            requires_grad=False,
        )

        # Rotation angle in degrees
        self.angle = nn.Parameter(
            torch.tensor(angle, dtype=self.dtype, device=self.device),
            requires_grad=False,
        )

        # Shift of the rotation center relative to the shift_center
        self.rotation_center_shift = nn.Parameter(
            torch.tensor(
                rotation_center_shift, dtype=self.dtype, device=self.device
            ),
            requires_grad=False,
        )

        self.verbose = verbose

        # Setting scaling to pixel size ratios
        scale = tuple(
            self.pixel_size_in[i] / pixel_size_out[i] for i in range(2)
        )[::-1]
        self.scale = torch.tensor(scale, dtype=self.dtype, device=device)

        # Setting the rotation center to the center of the input image
        rotation_center = (resolution_in[1] // 2, resolution_in[0] // 2)
        self.rotation_center = torch.tensor(
            rotation_center, dtype=self.dtype, device=device
        )

        # Shift moving the centre of the input image to the center of the 
        # output image
        shift_center = tuple(
            self.rotation_center[i] * (self.scale[i] - 1)
            + (resolution_out[i] - resolution_in[i] * self.scale[i]) / 2
            for i in range(2)
        )[::-1]
        self.shift_center = torch.tensor(
            shift_center, dtype=self.dtype, device=device
        )
        
        self.affine_matrix = self.get_affine_matrix()

    @property
    def pixel_size_out(self: PartialAffineTransform) -> tuple[float, float]:
        return self._pixel_size_out

    @property
    def resolution_out(self: PartialAffineTransform) -> tuple[int, int]:
        return self._resolution_out

    def get_affine_matrix(self) -> torch.Tensor:
        return get_affine_matrix2d(
            (self.shift_center + self.shift).unsqueeze(0),
            (self.rotation_center + self.rotation_center_shift).unsqueeze(0),
            (self.scale * self.scale_factor).unsqueeze(0),
            self.angle.unsqueeze(0),
        )

    def forward(
        self: PartialAffineTransform, input_field: torch.Tensor
    ) -> torch.Tensor:
        """Applies partial affine transformation to input_field."""
        if self.verbose:
            print("Scale:", self.scale.data)
            print("Shift:", self.shift.data)
            print("Angle:", self.angle.data)

        input_field = unsqueeze_to(input_field, 4)
        self.affine_matrix = self.get_affine_matrix()

        # Kornia does not support complex numbers in warp_perspective(),
        # so we need to split the real and imaginary parts and then
        # recombine them.
        output_real = warp_perspective(
            input_field.real, self.affine_matrix, self.resolution_out
        )
        output_imag = warp_perspective(
            input_field.imag, self.affine_matrix, self.resolution_out
        )
        
        transformed_field = (output_real + 1j * output_imag).squeeze()

        # Normalize to conserve optical power
        norm = self.scale.prod().sqrt()
        transformed_field /= norm
        return transformed_field


class SimpleLens(PropagatorBase):
    def __init__(
        self: SimpleLens,
        focal_length: float,
        aperture_radius: float,
        wavelength: float,
        resolution_in: tuple[int, int],
        pixel_pitch_in: float,
        device: str = "cpu",
    ) -> None:
        self.focal_length = focal_length
        self.wavelength = wavelength
        self.aperture_radius = aperture_radius

        super().__init__(
            wavelength=wavelength,
            resolution_in=resolution_in,
            pixel_size_in=(pixel_pitch_in, pixel_pitch_in),
            device=device,
        )

        spatial_grid = self.get_spatial_grid_input()

        self.lens_phase = lens_phase(
            spatial_grid[1],
            spatial_grid[0],
            self.focal_length,
            self.wavenumber,
        )

        self.lens_aperture = circular_mask(
            spatial_grid[1],
            spatial_grid[0],
            self.aperture_radius,
            shift_x=self.spatial_extent_in[0] / 2,
            shift_y=self.spatial_extent_in[1] / 2,
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


class Zernike(PropagatorBase):
    """Experimental module, currently not fully tested."""
    def __init__(
        self: Zernike,
        resolution_in: tuple[int, int],
        pixel_size_in: tuple[float, float],
        number_of_orders: int,
        initial_coefficients: torch.Tensor | None = None,
        norm: str | None = "noll",
        device: str = "cpu",
    ) -> None:
        super().__init__(
            resolution_in=resolution_in,
            pixel_size_in=pixel_size_in,
            device=device,
        )

        self.number_of_coefficients = (
            (number_of_orders + 1) * (number_of_orders + 2) // 2
        )

        if initial_coefficients is None:
            initial_coefficients = 0.1 * torch.rand(
                self.number_of_coefficients,
                dtype=self.dtype,
                device=self.device,
            )
        else:
            if initial_coefficients.shape[0] != self.number_of_coefficients:
                raise ValueError(
                    "Initial Zernike coefficients must have shape "
                    + f"({self.number_of_coefficients},), but got "
                    + f"{initial_coefficients.shape}."
                )
            else:
                initial_coefficients = initial_coefficients.to(
                    dtype=self.dtype, device=self.device
                )

        self.zernike_coefficients = nn.Parameter(
            initial_coefficients, requires_grad=True
        )

        unit_circle_diameter = (
            int(
                (self.resolution_in[0] ** 2 + self.resolution_in[1] ** 2)
                ** 0.5
            ) // 2 * 2
        )

        self.zernike_array = torch.tensor(
            crop_to_shape_2D(
                zernikeArray(
                    self.number_of_coefficients,
                    unit_circle_diameter,
                    norm=norm,
                ),
                self.resolution_in,
            ),
            dtype=self.dtype,
            device=device,
        )

        self.maximum_gradients = self.get_gradient_maxima()

    def get_gradient_maxima(self):
        n_index = torch.zeros(
            self.number_of_coefficients, dtype=self.dtype, device=self.device
        )
        m_index = torch.zeros(
            self.number_of_coefficients, dtype=self.dtype, device=self.device
        )

        for i in range(self.number_of_coefficients):
            n_index[i], m_index[i] = zernIndex(i + 1)

        delta = torch.zeros_like(m_index)
        delta[m_index == 0] = 1

        order = 0.5 * n_index * (n_index + 2) - m_index ** 2

        max_gradients = (2 * (n_index + 1) / (1 + delta)) * torch.maximum(
            order, m_index.abs()
        )
        return max_gradients

    def get_phase(self) -> torch.Tensor:
        zernike_array_padded = pad_to_shape_2D(
            self.zernike_array, self.resolution_in
        )
        phase = torch.sum(
            unsqueeze_to(self.zernike_coefficients, 3, dim=1)
            # / (1 + unsqueeze_to(self.maximum_gradients, 3, dim=1))
            * zernike_array_padded,
            dim=0,
        )
        return phase

    def forward(
        self: Zernike, input_field: torch.Tensor | None = None
    ) -> torch.Tensor:
        phase = self.get_phase()

        if input_field is None:
            return torch.exp(1j * phase).squeeze()
        else:
            input_field = unsqueeze_to(input_field, 3)
            return (input_field * torch.exp(1j * phase)).squeeze()


# TODO: Implement doublet lens module
