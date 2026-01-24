from __future__ import annotations

from jaxtyping import Float

import torch
from torch import Tensor
from torch.nn import Parameter

from torchkbnufft import KbNufft

from ..utils.tensor_utils import unsqueeze_to
from ..utils.fourier_utils import get_frequency_grid

from .abstract import PropagatorBase


class FourierLensNUFFT(PropagatorBase):
    def __init__(
        self: FourierLensNUFFT,
        focal_length: float,
        wavelength: float,
        resolution_in: tuple[int, int],
        pixel_size_in: tuple[float, float],
        resolution_out: tuple[int, int],
        pixel_size_out: tuple[float, float],
        calculation_resolution: tuple[int, int] | None = None,
        scale_factor: tuple[float, float] = (1, 1),
        shift: tuple[float, float] = (0, 0),
        angle: float = 0,
        nufft_kwargs: dict = {},
        device: str = "cpu",
    ) -> None:
        super().__init__(resolution_in, pixel_size_in, device)

        self.focal_length: float = focal_length
        self.wavelength: float = wavelength

        self._resolution_out: tuple[int, int] = resolution_out
        self._pixel_size_out: tuple[float, float] = pixel_size_out

        if calculation_resolution is None:
            calculation_resolution = tuple(
                resolution_in[i] * 2 for i in range(2)
            )
        self.calculation_resolution: tuple[int, int] = calculation_resolution

        # Calculating scale to achieve the desired output pixel size
        self.scale: Float[Tensor, "2"] = torch.tensor(
            [
                self.wavelength
                * self.focal_length
                / (self.pixel_size_in[i] * self.calculation_resolution[i])
                / self.pixel_size_out[i]
                for i in range(2)
            ],
            dtype=self.dtype_r,
            device=self.device,
        )

        self.scale_factor: Float[Tensor, "2"] = Parameter(
            torch.tensor(
                scale_factor,
                dtype=self.dtype_r,
                device=device,
            ),
            requires_grad=False,
        )

        # Shift from the center of the in pixels
        self.shift: Float[Tensor, "2"] = Parameter(
            torch.tensor(
                shift,
                dtype=self.dtype_r,
                device=device,
            ),
            requires_grad=False,
        )

        # Rotation angle in degrees
        self.angle: Float[Tensor, ""] = Parameter(
            torch.tensor(
                angle,
                dtype=self.dtype_r,
                device=device,
            ),
            requires_grad=False,
        )

        resolution_ratio: tuple[float, float] = tuple(
            self.calculation_resolution[i] / self.resolution_out[i]
            for i in range(2)
        )

        self.kbnufft: KbNufft = KbNufft(
            im_size=self.resolution_in,
            grid_size=self.calculation_resolution,
            device=self.device,
            dtype=self.dtype_r,
            **nufft_kwargs,
        )

        frequency_grid: tuple[Float[Tensor, "h w"], Float[Tensor, "h w"]] = (
            get_frequency_grid(
                self.resolution_out,
                resolution_ratio,
                self.device,
            )
        )

        # Flatten frequency grid
        self.frequencies: tuple[Float[Tensor, " hw"], Float[Tensor, " hw"]] = (
            tuple(frequency_grid[i].flatten() for i in range(2))
        )

        self.frequencies_flattened: Float[Tensor, "2 hw"] = (
            self._get_transformed_coordinates(
                self.scale_factor, self.shift, self.angle
            )
        )

    @property
    def pixel_size_out(self: FourierLensNUFFT) -> tuple[float, float]:
        return self._pixel_size_out

    @property
    def resolution_out(self: FourierLensNUFFT) -> tuple[int, int]:
        return self._resolution_out

    def _get_transformed_coordinates(
        self: FourierLensNUFFT,
        scale_factor: Float[Tensor, "2"],
        shift: Float[Tensor, "2"],
        angle: Float[Tensor, ""],
    ) -> Float[Tensor, "2 hw"]:
        scale_factor = scale_factor.abs() * self.scale

        shift_randians = tuple(
            2
            * torch.pi
            * shift[i]
            / (self.calculation_resolution[i] * self.scale[i])
            for i in range(2)
        )

        angle_radians = torch.deg2rad(angle)
        angle_sin = angle_radians.sin()
        angle_cos = angle_radians.cos()

        frequencies_transformed = (
            (
                self.frequencies[0] * angle_cos / scale_factor[1]
                - self.frequencies[1] * angle_sin / scale_factor[0]
                - shift_randians[1] * angle_cos
                + shift_randians[0] * angle_sin
            ),
            (
                self.frequencies[0] * angle_sin / scale_factor[1]
                + self.frequencies[1] * angle_cos / scale_factor[0]
                - shift_randians[1] * angle_sin
                - shift_randians[0] * angle_cos
            ),
        )

        return torch.stack(frequencies_transformed, dim=0)

    def forward(
        self: FourierLensNUFFT,
        input_field: Float[Tensor, "... h w"] | Float[Tensor, "h w"],
    ) -> Float[Tensor, "... h w"] | Float[Tensor, "h w"]:
        self.number_of_images = (
            input_field.shape[-3] if input_field.dim() > 2 else 1
        )
        input_field = unsqueeze_to(input_field, 3)

        output_field = self.kbnufft(
            unsqueeze_to(input_field, 4),
            unsqueeze_to(self.frequencies_flattened, 3),
        ).squeeze()

        return output_field.reshape((
            self.number_of_images,
            self.resolution_out[0],
            self.resolution_out[1],
        )).squeeze()
