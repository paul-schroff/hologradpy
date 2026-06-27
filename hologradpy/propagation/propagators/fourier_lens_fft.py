from __future__ import annotations

import torch
from torch import nn, Tensor

from ...utils import pad_to_shape_2D, crop_to_shape_2D
from ..fourier import fft_2d, ifft_2d

from ..optics_module import OpticsModule, SaveDict
from ..complex_amplitude import ComplexAmplitude


class FourierLensFFT(OpticsModule):
    def __init__(
        self,
        focal_length: float,
        pixel_size_out: tuple[float, float] | None = None,
        padded_resolution: tuple[int, int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(pixel_size_out, padded_resolution)

        if pixel_size_out is not None and padded_resolution is not None:
            raise ValueError(
                "Specify either pixel_size_out or padded_resolution, not both."
            )

        self.focal_length: float = nn.Parameter(
            torch.tensor(focal_length, dtype=torch.float32),
            requires_grad=False,
        )
        self._padded_resolution_init: tuple[int, int] | None = padded_resolution

        self.kwargs = kwargs

    def lazy_init(self, complex_amplitude: ComplexAmplitude) -> None:
        # Only padded_resolution is provided
        if self._padded_resolution_init is not None and self._pixel_size_out is None:
            # check if the provided padded resolution is at least
            # as large as the input resolution
            if (
                self._padded_resolution_init[0] < complex_amplitude.resolution[0]
                or self._padded_resolution_init[1] < complex_amplitude.resolution[1]
            ):
                raise ValueError(
                    "Padded resolution must be at least as large as input resolution."
                )

            # Check if padded_resolution is even
            parity = tuple(self._padded_resolution_init[i] % 2 for i in range(2))

            if parity[0] != 0 or parity[1] != 0:
                raise ValueError("Padded resolution must be even.")

            self._padded_resolution = torch.tensor(
                self._padded_resolution_init,
                device=complex_amplitude.device,
                dtype=torch.int64,
            )
            self._pixel_size_out = self._get_pixel_size_out(
                complex_amplitude.wavelength,
                self.focal_length,
                self.pixel_size_in,
                self._padded_resolution,
            )
        # Only pixel_size_out is provided
        elif (
            self._padded_resolution_init is None
            and self._pixel_size_out_init is not None
        ):
            requested_pixel_size_out = torch.tensor(
                self._pixel_size_out_init,
                device=complex_amplitude.device,
                dtype=torch.float32,
            )

            max_pixel_size_out = self._get_pixel_size_out(
                complex_amplitude.wavelength[0],
                self.focal_length,
                self.pixel_size_in[0],
                torch.tensor(
                    complex_amplitude.resolution,
                    device=complex_amplitude.device,
                ),
            )

            # Check if the requested pixel size is larger than the maximum
            # allowed by the input geometry and focal length
            if (
                requested_pixel_size_out[0] > max_pixel_size_out[0]
                or requested_pixel_size_out[1] > max_pixel_size_out[1]
            ):
                raise ValueError(
                    "Requested pixel size out is too large for the given "
                    "input geometry and focal length. Maximum pixel size out "
                    f"for the first wavelength is {max_pixel_size_out}."
                )

            # Uses the first wavelength and pixel size for calculating the
            # padded resolution.
            self._padded_resolution = self._get_padded_resolution(
                complex_amplitude.wavelength[0],
                self.focal_length,
                self.pixel_size_in[0],
                requested_pixel_size_out,
            )

            self._pixel_size_out = self._get_pixel_size_out(
                complex_amplitude.wavelength,
                self.focal_length,
                self.pixel_size_in,
                self._padded_resolution,
            )
        # If neither is provided, default to zero-padding to double the
        # input resolution
        else:
            self._padded_resolution_init = tuple(
                2 * complex_amplitude.resolution[i] for i in range(2)
            )
            self._padded_resolution = torch.tensor(
                self._padded_resolution_init,
                device=complex_amplitude.device,
                dtype=torch.int64,
            )
            self._pixel_size_out = self._get_pixel_size_out(
                complex_amplitude.wavelength,
                self.focal_length,
                self.pixel_size_in,
                self._padded_resolution,
            )

        self._resolution_out = tuple(self._padded_resolution.tolist())

    @property
    def padded_resolution(self) -> Tensor:
        return self._padded_resolution

    @staticmethod
    def _get_padded_resolution(
        wavelength: Tensor,
        focal_length: Tensor,
        pixel_size_in: Tensor,
        pixel_size_out: Tensor,
    ) -> Tensor[torch.int64]:
        padded_resolution = (
            wavelength * focal_length / (pixel_size_in * pixel_size_out) // 2 * 2
        )
        return padded_resolution.to(torch.int64)

    @staticmethod
    def _get_pixel_size_out(
        wavelength: Tensor,
        focal_length: Tensor,
        pixel_size_in: Tensor,
        padded_resolution: Tensor,
    ) -> Tensor:
        return wavelength * focal_length / (pixel_size_in * padded_resolution)

    @property
    def pixel_size_out(self) -> Tensor:
        return self._get_pixel_size_out(
            self.input_geometry.wavelength,
            self.focal_length,
            self.pixel_size_in,
            self.padded_resolution,
        )

    def save(self, path: str) -> None:
        save_dict: SaveDict = {
            "state_dict": self.state_dict(),
            "input_geometry": self.input_geometry,
            "resolution_out": self.resolution_out,
            "pixel_size_out": self.pixel_size_out,
            "kwargs": self.kwargs,
        }
        torch.save(save_dict, path)

    @classmethod
    def from_file(cls, path: str, device: torch.device = "cpu") -> FourierLensFFT:
        state: SaveDict = torch.load(path, map_location=device, weights_only=False)
        sd = state["state_dict"]
        module = cls(
            focal_length=sd["focal_length"].item(),
            padded_resolution=state["resolution_out"],
            **state.get("kwargs", {}),
        )
        return module

    def forward(self, complex_amplitude: ComplexAmplitude) -> ComplexAmplitude:
        padded_complex_amplitude = pad_to_shape_2D(
            complex_amplitude, self.resolution_out
        )

        # Perform 2D FFT and FFT shift if specified
        out = fft_2d(padded_complex_amplitude, **self.kwargs)

        return out.with_geometry(
            wavelength=complex_amplitude.wavelength,
            pixel_size=self.pixel_size_out,
        )

    def adjoint(self, complex_amplitude: ComplexAmplitude) -> ComplexAmplitude:
        self._ensure_initialized()

        # Perform inverse 2D FFT and FFT shift if specified
        padded_complex_amplitude = ifft_2d(complex_amplitude, **self.kwargs)

        out: ComplexAmplitude = crop_to_shape_2D(
            padded_complex_amplitude, self.resolution_in
        )

        return out.with_geometry(
            wavelength=complex_amplitude.wavelength,
            pixel_size=self.pixel_size_in,
        )
