from __future__ import annotations

from jaxtyping import Float

import torch
from torch import Tensor
from torch.nn import Parameter

from torchkbnufft import KbNufft, KbNufftAdjoint

from ..utils.fourier_utils import get_frequency_grid

from ..optics_module import OpticsModule
from ..complex_amplitude import ComplexAmplitude


class FourierLensNUFFT(OpticsModule):
    def __init__(
        self: FourierLensNUFFT,
        focal_length: float,
        resolution_out: tuple[int, int],
        pixel_size_out: tuple[float, float],
        padded_resolution: tuple[int, int] | None = None,
        shift: tuple[float, float] = (0, 0),
        angle: float = 0,
        nufft_kwargs: dict = {},
    ) -> None:
        super().__init__(pixel_size_out, resolution_out)

        self.focal_length: float = focal_length
        self._padded_resolution_init: tuple[int, int] | None = (
            padded_resolution
        )
        self.shift_init: tuple[float, float] = shift
        self.angle_init: float = angle
        self.nufft_kwargs = nufft_kwargs

    def lazy_init(
        self: FourierLensNUFFT, complex_amplitude: ComplexAmplitude
    ) -> None:
        self._pixel_size_out = torch.tensor(
            self._pixel_size_out_init,
            device=complex_amplitude.pixel_size.device,
            dtype=complex_amplitude.pixel_size.dtype,
        )

        if self._padded_resolution_init is not None:
            # check if the provided padded resolution is at least
            # as large as the input resolution
            if (
                self._padded_resolution_init[0]
                < complex_amplitude.resolution[0]
                or self._padded_resolution_init[1]
                < complex_amplitude.resolution[1]
            ):
                raise ValueError(
                    "Padded resolution must be at least as large as input "
                    "resolution."
                )

            # Check if padded_resolution is even
            parity = tuple(
                self._padded_resolution_init[i] % 2 for i in range(2)
            )

            if parity[0] != 0 or parity[1] != 0:
                raise ValueError("Padded resolution must be even.")
        else:
            self._padded_resolution_init = tuple(
                2 * complex_amplitude.resolution[i] for i in range(2)
            )

        self._padded_resolution: tuple[int, int] = self._padded_resolution_init
        self._padded_resolution_tensor = torch.tensor(
            self._padded_resolution,
            device=complex_amplitude.device,
            dtype=complex_amplitude.dtype_r,
        )

        # Calculating scale to achieve the desired output pixel size
        self._scale: Float[Tensor, "n_wavelenghts 2"] = (
            complex_amplitude.wavelength.unsqueeze(-1)
            * self.focal_length
            / (
                complex_amplitude.pixel_size
                * self._padded_resolution_tensor.unsqueeze(0)
            )
            / self._pixel_size_out.unsqueeze(0)
        )

        self.scale_factor: Float[Tensor, "2"] = Parameter(
            torch.ones(
                2,
                dtype=complex_amplitude.dtype_r,
                device=complex_amplitude.device,
            ),
            requires_grad=False,
        )

        # Shift from the center of the in pixels
        self.shift: Float[Tensor, "2"] = Parameter(
            torch.tensor(
                self.shift_init,
                dtype=complex_amplitude.dtype_r,
                device=complex_amplitude.device,
            ),
            requires_grad=False,
        )

        # Rotation angle in degrees
        self.angle: Float[Tensor, "1"] = Parameter(
            torch.tensor(
                self.angle_init,
                dtype=complex_amplitude.dtype_r,
                device=complex_amplitude.device,
            ),
            requires_grad=False,
        )

        # TODO: Add support for each wavelength
        resolution_ratio: tuple[float, float] = tuple(
            self._padded_resolution[i] / self.resolution_out[i]
            for i in range(2)
        )

        self._kbnufft: KbNufft = KbNufft(
            im_size=complex_amplitude.resolution,
            grid_size=self._padded_resolution,
            device=complex_amplitude.device,
            dtype=complex_amplitude.dtype_r,
            **self.nufft_kwargs,
        )

        self._kbnufft_adjoint: KbNufftAdjoint = KbNufftAdjoint(
            im_size=complex_amplitude.resolution,
            grid_size=self._padded_resolution,
            device=complex_amplitude.device,
            dtype=complex_amplitude.dtype_r,
            **self.nufft_kwargs,
        )

        frequency_grid: tuple[Float[Tensor, "h w"], Float[Tensor, "h w"]] = (
            get_frequency_grid(
                self.resolution_out,
                resolution_ratio,
                complex_amplitude.device,
            )
        )

        # Flatten frequency grid
        self.frequencies: tuple[
            Float[Tensor, "n_wavelenghts hw"],
            Float[Tensor, "n_wavelenghts hw"],
        ] = tuple(
            frequency_grid[i]
            .flatten()
            .expand(complex_amplitude.number_of_wavelengths, -1)
            for i in range(2)
        )

        self.frequencies_transformed: Float[Tensor, "n_wavelenghts 2 hw"] = (
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
        angle: Float[Tensor, "1"],
    ) -> Float[Tensor, "n_wavelenghts hw 2"]:
        scale_factor: Float[Tensor, "n_wavelenghts 2"] = (
            scale_factor.abs().unsqueeze(0) * self._scale
        )

        shift_randians: tuple[
            Float[Tensor, " n_wavelenghts"], Float[Tensor, " n_wavelenghts"]
        ] = tuple(
            2
            * torch.pi
            * shift[i]
            / (self._padded_resolution[i] * self._scale[:, i])
            for i in range(2)
        )

        angle_radians = torch.deg2rad(angle)
        angle_sin = angle_radians.sin()
        angle_cos = angle_radians.cos()

        frequencies_transformed: tuple[
            Float[Tensor, "n_wavelenghts hw"],
            Float[Tensor, "n_wavelenghts hw"],
        ] = (
            (
                self.frequencies[0]
                * angle_cos
                / scale_factor[:, 1].unsqueeze(-1)
                - self.frequencies[1]
                * angle_sin
                / scale_factor[:, 0].unsqueeze(-1)
                - shift_randians[1].unsqueeze(-1) * angle_cos
                + shift_randians[0].unsqueeze(-1) * angle_sin
            ),
            (
                self.frequencies[0]
                * angle_sin
                / scale_factor[:, 1].unsqueeze(-1)
                + self.frequencies[1]
                * angle_cos
                / scale_factor[:, 0].unsqueeze(-1)
                - shift_randians[1].unsqueeze(-1) * angle_sin
                - shift_randians[0].unsqueeze(-1) * angle_cos
            ),
        )

        return torch.stack(frequencies_transformed, dim=0)

    def _batched_k_trajectory(
        self: FourierLensNUFFT,
        number_of_images: int,
        number_of_wavelengths: int,
    ) -> Float[Tensor, "n_images_wl 2 hw"]:
        """Tile the per-wavelength k-space trajectory across the batch images.

        The stored trajectory is ``(2, n_wl, hw)``; this returns
        ``(n_images * n_wl, 2, hw)`` with wavelength alignment matching the
        row-major (image, wavelength) flattening used by ``flatten_batch``.
        """
        k_traj: Float[Tensor, "n_wl 2 hw"] = (
            self.frequencies_transformed.moveaxis(0, 1)
        )
        k_traj = k_traj.unsqueeze(0).expand(number_of_images, -1, -1, -1)
        return k_traj.reshape(
            number_of_images * number_of_wavelengths, 2, -1
        )

    def forward(
        self: FourierLensNUFFT,
        complex_amplitude: ComplexAmplitude,
    ) -> ComplexAmplitude:
        """Propagate a field of arbitrary batch rank ``(*batch, n_wl, H, W)``.

        All leading batch dimensions are collapsed into a single axis, the
        NUFFT is evaluated with the per-wavelength k-space trajectory tiled
        across that axis, and the original rank is restored on the way out.
        """
        number_of_wavelengths = complex_amplitude.number_of_wavelengths

        # Collapse all batch dimensions into a single leading axis:
        # (N, n_wavelengths, H, W).
        flat_field, batch_spec = complex_amplitude.flatten_batch()
        number_of_images = flat_field.shape[0]

        # torchkbnufft expects (batch, coil, H, W); merge (image, wavelength)
        # into the batch axis with a single coil.
        input_field: Float[Tensor, "n_images_wl 1 h w"] = flat_field.reshape(
            number_of_images * number_of_wavelengths,
            1,
            *complex_amplitude.resolution,
        )

        k_traj = self._batched_k_trajectory(
            number_of_images, number_of_wavelengths
        )

        output_field: Float[Tensor, "n_images_wl 1 hw"] = self._kbnufft(
            input_field, k_traj
        )

        # Restore canonical (N, n_wavelengths, H_out, W_out) layout.
        output_field = output_field.reshape(
            number_of_images,
            number_of_wavelengths,
            self.resolution_out[0],
            self.resolution_out[1],
        )

        return ComplexAmplitude.unflatten_batch(
            output_field,
            batch_spec,
            complex_amplitude.wavelength,
            self.pixel_size_out,
        )

    def adjoint(
        self: FourierLensNUFFT,
        complex_amplitude: ComplexAmplitude,
    ) -> ComplexAmplitude:
        """Adjoint NUFFT mapping an output-plane field back to the input plane.

        This is the conjugate transpose of :meth:`forward` (via
        ``KbNufftAdjoint``), not its inverse. The input is a field sampled on
        the output grid ``(*batch, n_wl, H_out, W_out)``; the output lives in
        the input plane with ``resolution_in`` / ``pixel_size_in``. The module
        must be initialised first (via ``forward`` or
        ``initialize_from_geometry``).
        """
        self._ensure_initialized()

        number_of_wavelengths = complex_amplitude.number_of_wavelengths

        # Collapse batch dims and flatten the output-plane image into the
        # k-space sample axis: (n_images * n_wl, 1, hw).
        flat_field, batch_spec = complex_amplitude.flatten_batch()
        number_of_images = flat_field.shape[0]

        samples: Float[Tensor, "n_images_wl 1 hw"] = flat_field.reshape(
            number_of_images * number_of_wavelengths, 1, -1
        )

        k_traj = self._batched_k_trajectory(
            number_of_images, number_of_wavelengths
        )

        input_field: Float[Tensor, "n_images_wl 1 h w"] = (
            self._kbnufft_adjoint(samples, k_traj)
        )

        # Restore canonical (N, n_wavelengths, H_in, W_in) layout.
        input_field = input_field.reshape(
            number_of_images,
            number_of_wavelengths,
            self.resolution_in[0],
            self.resolution_in[1],
        )

        return ComplexAmplitude.unflatten_batch(
            input_field,
            batch_spec,
            complex_amplitude.wavelength,
            self.pixel_size_in,
        )
