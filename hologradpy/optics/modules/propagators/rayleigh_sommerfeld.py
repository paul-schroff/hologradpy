from __future__ import annotations

import torch
from scipy.fft import next_fast_len
from torch import Tensor

from ....grids import get_spatial_grid
from ..abstract import OpticsModule, capture_init
from ...complex_amplitude import ComplexAmplitude

DEFAULT_BLOCK = 2**22


class RayleighSommerfeld(OpticsModule):
    """The Rayleigh-Sommerfeld diffraction integral, summed directly.

    No paraxial approximation and no band limit, making it the ground truth to benchmark
    the fast propagators against.
    """

    @capture_init
    def __init__(
        self,
        propagation_distance: float,
        pixel_size_out: Tensor | tuple[float, float] | None = None,
        resolution_out: tuple[int, int] | None = None,
        block: int = DEFAULT_BLOCK,
        convolution: bool = True,
    ) -> None:
        """
        Args:
            propagation_distance: How far to propagate, in metres.
            pixel_size_out: Output pixel size ``(height, width)`` in metres. Defaults
                to the input's.
            resolution_out: Output resolution ``(height, width)``. Defaults to the
                input's.
            block: Kernel entries evaluated per pass, output points times source
                points.
            convolution: Take the convolution route when both planes sample the same
                resolution and pitch.
        """
        super().__init__(
            pixel_size_out=pixel_size_out, resolution_out=resolution_out
        )
        self.propagation_distance = propagation_distance
        self.block = block
        self.convolution = convolution

    def _grids(
        self, complex_amplitude: ComplexAmplitude
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """The two sample grids, each flattened to ``(number_of_points,)``."""
        device = complex_amplitude.device
        pixel_in = complex_amplitude.pixel_size.reshape(-1, 2)[0].to(torch.float64)
        pixel_out = self.pixel_size_out.reshape(-1, 2)[0].to(torch.float64)

        source_x, source_y = get_spatial_grid(
            tuple(complex_amplitude.resolution), pixel_in, device
        )
        target_x, target_y = get_spatial_grid(
            tuple(self.resolution_out), pixel_out, device
        )
        return (
            source_x.flatten(),
            source_y.flatten(),
            target_x.flatten(),
            target_y.flatten(),
        )

    def _kernel(self, separation: Tensor, wavenumber: Tensor, area: float) -> Tensor:
        """The integrand at a set of separations.

        Args:
            separation: Distance between the two points, any shape.
            wavenumber: Wavenumber, broadcasting against ``separation``.
            area: Area of one source pixel, the quadrature weight.

        Returns:
            Tensor: The kernel, shaped by the broadcast.
        """
        distance = float(self.propagation_distance)
        return (
            (distance / separation)
            * torch.exp(1j * wavenumber * separation)
            / separation
            * (1 / separation - 1j * wavenumber)
            * (area / (2 * torch.pi))
        )

    def _shares_a_lattice(self, complex_amplitude: ComplexAmplitude) -> bool:
        """Checks if both planes sample the same lattice, so the sum is a convolution.

        Args:
            complex_amplitude: The field about to be propagated.

        Returns:
            bool: True when the convolution route applies.
        """
        pixel_in = complex_amplitude.pixel_size.reshape(-1, 2)[0].to(torch.float64)
        pixel_out = self.pixel_size_out.reshape(-1, 2)[0].to(torch.float64)
        return bool(torch.allclose(pixel_in, pixel_out, rtol=1e-12, atol=0.0))

    def _offset_axis(
        self,
        length_in: int,
        length_out: int,
        pitch: float,
        size: int,
        device: torch.device,
    ) -> Tensor:
        """Separations along one axis, as the convolution indexes them."""
        offset = length_in // 2 - length_out // 2
        index = torch.arange(size, device=device, dtype=torch.float64)
        return (index + offset - length_in + 1) * pitch

    def _by_convolution(
        self, complex_amplitude: ComplexAmplitude, conjugate: bool
    ) -> Tensor:
        """The same sum, evaluated as a convolution.

        Args:
            complex_amplitude: The field to propagate.
            conjugate: Correlate with the conjugate kernel instead, which is the
                conjugate transpose of the forward convolution.

        Returns:
            Tensor: The propagated field.
        """
        resolution_in = tuple(complex_amplitude.resolution)
        resolution_out = tuple(self.resolution_out)
        if conjugate:
            resolution_in, resolution_out = resolution_out, resolution_in

        pixel = complex_amplitude.pixel_size.reshape(-1, 2)[0]
        area = float(pixel[0]) * float(pixel[1])

        sizes = tuple(
            next_fast_len(resolution_in[axis] + resolution_out[axis] - 1)
            for axis in range(2)
        )
        device = complex_amplitude.device
        offsets_y = self._offset_axis(
            resolution_in[0], resolution_out[0], float(pixel[0]), sizes[0], device
        )
        offsets_x = self._offset_axis(
            resolution_in[1], resolution_out[1], float(pixel[1]), sizes[1], device
        )
        grid_x, grid_y = torch.meshgrid(offsets_x, offsets_y, indexing="xy")
        separation = torch.sqrt(
            grid_x**2 + grid_y**2 + float(self.propagation_distance) ** 2
        )

        field = complex_amplitude.as_tensor()
        original = field.shape
        points = original[-2] * original[-1]
        if field.ndim == 2:
            flat = field.reshape(1, 1, points)
        elif field.ndim == 3:
            flat = field.reshape(1, original[0], points)
        else:
            flat = field.reshape(-1, original[-3], points)
        flat = flat.to(torch.complex128).reshape(
            flat.shape[0], flat.shape[1], *original[-2:]
        )

        wavenumber = complex_amplitude.wavenumber.reshape(-1).to(torch.float64)
        top = resolution_in[0] - 1
        left = resolution_in[1] - 1
        outputs = []
        for index in range(flat.shape[1]):
            kernel = self._kernel(separation, wavenumber[index], area)
            spectrum = torch.fft.fft2(kernel)
            if conjugate:
                spectrum = torch.conj(spectrum)
                placed = torch.nn.functional.pad(
                    flat[:, index],
                    (left, sizes[1] - resolution_out[1] - left,
                     top, sizes[0] - resolution_out[0] - top),
                )
                product = torch.fft.ifft2(torch.fft.fft2(placed) * spectrum)
                outputs.append(product[..., : resolution_out[0], : resolution_out[1]])
                continue
            placed = torch.nn.functional.pad(
                flat[:, index],
                (0, sizes[1] - resolution_in[1], 0, sizes[0] - resolution_in[0]),
            )
            product = torch.fft.ifft2(torch.fft.fft2(placed) * spectrum)
            outputs.append(
                product[..., top : top + resolution_out[0],
                        left : left + resolution_out[1]]
            )

        stacked = torch.stack(outputs, dim=1)
        return stacked.reshape(*original[:-2], *resolution_out).to(field.dtype)

    def _apply(
        self,
        complex_amplitude: ComplexAmplitude,
        conjugate: bool,
    ) -> Tensor:
        """Sum the integral, in blocks of output points.

        Args:
            complex_amplitude: The field to propagate.
            conjugate: Take the conjugate kernel, which gives the conjugate transpose
                of the forward sum rather than its inverse.

        Returns:
            Tensor: The propagated field, shaped like the input but on the output grid.
        """
        source_x, source_y, target_x, target_y = self._grids(complex_amplitude)
        if conjugate:
            source_x, source_y, target_x, target_y = (
                target_x,
                target_y,
                source_x,
                source_y,
            )

        distance = float(self.propagation_distance)
        pixel_in = complex_amplitude.pixel_size.reshape(-1, 2)[0]
        area = float(pixel_in[0]) * float(pixel_in[1])
        wavenumber = complex_amplitude.wavenumber.reshape(-1).to(torch.float64)

        field = complex_amplitude.as_tensor()
        original = field.shape
        # A field may arrive as (H, W), (wavelengths, H, W) or with batches in front.
        # Normalizing to (batch, wavelengths, points) keeps one product below.
        points = original[-2] * original[-1]
        if field.ndim == 2:
            flat = field.reshape(1, 1, points)
        elif field.ndim == 3:
            flat = field.reshape(1, original[0], points)
        else:
            flat = field.reshape(-1, original[-3], points)
        flat = flat.to(torch.complex128)

        rows_out = len(target_x)
        # As many output points as keep one block within the budget.
        chunk = max(1, self.block // max(1, len(source_x)))
        pieces = []
        for start in range(0, rows_out, chunk):
            stop = min(start + chunk, rows_out)
            separation = torch.sqrt(
                (target_x[start:stop, None] - source_x[None, :]) ** 2
                + (target_y[start:stop, None] - source_y[None, :]) ** 2
                + distance**2
            )
            # (n_wavelengths, chunk, n_source).
            kernel = self._kernel(
                separation[None], wavenumber[:, None, None], area
            )
            if conjugate:
                kernel = torch.conj(kernel)
            # One matrix product per wavelength, broadcast over the batch.
            pieces.append(torch.einsum("bwi,wci->bwc", flat, kernel))

        propagated = torch.cat(pieces, dim=-1)
        shape = (
            complex_amplitude.resolution if conjugate else tuple(self.resolution_out)
        )
        return propagated.reshape(*original[:-2], *shape).to(field.dtype)

    def forward(self, complex_amplitude: ComplexAmplitude) -> ComplexAmplitude:
        """Propagate the field forward by ``propagation_distance``."""
        if self.convolution and self._shares_a_lattice(complex_amplitude):
            out = self._by_convolution(complex_amplitude, conjugate=False)
        else:
            out = self._apply(complex_amplitude, conjugate=False)
        return ComplexAmplitude(
            out, wavelength=complex_amplitude.wavelength, pixel_size=self.pixel_size_out
        )

    def adjoint(self, complex_amplitude: ComplexAmplitude) -> ComplexAmplitude:
        """The conjugate transpose of :meth:`forward`."""
        if self.convolution and self._shares_a_lattice(complex_amplitude):
            out = self._by_convolution(complex_amplitude, conjugate=True)
        else:
            out = self._apply(complex_amplitude, conjugate=True)
        return ComplexAmplitude(
            out, wavelength=complex_amplitude.wavelength, pixel_size=self.pixel_size_in
        )
