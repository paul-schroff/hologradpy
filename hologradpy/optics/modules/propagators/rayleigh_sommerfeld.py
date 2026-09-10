from __future__ import annotations

from dataclasses import replace

import torch
from jaxtyping import Complex
from scipy.fft import next_fast_len
from torch import Tensor

from ..abstract import OpticsModule, capture_init
from ...complex_amplitude import BatchSpec, ComplexAmplitude, FieldGeometry

DEFAULT_BLOCK = 2**22


def _in_double(geometry: FieldGeometry) -> FieldGeometry:
    """The same geometry with a float64 pitch, so its grid is built in double."""
    return replace(geometry, pixel_size=geometry.pixel_size.to(torch.float64))


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

    def _distance(self, device: torch.device) -> Tensor:
        """The propagation distance as a float64 tensor."""
        return torch.as_tensor(
            self.propagation_distance, dtype=torch.float64, device=device
        )

    def _sample_points(
        self,
        complex_amplitude: ComplexAmplitude,
        geometry: FieldGeometry | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Source and target sample points, each ``(number_of_points, 3)`` in metres."""
        device = complex_amplitude.device
        source = _in_double(complex_amplitude.geometry).positions()

        if geometry is None:
            pixel_out = self.pixel_size_out.reshape(-1, 2)[0].to(torch.float64)
            plane = FieldGeometry(
                wavelength=complex_amplitude.wavelength,
                pixel_size=pixel_out.unsqueeze(0),
                resolution=tuple(self.resolution_out),
            )
            distance = self._distance(device)
            zero = torch.zeros_like(distance)
            target = plane.positions() + torch.stack((zero, zero, distance))
        else:
            target = _in_double(geometry).positions()

        return source.reshape(-1, 3), target.reshape(-1, 3)

    def _kernel(
        self,
        separation: Tensor,
        obliquity: Tensor,
        wavenumber: Tensor,
        area: Tensor,
    ) -> Tensor:
        """The integrand at a set of separations.

        Args:
            separation: Distance between the two points, any shape.
            obliquity: Cosine between the separation and the source plane's normal.
            wavenumber: Wavenumber, broadcasting against ``separation``.
            area: Area of one source pixel, the quadrature weight.

        Returns:
            Tensor: The kernel, shaped by the broadcast.
        """
        return (
            obliquity
            * torch.exp(1j * wavenumber * separation)
            / separation
            * (1 / separation - 1j * wavenumber)
            * (area / (2 * torch.pi))
        )

    def _shares_a_grid(self, complex_amplitude: ComplexAmplitude) -> bool:
        """Checks if both planes sample the same grid, so the sum is a convolution.

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
        pitch: Tensor,
        size: int,
        device: torch.device,
    ) -> Tensor:
        """Separations along one axis, as the convolution indexes them."""
        offset = length_in // 2 - length_out // 2
        index = torch.arange(size, device=device, dtype=torch.float64)
        return (index + offset - length_in + 1) * pitch

    def _by_convolution(
        self, complex_amplitude: ComplexAmplitude, conjugate: bool
    ) -> tuple[Complex[Tensor, "N n_wavelengths H_out W_out"], BatchSpec]:
        """The same sum, evaluated as a convolution.

        Args:
            complex_amplitude: The field to propagate.
            conjugate: Correlate with the conjugate kernel, which is the conjugate
                transpose of the forward convolution.

        Returns:
            tuple[Tensor, BatchSpec]: The propagated fields, ``(N, n_wavelengths,
            H_out, W_out)``, and the spec that restores the input rank.
        """
        resolution_in = tuple(complex_amplitude.resolution)
        resolution_out = tuple(self.resolution_out)
        if conjugate:
            resolution_in, resolution_out = resolution_out, resolution_in

        pixel = complex_amplitude.pixel_size.reshape(-1, 2)[0].to(torch.float64)
        area = pixel[0] * pixel[1]

        sizes = tuple(
            next_fast_len(resolution_in[axis] + resolution_out[axis] - 1)
            for axis in range(2)
        )
        device = complex_amplitude.device
        offsets_y = self._offset_axis(
            resolution_in[0], resolution_out[0], pixel[0], sizes[0], device
        )
        offsets_x = self._offset_axis(
            resolution_in[1], resolution_out[1], pixel[1], sizes[1], device
        )
        grid_x, grid_y = torch.meshgrid(offsets_x, offsets_y, indexing="xy")
        distance = self._distance(device)
        separation = torch.sqrt(grid_x**2 + grid_y**2 + distance**2)

        flat, spec = complex_amplitude.flatten_batch()  # (N, n_wavelengths, H, W)
        flat = flat.to(torch.complex128)

        wavenumber = complex_amplitude.wavenumber.reshape(-1).to(torch.float64)
        top = resolution_in[0] - 1
        left = resolution_in[1] - 1
        outputs = []
        for index in range(flat.shape[1]):
            kernel = self._kernel(
                separation, distance / separation, wavenumber[index], area
            )
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
        return stacked.to(complex_amplitude.dtype), spec

    def _apply(
        self,
        complex_amplitude: ComplexAmplitude,
        conjugate: bool,
        geometry: FieldGeometry | None = None,
    ) -> tuple[Complex[Tensor, "N n_wavelengths H_out W_out"], BatchSpec]:
        """Sum the integral, in blocks of output points.

        Args:
            complex_amplitude: The field to propagate.
            conjugate: Take the conjugate kernel, which gives the conjugate
                transpose of the forward sum.
            geometry: Where to sample the result. Defaults to the output plane.

        Returns:
            tuple[Tensor, BatchSpec]: The propagated fields on the output grid,
            ``(N, n_wavelengths, H_out, W_out)``, and the spec that restores the input
            rank.
        """
        source, target = self._sample_points(complex_amplitude, geometry)
        if conjugate:
            source, target = target, source

        pixel_in = complex_amplitude.pixel_size.reshape(-1, 2)[0].to(torch.float64)
        area = pixel_in[0] * pixel_in[1]
        wavenumber = complex_amplitude.wavenumber.reshape(-1).to(torch.float64)

        flat, spec = complex_amplitude.flatten_batch()  # (N, n_wavelengths, H, W)
        number_of_fields, number_of_wavelengths = flat.shape[:2]
        
        flat = flat.reshape(number_of_fields, number_of_wavelengths, -1).to(
            torch.complex128
        )

        rows_out = len(target)
        # As many output points as keep one block within the budget.
        chunk = max(1, self.block // max(1, len(source)))
        pieces = []
        for start in range(0, rows_out, chunk):
            stop = min(start + chunk, rows_out)
            offset = target[start:stop, None, :] - source[None, :, :]
            separation = torch.linalg.vector_norm(offset, dim=-1)
            # The source plane is transverse, so its normal is z and the obliquity
            # is the z component of the separation over its length.
            obliquity = offset[..., 2] / separation
            # (n_wavelengths, chunk, n_source).
            kernel = self._kernel(
                separation[None],
                obliquity[None],
                wavenumber[:, None, None],
                area,
            )
            if conjugate:
                kernel = torch.conj(kernel)
            # One matrix product per wavelength, broadcast over the batch.
            pieces.append(torch.einsum("bwi,wci->bwc", flat, kernel))

        propagated = torch.cat(pieces, dim=-1)
        if conjugate:
            shape = tuple(complex_amplitude.resolution)
        elif geometry is not None:
            shape = tuple(geometry.resolution)
        else:
            shape = tuple(self.resolution_out)
        propagated = propagated.reshape(number_of_fields, number_of_wavelengths, *shape)
        return propagated.to(complex_amplitude.dtype), spec

    def propagate_to(
        self, complex_amplitude: ComplexAmplitude, geometry: FieldGeometry
    ) -> ComplexAmplitude:
        """Propagate the field onto a plane oriented according to ``geometry``.

        Args:
            complex_amplitude: The field to propagate.
            geometry: Geometry of the field.

        Returns:
            ComplexAmplitude: The field on ``geometry``, carrying its pose.
        """
        if complex_amplitude.is_vector:
            raise NotImplementedError(
                "The components of a field vector are given in the plane's own frame, "
                "and the target plane has a frame of its own, so they need rotating "
                "into it. Propagate a scalar field."
            )
        if not self.initialized:
            self._lazy_initialize(complex_amplitude)
        out, spec = self._apply(complex_amplitude, conjugate=False, geometry=geometry)
        return ComplexAmplitude.unflatten_batch(
            out, spec, complex_amplitude.wavelength, geometry.pixel_size
        ).with_geometry(origin=geometry.origin, rotation=geometry.rotation)

    def forward(self, complex_amplitude: ComplexAmplitude) -> ComplexAmplitude:
        """Propagate the field forward by ``propagation_distance``."""
        if self.convolution and self._shares_a_grid(complex_amplitude):
            out, spec = self._by_convolution(complex_amplitude, conjugate=False)
        else:
            out, spec = self._apply(complex_amplitude, conjugate=False)
        return ComplexAmplitude.unflatten_batch(
            out, spec, complex_amplitude.wavelength, self.pixel_size_out
        )

    def adjoint(self, complex_amplitude: ComplexAmplitude) -> ComplexAmplitude:
        """The conjugate transpose of :meth:`forward`."""
        if self.convolution and self._shares_a_grid(complex_amplitude):
            out, spec = self._by_convolution(complex_amplitude, conjugate=True)
        else:
            out, spec = self._apply(complex_amplitude, conjugate=True)
        return ComplexAmplitude.unflatten_batch(
            out, spec, complex_amplitude.wavelength, self.pixel_size_in
        )
