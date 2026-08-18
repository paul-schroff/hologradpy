"""Diagonal (per-pixel) elements that modify the electric field in place."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Parameter

from ...profiles.phase import (
    lens_phase,
    doublet_lens,
)
from ...profiles.masks import circular_mask
from ...grids import get_spatial_grid
from ...profiles.zernike import (
    DEFAULT_UNIT_DISK_MODE,
    Zernike,
    Conventions,
    make_per_wavelength_coefficients,
)
from .abstract import OpticsModule
from ..complex_amplitude import (
    ComplexAmplitude,
    broadcast_wavelength_operand,
)


class DiagonalElement(OpticsModule):
    """Base for diagonal (per-pixel) optical elements that preserve sampling.

    The output pixel size and resolution match the input. :meth:`forward`
    multiplies the field by the element's complex transmission
    ``(n_wavelengths, H, W)`` and :meth:`adjoint` by its conjugate (the
    conjugate transpose of a diagonal operator).

    Subclasses provide the transmission via :meth:`get_transmission`. The
    default returns a static ``transmission`` buffer (set in :meth:`lazy_init`,
    e.g. a fixed lens phase); subclasses whose transmission depends on
    parameters (e.g. learnable coefficients) override it to recompute each call.
    """

    transmission: Tensor

    def get_transmission(self: DiagonalElement) -> Tensor:
        return self.transmission

    def _modulate(
        self: DiagonalElement,
        complex_amplitude: ComplexAmplitude,
        transmission: Tensor,
    ) -> ComplexAmplitude:
        transmission = broadcast_wavelength_operand(
            transmission, complex_amplitude.ndim
        )
        modulated = complex_amplitude * transmission
        return modulated.with_geometry(
            wavelength=complex_amplitude.wavelength,
            pixel_size=self.pixel_size_out,
        )

    def forward(
        self: DiagonalElement, complex_amplitude: ComplexAmplitude
    ) -> ComplexAmplitude:
        return self._modulate(complex_amplitude, self.get_transmission())

    def adjoint(
        self: DiagonalElement, complex_amplitude: ComplexAmplitude
    ) -> ComplexAmplitude:
        """Conjugate transpose of :meth:`forward`."""
        return self._modulate(complex_amplitude, self.get_transmission().conj())



class SimpleLens(DiagonalElement):
    """Ideal thin lens: an apertured quadratic phase mask.

    The lens phase is built per-wavelength at lazy initialization.
    """

    def __init__(
        self: SimpleLens,
        focal_length: float,
        aperture_radius: float,
    ) -> None:
        super().__init__()

        self.focal_length: float = focal_length
        self.aperture_radius: float = aperture_radius

    def lazy_init(self: SimpleLens, complex_amplitude: ComplexAmplitude) -> None:
        grid_x, grid_y = get_spatial_grid(
            self.resolution_in,
            tuple(self.pixel_size_in[0].tolist()),
            complex_amplitude.device,
        )

        # Per-wavelength lens phase (n_wavelengths, H, W).
        wavenumber = complex_amplitude.wavenumber.reshape(-1, 1, 1)
        phase = lens_phase(
            grid_x.unsqueeze(0),
            grid_y.unsqueeze(0),
            self.focal_length,
            wavenumber,
        )

        # Aperture is wavelength-independent; centered on the grid.
        aperture = circular_mask(grid_x, grid_y, self.aperture_radius)

        # Complex transmission of the apertured lens.
        self.register_buffer("transmission", aperture * torch.exp(1j * phase))


class DoubletLens(DiagonalElement):
    """Achromatic doublet lens phase mask.

    Models the phase imparted by a cemented doublet (a crown and a flint
    element) as the optical path difference through three spherical surfaces,
    following the ``doublet`` profile on the main branch. The phase is built
    per-wavelength at lazy initialization (the wavenumber is per-wavelength;
    the refractive indices are treated as constant over the wavelength range).
    """

    def __init__(
        self: DoubletLens,
        refractive_index_flint: float,
        refractive_index_crown: float,
        radius_crown: float,
        radius_crown_flint: float,
        radius_flint: float,
        shift: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        """
        Args:
            refractive_index_flint: Refractive index of the flint element.
            refractive_index_crown: Refractive index of the crown element.
            radius_crown: Radius of curvature of the first crown surface [m].
            radius_crown_flint: Radius of curvature of the cemented
                crown/flint surface [m].
            radius_flint: Radius of curvature of the second flint surface [m].
            shift: Lateral offset ``(shift_x, shift_y)`` of the lens center [m].
        """
        super().__init__()

        self.refractive_index_flint: float = refractive_index_flint
        self.refractive_index_crown: float = refractive_index_crown
        self.radius_crown: float = radius_crown
        self.radius_crown_flint: float = radius_crown_flint
        self.radius_flint: float = radius_flint
        self.shift: tuple[float, float] = shift

    def lazy_init(self: DoubletLens, complex_amplitude: ComplexAmplitude) -> None:
        grid_x, grid_y = get_spatial_grid(
            self.resolution_in,
            tuple(self.pixel_size_in[0].tolist()),
            complex_amplitude.device,
        )

        # Per-wavelength doublet phase (n_wavelengths, H, W).
        wavenumber = complex_amplitude.wavenumber.reshape(-1, 1, 1)
        phase = doublet_lens(
            grid_x.unsqueeze(0),
            grid_y.unsqueeze(0),
            wavenumber,
            self.refractive_index_flint,
            self.refractive_index_crown,
            self.radius_crown,
            self.radius_crown_flint,
            self.radius_flint,
            shift_x=self.shift[0],
            shift_y=self.shift[1],
        )

        self.register_buffer("transmission", torch.exp(1j * phase))


class ZernikePhase(DiagonalElement):
    """Phase element imparting a per-wavelength Zernike phase.

    Learns an independent set of Zernike coefficients
    ``(n_wavelengths, n_coefficients)`` and applies ``exp(i * phase)``, where
    the phase for each wavelength is the coefficient-weighted sum of a fixed
    Zernike basis built lazily for the input resolution. The transmission is
    recomputed from the coefficients each call so gradients flow back to them.
    """

    def __init__(
        self: ZernikePhase,
        number_of_radial_orders: int = 5,
        initial_coefficients: torch.Tensor | None = None,
        convention: Conventions = "Noll",
        unit_disk_mode: str = DEFAULT_UNIT_DISK_MODE,
    ) -> None:
        """
        Args:
            number_of_radial_orders: Number of radial Zernike orders to
                include (orders ``0 .. number_of_radial_orders - 1``).
            initial_coefficients: Optional initial coefficients. May be a 1D
                tensor ``(n_coefficients,)`` broadcast across all wavelengths,
                or a 2D tensor ``(n_wavelengths, n_coefficients)`` to seed each
                wavelength independently. Defaults to small random values.
            convention: Zernike ordering/normalization convention.
            unit_disk_mode: How the unit disk maps onto the resolution
                (``"fill"`` covers the corners, ``"fit"`` inscribes it).
        """
        super().__init__()

        self.number_of_radial_orders: int = number_of_radial_orders
        self.initial_coefficients: torch.Tensor | None = initial_coefficients
        self.convention: Conventions = convention
        self.unit_disk_mode: str = unit_disk_mode

    def lazy_init(self: ZernikePhase, complex_amplitude: ComplexAmplitude) -> None:
        # Build the (n_coefficients, H, W) Zernike basis for the input
        # resolution and keep it as a buffer.
        zernike = Zernike(
            resolution=self.resolution_in,
            unit_disk_mode=self.unit_disk_mode,
            number_of_radial_orders=self.number_of_radial_orders,
            convention=self.convention,
            device=complex_amplitude.device,
        )
        self.register_buffer(
            "zernike_basis",
            zernike.zernike_array.to(dtype=complex_amplitude.dtype_r),
        )

        coefficients = make_per_wavelength_coefficients(
            self.initial_coefficients,
            complex_amplitude.number_of_wavelengths,
            zernike.number_of_zernikes,
            complex_amplitude.dtype_r,
            complex_amplitude.device,
        )
        self.zernike_coefficients = Parameter(coefficients, requires_grad=True)

    def get_phase(self: ZernikePhase) -> torch.Tensor:
        """Per-wavelength phase ``(n_wavelengths, H, W)`` from the learnable
        coefficients and the Zernike basis."""
        return torch.einsum(
            "lc,chw->lhw", self.zernike_coefficients, self.zernike_basis
        )

    def get_transmission(self: ZernikePhase) -> Tensor:
        return torch.exp(1j * self.get_phase())
