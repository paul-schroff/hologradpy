"""Diagonal (per-pixel) elements that modify the electric field in place."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import Parameter

from .utils.optics_utils import lens_phase, circular_mask, doublet_lens
from .utils.fourier_utils import get_spatial_grid
from .utils.zernike import (
    Zernike,
    Conventions,
    make_per_wavelength_coefficients,
)
from .optics_module import OpticsModule, SaveDict
from .complex_amplitude import (
    ComplexAmplitude,
    broadcast_wavelength_operand,
)

if TYPE_CHECKING:
    from ..calibration.wavefront.abstract import WavefrontCalibrationData


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
        self._ensure_initialized()
        return self._modulate(complex_amplitude, self.get_transmission().conj())


class StaticSLMField(DiagonalElement):
    def __init__(
        self: StaticSLMField,
        init_field: ComplexAmplitude | None = None,
    ) -> None:
        super().__init__()
        self.init_field: ComplexAmplitude | None = init_field

    def lazy_init(self: StaticSLMField, complex_amplitude: ComplexAmplitude) -> None:
        super().lazy_init(complex_amplitude)

        if self.init_field is None:
            number_of_wavelengths = complex_amplitude.number_of_wavelengths
            # A uniform default field is wavelength-independent, but the
            # ComplexAmplitude layout requires an explicit wavelength axis when
            # more than one wavelength is present.
            default_shape = (
                self.resolution_in
                if number_of_wavelengths == 1
                else (number_of_wavelengths, *self.resolution_in)
            )
            self.init_field = ComplexAmplitude(
                data=torch.ones(
                    default_shape,
                    dtype=complex_amplitude.dtype,
                    device=complex_amplitude.device,
                ),
                wavelength=complex_amplitude.wavelength,
                pixel_size=complex_amplitude.pixel_size,
            )

        self.phase = Parameter(
            torch.tensor(
                self.init_field.phase,
                dtype=complex_amplitude.dtype_r,
                device=complex_amplitude.device,
            ),
            requires_grad=False,
        )

        self.amplitude = Parameter(
            torch.tensor(
                self.init_field.amplitude,
                dtype=complex_amplitude.dtype_r,
                device=complex_amplitude.device,
            ),
            requires_grad=False,
        )

    @classmethod
    def from_file(cls, path: str, device: torch.device = "cpu") -> StaticSLMField:
        state: SaveDict = torch.load(path, map_location=device, weights_only=False)
        state_dict = state["state_dict"]
        geometry = state["input_geometry"]

        init_field_data: Tensor = state_dict["amplitude"] * torch.exp(
            1j * state_dict["phase"]
        )

        init_field = ComplexAmplitude(
            data=init_field_data.to(device),
            wavelength=geometry.wavelength.to(device),
            pixel_size=geometry.pixel_size.to(device),
        )
        return cls(init_field=init_field)

    @classmethod
    def from_calibration_data(
        cls, calibration_data: WavefrontCalibrationData
    ) -> StaticSLMField:
        return cls(init_field=calibration_data.complex_amplitude)

    def get_transmission(self: StaticSLMField) -> Tensor:
        """Complex transmission ``amplitude * exp(i * phase)`` — the stored
        constant field, applied as a per-pixel diagonal multiply."""
        return self.amplitude * torch.exp(1j * self.phase)


class SimpleLens(DiagonalElement):
    """Ideal thin lens: an apertured quadratic phase mask.

    The lens phase is built per-wavelength at lazy initialisation.
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
        super().lazy_init(complex_amplitude)

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

        # Aperture is wavelength-independent; centred on the grid.
        aperture = circular_mask(grid_x, grid_y, self.aperture_radius)

        # Complex transmission of the apertured lens.
        self.register_buffer("transmission", aperture * torch.exp(1j * phase))


class DoubletLens(DiagonalElement):
    """Achromatic doublet lens phase mask.

    Models the phase imparted by a cemented doublet (a crown and a flint
    element) as the optical path difference through three spherical surfaces,
    following the ``doublet`` profile on the main branch. The phase is built
    per-wavelength at lazy initialisation (the wavenumber is per-wavelength;
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
            shift: Lateral offset ``(shift_x, shift_y)`` of the lens centre [m].
        """
        super().__init__()

        self.refractive_index_flint: float = refractive_index_flint
        self.refractive_index_crown: float = refractive_index_crown
        self.radius_crown: float = radius_crown
        self.radius_crown_flint: float = radius_crown_flint
        self.radius_flint: float = radius_flint
        self.shift: tuple[float, float] = shift

    def lazy_init(self: DoubletLens, complex_amplitude: ComplexAmplitude) -> None:
        super().lazy_init(complex_amplitude)

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
        unit_disk_mode: str = "fill",
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
        super().lazy_init(complex_amplitude)

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
