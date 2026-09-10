from __future__ import annotations

import warnings
from dataclasses import replace

import torch
from torch import Tensor

from ....fourier_transforms import fft_2d, ifft_2d
from ....grids import get_frequency_grid
from ....utils import to_canvas

from ..abstract import OpticsModule
from ...complex_amplitude import (
    BatchSpec,
    ComplexAmplitude,
    FieldGeometry,
    broadcast_wavelength_operand,
)

try:
    from pytorch_finufft.functional import finufft_type1
except ImportError:
    # Only propagate_to() needs it, so the propagator itself imports without it.
    finufft_type1 = None

NUFFT_INSTALL_HINT = (
    "AngularSpectrumMethod.propagate_to needs pytorch-finufft, which is an optional "
    "dependency."
)


def _pose(geometry: FieldGeometry, device: torch.device) -> tuple[Tensor, Tensor]:
    """A geometry's rotation and origin as float64 tensors with defaults when absent."""
    options = {"dtype": torch.float64, "device": device}
    rotation = (
        torch.eye(3, **options)
        if geometry.rotation is None
        else geometry.rotation.to(**options)
    )
    origin = (
        torch.zeros(3, **options)
        if geometry.origin is None
        else geometry.origin.to(**options)
    )
    return rotation, origin


DEFAULT_ENCLOSED_ENERGY = 0.999
QUANTILE_BUCKETS = 1024


def _phase_step_quantile(
    step: Tensor, energy: Tensor, enclosed_energy: float
) -> Tensor:
    """The phase step ``enclosed_energy`` of the energy stays inside, per wavelength.

    Args:
        step: ``(n_wavelengths, n_waves)`` phase steps per sample, non-negative.
        energy: ``(n_wavelengths, n_waves)`` weight of each wave.
        enclosed_energy: The share of the energy that has to sit inside the answer.

    Returns:
        Tensor: ``(n_wavelengths,)`` phase steps in radians.
    """
    largest_step = step.amax(dim=-1, keepdim=True)

    bucket = (
        (step / largest_step.clamp(min=torch.finfo(step.dtype).tiny))
        .mul(QUANTILE_BUCKETS)
        .long()
        .clamp(max=QUANTILE_BUCKETS - 1)
    )
    histogram = torch.zeros(
        step.shape[0], QUANTILE_BUCKETS, dtype=energy.dtype, device=energy.device
    ).scatter_add_(1, bucket, energy)
    # Energy in this bucket and every one beyond it.
    tail_energy = histogram.flip(-1).cumsum(-1).flip(-1)
    within_allowance = tail_energy <= (1.0 - enclosed_energy) * tail_energy[:, :1]
    edge_bucket = torch.where(
        within_allowance.any(dim=-1),
        within_allowance.to(torch.uint8).argmax(dim=-1),
        torch.full_like(within_allowance[:, 0], QUANTILE_BUCKETS, dtype=torch.long),
    )
    return edge_bucket.to(step.dtype) / QUANTILE_BUCKETS * largest_step.squeeze(-1)


def _nyquist_ratio(
    points: Tensor, coefficients: Tensor, propagating: Tensor,
    enclosed_energy: float
) -> tuple[float, float]:
    """The pitch as a multiple of the Nyquist pitch, per output axis."""
    energy = (coefficients.abs() ** 2 * propagating).sum(dim=0)
    steps = torch.stack(
        [
            _phase_step_quantile(points[:, axis].abs(), energy, enclosed_energy)
            for axis in range(2)
        ],
        dim=-1,
    )
    return (float(steps[:, 0].max()) / torch.pi, float(steps[:, 1].max()) / torch.pi)


class AngularSpectrumMethod(OpticsModule):
    """Near-field propagation by the angular spectrum method.

    Propagates a field a distance ``propagation_distance`` while preserving the
    sampling (output pixel size and resolution equal the input). The field is
    zero-padded to ``padded_resolution`` before the transform to avoid circular
    convolution wraparound, then cropped back.

    The transform to and from the angular-spectrum domain is an orthonormal FFT,
    so it conserves energy and :meth:`adjoint`, which back-propagates with the
    conjugate transfer function, is the exact conjugate transpose of
    :meth:`forward`. A zoomed or off-axis angular spectrum is
    :class:`AngularSpectrumSAFT`.
    """

    def __init__(
        self: AngularSpectrumMethod,
        propagation_distance: float,
        padded_resolution: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()

        self.propagation_distance: float = propagation_distance
        self._padded_resolution_init: tuple[int, int] | None = padded_resolution

    def lazy_init(
        self: AngularSpectrumMethod, complex_amplitude: ComplexAmplitude
    ) -> None:
        resolution_in = complex_amplitude.resolution

        if self._padded_resolution_init is None:
            self._padded_resolution = tuple(2 * resolution_in[i] for i in range(2))
        else:
            if (
                self._padded_resolution_init[0] < resolution_in[0]
                or self._padded_resolution_init[1] < resolution_in[1]
            ):
                raise ValueError(
                    "Padded resolution must be at least as large as input resolution."
                )
            if any(self._padded_resolution_init[i] % 2 for i in range(2)):
                raise ValueError("Padded resolution must be even.")
            self._padded_resolution = self._padded_resolution_init

        self.register_buffer("phase_factor", self._get_phase_factor(complex_amplitude))

    def _get_phase_factor(
        self: AngularSpectrumMethod, complex_amplitude: ComplexAmplitude
    ) -> Tensor:
        """Angular-spectrum transfer function ``(n_wavelengths, H, W)``.

        The transfer function is evaluated on the padded frequency grid, in
        radians per metre, the same grid :meth:`_plane_wave_sum` decomposes the
        field over.
        """
        frequency_grid_x, frequency_grid_y = get_frequency_grid(
            self._padded_resolution,
            complex_amplitude.pixel_size[0],
            complex_amplitude.device,
        )

        # Per-wavelength wavenumber broadcast over the (padded) frequency grid.
        wavenumber = complex_amplitude.wavenumber.reshape(-1, 1, 1)
        axial_frequency_squared = (
            wavenumber**2
            - frequency_grid_x.unsqueeze(0) ** 2
            - frequency_grid_y.unsqueeze(0) ** 2
        )
        # ``+ 0j`` allows the square root to go imaginary for evanescent waves.
        return torch.exp(
            1j * self.propagation_distance
            * torch.sqrt(axial_frequency_squared + 0j)
        )

    def _propagate(
        self: AngularSpectrumMethod,
        complex_amplitude: ComplexAmplitude,
        transfer_function: Tensor,
    ) -> ComplexAmplitude:
        transfer_function = broadcast_wavelength_operand(
            transfer_function, complex_amplitude.ndim
        )
        padded = to_canvas(complex_amplitude, self._padded_resolution)
        spectrum = fft_2d(padded, norm="ortho")
        propagated = ifft_2d(spectrum * transfer_function, norm="ortho")
        out = to_canvas(propagated, self.resolution_out)
        return out.with_geometry(
            wavelength=complex_amplitude.wavelength,
            pixel_size=self.pixel_size_out,
        )

    def forward(
        self: AngularSpectrumMethod, complex_amplitude: ComplexAmplitude
    ) -> ComplexAmplitude:
        return self._propagate(complex_amplitude, self.phase_factor)

    def adjoint(
        self: AngularSpectrumMethod, complex_amplitude: ComplexAmplitude
    ) -> ComplexAmplitude:
        """Back-propagation by ``-propagation_distance``, the conjugate
        transpose of :meth:`forward`.
        """
        return self._propagate(complex_amplitude, self.phase_factor.conj())

    def _plane_wave_sum(
        self: AngularSpectrumMethod,
        complex_amplitude: ComplexAmplitude,
        geometry: FieldGeometry,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Where to evaluate the plane-wave sum, ``E(p) = sum_j A_j exp(i k_j . p)``.

        Each wave vector is projected onto the two axes of the output grid, ``(k . u,
        k . v)``. The result is a non-uniform point cloud, since the axial frequency,
        ``kz = sqrt(k^2 - kx^2 - ky^2)``, varies nonlinearly across the frequency grid.
        The sum is therefore evaluated with one type-1 NUFFT. On a plane of constant z,
        the projection is the frequency grid itself, so the usual route is an inverse
        FFT.

        The carrier, ``exp(i k z)``, is divided out before the transform, so one sample
        along z only has to resolve the bandwidth of ``kz`` (:meth:`nyquist_ratio`).

        Returns:
            tuple: Four tensors.

                - The sample points, ``(n_wavelengths, 2, n_waves)``, in radians per
                  output sample.
                - The phase carrying the section's centre away from the source plane,
                  ``(n_wavelengths, n_waves)``.
                - Which waves propagate, same shape.
                - The carrier, ``(n_wavelengths, *geometry.resolution)``.
        """
        device = complex_amplitude.device
        pitch_in = complex_amplitude.pixel_size.reshape(-1, 2)[0].to(torch.float64)
        frequency_x, frequency_y = get_frequency_grid(
            self._padded_resolution, pitch_in, device
        )
        frequency_x = frequency_x.reshape(-1)
        frequency_y = frequency_y.reshape(-1)
        wavenumber = complex_amplitude.wavenumber.reshape(-1, 1).to(torch.float64)

        transverse_squared = frequency_x**2 + frequency_y**2
        propagating = transverse_squared <= wavenumber**2
        axial_frequency = torch.sqrt(
            torch.clamp(wavenumber**2 - transverse_squared, min=0.0)
        )
        # Evanescent waves grow without bound in one direction along z, and a
        # section runs both ways, so they are set to zero. The carrier is taken
        # out here, which is what leaves the bandwidth to be resolved.
        reduced_axial = torch.where(
            propagating,
            axial_frequency - wavenumber,
            torch.zeros_like(axial_frequency),
        )

        wave_vectors = torch.stack(
            (
                frequency_x.unsqueeze(0).expand_as(reduced_axial),
                frequency_y.unsqueeze(0).expand_as(reduced_axial),
                reduced_axial,
            ),
            dim=1,
        )

        rotation, origin = _pose(geometry, device)
        _, source_origin = _pose(complex_amplitude.geometry, device)
        pitch_out = geometry.pixel_size.reshape(-1, 2)[0].to(torch.float64)

        # The wave vectors are projected onto the grid's column and row directions,
        # the first two columns of its rotation, and scaled to radians per sample.
        projected = torch.einsum("da,wdm->wam", rotation[:, :2], wave_vectors)
        points = torch.stack(
            (projected[:, 1] * pitch_out[0], projected[:, 0] * pitch_out[1]), dim=1
        )
        phase = torch.exp(
            1j * torch.einsum("wdm,d->wm", wave_vectors, origin - source_origin)
        )

        geometry_in_double = replace(
            geometry, pixel_size=geometry.pixel_size.to(torch.float64)
        )
        along_axis = geometry_in_double.positions()[..., 2] - source_origin[2]
        carrier = torch.exp(1j * wavenumber.reshape(-1, 1, 1) * along_axis)

        return points, phase, propagating, carrier

    def _spectrum(
        self: AngularSpectrumMethod, complex_amplitude: ComplexAmplitude
    ) -> tuple[Tensor, BatchSpec]:
        """The padded field as plane-wave coefficients ``(n_images, n_wl, n_waves)``,
        with the batch layout needed to put the result back together.
        """
        padded = to_canvas(complex_amplitude, self._padded_resolution)
        images, spec = padded.flatten_batch()
        number_of_modes = self._padded_resolution[0] * self._padded_resolution[1]
        coefficients = (fft_2d(images) / number_of_modes).reshape(
            images.shape[0], images.shape[1], -1
        )
        return coefficients, spec

    def nyquist_ratio(
        self: AngularSpectrumMethod,
        complex_amplitude: ComplexAmplitude,
        geometry: FieldGeometry,
        enclosed_energy: float = DEFAULT_ENCLOSED_ENERGY,
    ) -> tuple[float, float]:
        """The pitch of ``geometry`` as a multiple of the Nyquist pitch, per output
        axis, rows before columns.

        A ratio of at most 1 on both axes means the section resolves the field, since
        one sample then advances the phase of the fastest wave by at most ``pi``. A
        ratio of 3 means the pitch is three times too coarse, so the picture between
        the samples is a moire of the field. The sample values themselves stay exact at
        any pitch, since the sum is evaluated at the sample points
        (:meth:`propagate_to`).

        The fastest wave is taken at the quantile inside which ``enclosed_energy`` of
        the field's energy sits. Across a transverse output the ratio is plain Nyquist
        and reaches 1 when the output pitch matches the input pitch and the field fills
        the band. Along z it is set by the bandwidth of ``kz``, which for a beam of
        numerical aperture ``NA`` requires ``dz < pi / (k (1 - sqrt(1 - NA^2)))``.

        Args:
            complex_amplitude: The field that would be propagated.
            geometry: Where it would be sampled.
            enclosed_energy: The share of the field's energy that has to be
                resolved. The ratio follows the energy, so a wave carrying a
                millionth of the power weighs a millionth as much.

        Returns:
            tuple[float, float]: The ratio along the rows and along the columns.
        """
        if not self.initialized:
            self._lazy_initialize(complex_amplitude)
        points, _, propagating, _ = self._plane_wave_sum(complex_amplitude, geometry)
        coefficients, _ = self._spectrum(complex_amplitude)
        return _nyquist_ratio(points, coefficients, propagating, enclosed_energy)

    def _warn_if_unresolved(
        self: AngularSpectrumMethod,
        ratio: tuple[float, float],
        geometry: FieldGeometry,
    ) -> None:
        if max(ratio) <= 1.0:
            return
        pitch = geometry.pixel_size.reshape(-1, 2)[0]
        warnings.warn(
            "This section does not resolve the field it samples. Nyquist ratio "
            f"({ratio[0]:.2f}, {ratio[1]:.2f}) down the rows and across the "
            "columns, which has to be at most 1. The values returned are still "
            "exact, but a pitch of "
            f"{float(pitch[0]) * 1e6:.3f} x {float(pitch[1]) * 1e6:.3f} um steps "
            "over more than half a period of what is there, so the picture between "
            "the samples is a moire of the field. Ask for a finer pitch on "
            "the axis over 1. See nyquist_ratio().",
            RuntimeWarning,
            stacklevel=3,
        )

    def propagate_to(
        self: AngularSpectrumMethod,
        complex_amplitude: ComplexAmplitude,
        geometry: FieldGeometry,
        enclosed_energy: float = DEFAULT_ENCLOSED_ENERGY,
    ) -> ComplexAmplitude:
        """Propagate the field onto a grid at any location and angle.

        The targets are a regular grid placed and oriented in three dimensions by
        ``geometry``, usually an x-z or y-z cross section built by
        :meth:`~hologradpy.optics.complex_amplitude.FieldGeometry.cross_section`.

        Args:
            complex_amplitude: The field to propagate.
            geometry: Where to sample the propagated field.
            enclosed_energy: Passed to :meth:`nyquist_ratio`.

        Returns:
            ComplexAmplitude: The field on ``geometry``, carrying its pose.

        Raises:
            ImportError: ``pytorch-finufft`` is not installed.
            ValueError: The input field is not sampled on a transverse plane.
        """
        if finufft_type1 is None:
            raise ImportError(NUFFT_INSTALL_HINT)
        if complex_amplitude.is_vector:
            raise NotImplementedError(
                "The components of a field vector are given in the plane's own frame, "
                "and the target plane has a frame of its own, so they need rotating "
                "into it. Propagate a scalar field."
            )
        if not self.initialized:
            self._lazy_initialize(complex_amplitude)
        if not complex_amplitude.geometry.is_transverse:
            raise ValueError(
                "The angular spectrum decomposes a field over a transverse plane, and "
                "this input is sampled on one tilted out of x-y. Propagate from a "
                "transverse plane, or use RayleighSommerfeld.propagate_to, whose sum "
                "needs no plane at all."
            )

        points, phase, propagating, carrier = self._plane_wave_sum(
            complex_amplitude, geometry
        )
        coefficients, spec = self._spectrum(complex_amplitude)
        self._warn_if_unresolved(
            _nyquist_ratio(points, coefficients, propagating, enclosed_energy), geometry
        )
        # The modes a type-1 NUFFT sums against are integers, so a point outside one
        # period is the same point inside and the fold changes nothing.
        points = torch.remainder(points + torch.pi, 2 * torch.pi) - torch.pi

        resolution_out = tuple(geometry.resolution)
        real_dtype = coefficients.real.dtype
        retained = (phase * propagating).to(coefficients.dtype)
        samples = [
            finufft_type1(
                points[index].to(real_dtype).contiguous(),
                (coefficients[:, index] * retained[index]).contiguous(),
                resolution_out,
                isign=1,
                modeord=0,
            )
            for index in range(coefficients.shape[1])
        ]
        out = torch.stack(samples, dim=1) * carrier.to(coefficients.dtype)

        field = ComplexAmplitude.unflatten_batch(
            out.to(complex_amplitude.dtype_c),
            spec,
            complex_amplitude.wavelength,
            geometry.pixel_size,
        )
        return field.with_geometry(
            origin=geometry.origin, rotation=geometry.rotation
        )
