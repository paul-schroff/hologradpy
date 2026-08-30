from __future__ import annotations

import torch
from torch import Tensor
from torch.fft import fftshift

from ..utils import to_canvas
from .abstract import FourierBase
from .fft import fft_2d, ifft_2d


def transformed_curvature(
    curvature: tuple[float, float, float],
) -> tuple[float, float, float]:
    """The coefficients a quadratic phase has once it is Fourier transformed.

    The transform of a quadratic phase is another quadratic phase. Writing it as
    ``rho^T A rho`` with ``A = [[Dx, C/2], [C/2, Dy]]``, the transform of
    ``exp(i rho^T A rho)`` carries ``-A^-1 / 4`` in place of ``A``. Inverting like
    that means a curvature that is strong in one domain is weak in the other, which is
    what the method relies on.

    Args:
        curvature: ``(Dx, C, Dy)`` in radians per sample squared.

    Returns:
        tuple[float, float, float]: The transformed ``(Dx, C, Dy)``.

    Raises:
        ValueError: The quadratic form is degenerate, which leaves it with no inverse.
    """
    curvature_x, cross, curvature_y = curvature
    determinant = curvature_x * curvature_y - cross**2 / 4
    if determinant == 0:
        raise ValueError(
            "A quadratic phase with Dx * Dy == C**2 / 4 is one dimensional, so it has "
            "no two dimensional transform. Rotate it onto an axis and transform along "
            "that axis instead."
        )
    return (
        -curvature_y / (4 * determinant),
        cross / (4 * determinant),
        -curvature_x / (4 * determinant),
    )


class SemiAnalyticalFourierTransform(FourierBase):
    """The DFT of a field carrying a quadratic phase, without sampling that phase.

    A field with a strong wavefront curvature needs a large number of samples before its
    phase is resolved. Splitting the field as ``V = U exp(i psi_q)`` with ``psi_q = Dx
    x^2 + C xy + Dy y^2`` leaves a residual ``U`` that is smooth, and the quadratic
    factor is carried analytically instead of being sampled at all.

    The method is the implementation of Z. Wang, S. Zhang, O. Baladron-Zorita, C.
    Hellmann and F. Wyrowski, "Application of the semi-analytical Fourier transform to
    electromagnetic modeling", Opt. Express 27, 15335 (2019),
    https://doi.org/10.1364/OE.27.015335.
    """

    def __init__(
        self,
        resolution: tuple[int, int],
        curvature: tuple[float, float, float],
        device: torch.device = "cpu",
    ) -> None:
        """
        Args:
            resolution: ``(height, width)`` of the residual field.
            curvature: ``(Dx, C, Dy)`` of the phase being carried, in radians per
                sample squared.
            device: Where to build the chirp.

        Raises:
            ValueError: The curvature is degenerate. See :func:`transformed_curvature`.
        """
        transformed_curvature(curvature)  # Refuses a degenerate form up front.

        super().__init__(
            resolution,
            frequencies=None,
            is_gridded=True,
            resolution_out=resolution,
            device=device,
        )
        self.curvature = curvature

        # Padded to twice the resolution, so the convolution does not wrap.
        self._padded = tuple(2 * length for length in resolution)
        rows = torch.arange(
            -(self._padded[0] // 2),
            self._padded[0] - self._padded[0] // 2,
            device=device,
            dtype=torch.float64,
        )
        columns = torch.arange(
            -(self._padded[1] // 2),
            self._padded[1] - self._padded[1] // 2,
            device=device,
            dtype=torch.float64,
        )
        grid_x, grid_y = torch.meshgrid(columns, rows, indexing="xy")

        curvature_x, cross, curvature_y = curvature
        chirp = torch.exp(
            1j
            * (
                curvature_x * grid_x**2
                + cross * grid_x * grid_y
                + curvature_y * grid_y**2
            )
        )
        # Kept in the transform's own bin order rather than centred. A shift is a
        # permutation, so it passes through the pointwise product, and doing it
        # once here rather than twice per call leaves the two convolution
        # transforms with no shifting to do at all.
        self.register_buffer(
            "_chirp_spectrum",
            fftshift(fft_2d(chirp), dim=(-2, -1)),
            persistent=False,
        )
        self._chirp_spectrum: Tensor

        out_x, out_y = self._lattice()
        self.register_buffer(
            "_residual_phase",
            torch.exp(
                -1j
                * (
                    curvature_x * out_x**2
                    + cross * out_x * out_y
                    + curvature_y * out_y**2
                )
            ),
            persistent=False,
        )
        self._residual_phase: Tensor

    def _lattice(self) -> tuple[Tensor, Tensor]:
        """The integer lattice the output is indexed by, as an ``(x, y)`` pair."""
        rows = torch.arange(
            -(self.resolution[0] // 2),
            self.resolution[0] - self.resolution[0] // 2,
            device=self.device,
            dtype=torch.float64,
        )
        columns = torch.arange(
            -(self.resolution[1] // 2),
            self.resolution[1] - self.resolution[1] // 2,
            device=self.device,
            dtype=torch.float64,
        )
        return torch.meshgrid(columns, rows, indexing="xy")

    def _convolve(self, padded: Tensor, spectrum: Tensor) -> Tensor:
        """Convolve a padded field with a chirp held in bin order. No fftshifts are
        applied.

        Args:
            padded: The field, centred on the padded grid.
            spectrum: The chirp's spectrum, in the same bin order the transforms use.

        Returns:
            Tensor: The convolution, offset by half the padded grid.
        """
        transformed = fft_2d(padded, fft_shift=False)
        return ifft_2d(transformed * spectrum, fft_shift=False)

    def _reflected_corner(self) -> tuple[int, int]:
        """Where the output window sits once the convolution has been flipped.

        The convolution is wanted at ``-m``, and flipping an axis of length ``L``
        sends ``i`` to ``L - 1 - i``, which is half a sample off the centre an even
        axis has. So this is the centred corner :func:`to_canvas` would pick, moved
        one sample for each even axis.
        """
        height, width = self.resolution
        return (
            self._padded[0] // 2 - height // 2 - (1 - height % 2),
            self._padded[1] // 2 - width // 2 - (1 - width % 2),
        )

    def _build_frequencies(self) -> Tensor:
        """The sample points ``k = 2 A m``, in radians per sample.

        Sheared rather than rectangular whenever the cross term is present.
        """
        grid_x, grid_y = self._lattice()
        curvature_x, cross, curvature_y = self.curvature
        return torch.stack(
            (
                (2 * curvature_x * grid_x + cross * grid_y).flatten(),
                (cross * grid_x + 2 * curvature_y * grid_y).flatten(),
            ),
            dim=0,
        )

    def sampling_margin(self) -> tuple[float, float]:
        """How far the sample points reach, as a fraction of one period.

        Returns:
            tuple[float, float]: ``max|k| / pi`` along ``x`` and along ``y``.
        """
        frequencies = self.frequencies
        return (
            float(frequencies[0].abs().max() / torch.pi),
            float(frequencies[1].abs().max() / torch.pi),
        )

    def forward(self, input: Tensor) -> Tensor:
        """Transform a residual field, returning the transform of it times the chirp.

        Args:
            input: The residual ``U``, on the transform's resolution. The quadratic
                phase is supplied at construction, so leave it out of this.

        Returns:
            Tensor: The transform on the lattice :attr:`frequencies`.
        """
        height, width = self.resolution
        padded = to_canvas(input, self._padded)

        convolved = self._convolve(padded, self._chirp_spectrum)
        convolved = torch.flip(convolved, dims=(-2, -1))
        top, left = self._reflected_corner()
        cropped = convolved[..., top : top + height, left : left + width]
        return self._residual_phase * cropped

    def adjoint(self, input: Tensor) -> Tensor:
        """The conjugate transpose of :meth:`forward`."""
        height, width = self.resolution
        undone = torch.conj(self._residual_phase) * input

        top, left = self._reflected_corner()
        padded = torch.nn.functional.pad(
            undone,
            (
                left,
                self._padded[1] - width - left,
                top,
                self._padded[0] - height - top,
            ),
        )
        padded = torch.flip(padded, dims=(-2, -1))

        convolved = self._convolve(padded, torch.conj(self._chirp_spectrum))
        return to_canvas(convolved, self.resolution)
