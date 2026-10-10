"""An elliptical Gaussian beam, drawn on a grid of pixels and fitted to an intensity.

Like :class:`~hologradpy.profiles.zernike.Zernike`, :class:`Gaussian` is a model on a
grid. Its :meth:`~Gaussian.get_intensity` draws a :class:`GaussianBeam`, and its
:meth:`~Gaussian.fit` is the inverse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
from numpy.typing import NDArray
import torch
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit

from ..grids import get_spatial_grid
from ..serialization import record_type
from .amplitude import ArrayLike, _minor_radius, gaussian_beam_intensity

if TYPE_CHECKING:
    from ..roi import ROI


@record_type("gaussian_beam")
@dataclass(frozen=True)
class GaussianBeam:
    """An elliptical Gaussian beam intensity.

    The fields are the arguments of
    :func:`~hologradpy.profiles.amplitude.gaussian_beam_intensity`, with its defaults.
    A minor radius of None makes the beam round. :meth:`evaluate` evaluates the
    function with them.

    Raises:
        ValueError: The minor radius is larger than the major radius.
    """

    beam_radius_major: float
    beam_radius_minor: float | None = None
    shift_x: float = 0.0
    shift_y: float = 0.0
    intensity: float = 1.0
    offset: float = 0.0
    angle_degrees: float = 0.0

    def __post_init__(self) -> None:
        _minor_radius(self.beam_radius_major, self.beam_radius_minor)

    def evaluate(self, x: ArrayLike, y: ArrayLike) -> ArrayLike:
        """Return the intensity of the beam at the points ``x`` and ``y``."""
        return gaussian_beam_intensity(
            x,
            y,
            self.beam_radius_major,
            self.beam_radius_minor,
            self.shift_x,
            self.shift_y,
            self.intensity,
            self.offset,
            self.angle_degrees,
        )

    @property
    def mean_radius(self) -> float:
        """The geometric mean of the major and minor radii."""
        return math.sqrt(self.beam_radius_major * self._minor)

    @property
    def beam_radius_x(self) -> float:
        """The 1/e^2 intensity radius of the beam summed over y."""
        angle = math.radians(self.angle_degrees)
        return math.hypot(
            self.beam_radius_major * math.cos(angle), self._minor * math.sin(angle)
        )

    @property
    def beam_radius_y(self) -> float:
        """The 1/e^2 intensity radius of the beam summed over x."""
        angle = math.radians(self.angle_degrees)
        return math.hypot(
            self.beam_radius_major * math.sin(angle), self._minor * math.cos(angle)
        )

    @property
    def _minor(self) -> float:
        """The minor radius, which is the major radius for a round beam."""
        return _minor_radius(self.beam_radius_major, self.beam_radius_minor)


@record_type("gaussian_fit")
@dataclass(frozen=True, eq=False)
class GaussianFit:
    """A Gaussian beam fitted by :meth:`Gaussian.fit`.

    The fields are the fit's own parameters and their covariance. The parameters are
    the entries ``a``, ``s`` and ``c`` of ``L`` (see :meth:`Gaussian.fit`), then the
    beam's ``shift_x``, ``shift_y``, ``intensity`` and ``offset``. :attr:`beam` turns
    them into the fitted beam.
    """

    parameters: NDArray[np.float64]
    covariance: NDArray[np.float64]

    @property
    def beam(self) -> GaussianBeam:
        """The fitted beam."""
        return _beam(self.parameters)


class Gaussian:
    """A Gaussian beam on a grid of pixels, in metres.

    The grid comes from :func:`~hologradpy.grids.get_spatial_grid`, cropped to an
    optional ROI. :meth:`get_intensity` returns, and :meth:`fit` takes, an intensity
    of the grid's shape.
    """

    def __init__(
        self,
        resolution: tuple[int, int],
        pixel_size: tuple[float, float],
        roi: ROI | None = None,
    ) -> None:
        """
        Args:
            resolution: The ``(height, width)`` of the whole grid, in pixels.
            pixel_size: The ``(height, width)`` pitch in metres, as
                :func:`~hologradpy.grids.get_spatial_grid` takes it.
            roi: The region of the grid to keep. The whole grid is kept when None.
        """
        x, y = get_spatial_grid(resolution, pixel_size)
        if roi is not None:
            x, y = roi.crop(x), roi.crop(y)
        self.x: torch.Tensor = x
        self.y: torch.Tensor = y

    def get_intensity(self, beam: GaussianBeam) -> torch.Tensor:
        """Return the intensity of ``beam`` on the grid (the inverse of :meth:`fit`)."""
        return beam.evaluate(self.x, self.y)

    def fit(
        self,
        intensity: NDArray,
        beam_radius_guess: float,
        mask: NDArray[np.bool_] | None = None,
        blur_sigma: float = 10,
    ) -> GaussianFit:
        """Fit a Gaussian beam with a constant offset to ``intensity`` on the grid (the
        inverse of :meth:`get_intensity`).

        The beam's exponent is ``-2 d^T L L^T d``, where ``d`` is the position from the
        centre and ``L = [[a, 0], [s, c]]``. The fit varies ``a``, ``s`` and ``c``, the
        centre, the peak and the offset. The exponent is quadratic in ``L``, so no
        angle wraps, and a round beam is an ordinary point of the fit.

        The fit starts round, with ``beam_radius_guess`` as the radius, at the peak of
        ``intensity`` after a Gaussian blur of ``blur_sigma`` pixels. With a ``mask``,
        only the pixels where it is True enter the fit.

        Args:
            intensity: The measured intensity, of the grid's shape.
            beam_radius_guess: The starting 1/e^2 intensity radius, in metres.
            mask: True at the pixels to fit, of the grid's shape. Every pixel is
                fitted when None.
            blur_sigma: The standard deviation of the Gaussian blur, in pixels. The
                starting peak is located on the blurred intensity.

        Returns:
            GaussianFit: The fitted parameters and their covariance.
        """
        x = np.asarray(self.x, dtype=np.float64)
        y = np.asarray(self.y, dtype=np.float64)
        data = np.asarray(intensity, dtype=np.float64)
        if mask is None:
            data_blurred = gaussian_filter(data, blur_sigma)
            index = np.unravel_index(np.argmax(data_blurred), data.shape)
            peak = np.max(data)
        else:
            mask = np.asarray(mask, dtype=bool)
            blurred_mask = gaussian_filter(mask.astype(np.float64), blur_sigma)
            data_blurred = gaussian_filter(
                np.where(mask, data, 0.0), blur_sigma
            ) / np.maximum(blurred_mask, np.finfo(np.float64).tiny)
            index = np.unravel_index(
                np.argmax(np.where(mask, data_blurred, -np.inf)), data.shape
            )
            peak = np.max(data[mask])

        start = 1.0 / beam_radius_guess
        p0 = (start, 0.0, start, x[index], y[index], peak, 0.0)
        if mask is not None:
            x, y, data = x[mask], y[mask], data[mask]
        parameters, covariance = curve_fit(
            _model, np.stack((x.ravel(), y.ravel())), data.ravel(), p0=p0
        )
        return GaussianFit(parameters, covariance)


def _model(points: NDArray[np.float64], *parameters: float) -> NDArray[np.float64]:
    """Return the intensity of the beam of the fit's ``parameters`` at the ``(x, y)``
    ``points``.
    """
    return _beam(parameters).evaluate(*points)


def _beam(parameters: Sequence[float]) -> GaussianBeam:
    """Return the beam of the fit's ``parameters``.

    The eigenvector of the smaller eigenvalue of ``L L^T`` lies along the long axis.
    The signs of ``L``'s columns leave ``L L^T`` unchanged, so they don't matter.
    """
    a, s, c, shift_x, shift_y, intensity, offset = (
        float(value) for value in parameters
    )
    factor = np.array([[a, 0.0], [s, c]])
    eigenvalues, eigenvectors = np.linalg.eigh(factor @ factor.T)
    major, minor = 1.0 / np.sqrt(eigenvalues)
    along_x, along_y = eigenvectors[:, 0]
    angle = math.degrees(math.atan2(along_y, along_x))
    return GaussianBeam(
        float(major),
        float(minor),
        shift_x,
        shift_y,
        intensity,
        offset,
        90.0 - (90.0 - angle) % 180.0,
    )
