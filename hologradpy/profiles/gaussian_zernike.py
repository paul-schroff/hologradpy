"""A Gaussian beam whose phase is a sum of Zernike polynomials.

A wavefront calibration holds one as its fitted beam (see
:meth:`~hologradpy.calibration.wavefront.abstract.WavefrontCalibratorBase.fit_beam`).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..serialization import SaveableRecord, record_type
from .gaussian import GaussianBeam


@record_type("gaussian_zernike_beam")
@dataclass(frozen=True)
class GaussianZernikeBeam(SaveableRecord):
    """A beam with a Gaussian intensity and a phase made of Zernike polynomials.

    ``gaussian`` is the intensity. The phase is a sum of Zernike polynomials on a disk
    of radius ``zernike_radius`` in metres, centred on the Gaussian's centre.

    - ``zernike_indices`` holds each polynomial's radial order n and azimuthal
      frequency m.
    - ``zernike_coefficients`` holds each polynomial's coefficient in radians.
    - Each polynomial has unit RMS over the disk, as in the OSA/ANSI convention of
      :class:`~hologradpy.profiles.zernike.Zernike`.
    - The angle runs counterclockwise from +x, with y up. A negative m gives a sine
      term.
    """

    gaussian: GaussianBeam
    zernike_radius: float
    zernike_indices: tuple[tuple[int, int], ...] = ()
    zernike_coefficients: tuple[float, ...] = ()
