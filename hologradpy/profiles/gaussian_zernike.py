"""A Gaussian beam whose phase is a sum of Zernike polynomials.

A wavefront calibration holds one as its fitted beam (see
:meth:`~hologradpy.calibration.wavefront.abstract.WavefrontCalibratorBase.fit_beam`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from ..serialization import SaveableRecord, record_type
from .gaussian import Gaussian, GaussianBeam
from .zernike import Zernike


@record_type("gaussian_zernike_beam")
@dataclass(frozen=True)
class GaussianZernikeBeam(SaveableRecord):
    """A beam with a Gaussian intensity and a phase made of Zernike polynomials.

    ``gaussian`` is the intensity. ``cutoff_radius`` cuts the beam to 0 beyond that
    distance from the Gaussian's centre, in metres. None leaves the beam uncut.

    The phase is a sum of Zernike polynomials on a disk of radius ``zernike_radius`` in
    metres, centred on the Gaussian's centre.

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
    cutoff_radius: float | None = None

    def get_wavefront(
        self, resolution: tuple[int, int], pixel_size: tuple[float, float]
    ) -> torch.Tensor:
        """Return the complex field of the beam on a grid of pixels.

        The grid is that of :class:`~hologradpy.profiles.gaussian.Gaussian`. The
        amplitude is the square root of the Gaussian's intensity without its offset.
        The phase is 0 outside the Zernike disk.

        Args:
            resolution: The ``(height, width)`` of the grid, in pixels.
            pixel_size: The ``(height, width)`` pitch in metres.

        Returns:
            torch.Tensor: The complex64 field, of shape ``resolution``.
        """
        gaussian = Gaussian(resolution, pixel_size)
        intensity = gaussian.get_intensity(replace(self.gaussian, offset=0.0))
        if self.cutoff_radius is not None:
            distance = torch.hypot(
                gaussian.x - self.gaussian.shift_x, gaussian.y - self.gaussian.shift_y
            )
            intensity = intensity * (distance <= self.cutoff_radius)
        phase = torch.zeros_like(intensity)
        if self.zernike_indices:
            pitch_y, pitch_x = (float(pitch) for pitch in pixel_size)
            zernike = Zernike(
                resolution,
                unit_disk_radius=self.zernike_radius / pitch_x,
                unit_disk_center=(
                    self.gaussian.shift_x / pitch_x,
                    self.gaussian.shift_y / pitch_y,
                ),
                indices=list(self.zernike_indices),
            )
            phase = zernike.get_phase(torch.tensor(self.zernike_coefficients))
        return intensity.sqrt() * torch.exp(1j * phase)
