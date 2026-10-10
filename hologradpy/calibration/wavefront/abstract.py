from __future__ import annotations
from abc import abstractmethod
from datetime import datetime
from dataclasses import dataclass, field

import math

import numpy as np
from numpy.typing import NDArray
import torch

from ...analysis.unwrapping import unwrap_2d_poisson
from ...optics.complex_amplitude import ComplexAmplitude, FieldGeometry
from ...profiles.gaussian import GaussianBeam
from ...profiles.gaussian_zernike import GaussianZernikeBeam
from ...profiles.zernike import Zernike
from ...serialization import SaveableRecord, record_type
from ...visualizer import VisualizationData

from ..abstract import CalibratorBase


@record_type("wavefront_calibration", version=3)
@dataclass
class WavefrontCalibrationData(SaveableRecord):
    timestamp: datetime
    name: str
    complex_amplitude: ComplexAmplitude
    beam: GaussianZernikeBeam | None = None
    metadata: dict = field(default_factory=dict)
    visualization_data: VisualizationData | None = None

    # save / load come from SaveableRecord.

    @classmethod
    def from_beam(
        cls, beam: GaussianZernikeBeam, geometry: FieldGeometry
    ) -> WavefrontCalibrationData:
        """Return a calibration whose field is ``beam`` drawn on ``geometry``.

        The calibration is named "gaussian", and it holds ``beam``. Its timestamp is
        now.
        """
        field = beam.get_wavefront(geometry.resolution, geometry.pixel_size[0])
        return cls(
            timestamp=datetime.now(),
            name="gaussian",
            complex_amplitude=ComplexAmplitude.from_geometry(geometry, field),
            beam=beam,
        )

    @classmethod
    def _migrate(cls, version: int, stored: dict) -> dict:
        """Convert a calibration saved by an older version.

        - Version 1 was saved while y increased with increasing row index. The field
          is indexed by row and column, so it stays as it is. The y of the beam centre
          ``shift_y`` and of the raster tilts changes sign, since y now increases with
          decreasing row index.
        - Versions 1 and 2 kept the fitted beam in the metadata, as ``beam_radius``,
          ``shift_x`` and ``shift_y``. These keys become ``beam``, a round beam
          without Zernike terms. Its disk's radius is sqrt(2) times the beam radius,
          as :meth:`WavefrontCalibratorBase.fit_beam` sets it.
        """
        metadata = dict(stored.get("metadata") or {})
        if version < 2:
            if metadata.get("shift_y") is not None:
                metadata["shift_y"] = -metadata["shift_y"]
            for key in ("linear_phase_tilt", "lattice_phase_tilt"):
                if metadata.get(key) is not None:
                    x, y = metadata[key]
                    metadata[key] = (x, -y)
        if version < 3:
            beam_radius = metadata.pop("beam_radius", None)
            shift_x = metadata.pop("shift_x", None) or 0.0
            shift_y = metadata.pop("shift_y", None) or 0.0
            if beam_radius is not None:
                stored["beam"] = GaussianZernikeBeam(
                    GaussianBeam(beam_radius, None, shift_x, shift_y),
                    math.sqrt(2) * beam_radius,
                )
        stored["metadata"] = metadata
        return stored


class WavefrontCalibratorBase(CalibratorBase):
    """Base for calibrators that measure the amplitude and phase at the SLM."""

    @abstractmethod
    def calibrate(self, *args, **kwargs) -> WavefrontCalibrationData:
        """Run the calibration and return the measured SLM-plane wavefront.

        The arguments are specific to each calibrator, since the measurement
        strategies have little in common. Only the return type is part of the
        contract.
        """

    def fit_beam(self, field: NDArray[np.complex128]) -> GaussianZernikeBeam:
        """Fit a Gaussian beam with a Zernike phase to an SLM-plane field.

        :meth:`fit_gaussian_beam` fits the intensity. The Zernike disk is centred on
        the fitted centre. Its radius is sqrt(2) times the major radius, where the
        intensity along the long axis falls to e^-4 of its peak.

        The phase is fitted inside the disk, where the intensity exceeds e^-4 of its
        maximum. This threshold is fixed, so a speckle calibration's
        ``beam_mask_threshold`` doesn't change it. The phase is unwrapped there first.
        The fit covers the radial orders 0 to 6, 28 polynomials.

        Args:
            field: The complex field on the SLM's grid.

        Returns:
            GaussianZernikeBeam: The fitted beam, in metres on the SLM.

        Raises:
            RuntimeError: A fit did not converge.
            ValueError: The field is not finite.
        """
        intensity = np.abs(field) ** 2
        gaussian = self.fit_gaussian_beam(intensity)
        pitch_y, pitch_x = (float(pitch) for pitch in self.slm.pixel_size)
        radius = math.sqrt(2) * gaussian.beam_radius_major
        zernike = Zernike(
            self.slm.resolution,
            unit_disk_radius=radius / pitch_x,
            unit_disk_center=(gaussian.shift_x / pitch_x, gaussian.shift_y / pitch_y),
            number_of_radial_orders=7,
        )
        mask = zernike.mask.numpy() & (intensity > np.exp(-4.0) * intensity.max())
        phase = unwrap_2d_poisson(np.angle(field), mask)
        coefficients = zernike.fit(torch.from_numpy(phase), torch.from_numpy(mask))
        return GaussianZernikeBeam(
            gaussian, radius, tuple(zernike.indices), tuple(coefficients.tolist())
        )
