"""The :class:`CameraMapping` result value object, produced by a camera mapper.

Kept separate from the mapper (``abstract.py``), which needs an optical model and so
pulls in :mod:`hologradpy.optics`. This module has no such dependency, so it can
be imported from lower layers (e.g. as a type in a propagation model) without a cycle.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TypeVar
from datetime import datetime

import numpy as np
import torch
from numpy.typing import NDArray

from ...geometry import AffineTransform, PartialAffineTransform
from ...serialization import SaveableRecord
from ...visualizer import VisualizationData
from ...hardware.camera import CameraData

ArrayLike = TypeVar("ArrayLike", torch.Tensor, NDArray)


@dataclass
class CameraMapping(SaveableRecord):
    timestamp: datetime
    name: str
    transform: ArrayLike
    inverse_transform: ArrayLike
    detected_points: list[tuple[float, float]]
    calculated_points: list[tuple[float, float]]
    camera_images: list[ArrayLike]
    simulated_images: list[ArrayLike]
    zeroth_order_position: tuple[float, float]
    focal_spot_radius: float
    reprojection_errors: ArrayLike | None = None
    reprojection_rms: float | None = None
    spot_fit_parameters: list[ArrayLike] | None = None
    spot_fit_covariances: list[ArrayLike] | None = None
    average_waist: float | None = None
    average_waist_uncertainty: float | None = None
    zeroth_order_mask: ArrayLike | None = None
    excluded_points: list[tuple[float, float]] | None = None
    excluded_reasons: list[str] | None = None
    visualization_data: VisualizationData | None = None
    camera_data: CameraData | None = None
    suggested_orientation: dict | None = None
    residual_transform: ArrayLike | None = None
    metadata = []

    @property
    def affine(self) -> AffineTransform:
        """The camera -> model transform as an :class:`AffineTransform` value object
        (the single home for the rotation / mirror / scale decomposition)."""
        return AffineTransform.from_matrix(
            np.asarray(self.transform, dtype=np.float64)
        )

    @property
    def partial_affine(self) -> PartialAffineTransform:
        """The camera -> model mapping refit as a similarity (uniform scale +
        rotation + translation, no shear or mirror), from the same detected /
        calculated point pairs.

        This is the ``(scale, angle, shift)`` parameterization the differentiable
        Fourier lenses and the field warp calibrate against, so it is fit directly
        from the correspondences rather than reduced from the 6-DOF
        :attr:`affine` (which would discard shear inconsistently)."""
        return PartialAffineTransform.fit(
            self.detected_points, self.calculated_points
        )

    @property
    def is_mirrored(self) -> bool:
        """True if the camera view is mirrored (the transform flips handedness)."""
        return self.affine.is_mirrored

    @property
    def rotation_degrees(self) -> float:
        """Rotation of the camera axes relative to the model plane in degrees, from the
        polar decomposition of the transform (reflection factored out for a mirrored
        camera)."""
        return self.affine.rotation_degrees

    @property
    def scales(self) -> tuple[float, float]:
        """Scale factors of the transform (singular values, major then minor)."""
        return self.affine.scales

    # save / load come from SaveableRecord.
