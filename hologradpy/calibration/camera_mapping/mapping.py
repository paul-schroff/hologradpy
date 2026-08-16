from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import TypeVar
from datetime import datetime

import numpy as np
import torch
from numpy.typing import NDArray

from ...geometry import AffineTransform, PartialAffineTransform
from ...serialization import SaveableRecord, record_type
from ...visualizer import VisualizationData
from ...hardware.camera import CameraData, CameraOrientation

ArrayLike = TypeVar("ArrayLike", torch.Tensor, NDArray)


@record_type("focal_spot_fit")
@dataclass(frozen=True)
class FocalSpotFit:
    waist: float
    waist_uncertainty: float | None = None
    parameters: list[ArrayLike] | None = None
    covariances: list[ArrayLike] | None = None


@record_type("mapping_fit")
@dataclass(frozen=True)
class MappingFit:
    reprojection_errors: ArrayLike
    reprojection_rms: float
    excluded_points: list[tuple[float, float]] | None = None


@record_type("orientation_suggestion")
@dataclass(frozen=True)
class OrientationSuggestion:
    """The camera orientation that would align the sensor with the model plane.

    ``residual_transform`` is the mapping that would remain after adopting it, via
    :meth:`hologradpy.hardware.camera.Camera.set_orientation`.
    """

    suggested: CameraOrientation
    residual_transform: ArrayLike


@record_type("camera_mapping")
@dataclass
class CameraMapping(SaveableRecord):
    """The coordinate mapping between camera pixels and the simulated image."""

    timestamp: datetime
    name: str
    transform: ArrayLike
    detected_points: list[tuple[float, float]]
    calculated_points: list[tuple[float, float]]
    zeroth_order_position: tuple[float, float]
    spot_fit: FocalSpotFit
    fit: MappingFit | None = None
    orientation: OrientationSuggestion | None = None
    camera_data: CameraData | None = None
    metadata: dict = field(default_factory=dict)
    visualization_data: VisualizationData | None = None

    @property
    def affine(self) -> AffineTransform:
        """The camera -> model transform as an :class:`AffineTransform` value object
        (the single home for the rotation / mirror / scale decomposition)."""
        return AffineTransform.from_matrix(
            np.asarray(self.transform, dtype=np.float64)
        )

    @property
    def inverse_transform(self) -> NDArray:
        """The model -> camera transform."""
        return self.affine.inverse().as_matrix(homogeneous=False)

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

    def lean(self) -> CameraMapping:
        """A copy without the frames it was fit from, for embedding in another record.
        """
        return replace(self, visualization_data=None)
