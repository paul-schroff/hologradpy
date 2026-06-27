# TODO: This file is not currently in use but is a placeholder for the future.
# The idea is to have a base class for geometric transformations that can be
# extended for different transformation types (affine, perspective, homography,
# etc.). It should handle fitting a transformation to two sets of points, and
# then transform points and 2D images.
from typing import Literal, TypeVar

import numpy as np
from numpy.typing import NDArray

TransformType = TypeVar(
    "TransformTypes",
    bound=Literal["affine", "partial_affine", "perspective", "homography"],
)


class GeometricTransformer:
    def __init__(self):
        self.camera_points: NDArray[np.float_] | None = None
        self.calculated_points: NDArray[np.float_] | None = None

    def find_transform(
        self,
        camera_points: NDArray[np.float_],
        calculated_points: NDArray[np.float_],
    ) -> NDArray[np.float_]:
        raise NotImplementedError("The find_transform() method is not implemented.")

    @property
    def inverse_transform(self) -> NDArray:
        raise NotImplementedError(
            "Inverse transform is not implemented for this transformation type."
        )

    def transform_points(
        self, points: NDArray[np.float_], inverse: bool = False
    ) -> NDArray[np.float_]:
        raise NotImplementedError("Transform method is not implemented.")
