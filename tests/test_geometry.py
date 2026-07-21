"""Tests for the geometric-transform value objects."""

import numpy as np
import pytest

from hologradpy.geometry import (
    GeometricTransform,
    AffineTransform,
    PartialAffineTransform,
)

POINTS = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, -3.0], [-4.0, 5.0]])


def test_degrees_of_freedom():
    assert PartialAffineTransform.from_components().degrees_of_freedom == 4
    assert AffineTransform.from_components().degrees_of_freedom == 6


def test_matrix_accepts_2x3_and_pads():
    transform = AffineTransform.from_matrix([[1.0, 0.0, 5.0], [0.0, 1.0, -2.0]])
    assert transform.matrix.shape == (3, 3)
    np.testing.assert_allclose(transform.matrix[2], [0.0, 0.0, 1.0])


def test_transform_points_known_case():
    # scale 2, rotate 90 deg, shift (1, 0): (1, 0) -> 2*R@(1,0) + (1,0) = (1, 2).
    transform = PartialAffineTransform.from_components(
        scale=2.0, angle_deg=90.0, shift=(1.0, 0.0)
    )
    np.testing.assert_allclose(transform.transform_points([[1.0, 0.0]]), [[1.0, 2.0]])


def test_partial_affine_components_roundtrip():
    transform = PartialAffineTransform.from_components(
        scale=1.3, angle_deg=25.0, shift=(3.0, -2.0)
    )
    assert transform.scale == pytest.approx(1.3)
    assert transform.angle_degrees == pytest.approx(25.0)
    np.testing.assert_allclose(transform.translation, [3.0, -2.0])
    assert transform.is_mirrored is False


def test_affine_decomposition():
    transform = AffineTransform.from_components(
        scale=(1.2, 0.8), angle_deg=15.0, mirror=True
    )
    assert transform.is_mirrored is True
    # Scales are the singular values (order-independent), so compare as a set.
    np.testing.assert_allclose(sorted(transform.scales), sorted((1.2, 0.8)), atol=1e-9)
    # rotation_matrix is orthonormal.
    rotation = transform.rotation_matrix
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(2), atol=1e-9)


def test_inverse_roundtrips_points_and_preserves_type():
    transform = PartialAffineTransform.from_components(
        scale=1.7, angle_deg=40.0, shift=(2.0, 5.0)
    )
    inverse = transform.inverse()
    assert isinstance(inverse, PartialAffineTransform)
    np.testing.assert_allclose(
        inverse.transform_points(transform.transform_points(POINTS)), POINTS, atol=1e-9
    )


def test_compose_matches_sequential_application_and_promotes_type():
    partial = PartialAffineTransform.from_components(scale=1.4, angle_deg=10.0)
    affine = AffineTransform.from_components(shear=0.3, shift=(1.0, -1.0))
    composed = affine.compose(partial)  # affine after partial
    np.testing.assert_allclose(
        composed.transform_points(POINTS),
        affine.transform_points(partial.transform_points(POINTS)),
        atol=1e-9,
    )
    # The more general type wins.
    assert type(composed) is AffineTransform
    assert type(partial.compose(partial)) is PartialAffineTransform


def test_fit_recovers_partial_affine():
    source = np.random.default_rng(0).uniform(-10, 10, size=(12, 2))
    truth = PartialAffineTransform.from_components(
        scale=1.3, angle_deg=20.0, shift=(3.0, -2.0)
    )
    fitted = PartialAffineTransform.fit(source, truth.transform_points(source))
    np.testing.assert_allclose(fitted.matrix, truth.matrix, atol=1e-4)


def test_fit_recovers_affine_with_shear():
    source = np.random.default_rng(1).uniform(-10, 10, size=(12, 2))
    truth = AffineTransform.from_components(
        scale=(1.2, 0.9), angle_deg=15.0, shift=(2.0, 1.0), shear=0.3
    )
    fitted = AffineTransform.fit(source, truth.transform_points(source))
    np.testing.assert_allclose(fitted.matrix, truth.matrix, atol=1e-4)


def test_reprojection_error():
    transform = AffineTransform.from_components(shift=(1.0, 0.0))
    destination = POINTS + np.array([1.0, 0.0])  # exactly the mapped points
    errors, rms = transform.reprojection_error(POINTS, destination)
    np.testing.assert_allclose(errors, 0.0, atol=1e-12)
    assert rms == pytest.approx(0.0, abs=1e-12)


def test_geometric_transform_is_abstract():
    with pytest.raises(TypeError):
        GeometricTransform(np.eye(3))
