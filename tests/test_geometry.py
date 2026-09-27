"""Tests for the geometric-transform value objects."""

import numpy as np
import pytest
import torch

from hologradpy.geometry import (
    GeometricTransform,
    AffineTransform,
    PartialAffineTransform,
    homogeneous_matrix,
    rotation_matrix_from_angle,
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


def test_rotation_turns_the_x_axis_towards_the_y_axis():
    quarter_turn = rotation_matrix_from_angle(np.asarray(90.0))
    np.testing.assert_allclose(quarter_turn, [[0.0, -1.0], [1.0, 0.0]], atol=1e-15)


def test_homogeneous_matrix_keeps_the_center_fixed_before_the_shift():
    linear = rotation_matrix_from_angle(np.asarray(30.0)) @ np.diag([1.5, 0.5])
    shift, center = np.array([4.0, -2.0]), np.array([10.0, 6.0])
    matrix = homogeneous_matrix(linear, shift, center)
    point = np.array([3.0, 7.0])
    expected = linear @ (point - center) + center + shift
    np.testing.assert_allclose(matrix @ np.append(point, 1.0), np.append(expected, 1.0))
    np.testing.assert_array_equal(matrix[2], [0.0, 0.0, 1.0])


def test_partial_affine_components_use_the_matrix_builders():
    transform = PartialAffineTransform.from_components(
        scale=1.3, angle_deg=25.0, shift=(3.0, -2.0), center=(5.0, 1.0)
    )
    expected = homogeneous_matrix(
        1.3 * rotation_matrix_from_angle(np.asarray(25.0)),
        np.array([3.0, -2.0]),
        np.array([5.0, 1.0]),
    )
    np.testing.assert_array_equal(transform.matrix, expected)


@pytest.mark.parametrize(
    ("dtype", "tolerance"), [(torch.float64, 1e-10), (torch.float32, 1e-4)]
)
def test_matrix_builders_agree_between_numpy_and_torch(dtype, tolerance):
    generator = np.random.default_rng(0)
    angles = generator.uniform(-180.0, 180.0, 4)
    scales = generator.uniform(0.5, 2.0, (4, 1, 2))
    shifts = generator.normal(0.0, 10.0, (4, 2))
    centers = generator.uniform(0.0, 100.0, (4, 2))
    expected = homogeneous_matrix(
        rotation_matrix_from_angle(angles) * scales, shifts, centers
    )

    def as_tensor(array):
        return torch.as_tensor(array, dtype=dtype)

    matrices = homogeneous_matrix(
        rotation_matrix_from_angle(as_tensor(angles)) * as_tensor(scales),
        as_tensor(shifts),
        as_tensor(centers),
    )
    assert matrices.dtype == dtype
    assert matrices.shape == (4, 3, 3)
    np.testing.assert_allclose(matrices.numpy(), expected, atol=tolerance)


def test_matrix_builders_are_differentiable_in_torch():
    def build(angle, scale, shift, center):
        linear = rotation_matrix_from_angle(angle) * scale[..., None, :]
        return homogeneous_matrix(linear, shift, center)

    inputs = (
        torch.tensor([20.0, -35.0], dtype=torch.float64),
        torch.tensor([[1.2, 0.8], [0.9, 1.1]], dtype=torch.float64),
        torch.tensor([[3.0, -1.0], [0.5, 2.0]], dtype=torch.float64),
        torch.tensor([[10.0, 4.0], [-3.0, 7.0]], dtype=torch.float64),
    )
    for tensor in inputs:
        tensor.requires_grad_(True)
    assert torch.autograd.gradcheck(build, inputs)
