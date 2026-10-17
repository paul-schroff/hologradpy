"""Tests for the field warp, GeometricWarp: its matrix convention and the resampling."""

from __future__ import annotations

import numpy as np
import torch

from hologradpy.geometry import PartialAffineTransform
from hologradpy.optics.complex_amplitude import ComplexAmplitude
from hologradpy.optics.modules.geometric_transforms import GeometricWarp

WAVELENGTH = 650e-9
PIXEL_SIZE = (8e-6, 8e-6)


def _random_plane(resolution: tuple[int, int], seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.complex(
        torch.randn(resolution, generator=generator, dtype=torch.float64),
        torch.randn(resolution, generator=generator, dtype=torch.float64),
    )


def _warped(data: torch.Tensor, **parameters) -> torch.Tensor:
    """Warp a plane at equal input and output resolution and pixel size."""
    warp = GeometricWarp(tuple(data.shape), PIXEL_SIZE, **parameters)
    output = warp(ComplexAmplitude(data, WAVELENGTH, PIXEL_SIZE))
    return output.as_tensor().reshape(data.shape)


def test_matrix_matches_the_partial_affine_value_object():
    resolution = (33, 41)
    warp = GeometricWarp(
        resolution, PIXEL_SIZE, scale_factor=(1.2, 1.2), shift=(3.0, -2.0), angle=17.0
    )
    warp(ComplexAmplitude(_random_plane(resolution), WAVELENGTH, PIXEL_SIZE))
    center = warp.rotation_center[0]
    shift = (warp.shift_center + warp.focal_plane_partial_affine.shift)[0]
    expected = PartialAffineTransform.from_components(
        scale=1.2,
        angle_deg=17.0,
        shift=tuple(shift.tolist()),
        center=tuple(center.tolist()),
    ).matrix
    np.testing.assert_allclose(
        warp.get_affine_matrix()[0].detach().numpy(), expected, atol=1e-12
    )


def test_the_centre_of_a_non_square_input_lands_on_the_centre_of_the_output():
    """Without a shift, the warp carries the centre pixel of the input plane onto the
    centre pixel of the output plane, also for a non-square input at a pitch ratio
    other than one.
    """
    data = torch.zeros((4, 8), dtype=torch.complex128)
    data[2, 4] = 1.0  # the centre (x, y) = (4, 2) of a 4 x 8 plane
    warp = GeometricWarp((16, 16), PIXEL_SIZE)
    coarse_pitch = (2 * PIXEL_SIZE[0], 2 * PIXEL_SIZE[1])

    output = warp(ComplexAmplitude(data, WAVELENGTH, coarse_pitch)).as_tensor()

    row, column = np.unravel_index(int(output.abs().argmax()), (16, 16))
    assert (row, column) == (8, 8)


def test_default_warp_returns_the_field():
    data = _random_plane((31, 45))
    torch.testing.assert_close(_warped(data), data, rtol=0.0, atol=1e-12)


def test_whole_pixel_shift_moves_the_field_and_fills_zeros():
    data = _random_plane((31, 45))
    # The shift is (x, y), so the field moves 3 columns right and 2 rows down.
    expected = torch.zeros_like(data)
    expected[2:, 3:] = data[:-2, :-3]
    torch.testing.assert_close(
        _warped(data, shift=(3.0, 2.0)), expected, rtol=0.0, atol=1e-12
    )


def test_quarter_turn_about_the_center_matches_rot90():
    data = _random_plane((31, 31))
    # A positive angle turns +x towards +y. With rows running along +y, that is a
    # clockwise quarter turn of the array, torch.rot90 with k=-1.
    expected = torch.rot90(data, k=-1, dims=(0, 1))
    torch.testing.assert_close(
        _warped(data, angle=90.0), expected, rtol=0.0, atol=1e-12
    )


def test_real_and_imaginary_parts_warp_independently():
    parameters = dict(scale_factor=(1.1, 0.9), shift=(1.3, -0.7), angle=12.0)
    data = _random_plane((29, 37), seed=1)
    real = data.real.to(data.dtype)
    imaginary = data.imag.to(data.dtype)
    torch.testing.assert_close(
        _warped(data, **parameters),
        _warped(real, **parameters) + 1j * _warped(imaginary, **parameters),
        rtol=0.0,
        atol=1e-12,
    )
