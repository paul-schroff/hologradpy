"""Tests for the field that a Fourier lens makes at focal-plane positions."""

from __future__ import annotations

import torch

from hologradpy.fourier_optics import focal_field, focal_field_adjoint

RESOLUTION = (12, 16)
PIXEL_SIZE = (8e-6, 6e-6)
WAVELENGTH = 800e-9
FOCAL_LENGTH = 0.1


def test_focal_field_adjoint_is_the_adjoint_of_focal_field() -> None:
    """For any field and spot fields, the inner product of the spot fields with the
    focal field equals that of the adjoint's field with the field.
    """
    generator = torch.Generator().manual_seed(0)
    field = torch.randn(RESOLUTION, dtype=torch.complex128, generator=generator)
    spots = torch.randn(5, dtype=torch.complex128, generator=generator)
    offsets = torch.rand((5, 2), dtype=torch.float64, generator=generator) - 0.5
    positions = 2e-4 * offsets
    lens = (PIXEL_SIZE, WAVELENGTH, FOCAL_LENGTH)

    forward = focal_field(field, positions, *lens)
    backward = focal_field_adjoint(spots, positions, RESOLUTION, *lens)

    torch.testing.assert_close(
        torch.vdot(spots, forward), torch.vdot(backward.flatten(), field.flatten())
    )
