"""ZeroPadding: a field carried onto a larger plane of zeros at the same pitch."""

from __future__ import annotations

import pytest
import torch

from hologradpy.optics.complex_amplitude import ComplexAmplitude
from hologradpy.optics.modules import AngularSpectrumMethod, ZeroPadding

PITCH = 12.5e-6
WAVELENGTH = 1039e-9


def _field(
    resolution: tuple[int, int], pitch: float = PITCH, seed: int = 0
) -> ComplexAmplitude:
    generator = torch.Generator().manual_seed(seed)
    values = torch.complex(
        torch.randn(resolution, generator=generator, dtype=torch.float64),
        torch.randn(resolution, generator=generator, dtype=torch.float64),
    )
    return ComplexAmplitude(values, wavelength=WAVELENGTH, pixel_size=(pitch, pitch))


def test_the_field_sits_at_the_centre_of_the_plane() -> None:
    """The field's centre pixel, (3, 4), lands on the plane's, (8, 10), with zeros
    around it and the pitch unchanged.
    """
    field = _field((6, 8))
    padding = ZeroPadding((16 * PITCH, 20 * PITCH))
    padded = padding(field).as_tensor().reshape(16, 20)

    assert padding.resolution_out == (16, 20)
    torch.testing.assert_close(padded[5:11, 6:14], field.as_tensor().reshape(6, 8))
    around = torch.ones(16, 20, dtype=torch.bool)
    around[5:11, 6:14] = False
    assert torch.count_nonzero(padded[around]) == 0
    pitch = padding.pixel_size_out.reshape(-1, 2)[0].tolist()
    assert pitch == pytest.approx([PITCH, PITCH])


def test_the_plane_keeps_its_extent_on_a_finer_grid() -> None:
    padding = ZeroPadding((16 * PITCH, 20 * PITCH))
    padding(_field((12, 16), pitch=PITCH / 2))
    assert padding.resolution_out == (32, 40)


def test_the_adjoint_is_the_conjugate_transpose() -> None:
    field = _field((6, 8))
    padding = ZeroPadding((16 * PITCH, 20 * PITCH))
    padded = padding(field)
    other = _field((16, 20), seed=1)

    cropped = padding.adjoint(other)
    forward_product = (padded.as_tensor().conj() * other.as_tensor()).sum()
    adjoint_product = (field.as_tensor().conj() * cropped.as_tensor()).sum()
    torch.testing.assert_close(forward_product, adjoint_product)


def test_a_plane_smaller_than_the_field_is_refused() -> None:
    with pytest.raises(ValueError, match="only enlarges"):
        ZeroPadding((4 * PITCH, 20 * PITCH))(_field((6, 8)))


def test_the_padded_plane_keeps_the_light_that_leaves_the_field() -> None:
    """A beam tilted by 30 mrad walks 48 pixels over 20 mm, past the edge of a field 64
    pixels across. The angular spectrum method keeps its grid, so on the field alone the
    light that walks off is lost. On the padded plane, the light stays on the grid.
    """
    resolution = (64, 64)
    distance = 0.02
    angle = 30e-3
    rows, columns = torch.meshgrid(
        torch.arange(resolution[0], dtype=torch.float64) - resolution[0] // 2,
        torch.arange(resolution[1], dtype=torch.float64) - resolution[1] // 2,
        indexing="ij",
    )
    x, y = columns * PITCH, rows * PITCH
    beam = torch.exp(-(x**2 + y**2) / (8 * PITCH) ** 2) * torch.exp(
        2j * torch.pi / WAVELENGTH * angle * x
    )
    field = ComplexAmplitude(beam, wavelength=WAVELENGTH, pixel_size=(PITCH, PITCH))
    power = float((field.as_tensor().abs() ** 2).sum())

    plane = ZeroPadding((256 * PITCH, 256 * PITCH))(field)
    unpadded = AngularSpectrumMethod(distance)(field).as_tensor()
    padded = AngularSpectrumMethod(distance)(plane).as_tensor()

    assert float((unpadded.abs() ** 2).sum()) < 0.5 * power
    assert float((padded.abs() ** 2).sum()) == pytest.approx(power, rel=1e-6)
