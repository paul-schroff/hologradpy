"""The helpers in :mod:`hologradpy.utils`.

The two progress helpers exist so a long loop can show a bar without the loop having to
know whether one is wanted, so what matters is that turning the bar off changes nothing
except the output. ``as_image`` is the one way the package turns a field into a 2-D
image, so it has to refuse to drop anything but a singleton axis.
``load_grayscale_image`` reads picture files written by Pillow, so each test pins how
one kind of file becomes values in [0, 1].
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from hologradpy.optics.complex_amplitude import ComplexAmplitude
from hologradpy.utils import ProgressBar, as_image, load_grayscale_image, progress


def test_progress_yields_every_element_either_way() -> None:
    assert list(progress(range(5), verbose=False)) == [0, 1, 2, 3, 4]
    assert list(progress(range(5), verbose=True, description="counting")) == [
        0,
        1,
        2,
        3,
        4,
    ]


def test_progress_returns_the_iterable_untouched_when_off() -> None:
    """Not merely equal: the same object, so nothing is consumed or copied on the way
    through.
    """
    values = [3, 1, 4]
    assert progress(values, verbose=False) is values


def test_progress_bar_is_a_no_op_when_off() -> None:
    with ProgressBar(total=2, description="quiet", verbose=False) as bar:
        assert bar._bar is None
        bar.update(loss=1.0)
        bar.update()
    assert bar._bar is None


def test_progress_bar_updates_and_closes() -> None:
    """disable=False because pytest captures stderr, where tqdm would otherwise
    suppress the bar and leave its counter at zero. This is about the mechanics.
    """
    with ProgressBar(total=3, description="loud", verbose=True, disable=False) as bar:
        for value in (1e10, 1e8, 1e6):
            bar.update(loss=value)
        assert bar._bar.n == 3
    assert bar._bar is None


def test_progress_bar_closes_on_exception() -> None:
    """A search that raises should not leave a bar open and the terminal wedged."""
    bar = ProgressBar(total=3, description="raising", verbose=True)
    with pytest.raises(RuntimeError):
        with bar:
            bar.update()
            raise RuntimeError("optimizer blew up")
    assert bar._bar is None


def test_progress_bar_formats_numeric_postfix() -> None:
    """A loss running from 1e10 to 1e6 keeps a constant width, so the bar does not
    jitter.
    """
    with ProgressBar(total=1, verbose=True, disable=False) as bar:
        bar.update(loss=1.23456789e10, label="text")
        postfix = bar._bar.postfix
    assert "1.235e+10" in postfix
    assert "text" in postfix


# --- Suppression where a bar cannot render -----------------------------------------


def test_bars_are_left_for_tqdm_to_suppress() -> None:
    """Bars pass disable=None, which is tqdm's own non-terminal rule.

    tqdm then suppresses a text bar when the stream is not a terminal and keeps a widget
    bar regardless. Doing this ourselves is what the removed nested_progress_supported()
    was for, and tqdm already knows the answer.
    """
    with ProgressBar(total=2, verbose=True) as bar:
        assert bar._bar.disable is not None


def test_as_image_drops_singleton_leading_axes() -> None:
    image = torch.arange(6.0).reshape(1, 1, 2, 3)
    assert as_image(image).shape == (2, 3)
    assert as_image(image).data_ptr() == image.data_ptr()
    assert as_image(np.zeros((1, 2, 3))).shape == (2, 3)
    assert as_image(torch.zeros(2, 3)).shape == (2, 3)


def test_as_image_refuses_to_drop_data() -> None:
    with pytest.raises(ValueError, match="length one"):
        as_image(torch.zeros(2, 4, 4))
    # The suite's runtime shape checks refuse a 1-D input before the function does.
    with pytest.raises((ValueError, TypeError)):
        as_image(torch.zeros(4))


def test_as_image_reads_a_field_as_a_plain_tensor() -> None:
    field = ComplexAmplitude(
        torch.ones(4, 4, dtype=torch.complex64), 800e-9, (1e-5, 1e-5)
    )
    image = as_image(field)
    assert type(image) is torch.Tensor
    assert image.shape == (4, 4)


def _saved(folder: Path, name: str, pixels: np.ndarray) -> Path:
    """``pixels`` saved by Pillow as ``name`` in ``folder``. The dtype and shape set the
    mode: uint8 gray, RGB or RGBA, uint16 for 16 bits, float32 for floats.
    """
    path = folder / name
    Image.fromarray(pixels).save(path)
    return path


def test_an_8_bit_picture_is_divided_by_255(tmp_path: Path) -> None:
    pixels = np.array([[0, 128, 255]], dtype=np.uint8)
    image = load_grayscale_image(_saved(tmp_path, "gray.png", pixels))
    assert image.dtype == np.float64
    np.testing.assert_allclose(image, [[0, 128 / 255, 1]])


def test_the_first_row_is_the_top_of_the_picture(tmp_path: Path) -> None:
    pixels = np.zeros((3, 2), dtype=np.uint8)
    pixels[0] = 255
    image = load_grayscale_image(_saved(tmp_path, "top.png", pixels))
    np.testing.assert_array_equal(image, pixels / 255)


def test_colour_becomes_luma(tmp_path: Path) -> None:
    green = np.array([[[0, 255, 0]]], dtype=np.uint8)
    image = load_grayscale_image(_saved(tmp_path, "green.png", green))
    assert abs(image[0, 0] - 0.587) <= 1 / 255


def test_a_transparent_pixel_is_dark(tmp_path: Path) -> None:
    pixels = np.full((1, 2, 4), 255, dtype=np.uint8)
    pixels[0, 0, 3] = 0
    image = load_grayscale_image(_saved(tmp_path, "alpha.png", pixels))
    np.testing.assert_allclose(image, [[0, 1]])


def test_a_16_bit_picture_is_divided_by_65535(tmp_path: Path) -> None:
    pixels = np.array([[0, 32768, 65535]], dtype=np.uint16)
    image = load_grayscale_image(_saved(tmp_path, "deep.png", pixels))
    np.testing.assert_allclose(image, [[0, 32768 / 65535, 1]])


def test_a_float_picture_is_kept_as_it_is(tmp_path: Path) -> None:
    pixels = np.array([[0, 0.25, 1]], dtype=np.float32)
    image = load_grayscale_image(_saved(tmp_path, "float.tif", pixels))
    np.testing.assert_allclose(image, pixels)


def test_a_float_picture_outside_0_to_1_is_refused(tmp_path: Path) -> None:
    pixels = np.array([[0, 2]], dtype=np.float32)
    with pytest.raises(ValueError, match="outside"):
        load_grayscale_image(_saved(tmp_path, "bright.tif", pixels))


def test_a_jpeg_loads_as_one_gray_channel(tmp_path: Path) -> None:
    pixels = np.random.default_rng(0).integers(0, 256, (8, 6, 3), dtype=np.uint8)
    image = load_grayscale_image(_saved(tmp_path, "photo.jpg", pixels))
    assert image.shape == (8, 6)
    assert 0 <= image.min() and image.max() <= 1


def test_a_device_gives_a_float32_tensor(tmp_path: Path) -> None:
    path = _saved(tmp_path, "gray.png", np.array([[0, 255]], dtype=np.uint8))
    image = load_grayscale_image(path, device=torch.device("cpu"))
    assert type(image) is torch.Tensor
    assert image.dtype == torch.float32
    assert image.device == torch.device("cpu")
    np.testing.assert_allclose(image.numpy(), load_grayscale_image(path))
