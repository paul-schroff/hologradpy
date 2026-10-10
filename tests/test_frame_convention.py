"""The physical frame and the pixel frame of HoloGradPy.

Positions, displacements and angles are physical. Arrays are indexed in the pixel
frame, by ``(column, row)``. Only :mod:`hologradpy.grids` converts between the two.

The consistency tests hold whichever way y points. Each way of putting light at a
position has to agree with the model's output grid: a linear phase and the weighted
Gerchberg-Saxton retriever put a spot on the pixel of its position. The
conjugate-gradient retriever is left out, since it optimises the image on that grid
itself. The vortex detector reads a vortex with the handedness that
:func:`~hologradpy.profiles.phase.vortex_phase` gives it, so its anti-vortex cancels it.

The orientation tests define the physical frame: x points right and y points up on a
screen that draws row 0 at the top, and a positive angle turns counterclockwise.
"""

from __future__ import annotations

import importlib.util
import math
from datetime import datetime

import numpy as np
import pytest
import torch

from hologradpy.calibration.camera_mapping import CameraMapping, FocalSpotFit
from hologradpy.geometry import AffineTransform
from hologradpy.grids import (
    coordinates_to_indices,
    get_spatial_grid,
    metres_to_pixel,
    plane_center,
)
from hologradpy.holography.phase_retrieval import WeightedGerchbergSaxtonPhaseRetriever
from hologradpy.holography.vortices import VortexDetector
from hologradpy.holography.vortices.vortex_detection import find_vortex_charge
from hologradpy.optics import SLMFourierLensModel
from hologradpy.optics.complex_amplitude import ComplexAmplitude, FieldGeometry
from hologradpy.optics.modules.slm_fields import PixelwiseSLMField
from hologradpy.optics.modules.virtual_slms import VirtualSLM
from hologradpy.optics.systems import SLMCZT, SLMFFT, SLMFFTAffine, SLMNUFFT
from hologradpy.profiles.amplitude import gaussian_beam_intensity
from hologradpy.profiles.phase import linear_phase, vortex_field, vortex_phase
from hologradpy.utils import as_image

SLM_RESOLUTION = (64, 80)
PITCH = 12.5e-6
WAVELENGTH = 1e-6
FOCAL_LENGTH = 0.1
PADDED = (256, 256)
FOCAL_PITCH = WAVELENGTH * FOCAL_LENGTH / (PADDED[0] * PITCH)  # 31.25 um
# Off both axes and off the diagonals, so a mirror or a transpose misses its pixel.
POSITION = (-20 * FOCAL_PITCH, 7 * FOCAL_PITCH)
VORTEX_RESOLUTION = (32, 32)

WITH_NUFFT = pytest.mark.skipif(
    importlib.util.find_spec("pytorch_finufft") is None,
    reason="SLMNUFFT needs pytorch-finufft.",
)
CAMERA_SYSTEMS = [SLMCZT, SLMFFTAffine, pytest.param(SLMNUFFT, marks=WITH_NUFFT)]


def _model(system_class=SLMFFT, **camera) -> SLMFourierLensModel:
    """A Gaussian beam on a small SLM, focused by the lens of ``system_class``.

    The systems with a camera image one pixel of the FFT plane on each camera pixel.
    ``camera`` holds their ``camera_angle`` and ``camera_shift``.
    """
    geometry = FieldGeometry(
        resolution=SLM_RESOLUTION,
        pixel_size=torch.tensor([PITCH, PITCH]),
        wavelength=torch.tensor(WAVELENGTH),
    )
    x, y = geometry.get_spatial_grid()
    amplitude = gaussian_beam_intensity(x, y, 0.3e-3).sqrt()
    stages = {
        "input_geometry": geometry,
        "virtual_slm": VirtualSLM(full_scale_cycles=1.0),
        "slm_field": PixelwiseSLMField(
            ComplexAmplitude.from_geometry(geometry, data=amplitude + 0j)
        ),
        "focal_length": FOCAL_LENGTH,
    }
    if system_class is SLMFFT:
        model = SLMFFT(**stages, padded_resolution=PADDED)
    else:
        if system_class is SLMFFTAffine:
            camera["padded_resolution"] = PADDED
        model = system_class(
            **stages,
            camera_resolution=PADDED,
            camera_pixel_size=(FOCAL_PITCH, FOCAL_PITCH),
            **camera,
        )
    model()  # Builds the parameters of the virtual SLM.
    return model


def _peak(model: SLMFourierLensModel) -> tuple[int, int]:
    """The ``(row, column)`` of the brightest output pixel."""
    with torch.no_grad():
        intensity = as_image(model().intensity)
    row, column = np.unravel_index(int(intensity.argmax()), tuple(intensity.shape))
    return int(row), int(column)


def _spot(model: SLMFourierLensModel) -> tuple[float, float]:
    """The ``(row, column)`` centroid of the pixels within 5 of the brightest one."""
    with torch.no_grad():
        intensity = as_image(model().intensity).to(torch.float64)
    row, column = _peak(model)
    window = intensity[row - 5 : row + 6, column - 5 : column + 6]
    offsets = torch.arange(-5, 6, dtype=torch.float64)
    total = window.sum()
    return (
        row + float((window * offsets[:, None]).sum() / total),
        column + float((window * offsets[None, :]).sum() / total),
    )


def _pixel_of(
    model: SLMFourierLensModel, position: tuple[float, float]
) -> tuple[int, int]:
    """The ``(row, column)`` of ``position`` on the output grid of the model."""
    x, y = model.fourier_lens.get_spatial_grid_output()
    ((row, column),) = coordinates_to_indices(
        x, y, torch.tensor([position], dtype=x.dtype)
    )
    return int(row), int(column)


def _tilt(model: SLMFourierLensModel, position: tuple[float, float]) -> None:
    """Display the linear phase that steers the beam to ``position``."""
    x, y = model.input_geometry.get_spatial_grid()
    model.virtual_slm.set_phase(
        linear_phase(
            x,
            y,
            *position,
            wavenumber=2 * math.pi / WAVELENGTH,
            focal_length=FOCAL_LENGTH,
        )
    )


def _linear_phase(model: SLMFFT) -> None:
    _tilt(model, POSITION)


def _weighted_gerchberg_saxton(model: SLMFFT) -> None:
    WeightedGerchbergSaxtonPhaseRetriever(
        model, np.array([POSITION]), np.array([1.0])
    ).retrieve_phase(10, verbose=False)


@pytest.mark.parametrize(
    "place",
    [_linear_phase, _weighted_gerchberg_saxton],
    ids=["linear phase", "weighted Gerchberg-Saxton"],
)
def test_each_way_of_placing_a_spot_puts_it_on_the_pixel_of_its_position(place):
    model = _model()

    place(model)

    assert _peak(model) == _pixel_of(model, POSITION)


def _field(data: torch.Tensor) -> ComplexAmplitude:
    geometry = FieldGeometry(
        resolution=VORTEX_RESOLUTION,
        pixel_size=torch.tensor([1e-5, 1e-5]),
        wavelength=torch.tensor(1e-6),
    )
    return ComplexAmplitude.from_geometry(geometry, data=data.to(torch.complex64))


def test_a_vortex_of_charge_one_is_read_as_charge_one_and_cancelled():
    """The detector reads the charge on its own grid, so the anti-vortex it builds from
    that charge cancels the winding instead of doubling it.
    """
    detector = VortexDetector(VORTEX_RESOLUTION)
    x, y = detector.pixel_grid
    # Half a pixel off the samples, where the detector can see the zero.
    field = vortex_field(x, y, torch.tensor([[3.5, -6.5]]), torch.tensor([1]))

    detector.detect_vortices(
        _field(field), target_intensity=torch.ones(VORTEX_RESOLUTION)
    )
    charges = detector.charges.reshape(-1)
    centers = [
        center for center, charge in zip(detector.center_indices, charges) if charge
    ]

    assert charges[charges != 0].tolist() == [1]
    cancelled = field * detector.generate_anti_vortex_field()
    assert find_vortex_charge(_field(cancelled), x, y, centers).tolist() == [0]


# Orientation. Row 0 is the top row of the screen, so "up" is towards row 0.


@pytest.mark.parametrize("resolution", [(8, 10), (7, 9)])
def test_row_zero_holds_the_largest_y(resolution):
    """A step down the rows is a step of one pitch towards -y."""
    _, y = get_spatial_grid(resolution, (PITCH, PITCH))

    assert float(y[0, 0]) == float(y.max())
    assert float(y[1, 0] - y[0, 0]) == pytest.approx(-PITCH, rel=1e-5)


def test_a_position_up_from_the_origin_is_a_row_above_the_centre():
    resolution = (8, 10)
    center_column, center_row = plane_center(resolution)

    pixel = metres_to_pixel((0.0, PITCH), (PITCH, PITCH), resolution)

    assert pixel == pytest.approx((center_column, center_row - 1))


@pytest.mark.parametrize("system_class", [SLMFFT, *CAMERA_SYSTEMS])
def test_a_tilt_towards_plus_y_lands_above_the_centre(system_class):
    model = _model(system_class)
    center_column, center_row = plane_center(PADDED)

    _tilt(model, (0.0, 10 * FOCAL_PITCH))

    assert _spot(model) == pytest.approx((center_row - 10, center_column), abs=0.5)


@pytest.mark.parametrize("system_class", CAMERA_SYSTEMS)
def test_a_positive_camera_angle_turns_a_spot_counterclockwise(system_class):
    """A spot along +x turns towards the top of the screen."""
    center_column, center_row = plane_center(PADDED)
    angles = []
    for camera_angle in (0.0, 30.0):
        model = _model(system_class, camera_angle=camera_angle)
        _tilt(model, (30 * FOCAL_PITCH, 0.0))
        row, column = _spot(model)
        angle = math.atan2(center_row - row, column - center_column)
        angles.append(math.degrees(angle))

    assert angles[1] - angles[0] == pytest.approx(30.0, abs=1.0)


@pytest.mark.parametrize("system_class", CAMERA_SYSTEMS)
def test_a_camera_shift_towards_plus_y_moves_the_image_up(system_class):
    """The undiffracted spot of a flat phase moves up with the image."""
    model = _model(system_class, camera_shift=(0.0, 10 * FOCAL_PITCH))
    center_column, center_row = plane_center(PADDED)

    assert _spot(model) == pytest.approx((center_row - 10, center_column), abs=0.5)


def test_a_gaussian_turned_by_45_degrees_is_brighter_up_right_than_down_right():
    resolution = (33, 33)
    center_column, center_row = plane_center(resolution)
    x, y = get_spatial_grid(resolution, (1.0, 1.0))

    intensity = gaussian_beam_intensity(x, y, 8.0, 2.0, angle_degrees=45.0)

    up_right = float(intensity[center_row - 5, center_column + 5])
    down_right = float(intensity[center_row + 5, center_column + 5])
    assert up_right > down_right


def test_a_vortex_of_charge_one_is_a_quarter_turn_on_above_the_centre():
    """The phase of a charge-1 vortex is the polar angle, pi/2 straight up."""
    resolution = (33, 33)
    center_column, center_row = plane_center(resolution)
    x, y = get_spatial_grid(resolution, (1.0, 1.0))

    phase = vortex_phase(x, y, torch.tensor([0.0, 0.0]), torch.tensor(1.0))

    assert float(phase[center_row - 5, center_column]) == pytest.approx(math.pi / 2)


def _mapping(
    camera_points: np.ndarray,
    model_points: np.ndarray,
    output_resolution: tuple[int, int],
) -> CameraMapping:
    """A camera mapping fitted to ``(column, row)`` point pairs, with its output plane
    at one pitch per pixel.
    """
    transform = AffineTransform.fit(camera_points, model_points, robust=False)
    return CameraMapping(
        timestamp=datetime.now(),
        name="frame convention",
        transform=transform.as_matrix(homogeneous=False),
        detected_points=[tuple(point) for point in camera_points.tolist()],
        calculated_points=[tuple(point) for point in model_points.tolist()],
        zeroth_order_position=(output_resolution[0] // 2, output_resolution[1] // 2),
        spot_fit=FocalSpotFit(waist=PITCH),
        output_pixel_size=(PITCH, PITCH),
        output_resolution=output_resolution,
    )


def test_a_camera_turned_counterclockwise_reports_minus_its_angle():
    """``numpy.rot90`` turns an image a quarter turn counterclockwise. A camera that
    sees the model plane turned that way maps back to it with a quarter turn clockwise,
    -90 degrees.
    """
    height, width = 6, 8
    model_points = np.array([[1.0, 2.0], [6.0, 1.0], [3.0, 5.0], [7.0, 4.0]])
    # numpy.rot90 moves the pixel (column, row) to (row, width - 1 - column).
    camera_points = np.stack([model_points[:, 1], width - 1 - model_points[:, 0]], 1)
    marker = np.zeros((height, width))
    marker[2, 1] = 1.0
    assert np.rot90(marker)[width - 1 - 1, 2] == 1.0

    mapping = _mapping(camera_points, model_points, (height, width))

    assert mapping.rotation_degrees == pytest.approx(-90.0)


def test_an_identity_mapping_puts_plus_y_above_the_centre():
    resolution = (8, 10)
    center_column, center_row = plane_center(resolution)
    points = np.array([[1.0, 2.0], [6.0, 1.0], [3.0, 5.0], [7.0, 4.0]])
    mapping = _mapping(points, points, resolution)

    row, column = mapping.image_plane_to_sensor((0.0, PITCH))[0]

    assert (row, column) == pytest.approx((center_row - 1, center_column))
