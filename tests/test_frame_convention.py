"""The physical frame and the pixel frame of HoloGradPy.

Positions, displacements and angles are physical. Arrays are indexed
``image[row, column]``, and pixel positions are ``(column, row)``. Only
:mod:`hologradpy.grids` converts between the two.

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

import math
from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest
import torch

from hologradpy.calibration.camera_mapping import CameraMapping, FocalSpotFit
from hologradpy.calibration.wavefront.abstract import WavefrontCalibrationData
from hologradpy.calibration.wavefront.raster_calibration.visualizer import (
    RasterVisualizationData,
)
from hologradpy.geometry import AffineTransform
from hologradpy.grids import (
    coordinates_to_indices,
    get_spatial_grid,
    metres_to_pixel,
    plane_center,
)
from hologradpy.holography.phase_retrieval import (
    PixelwisePhaseRetriever,
    WeightedGerchbergSaxtonPhaseRetriever,
)
from hologradpy.holography.camera_feedback import (
    CameraFeedbackData,
    TargetPlacementData,
)
from hologradpy.holography.vortices import VortexAnnihilationData, VortexDetector
from hologradpy.holography.vortices.vortex_detection import find_vortex_charge
from hologradpy.optics import SLMFourierLensModel
from hologradpy.optics.complex_amplitude import ComplexAmplitude, FieldGeometry
from hologradpy.optics.modules.slm_fields import PixelwiseSLMField
from hologradpy.optics.modules.virtual_slms import VirtualSLM
from hologradpy.optics.systems import SLMCZT, SLMFFT, SLMFFTAffine, SLMNUFFT
from hologradpy.profiles.amplitude import gaussian_beam_intensity
from hologradpy.profiles.gaussian import GaussianBeam
from hologradpy.profiles.gaussian_zernike import GaussianZernikeBeam
from hologradpy.profiles.phase import linear_phase, vortex_field, vortex_phase
from hologradpy.utils import as_image

from .registry import requires_nufft

SLM_RESOLUTION = (64, 80)
PITCH = 12.5e-6
WAVELENGTH = 1e-6
FOCAL_LENGTH = 0.1
PADDED = (256, 256)
FOCAL_PITCH = WAVELENGTH * FOCAL_LENGTH / (PADDED[0] * PITCH)  # 31.25 um
# Off both axes and off the diagonals, so a mirror or a transpose misses its pixel.
POSITION = (-20 * FOCAL_PITCH, 7 * FOCAL_PITCH)
VORTEX_RESOLUTION = (32, 32)

CAMERA_SYSTEMS = [SLMCZT, SLMFFTAffine, pytest.param(SLMNUFFT, marks=requires_nufft)]


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


def test_a_cg_search_that_goes_non_finite_keeps_its_last_finite_levels():
    """From a flat start, CG on the spot's pixel goes NaN in its second iteration:
    torchmin's line search overflows in float32. The search stops there, instead of
    spending the other 18 iterations on NaN, about 25 evaluations each. The retriever
    restores the last finite levels and warns.
    """
    model = _model()
    target = torch.zeros(PADDED)
    target[_pixel_of(model, POSITION)] = 1.0
    retriever = PixelwisePhaseRetriever(model, target)

    with pytest.warns(UserWarning, match="non-finite"):
        retriever.retrieve_phase(20, method="cg", verbose=False)

    assert retriever.iteration <= 3
    assert len(retriever.run.loss_history) < 100
    assert torch.isfinite(model.virtual_slm.levels).all()


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
    """A step to the next row index is a step of one pitch towards -y."""
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
        zeroth_order_position=plane_center(output_resolution),
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

    column, row = mapping.image_plane_to_sensor((0.0, PITCH))[0]

    assert (column, row) == pytest.approx((center_column, center_row - 1))


def test_a_turned_scaled_mapping_round_trips_through_the_image_plane():
    """On the non-square 8 x 10 plane, sensor ``(column, row)`` pixels go to image-plane
    metres and back to the same pixels. The zeroth order lands where the inverse of the
    mapping puts the plane's centre pixel.
    """
    resolution = (8, 10)
    camera_points = np.array([[1.0, 2.0], [6.0, 1.0], [3.0, 5.0], [7.0, 4.0]])
    similarity = AffineTransform.from_components(
        scale=1.3, angle_deg=25.0, shift=(0.5, -1.0)
    )
    mapping = _mapping(
        camera_points, similarity.transform_points(camera_points), resolution
    )
    pixels = np.array([[0.0, 0.0], [9.0, 7.0], [2.5, 6.0]])

    positions = mapping.sensor_to_image_plane(pixels)

    np.testing.assert_allclose(
        mapping.image_plane_to_sensor(positions), pixels, atol=1e-9
    )
    np.testing.assert_allclose(
        mapping.image_plane_to_sensor((0.0, 0.0))[0],
        similarity.inverse().transform_points([plane_center(resolution)])[0],
        atol=1e-9,
    )


def test_a_crop_from_a_turned_camera_comes_out_in_the_model_orientation():
    """The camera is turned against the model: a step along the camera's columns is a
    step along the model's rows. The crop's sample 2 columns right of its centre then
    lands 2 rows below its centre.
    """
    turn = np.array([[0.0, -1.0], [1.0, 0.0]])
    camera_points = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [7.0, 3.0]])
    crop = np.zeros((9, 9))
    crop[4, 6] = 1.0
    mapping = _mapping(camera_points, camera_points @ turn.T + (50.0, 20.0), crop.shape)

    resampled = mapping.sensor_crop_to_model(crop)

    assert np.unravel_index(int(resampled.argmax()), resampled.shape) == (6, 4)
    assert resampled.max() == pytest.approx(1.0)


# Saved data. Files saved while y increased with increasing row index load converted.


def _calibration() -> WavefrontCalibrationData:
    """A calibration with a tilt and a lattice drift, and a beam in its metadata, as
    versions 1 and 2 kept it.
    """
    field = ComplexAmplitude(
        torch.linspace(0, 1, 12).reshape(3, 4) + 0j,
        wavelength=torch.tensor(WAVELENGTH),
        pixel_size=torch.tensor([PITCH, PITCH]),
    )
    drift = np.array([1e-6, -2e-6])
    return WavefrontCalibrationData(
        timestamp=datetime(2026, 10, 1),
        name="frame convention",
        complex_amplitude=field,
        metadata={
            "beam_radius": 3e-3,
            "shift_x": 1e-4,
            "shift_y": 2e-4,
            "linear_phase_tilt": (3e-4, 4e-4),
            "lattice_phase_tilt": None,
        },
        visualization_data=RasterVisualizationData(
            camera_images=np.zeros((2, 3, 3)),
            fitted_images=np.zeros((2, 3, 3)),
            measured_phase=np.zeros((3, 4)),
            superpixel_coordinates=np.zeros((2, 2)),
            lattice_shift_x=drift,
            lattice_shift_y=drift,
        ),
    )


def test_a_version_1_calibration_loads_with_its_y_negated(tmp_path, monkeypatch):
    """Version 1 of a calibration was saved while y increased with increasing row
    index. The y of its beam centre, tilt and lattice drift changes sign, and its field
    stays as it is. Its beam moves from the metadata into ``beam``.
    """
    original = _calibration()
    path = tmp_path / "version_1.asdf"
    with monkeypatch.context() as patch:
        patch.setattr(WavefrontCalibrationData, "RECORD_VERSION", 1)
        patch.setattr(RasterVisualizationData, "RECORD_VERSION", 1)
        original.save(path)

    loaded = WavefrontCalibrationData.load(path)

    assert loaded.beam == GaussianZernikeBeam(
        GaussianBeam(3e-3, None, 1e-4, -2e-4), math.sqrt(2) * 3e-3
    )
    assert not {"beam_radius", "shift_x", "shift_y"} & loaded.metadata.keys()
    assert tuple(loaded.metadata["linear_phase_tilt"]) == (3e-4, -4e-4)
    assert loaded.metadata["lattice_phase_tilt"] is None
    assert torch.equal(
        loaded.complex_amplitude.as_tensor(), original.complex_amplitude.as_tensor()
    )
    drift = original.visualization_data.lattice_shift_x
    np.testing.assert_array_equal(loaded.visualization_data.lattice_shift_x, drift)
    np.testing.assert_array_equal(loaded.visualization_data.lattice_shift_y, -drift)


def test_a_version_2_calibration_keeps_its_y(tmp_path, monkeypatch):
    """Version 2 of a calibration was saved with y up, so its y stays. Its beam moves
    from the metadata into ``beam``.
    """
    original = _calibration()
    path = tmp_path / "version_2.asdf"
    with monkeypatch.context() as patch:
        patch.setattr(WavefrontCalibrationData, "RECORD_VERSION", 2)
        original.save(path)

    loaded = WavefrontCalibrationData.load(path)

    assert loaded.beam.gaussian.shift_y == 2e-4
    assert tuple(loaded.metadata["linear_phase_tilt"]) == (3e-4, 4e-4)
    np.testing.assert_array_equal(
        loaded.visualization_data.lattice_shift_y,
        original.visualization_data.lattice_shift_y,
    )


def _version_1_mapping() -> CameraMapping:
    """A mapping whose zeroth order is row 3 and column 7, stored ``(row, column)`` as
    version 1 kept it.
    """
    points = np.array([[1.0, 2.0], [6.0, 1.0], [3.0, 5.0], [7.0, 4.0]])
    return replace(_mapping(points, points, (8, 10)), zeroth_order_position=(3.0, 7.0))


def test_a_version_1_camera_mapping_loads_with_its_zeroth_order_swapped(
    tmp_path, monkeypatch
):
    """Version 1 kept the zeroth order as ``(row, column)``. It loads as
    ``(column, row)``.
    """
    path = tmp_path / "mapping.asdf"
    with monkeypatch.context() as patch:
        patch.setattr(CameraMapping, "RECORD_VERSION", 1)
        _version_1_mapping().save(path)

    assert CameraMapping.load(path).zeroth_order_position == (7.0, 3.0)


def test_a_camera_feedback_run_with_a_version_1_mapping_loads_it_swapped(
    tmp_path, monkeypatch
):
    """The run nests the mapping as its own tagged record, so the mapping converts as
    it loads.
    """
    run = CameraFeedbackData(
        timestamp=datetime(2026, 10, 1),
        name="frame convention",
        target=np.zeros((2, 2)),
        signal_region=np.ones((2, 2), dtype=bool),
        camera_mapping=_version_1_mapping(),
    )
    path = tmp_path / "run.asdf"
    with monkeypatch.context() as patch:
        patch.setattr(CameraMapping, "RECORD_VERSION", 1)
        run.save(path)

    loaded = CameraFeedbackData.load(path)

    assert loaded.camera_mapping.zeroth_order_position == (7.0, 3.0)


def _loaded_at_version_1(data, monkeypatch, tmp_path):
    """``data``, saved at version 1 inside a camera mapping, and loaded back."""
    points = np.array([[1.0, 2.0], [6.0, 1.0], [3.0, 5.0], [7.0, 4.0]])
    mapping = replace(_mapping(points, points, (8, 10)), visualization_data=data)
    path = tmp_path / "nested.asdf"
    with monkeypatch.context() as patch:
        patch.setattr(type(data), "RECORD_VERSION", 1)
        mapping.save(path)
    return CameraMapping.load(path).visualization_data


def test_a_version_1_target_placement_loads_with_its_pixels_swapped(
    tmp_path, monkeypatch
):
    """Version 1 kept the zeroth order, the target centre and the addressable corners
    as ``(row, column)``. They load as ``(column, row)``, and the patch shape stays
    ``(height, width)``.
    """
    placement = TargetPlacementData(
        target=np.zeros((2, 2)),
        signal_region=np.ones((2, 2), dtype=bool),
        zeroth_order=(3.0, 7.0),
        target_center=(4.0, 9.0),
        patch_shape=(2, 4),
        addressable_corners=np.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        ),
    )

    loaded = _loaded_at_version_1(placement, monkeypatch, tmp_path)

    assert loaded.zeroth_order == (7.0, 3.0)
    assert loaded.target_center == (9.0, 4.0)
    assert loaded.patch_shape == (2, 4)
    np.testing.assert_array_equal(
        loaded.addressable_corners, [[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0]]
    )


def test_a_version_1_vortex_annihilation_loads_with_its_positions_swapped(
    tmp_path, monkeypatch
):
    """Version 1 kept the vortex positions as ``(row, column)``. They load as
    ``(column, row)``.
    """
    run = VortexAnnihilationData(
        counts=[2, 0], initial_positions=np.array([[3, 7], [5, 1]])
    )

    loaded = _loaded_at_version_1(run, monkeypatch, tmp_path)

    np.testing.assert_array_equal(loaded.initial_positions, [[7, 3], [1, 5]])
    assert loaded.final_positions is None


def test_a_calibration_keeps_its_beam(tmp_path):
    """A beam with Zernike terms and a cutoff is saved with its calibration and loads
    back equal.
    """
    beam = GaussianZernikeBeam(
        GaussianBeam(3e-3, 2e-3, 1e-4, 2e-4, angle_degrees=30.0),
        4e-3,
        ((0, 0), (2, -2), (2, 0)),
        (0.1, -0.6, 1.5),
        cutoff_radius=5e-3,
    )
    path = tmp_path / "calibration.asdf"
    replace(_calibration(), beam=beam, metadata={}).save(path)

    assert WavefrontCalibrationData.load(path).beam == beam
