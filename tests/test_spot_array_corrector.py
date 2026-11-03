"""Tests for measure_spot_intensities and the SpotArrayCorrector.

The measurement runs on a camera that renders Gaussian spots of known powers. The
corrector runs on a simulated bench whose SLM has pixel crosstalk, which the model
lacks. The crosstalk dims the spots far from the zeroth order, so the measured spots
differ from the model's. On the distorted bench, the camera also shows the spots off
where its mapping places them.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates

from hologradpy.calibration.camera_mapping import CameraMapping, FocalSpotFit
from hologradpy.hardware import SimulatedCameraTorch, SimulatedSLMTorch
from hologradpy.hardware.camera import CameraData
from hologradpy.holography.camera_feedback import (
    SpotArrayCorrectionData,
    SpotArrayCorrector,
    measure_spot_intensities,
)
from hologradpy.holography.phase_retrieval import (
    LinearSuperpositionPhaseRetriever,
    PixelwisePhaseRetriever,
    WeightedGerchbergSaxtonPhaseRetriever,
)
from hologradpy.loss_functions import LossAbsoluteIntensityMSE
from hologradpy.optics.complex_amplitude import ComplexAmplitude, FieldGeometry
from hologradpy.optics.modules.pixel_crosstalk import SuperGaussianCrosstalk
from hologradpy.optics.modules.slm_fields import PixelwiseSLMField
from hologradpy.optics.modules.virtual_slms import VirtualSLM
from hologradpy.optics.systems import SLMCZT
from hologradpy.profiles.amplitude import gaussian_beam_intensity
from hologradpy.utils import as_image, gpu_to_numpy

from tests.native_camera_fakes import CroppingCamera

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

# The sensor of the camera that renders spots, whose pitch is 1 um.
SPOT_CAMERA_SENSOR = (120, 160)
SPOT_CAMERA_PITCH = 1e-6
# The offset the spot camera adds to every pixel, in counts.
DARK_OFFSET = 10.0

# The simulated bench.
CAMERA_RESOLUTION = (240, 320)
CAMERA_PITCH = 15e-6
FOCAL_LENGTH = 0.25
# The waist of the bench's focal spot, in metres: lambda f / (pi w) for the 1 mm beam.
FOCAL_SPOT_RADIUS = 50e-6
# The distortion of the distorted bench's camera, in px^-2. A spot 120 px from the
# sensor's centre appears 1.7 px further out.
DISTORTION = 1e-6


class SpotCamera(CroppingCamera):
    """A camera that renders Gaussian spots at sensor pixels, each with a power per
    second of equivalent exposure, on a constant offset.
    """

    def __init__(
        self,
        sensor_pixels: NDArray,
        powers: NDArray,
        waist: float,
    ) -> None:
        super().__init__(SPOT_CAMERA_SENSOR, max_pixel_value=4095)
        rows, columns = np.indices(SPOT_CAMERA_SENSOR)
        self.profile = np.zeros(SPOT_CAMERA_SENSOR)
        for (column, row), power in zip(sensor_pixels, powers):
            radius_squared = (columns - column) ** 2 + (rows - row) ** 2
            # A Gaussian that sums to its power over the plane.
            self.profile += (
                power * 2 / (np.pi * waist**2) * np.exp(-2 * radius_squared / waist**2)
            )

    def render_sensor_frame(self) -> NDArray:
        return self.profile * self.get_equivalent_exposure() + DARK_OFFSET


def _spot_camera_mapping(waist_px: float) -> CameraMapping:
    """A mapping of the spot camera, with the zeroth order at the centre pixel."""
    points = [(10.0, 12.0), (150.0, 15.0), (20.0, 110.0), (155.0, 115.0)]
    return CameraMapping(
        timestamp=datetime(2026, 1, 1),
        name="identity",
        transform=np.eye(3),
        detected_points=points,
        calculated_points=points,
        spot_fit=FocalSpotFit(waist=waist_px * SPOT_CAMERA_PITCH),
        output_pixel_size=(SPOT_CAMERA_PITCH, SPOT_CAMERA_PITCH),
        output_resolution=SPOT_CAMERA_SENSOR,
    )


def _spot_camera(
    positions: NDArray, powers: NDArray, waist_px: float
) -> tuple[SpotCamera, CameraMapping]:
    """A spot camera with spots of ``powers`` at ``positions``, and its mapping."""
    mapping = _spot_camera_mapping(waist_px)
    sensor_pixels = mapping.image_plane_to_sensor(positions)
    return SpotCamera(sensor_pixels, powers, waist_px), mapping


def _peaks(powers: NDArray, waist: float) -> NDArray:
    """The peaks of Gaussian spots of ``powers`` and ``waist`` in pixels, smoothed over
    a pixel. A Gaussian of sigma 1 px lowers a peak by a factor of 1 + 4 / waist^2.
    """
    return 2 * powers / (np.pi * waist**2) / (1 + 4 / waist**2)


# Four spots at least 46 px apart.
FAR_APART = np.array(
    [[-40e-6, 20e-6], [30e-6, 25e-6], [-35e-6, -30e-6], [40e-6, -20e-6]]
)


def test_the_measured_intensities_are_the_spots_peaks():
    """Each reading is the spot's peak smoothed over a pixel, per second of equivalent
    exposure, at the spot's centre.
    """
    powers = np.array([1.0, 2.0, 3.0, 1.5]) * 1e6
    camera, mapping = _spot_camera(FAR_APART, powers, waist_px=3.0)

    measured = measure_spot_intensities(camera, mapping, FAR_APART)

    np.testing.assert_allclose(measured.intensities, _peaks(powers, 3.0), rtol=1e-3)
    np.testing.assert_allclose(
        measured.sensor_positions, mapping.image_plane_to_sensor(FAR_APART), atol=0.01
    )
    assert measured.background == pytest.approx(DARK_OFFSET)
    assert not measured.overexposed


def test_close_spots_still_read_their_peaks():
    """Spots two waists apart read their own peaks, at a fixed exposure."""
    positions = np.array([[0.0, 0.0], [12e-6, 0.0], [0.0, 12e-6]])
    powers = np.array([1.0, 1.0, 2.0]) * 1e6
    camera, mapping = _spot_camera(positions, powers, waist_px=6.0)

    measured = measure_spot_intensities(camera, mapping, positions, exposure=1e-3)

    assert measured.exposure == pytest.approx(1e-3)
    np.testing.assert_allclose(measured.intensities, _peaks(powers, 6.0), rtol=1e-2)


def test_a_mapping_off_by_two_pixels_still_finds_the_spots():
    """The search moves from the misplaced sites to the spots' maxima."""
    powers = np.array([1.0, 2.0, 3.0, 1.5]) * 1e6
    camera, mapping = _spot_camera(FAR_APART, powers, waist_px=3.0)

    measured = measure_spot_intensities(
        camera, mapping, FAR_APART + np.array([2e-6, -1.5e-6])
    )

    np.testing.assert_allclose(
        measured.sensor_positions, mapping.image_plane_to_sensor(FAR_APART), atol=0.05
    )
    np.testing.assert_allclose(measured.intensities, _peaks(powers, 3.0), rtol=1e-3)


def test_a_mapping_off_by_most_of_a_spacing_is_refused():
    """Shifted by 0.55 of the spacing, two sites find the same spot."""
    positions = np.array([[-20e-6, 0.0], [0.0, 0.0], [20e-6, 0.0]])
    camera, mapping = _spot_camera(positions, np.full(3, 1e6), waist_px=4.0)

    with pytest.raises(ValueError, match="find no maximum of their own"):
        measure_spot_intensities(camera, mapping, positions + np.array([11e-6, 0.0]))


def test_a_spot_off_the_sensor_is_refused():
    """A window has to lie on the sensor whole."""
    positions = np.array([[0.0, 0.0], [78e-6, 0.0]])
    camera, mapping = _spot_camera(positions, np.array([1e6, 1e6]), waist_px=3.0)

    with pytest.raises(ValueError, match=r"spots \[1\].*run off the sensor"):
        measure_spot_intensities(camera, mapping, positions)


def test_a_spot_without_light_is_refused():
    """A dark spot points to a mapping that no longer fits the camera."""
    positions = np.array([[-40e-6, 0.0], [40e-6, 0.0]])
    camera, mapping = _spot_camera(positions[:1], np.array([1e6]), waist_px=3.0)

    refusal = r"Spots \[1\] (hold no light|find no maximum)"
    with pytest.raises(ValueError, match=refusal):
        measure_spot_intensities(camera, mapping, positions)


# --- SpotArrayCorrector on the simulated bench ----------------------------------------


def _geometry() -> FieldGeometry:
    return FieldGeometry(
        resolution=(256, 320),
        pixel_size=torch.tensor([12.5e-6, 12.5e-6]),
        wavelength=torch.tensor(0.630e-6),
    )


def _beam(geometry: FieldGeometry) -> ComplexAmplitude:
    intensity = gaussian_beam_intensity(
        *geometry.get_spatial_grid(), beam_radius_major=1e-3
    )
    return ComplexAmplitude(
        intensity.sqrt() + 0j,
        wavelength=geometry.wavelength,
        pixel_size=geometry.pixel_size,
    )


def _model(geometry: FieldGeometry, virtual_slm: VirtualSLM) -> SLMCZT:
    """A model whose output plane is the bench's sensor, centred on the zeroth order."""
    return SLMCZT(
        input_geometry=geometry,
        virtual_slm=virtual_slm,
        camera_resolution=CAMERA_RESOLUTION,
        camera_pixel_size=(CAMERA_PITCH, CAMERA_PITCH),
        focal_length=FOCAL_LENGTH,
        slm_field=PixelwiseSLMField(_beam(geometry)),
    )


def _model_pixels(camera_pixels: NDArray) -> NDArray:
    """The ``(column, row)`` pixels of the model's image that the distorted camera
    shows at ``camera_pixels``. A camera pixel ``r`` from the sensor's centre shows the
    model's pixel ``r (1 - DISTORTION r^2)``.
    """
    centre = (np.array(CAMERA_RESOLUTION[::-1]) - 1) / 2
    offsets = camera_pixels - centre
    squared = np.sum(offsets**2, axis=-1, keepdims=True)
    return centre + offsets * (1 - DISTORTION * squared)


class DistortedCamera(SimulatedCameraTorch):
    """The bench's camera behind a radial distortion that the model lacks. Each pixel
    shows the model's image at :func:`_model_pixels`.
    """

    def _capture_frame(self) -> torch.Tensor:
        frame = super()._capture_frame()
        rows, columns = np.indices(CAMERA_RESOLUTION, dtype=np.float64)
        source = _model_pixels(np.stack([columns, rows], axis=-1))
        image = gpu_to_numpy(frame).reshape(CAMERA_RESOLUTION)
        warped = map_coordinates(
            image, [source[..., 1], source[..., 0]], mode="nearest"
        )
        warped = torch.as_tensor(warped, dtype=frame.dtype, device=frame.device)
        return warped.reshape(frame.shape)


def _bench(
    distorted: bool = False,
) -> tuple[SimulatedSLMTorch, SimulatedCameraTorch, SLMCZT]:
    """The SLM, with pixel crosstalk, the camera, and a model without the crosstalk. If
    ``distorted`` is True, the camera is a :class:`DistortedCamera`.
    """
    torch.manual_seed(0)
    geometry = _geometry()
    slm = SimulatedSLMTorch(
        input_geometry=geometry,
        bitdepth=8,
        pixel_crosstalk=SuperGaussianCrosstalk(upscale_factor=3, order=2.0, width=1.0),
    )
    camera_type = DistortedCamera if distorted else SimulatedCameraTorch
    camera = camera_type(_model(geometry, slm.virtual_slm), add_noise=False)
    camera.set_exposure(1e-3)
    camera.get_image()
    return slm, camera, _model(geometry, VirtualSLM(full_scale_cycles=1.0))


def _bench_mapping(distorted: bool = False) -> CameraMapping:
    """The bench's mapping, fitted to a grid of calibration spots on the camera.

    - If ``distorted`` is False, the camera agrees with the model, so the mapping is
      the identity.
    - If ``distorted`` is True, each spot shows the model's pixel at
      :func:`_model_pixels`. The mapping's similarity takes up the distortion's mean
      zoom, and the rest moves the spots off where the mapping places them.
    """
    columns, rows = np.meshgrid(np.linspace(30, 290, 9), np.linspace(20, 220, 7))
    detected = np.column_stack([columns.ravel(), rows.ravel()])
    calculated = _model_pixels(detected) if distorted else detected
    return CameraMapping(
        timestamp=datetime(2026, 1, 1),
        name="bench",
        transform=np.eye(3),
        detected_points=[tuple(point) for point in detected],
        calculated_points=[tuple(point) for point in calculated],
        spot_fit=FocalSpotFit(waist=FOCAL_SPOT_RADIUS),
        output_pixel_size=(CAMERA_PITCH, CAMERA_PITCH),
        output_resolution=CAMERA_RESOLUTION,
    )


def _positions() -> NDArray:
    """A 3 x 4 array, 0.4 mm apart, clear of the zeroth order."""
    xs, ys = np.meshgrid([0.6e-3, 1.0e-3, 1.4e-3, 1.8e-3], [-0.4e-3, 0.0, 0.4e-3])
    return np.column_stack([xs.ravel(), ys.ravel()])


def _wgs_corrector(
    target_intensities: NDArray | None = None, distorted: bool = False
) -> SpotArrayCorrector:
    """A corrector driving weighted Gerchberg-Saxton, which holds the positions."""
    slm, camera, model = _bench(distorted)
    retriever = WeightedGerchbergSaxtonPhaseRetriever(
        model, _positions(), target_intensities
    )
    return SpotArrayCorrector(slm, camera, retriever, _bench_mapping(distorted))


def _spot_target(model: SLMCZT, positions: NDArray) -> torch.Tensor:
    """Gaussian spots of the bench's focal-spot radius, which together hold the
    incident power, as hologradpy-square's compute_spot_array builds them.
    """
    x, y = model.focal_plane_positions(*model[-1].get_spatial_grid_output())
    target = sum(
        gaussian_beam_intensity(x, y, FOCAL_SPOT_RADIUS, None, *position)
        for position in positions
    )
    return target * (model.incident_power() / model.output_pixel_area() / target.sum())


def _pixelwise_retriever(model: SLMCZT, positions: NDArray) -> PixelwisePhaseRetriever:
    """A pixelwise retriever towards Gaussian spots, read in absolute units, from a
    linear superposition with random phases.
    """
    start = LinearSuperpositionPhaseRetriever(
        model, positions, target_phases="random"
    ).retrieve_phase(verbose=False)
    retriever = PixelwisePhaseRetriever(
        model, _spot_target(model, positions), init_slm_phase=start
    )
    retriever.set_loss_factory(LossAbsoluteIntensityMSE)
    return retriever


def _pixelwise_corrector(distorted: bool = False) -> SpotArrayCorrector:
    """A corrector driving a pixelwise retriever, whose target is an image."""
    slm, camera, model = _bench(distorted)
    positions = _positions()
    return SpotArrayCorrector(
        slm,
        camera,
        _pixelwise_retriever(model, positions),
        _bench_mapping(distorted),
        positions,
    )


@pytest.fixture(scope="module")
def wgs_run() -> SpotArrayCorrectionData:
    return _wgs_corrector().run(retriever_iterations=(40,) * 3, verbose=False)


def test_wgs_feedback_evens_out_the_spots(wgs_run):
    """The crosstalk leaves the first hologram a few percent off. Feedback brings the
    spots to a third of that.
    """
    non_uniformity = wgs_run.non_uniformity
    assert non_uniformity[0] > 0.02
    assert non_uniformity[-1] < non_uniformity[0] / 3


def test_pixelwise_feedback_evens_out_the_spots():
    data = _pixelwise_corrector().run(retriever_iterations=(40,) * 5, verbose=False)

    assert data.non_uniformity[0] > 0.02
    assert data.non_uniformity[-1] < data.non_uniformity[0] / 3


def test_the_pixelwise_target_keeps_its_power_as_its_spots_move():
    """Each spot's weight in the target moves by its factor, and the target keeps its
    total, which the cost reads in absolute units.
    """
    corrector = _pixelwise_corrector()
    total = float(corrector.phase_retriever.target.sum())
    data = corrector.run(retriever_iterations=(20,) * 2, verbose=False)

    final_total = float(corrector.phase_retriever.target.sum())
    assert final_total == pytest.approx(total, rel=1e-4)
    before, after = data.adjusted_weights
    measured = data.measured_intensities[0]
    requested = data.requested_weights
    expected = before * requested / (measured / measured.mean())
    np.testing.assert_allclose(after, expected / expected.mean(), rtol=1e-3)


def test_the_target_is_read_between_pixels():
    """At positions between the pixels, the read follows the exact sum of the target's
    Gaussian spots.
    """
    slm, camera, model = _bench()
    model.build()
    x, _ = model.focal_plane_positions(*model[-1].get_spatial_grid_output())
    pitch = float(x[0, 1] - x[0, 0])
    rng = np.random.default_rng(0)
    positions = _positions() + rng.uniform(-0.5, 0.5, (12, 2)) * pitch
    retriever = _pixelwise_retriever(model, positions)
    corrector = SpotArrayCorrector(slm, camera, retriever, _bench_mapping(), positions)

    read = corrector._target_at_positions()

    squared = np.sum((positions[:, None, :] - positions[None, :, :]) ** 2, axis=-1)
    exact = np.exp(-2 * squared / FOCAL_SPOT_RADIUS**2).sum(axis=1)
    np.testing.assert_allclose(read / read.mean(), exact / exact.mean(), rtol=5e-3)


def test_the_target_scales_smoothly_between_close_spots():
    """Spots two waists apart, scaled by 1.1 and 0.9, get those factors at their
    positions and their mean halfway, up to a common renormalization.
    """
    slm, camera, model = _bench()
    model.build()
    x, y = model.focal_plane_positions(*model[-1].get_spatial_grid_output())
    # An output pixel lies halfway between the spots.
    middle = (120, 200)
    halfway = np.array([float(x[middle]), float(y[middle])])
    offset = np.array([FOCAL_SPOT_RADIUS, 0.0])
    positions = np.array([halfway - offset, halfway + offset])
    retriever = _pixelwise_retriever(model, positions)
    corrector = SpotArrayCorrector(slm, camera, retriever, _bench_mapping(), positions)
    before = as_image(retriever.target).clone()

    corrector._scale_weights(np.array([1.1, 0.9]))

    ratio = as_image(retriever.target) / before
    nearest = [
        np.unravel_index(int(torch.argmin((x - px) ** 2 + (y - py) ** 2)), x.shape)
        for px, py in positions
    ]
    at_middle = float(ratio[middle])
    assert float(ratio[nearest[0]]) / at_middle == pytest.approx(1.1, rel=1e-3)
    assert float(ratio[nearest[1]]) / at_middle == pytest.approx(0.9, rel=1e-3)


def test_moving_the_spots_resamples_the_target():
    """Two spots two waists apart move by different sub-pixel errors. Within half a
    waist of each spot, the target then matches spots built at the moved positions.
    The target keeps its total power, so the stretch between the spots lowers both
    spots alike.
    """
    slm, camera, model = _bench()
    positions = np.array([[1.0e-3, 0.0], [1.1e-3, 0.0]])
    retriever = _pixelwise_retriever(model, positions)
    corrector = SpotArrayCorrector(slm, camera, retriever, _bench_mapping(), positions)
    before = corrector._target_at_positions()
    errors = np.array([[0.4, -0.3], [-0.3, 0.5]]) * CAMERA_PITCH

    corrector._move_spots(errors)

    moved = positions - errors
    np.testing.assert_allclose(corrector._targets, moved)
    lowered = corrector._target_at_positions() / before
    assert lowered[0] == pytest.approx(lowered[1], rel=2e-3)
    x, y = model.focal_plane_positions(*model[-1].get_spatial_grid_output())
    squared = torch.stack([(x - px) ** 2 + (y - py) ** 2 for px, py in moved])
    near = (squared < (FOCAL_SPOT_RADIUS / 2) ** 2).any(0)
    expected = as_image(_spot_target(model, moved)) * float(lowered.mean())
    difference = (as_image(retriever.target) - expected)[near].abs().max()
    assert float(difference) < 1e-2 * float(expected.max())


def test_feedback_reaches_unequal_intensities():
    """Spots alternate between 1 and 2, and the measured ones follow."""
    data = _wgs_corrector(np.tile([1.0, 2.0], 6)).run(
        retriever_iterations=(40,) * 3, verbose=False
    )

    assert data.non_uniformity[-1] < 0.01
    measured = data.measured_intensities[-1]
    assert measured[1::2].mean() / measured[::2].mean() == pytest.approx(2.0, rel=0.02)


@pytest.mark.parametrize("corrector", [_wgs_corrector, _pixelwise_corrector])
def test_feedback_moves_the_spots_onto_their_positions(corrector):
    """The distorted camera shows the spots about half a pixel off where the mapping
    places them. Moving the targets brings the spots within a tenth of a pixel, and
    the intensities still even out.
    """
    data = corrector(distorted=True).run(retriever_iterations=(40,) * 3, verbose=False)

    error = np.array(data.position_error) / CAMERA_PITCH
    assert error[0] > 0.3
    assert error[-1] < 0.1
    assert data.non_uniformity[-1] < data.non_uniformity[0] / 2


def test_feedback_can_leave_the_positions_alone():
    """Without the position correction, the targets stay at the positions, and the
    distorted camera keeps showing the spots off them.
    """
    data = _wgs_corrector(distorted=True).run(
        retriever_iterations=(40,) * 2, verbose=False, correct_positions=False
    )

    for adjusted in data.adjusted_positions:
        np.testing.assert_array_equal(adjusted, data.positions)
    assert data.position_error[-1] > 0.3 * CAMERA_PITCH


def test_the_weights_move_by_the_requested_over_the_measured(wgs_run):
    """Each hologram after the first is computed for the weights before it, times the
    requested over the measured intensities.
    """
    for before, measured, after in zip(
        wgs_run.adjusted_weights,
        wgs_run.measured_intensities,
        wgs_run.adjusted_weights[1:],
    ):
        expected = before / (measured / measured.mean())
        np.testing.assert_allclose(after, expected / expected.mean())


def test_the_record_holds_one_entry_per_iteration(wgs_run):
    assert wgs_run.number_of_iterations == 3
    for series in (
        wgs_run.adjusted_weights,
        wgs_run.adjusted_positions,
        wgs_run.measured_intensities,
        wgs_run.measured_positions,
        wgs_run.predicted_intensities,
        wgs_run.retrievals,
        wgs_run.exposures,
        wgs_run.overexposed,
        wgs_run.position_error,
    ):
        assert len(series) == 3
    assert wgs_run.final_camera_image.shape == CAMERA_RESOLUTION
    assert wgs_run.measured_positions[-1].shape == (12, 2)


def test_the_record_survives_a_round_trip(wgs_run, tmp_path):
    path = tmp_path / "spot_array_correction.asdf"
    wgs_run.save(path)
    loaded = SpotArrayCorrectionData.load(path)

    assert loaded.retriever == "WeightedGerchbergSaxtonPhaseRetriever"
    np.testing.assert_allclose(loaded.positions, wgs_run.positions)
    np.testing.assert_allclose(
        loaded.measured_intensities[-1], wgs_run.measured_intensities[-1]
    )
    np.testing.assert_allclose(
        loaded.measured_positions[-1], wgs_run.measured_positions[-1]
    )
    np.testing.assert_allclose(
        loaded.adjusted_positions[-1], wgs_run.adjusted_positions[-1]
    )
    assert loaded.non_uniformity == pytest.approx(wgs_run.non_uniformity)
    np.testing.assert_allclose(loaded.retrievals[-1].phase, wgs_run.retrievals[-1].phase)


def test_the_corrector_refuses_a_retriever_it_cannot_reweight():
    """A linear superposition does not iterate, so the corrector cannot drive it."""
    slm, camera, model = _bench()
    retriever = LinearSuperpositionPhaseRetriever(model, _positions())

    with pytest.raises(TypeError, match="LinearSuperpositionPhaseRetriever"):
        SpotArrayCorrector(slm, camera, retriever, _bench_mapping(), _positions())


def test_a_gradient_retriever_needs_the_positions_and_a_target():
    slm, camera, model = _bench()
    positions = _positions()

    with pytest.raises(ValueError, match="positions of its spots"):
        SpotArrayCorrector(
            slm, camera, _pixelwise_retriever(model, positions), _bench_mapping()
        )
    with pytest.raises(ValueError, match="has no target"):
        SpotArrayCorrector(
            slm, camera, PixelwisePhaseRetriever(model), _bench_mapping(), positions
        )


def test_a_spot_the_target_leaves_dark_is_refused():
    """The target holds 11 of the 12 spots, so the 12th has no weight to scale."""
    slm, camera, model = _bench()
    positions = _positions()
    retriever = _pixelwise_retriever(model, positions)
    retriever.set_target(_spot_target(model, positions[:-1]))

    with pytest.raises(ValueError, match=r"dark at spots \[11\]"):
        SpotArrayCorrector(slm, camera, retriever, _bench_mapping(), positions)


def test_the_corrector_refuses_positions_that_are_not_pairs():
    slm, camera, model = _bench()
    retriever = WeightedGerchbergSaxtonPhaseRetriever(model, _positions())

    with pytest.raises(ValueError, match=r"\(N, 2\) array"):
        SpotArrayCorrector(slm, camera, retriever, _bench_mapping(), np.zeros((4, 3)))


def test_the_corrector_refuses_a_mapping_of_another_camera():
    slm, camera, model = _bench()
    retriever = WeightedGerchbergSaxtonPhaseRetriever(model, _positions())
    other = SpotCamera(np.zeros((0, 2)), np.zeros(0), waist=1.0)
    mapping = replace(_bench_mapping(), camera_data=CameraData.from_camera(other))

    with pytest.raises(ValueError, match="Map the camera again"):
        SpotArrayCorrector(slm, camera, retriever, mapping)
