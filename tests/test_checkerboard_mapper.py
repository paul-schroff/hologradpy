"""Tests for the CheckerboardMapper camera-mapping pipeline.

Covers coarse-driven placement (build-internally and explicit-coarse), the
quarter-Nyquist square-size cap, zeroth-order clearance, and a camera-rotation
characterisation. The checkerboard corner correspondence assumes a near-axis-
aligned board, so the mapper is robust up to a +/-45 deg camera rotation and
aliases beyond it (the board's 90 deg orientation ambiguity flips past the 45 deg
bisector). The sweep encodes that breaking point. All setups are clean (no
aberrations/noise), which is the mapper's working regime.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from hologradpy.hardware import SimulatedSLMTorch, SimulatedCameraTorch  # noqa: E402
from hologradpy.propagation.complex_amplitude import (  # noqa: E402
    ComplexAmplitude,
    FieldGeometry,
)
from hologradpy.propagation.optical_systems import SLMFFT, SLMFFTAffine  # noqa: E402
from hologradpy.propagation.diagonal_elements import StaticSLMField  # noqa: E402
from hologradpy.propagation.virtual_slms import VirtualSLM  # noqa: E402
from hologradpy.propagation.amplitude_profiles import (  # noqa: E402
    gaussian_beam_intensity,
)
from hologradpy.propagation.zernike import Zernike  # noqa: E402
from hologradpy.calibration.camera_mapping import (  # noqa: E402
    CheckerboardMapper,
    CoarseMapper,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

DEVICE = torch.device("cpu")
NUMBER_OF_SQUARES = (5, 7)  # -> (4, 6) = 24 inner corners
CG_ITERATIONS = 40


def _aberration_phase(
    resolution: tuple[int, int],
    intensity: torch.Tensor,
    rms_radians: float,
    seed: int,
) -> torch.Tensor:
    """Random Zernike phase (astigmatism and up), normalised so its intensity-
    weighted RMS over the beam equals ``rms_radians`` (0 -> flat)."""
    if rms_radians == 0.0:
        return torch.zeros(resolution, device=DEVICE)
    zernike = Zernike(resolution, number_of_radial_orders=5, convention="ANSI")
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    coefficients = torch.randn(zernike.number_of_zernikes, generator=generator)
    coefficients[:3] = 0.0  # drop piston + tip/tilt (absorbed by the affine)
    phase = zernike.get_phase(coefficients)
    weight = intensity / intensity.sum()
    mean = (weight * phase).sum()
    variance = (weight * (phase - mean) ** 2).sum()
    return (phase - mean) * (rms_radians / variance.sqrt())


def _build_setup(
    camera_angle: float = 0.0,
    aberration_rms: float = 0.0,
    aberration_seed: int = 1,
    background_power_ratio: float = 0.0,
):
    """A small simulated SLM + camera + hologram model.

    The mapper's model always sees the ideal beam; ``aberration_rms`` (radians,
    intensity-weighted RMS) adds an *unknown* static Zernike wavefront error to
    the hardware beam only, and ``background_power_ratio`` adds static laser-
    speckle stray light of that fraction of the total beam power at the sensor.
    """
    slm_geometry = FieldGeometry(
        resolution=(256, 320),
        pixel_size=torch.tensor([12.5e-6, 12.5e-6], device=DEVICE),
        wavelength=torch.tensor(0.630e-6, device=DEVICE),
    )
    slm = SimulatedSLMTorch(input_geometry=slm_geometry, bitdepth=8)

    gaussian_intensity = gaussian_beam_intensity(
        *slm_geometry.get_spatial_grid(), beam_radius=1e-3
    )
    ideal_beam = ComplexAmplitude(
        gaussian_intensity.sqrt() + 0j,
        wavelength=slm_geometry.wavelength,
        pixel_size=slm_geometry.pixel_size,
    )
    phase = _aberration_phase(
        slm_geometry.resolution, gaussian_intensity, aberration_rms, aberration_seed
    )
    hardware_beam = ComplexAmplitude(
        gaussian_intensity.sqrt() * torch.exp(1j * phase),
        wavelength=slm_geometry.wavelength,
        pixel_size=slm_geometry.pixel_size,
    )

    simulated_camera_model = SLMFFTAffine(
        input_geometry=slm_geometry,
        virtual_slm=slm.virtual_slm,
        camera_resolution=(240, 320),
        camera_pixel_size=(30e-6, 30e-6),
        focal_length=0.25,
        static_slm_field=StaticSLMField(hardware_beam),
        padded_resolution=(512, 512),
        camera_angle=camera_angle,
        camera_shift=(0, 0),
    )
    background_power = None
    if background_power_ratio > 0.0:
        pixel_area = float(slm_geometry.pixel_size[0] * slm_geometry.pixel_size[1])
        total_beam_power = float(gaussian_intensity.sum()) * pixel_area
        background_power = background_power_ratio * total_beam_power
    camera = SimulatedCameraTorch(
        simulated_camera_model,
        background_scatter_power=background_power,
        background_scatter_grain_radius=30e-6,
        background_scatter_seed=2,
    )
    camera.set_exposure(1e-3)
    # Forward-initialize the shared slm.virtual_slm so set_phase() works later.
    camera.get_image()

    slm_camera_model = SLMFFT(
        input_geometry=slm_geometry,
        virtual_slm=VirtualSLM(phase_scaling=1.0),
        static_slm_field=StaticSLMField(ideal_beam),
        focal_length=0.25,
        padded_resolution=(512, 512),
    )
    return slm, camera, slm_camera_model


def _recovered_rotation(transform: np.ndarray) -> float:
    """Camera->model rotation of an affine, in degrees."""
    return float(np.degrees(np.arctan2(transform[1, 0], transform[0, 0])))


@pytest.fixture(scope="module")
def simulated_setup():
    return _build_setup()


@pytest.fixture(scope="module")
def mapper(simulated_setup):
    slm, camera, slm_camera_model = simulated_setup
    return CheckerboardMapper(slm, camera, slm_camera_model)


@pytest.fixture(scope="module")
def coarse_mapping(simulated_setup):
    slm, camera, slm_camera_model = simulated_setup
    return CoarseMapper(slm, camera, slm_camera_model).map_camera()


@pytest.fixture(scope="module")
def checkerboard_mapping(mapper):
    """Run the mapper once (building its coarse mapping internally)."""
    return mapper.map_camera(
        number_of_squares=NUMBER_OF_SQUARES,
        number_of_cg_iterations=CG_ITERATIONS,
    )


# --- end-to-end (build coarse internally) -------------------------------------


def test_map_camera_builds_coarse_and_recovers_affine(checkerboard_mapping):
    mapping = checkerboard_mapping
    assert mapping.name == "checkerboard"

    detected = np.asarray(mapping.detected_points, dtype=float)
    calculated = np.asarray(mapping.calculated_points, dtype=float)
    # (4, 6) inner corners detected in both the camera and simulated images.
    assert detected.shape == (24, 2)
    assert detected.shape == calculated.shape

    # No camera rotation -> the camera->model transform is (near) rotation-free.
    assert _recovered_rotation(np.asarray(mapping.transform)) == pytest.approx(
        0.0, abs=1.0
    )

    # The fitted affine maps detected (camera) corners onto the simulated-plane
    # corners within a fraction of a pixel.
    transform = np.asarray(mapping.transform, dtype=float)
    mapped = detected @ transform[:, :2].T + transform[:, 2]
    rms = np.sqrt(np.mean(np.sum((mapped - calculated) ** 2, axis=1)))
    assert rms < 2.0
    assert mapping.reprojection_rms == pytest.approx(rms)


def test_zeroth_order_clears_board_footprint(checkerboard_mapping):
    """The board is placed clear of the zeroth order: the DC pixel lies outside
    the detected board's camera footprint."""
    mapping = checkerboard_mapping
    detected = np.asarray(mapping.detected_points, dtype=float)
    board_centre = detected.mean(axis=0)
    half_diagonal = 0.5 * float(
        np.hypot(np.ptp(detected[:, 0]), np.ptp(detected[:, 1]))
    )
    # zeroth_order_position is stored (y, x) in camera pixels.
    zeroth_y, zeroth_x = mapping.zeroth_order_position
    distance = float(np.hypot(zeroth_x - board_centre[0], zeroth_y - board_centre[1]))
    assert distance > half_diagonal


# --- explicit coarse mapping --------------------------------------------------


def test_map_camera_accepts_explicit_coarse_mapping(mapper, coarse_mapping):
    """A supplied coarse mapping is used directly (no internal coarse build)."""
    mapping = mapper.map_camera(
        number_of_squares=NUMBER_OF_SQUARES,
        number_of_cg_iterations=CG_ITERATIONS,
        coarse_mapping=coarse_mapping,
    )
    assert mapping.name == "checkerboard"
    assert mapping.reprojection_rms < 2.0
    # The coarse mapping's focal-spot radius is carried through.
    assert mapping.focal_spot_radius == pytest.approx(
        abs(coarse_mapping.focal_spot_radius)
    )


# --- placement caps -----------------------------------------------------------


def test_auto_square_size_within_quarter_nyquist(mapper, coarse_mapping):
    """The auto square size keeps the board width within a quarter of the SLM's
    Nyquist-rectangle width on both axes."""
    lens = mapper.slm_camera_model.fourier_lens
    pixel_size_out = lens.pixel_size_out.tolist()[0]  # (y, x) metres
    resolution_out = tuple(lens.resolution_out)
    focal_length = float(lens.focal_length)
    camera_shape = tuple(mapper.camera.resolution)

    _, square_size, _ = mapper._place_checkerboard(
        NUMBER_OF_SQUARES,
        None,
        coarse_mapping,
        focal_length,
        pixel_size_out,
        resolution_out,
        camera_shape,
    )
    rows, columns = NUMBER_OF_SQUARES
    addressable = mapper.slm_camera_model.addressable_half_extent()  # (x, y) m

    assert square_size >= 4
    # Board width (model px) <= 1/4 Nyquist-rectangle width = 0.5 * addressable
    # / pixel_size, per axis.
    assert columns * square_size <= 0.5 * addressable[0] / pixel_size_out[1] + 1e-6
    assert rows * square_size <= 0.5 * addressable[1] / pixel_size_out[0] + 1e-6


# --- camera-rotation characterisation -----------------------------------------


@pytest.mark.parametrize("camera_angle", [15.0, 30.0, 40.0])
def test_map_camera_recovers_rotated_camera(camera_angle):
    """The mapper recovers the camera rotation up to the +/-45 deg limit (tested
    at 45 deg during development; 40 deg leaves margin)."""
    slm, camera, slm_camera_model = _build_setup(camera_angle=camera_angle)
    mapping = CheckerboardMapper(
        slm, camera, slm_camera_model
    ).map_camera(
        number_of_squares=NUMBER_OF_SQUARES,
        number_of_cg_iterations=CG_ITERATIONS,
    )
    transform = np.asarray(mapping.transform, dtype=float)
    # A +angle camera view yields the inverse rotation on the camera->model map.
    assert _recovered_rotation(transform) == pytest.approx(-camera_angle, abs=1.5)
    assert mapping.reprojection_rms < 2.0


@pytest.mark.xfail(
    reason="Beyond +/-45 deg the checkerboard's 90 deg orientation ambiguity flips "
    "the detected corner ordering, so the recovered rotation aliases to the wrong "
    "sign (documented breaking point).",
    strict=True,
)
def test_map_camera_breaks_beyond_45_degrees():
    slm, camera, slm_camera_model = _build_setup(camera_angle=60.0)
    mapping = CheckerboardMapper(
        slm, camera, slm_camera_model
    ).map_camera(
        number_of_squares=NUMBER_OF_SQUARES,
        number_of_cg_iterations=CG_ITERATIONS,
    )
    transform = np.asarray(mapping.transform, dtype=float)
    assert _recovered_rotation(transform) == pytest.approx(-60.0, abs=2.0)


# --- aberration / speckle robustness ------------------------------------------


def test_tolerates_mild_aberration_and_speckle():
    """A mild unknown wavefront error (~0.3 rad RMS, below the ~0.5-1.0 rad break)
    together with a static speckle background at half the beam power is still
    recovered: corner detection and the affine fit survive."""
    slm, camera, slm_camera_model = _build_setup(
        aberration_rms=0.3, aberration_seed=1, background_power_ratio=0.5
    )
    mapping = CheckerboardMapper(
        slm, camera, slm_camera_model
    ).map_camera(
        number_of_squares=NUMBER_OF_SQUARES,
        number_of_cg_iterations=CG_ITERATIONS,
    )
    assert np.asarray(mapping.detected_points, dtype=float).shape == (24, 2)
    assert mapping.reprojection_rms < 2.0
    assert _recovered_rotation(np.asarray(mapping.transform)) == pytest.approx(
        0.0, abs=1.5
    )


@pytest.mark.xfail(
    reason="A strong unknown wavefront error (~1.5 rad RMS) distorts the focal "
    "spots enough that corner detection collapses -- the mapper's documented "
    "sensitivity to aberration (fails by ~1 rad RMS, seed-dependent from ~0.7).",
    strict=True,
)
def test_map_camera_breaks_under_strong_aberration():
    slm, camera, slm_camera_model = _build_setup(
        aberration_rms=1.5, aberration_seed=1
    )
    mapping = CheckerboardMapper(
        slm, camera, slm_camera_model
    ).map_camera(
        number_of_squares=NUMBER_OF_SQUARES,
        number_of_cg_iterations=CG_ITERATIONS,
    )
    # Detection collapses under strong aberration; if it somehow returns, the
    # 24-corner assertion catches the degraded result.
    assert np.asarray(mapping.detected_points, dtype=float).shape == (24, 2)
    assert mapping.reprojection_rms < 2.0
