"""The consumers of the devices, held to the contract set by a bench.

Every routine that takes a camera or an SLM accepts a raw slmsuite driver and wraps it
with ``as_camera`` or ``as_slm``. A routine that changes the exposure or the region of
interest puts both back, asks for no exposure outside the camera's limits, and works
on the integer frames returned by a real camera. The raw devices are a minimal slmsuite
camera watching one Gaussian spot (``_RawSpotCamera``) and a minimal raw slmsuite SLM.
The bench is the simulator held to that contract (``StrictSimulatedCamera``).

The spot-seeking steps are ``expose_until_spot``, the main-order check of
``CoarseMapper``, and ``get_diffraction_spot_position`` with its ``mask`` and
``search_roi``. They run on small native cameras whose frames follow the exposure.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from slmsuite.hardware.cameras.camera import Camera as SLMSuiteCamera

from hologradpy.calibration.camera_mapping import (
    CameraMapping,
    CoarseMapper,
    FocalSpotFit,
)
from hologradpy.calibration.exposure import expose_until_spot
from hologradpy.calibration.speckle import DatasetGenerator
from hologradpy.calibration.spot_detection import (
    capture_focal_spot,
    detect_spot,
    get_diffraction_spot_position,
    has_prominent_peak,
    tilt_to_sensor_center,
)
from hologradpy.calibration.wavefront.raster_calibration import RasterCalibrator
from hologradpy.calibration.wavefront.speckle_calibration import (
    PSFSpeckleCalibrator,
)
from hologradpy.geometry import PartialAffineTransform
from hologradpy.grids import pixel_to_metres, plane_center
from hologradpy.hardware import (
    SLM,
    Camera,
    SimulatedCameraTorch,
    SimulatedSLMTorch,
    as_camera,
    as_slm,
)
from hologradpy.hardware.camera import CameraData
from hologradpy.holography.camera_feedback import SimpleFeedbackCorrector
from hologradpy.optics.complex_amplitude import ComplexAmplitude, FieldGeometry
from hologradpy.optics.modules.slm_fields import PixelwiseSLMField
from hologradpy.optics.modules.virtual_slms import VirtualSLM
from hologradpy.optics.systems import SLMCZT, SLMFFT
from hologradpy.profiles.amplitude import gaussian_beam_intensity
from hologradpy.profiles.masks import disc_mask
from hologradpy.roi import ROI
from tests.native_camera_fakes import CroppingCamera, StrictSimulatedCamera
from tests.test_device_interface import _RawSlm

pytestmark = [
    pytest.mark.filterwarnings("ignore:Autoexposure did not reach:UserWarning"),
    pytest.mark.filterwarnings("ignore:The zeroth order at:UserWarning"),
]

# --- Raw slmsuite devices ------------------------------------------------------------

# The raw camera's sensor, and the position (row, col) of the spot in its scene.
RAW_SENSOR_RESOLUTION = (48, 64)
RAW_SPOT_CENTER = (24, 32)
RAW_PIXEL_PITCH = 3.45e-6
# The Gaussian spot of the scene has a standard deviation of 1.5 pixels, so its 1/e^2
# radius, the waist, is 3 pixels.
RAW_SPOT_SIGMA_PX = 1.5
RAW_SPOT_WAIST = 2 * RAW_SPOT_SIGMA_PX * RAW_PIXEL_PITCH
# The counts per second at the spot's peak. The raw camera opens at an exposure of
# 10 ms, where the peak reads 600 of 1023.
RAW_SPOT_PEAK_RATE = 6e4
RAW_FOCAL_LENGTH = 0.1


def _raw_spot_scene() -> NDArray:
    """The counts per second of the raw camera's scene, one Gaussian spot."""
    rows, columns = np.mgrid[0 : RAW_SENSOR_RESOLUTION[0], 0 : RAW_SENSOR_RESOLUTION[1]]
    squared_distance = (rows - RAW_SPOT_CENTER[0]) ** 2 + (
        columns - RAW_SPOT_CENTER[1]
    ) ** 2
    return RAW_SPOT_PEAK_RATE * np.exp(-squared_distance / (2 * RAW_SPOT_SIGMA_PX**2))


class _RawSpotCamera(SLMSuiteCamera):
    """A minimal 10-bit slmsuite camera watching :func:`_raw_spot_scene`.

    Its counts are the scene times the exposure, floored and clipped to full scale in
    ``uint16``, so they follow the exposure as a sensor's do. A frame is the part inside
    the window of interest. It opens at an exposure of 10 ms and states the exposure
    bounds of a Zelux, capped at 1 s. The driver holds no frames, so ``flush`` captures
    nothing.
    """

    def __init__(self) -> None:
        self._scene = _raw_spot_scene()
        self._exposure_s = 10e-3
        height, width = RAW_SENSOR_RESOLUTION
        pitch_um = RAW_PIXEL_PITCH * 1e6
        super().__init__(
            resolution=(width, height),
            bitdepth=10,
            pitch_um=(pitch_um, pitch_um),
            name="raw_spot",
            exposure_bounds_s=(40e-6, 1.0),
        )

    def _get_image_hw(self, timeout_s=None):
        counts = np.floor(self._scene * self._exposure_s)
        frame = np.clip(counts, 0, 2**self.bitdepth - 1).astype(np.uint16)
        x0, width, y0, height = self.woi
        return frame[y0 : y0 + height, x0 : x0 + width]

    def flush(self, timeout_s=1):
        pass

    def _get_exposure_hw(self):
        return self._exposure_s

    def _set_exposure_hw(self, exposure_s):
        self._exposure_s = float(exposure_s)

    def set_woi(self, woi=None):
        if woi is None:
            height, width = RAW_SENSOR_RESOLUTION
            woi = (0, width, 0, height)
        self.woi = tuple(int(value) for value in woi)
        self.shape = np.shape(self.transform(np.zeros((self.woi[3], self.woi[1]))))
        return self.woi

    def close(self):
        pass


def _raw_slm() -> _RawSlm:
    """A raw 8-bit slmsuite SLM of 32 x 32 square pixels."""
    return _RawSlm(
        resolution=(32, 32), bitdepth=8, wav_um=0.633, pitch_um=(12.5, 12.5)
    )


def _mapping_centered_on(
    zeroth_order_position: tuple[float, float],
    waist: float,
    pixel_pitch: float,
    sensor_resolution: tuple[int, int],
) -> CameraMapping:
    """A mapping without rotation or scale that sends the zeroth order at
    ``zeroth_order_position``, ``(row, col)``, to the centre of an output plane of the
    sensor's size, sampled at ``pixel_pitch``, the camera's. The spot has a waist of
    ``waist`` metres.
    """
    center_x, center_y = plane_center(sensor_resolution)
    truth = PartialAffineTransform.from_components(
        shift=(
            center_x - zeroth_order_position[1],
            center_y - zeroth_order_position[0],
        )
    )
    detected = np.random.default_rng(0).uniform(-20, 20, size=(8, 2))
    return CameraMapping(
        timestamp=datetime(2026, 1, 1),
        name="synthetic",
        transform=truth.as_matrix(homogeneous=False),
        detected_points=detected.tolist(),
        calculated_points=truth.transform_points(detected).tolist(),
        zeroth_order_position=zeroth_order_position,
        spot_fit=FocalSpotFit(waist=waist),
        output_pixel_size=(pixel_pitch, pixel_pitch),
        output_resolution=sensor_resolution,
    )


def _raw_mapping() -> CameraMapping:
    return _mapping_centered_on(
        (float(RAW_SPOT_CENTER[0]), float(RAW_SPOT_CENTER[1])),
        RAW_SPOT_WAIST,
        RAW_PIXEL_PITCH,
        RAW_SENSOR_RESOLUTION,
    )


def _model_for(slm: SLM, sensor_resolution: tuple[int, int], pitch: float) -> SLMCZT:
    """A model of the native ``slm`` in front of a sensor of ``sensor_resolution``."""
    dtype = torch.get_default_dtype()
    geometry = FieldGeometry(
        resolution=tuple(slm.resolution),
        pixel_size=torch.tensor(tuple(slm.pixel_size), dtype=dtype),
        wavelength=torch.tensor(slm.wavelength, dtype=dtype),
    )
    return SLMCZT(
        input_geometry=geometry,
        virtual_slm=VirtualSLM.from_slm(slm),
        camera_resolution=sensor_resolution,
        camera_pixel_size=(pitch, pitch),
        focal_length=RAW_FOCAL_LENGTH,
        slm_field=PixelwiseSLMField(),
    )


def _raw_model(raw_slm: _RawSlm) -> SLMCZT:
    return _model_for(as_slm(raw_slm), RAW_SENSOR_RESOLUTION, RAW_PIXEL_PITCH)


def _holds_native_devices(consumer: object) -> None:
    assert isinstance(consumer.slm, SLM)
    assert isinstance(consumer.camera, Camera)


def _detect_spot(raw_slm, raw_camera, tmp_path) -> None:
    frame = as_camera(raw_camera).get_image()
    assert detect_spot(frame, RAW_SPOT_WAIST, raw_camera) == RAW_SPOT_CENTER


def _has_prominent_peak(raw_slm, raw_camera, tmp_path) -> None:
    frame = as_camera(raw_camera).get_image()
    assert has_prominent_peak(frame, raw_camera)


def _tilt_to_sensor_center(raw_slm, raw_camera, tmp_path) -> None:
    tilt = tilt_to_sensor_center(raw_camera, _raw_mapping())
    assert tilt == pytest.approx((0.0, 0.0), abs=1e-15)


def _expose_until_spot(raw_slm, raw_camera, tmp_path) -> None:
    frame = expose_until_spot(raw_camera, RAW_SPOT_WAIST)
    assert frame is not None
    assert frame.shape == RAW_SENSOR_RESOLUTION


def _capture_focal_spot(raw_slm, raw_camera, tmp_path) -> None:
    kernel = capture_focal_spot(raw_slm, raw_camera, _raw_mapping(), 0.1, 15)
    assert np.unravel_index(int(np.argmax(kernel)), kernel.shape) == (7, 7)


def _dataset_generator(raw_slm, raw_camera, tmp_path) -> None:
    generator = DatasetGenerator(
        raw_slm,
        raw_camera,
        _raw_mapping(),
        RAW_FOCAL_LENGTH,
        tmp_path / "dataset.asdf",
    )
    _holds_native_devices(generator)
    assert generator.largest_extent_on_sensor() == pytest.approx(
        tuple(2 * center * RAW_PIXEL_PITCH for center in RAW_SPOT_CENTER)
    )


def _simple_feedback_corrector(raw_slm, raw_camera, tmp_path) -> None:
    feedback = SimpleFeedbackCorrector(
        slm=raw_slm,
        camera=raw_camera,
        slm_camera_model=_raw_model(raw_slm),
        target=torch.ones(6, 6),
        target_position=(40e-6, 0.0),
        camera_mapping=_raw_mapping(),
    )
    _holds_native_devices(feedback)
    feedback._check_grids_match()
    assert tuple(feedback.target.shape) == RAW_SENSOR_RESOLUTION


def _coarse_mapper(raw_slm, raw_camera, tmp_path) -> None:
    _holds_native_devices(CoarseMapper(raw_slm, raw_camera, _raw_model(raw_slm)))


def _psf_speckle_calibrator(raw_slm, raw_camera, tmp_path) -> None:
    calibrator = PSFSpeckleCalibrator(
        slm=raw_slm,
        camera=raw_camera,
        slm_camera_model=_raw_model(raw_slm),
        dataset_path=tmp_path / "dataset.asdf",
        camera_mapping=_raw_mapping(),
        number_of_random_patterns=1,
    )
    _holds_native_devices(calibrator)
    _holds_native_devices(calibrator.dataset_generator)


def _raster_calibrator(raw_slm, raw_camera, tmp_path) -> None:
    _holds_native_devices(RasterCalibrator(raw_slm, raw_camera, RAW_FOCAL_LENGTH))


RAW_DEVICE_CONSUMERS: dict[str, Callable[..., None]] = {
    "detect_spot": _detect_spot,
    "has_prominent_peak": _has_prominent_peak,
    "tilt_to_sensor_center": _tilt_to_sensor_center,
    "expose_until_spot": _expose_until_spot,
    "capture_focal_spot": _capture_focal_spot,
    "DatasetGenerator": _dataset_generator,
    "SimpleFeedbackCorrector": _simple_feedback_corrector,
    "CoarseMapper": _coarse_mapper,
    "PSFSpeckleCalibrator": _psf_speckle_calibrator,
    "RasterCalibrator": _raster_calibrator,
}


def _mapping_of_another_camera(raw_camera) -> CameraMapping:
    """The mapping of the raw bench, recorded on a sensor of another resolution."""
    recorded = CameraData.from_camera(as_camera(raw_camera))
    return replace(
        _raw_mapping(), camera_data=replace(recorded, sensor_resolution=(24, 32))
    )


MAPPING_CONSUMERS: dict[str, Callable[..., object]] = {
    "tilt_to_sensor_center": lambda slm, camera, mapping, tmp_path: (
        tilt_to_sensor_center(camera, mapping)
    ),
    "DatasetGenerator": lambda slm, camera, mapping, tmp_path: DatasetGenerator(
        slm, camera, mapping, RAW_FOCAL_LENGTH, tmp_path / "dataset.asdf"
    ),
    "SimpleFeedbackCorrector": lambda slm, camera, mapping, tmp_path: (
        SimpleFeedbackCorrector(
            slm=slm,
            camera=camera,
            slm_camera_model=_raw_model(slm),
            target=torch.ones(6, 6),
            camera_mapping=mapping,
        )
    ),
    "PSFSpeckleCalibrator": lambda slm, camera, mapping, tmp_path: (
        PSFSpeckleCalibrator(
            slm=slm,
            camera=camera,
            slm_camera_model=_raw_model(slm),
            dataset_path=tmp_path / "dataset.asdf",
            camera_mapping=mapping,
        )
    ),
    "RasterCalibrator": lambda slm, camera, mapping, tmp_path: RasterCalibrator(
        slm, camera, RAW_FOCAL_LENGTH
    )._ensure_camera_mapping(mapping, None),
}


@pytest.mark.parametrize(
    "consumer", MAPPING_CONSUMERS.values(), ids=MAPPING_CONSUMERS.keys()
)
def test_every_consumer_refuses_a_mapping_of_another_camera(
    consumer: Callable[..., object], tmp_path
) -> None:
    """A consumer checks a supplied mapping against its camera before it places
    anything with it.
    """
    raw_camera = _RawSpotCamera()
    mapping = _mapping_of_another_camera(raw_camera)

    with pytest.raises(ValueError, match="24 x 32 sensor"):
        consumer(_raw_slm(), raw_camera, mapping, tmp_path)


@pytest.mark.parametrize(
    "consumer", RAW_DEVICE_CONSUMERS.values(), ids=RAW_DEVICE_CONSUMERS.keys()
)
def test_every_consumer_wraps_a_raw_slmsuite_device(
    consumer: Callable[..., None], tmp_path
) -> None:
    """A raw slmsuite camera or SLM goes into every consumer as it is, so the
    consumers read the native geometry provided by the adapters.
    """
    raw_camera = _RawSpotCamera()
    assert not isinstance(raw_camera, Camera)
    raw_slm = _raw_slm()

    consumer(raw_slm, raw_camera, tmp_path)


# --- Native cameras whose frames follow the exposure ---------------------------------

SPOT_SENSOR_RESOLUTION = (64, 96)
SPOT_PIXEL_PITCH = 20e-6


class _SpotCamera(CroppingCamera):
    """An 8-bit native camera whose counts are a scene of counts per second times the
    exposure, floored and clipped at full scale in ``uint16``.

    It records every requested exposure and the exposure of every rendered frame.

    Args:
        scene: A function returning the counts per second at each pixel, called for
            every frame, so the scene can follow what an SLM shows.
        exposure_bounds: The ``(min, max)`` exposure stated by the camera.
        exposure_s: The starting exposure in seconds.
        strict: Refuse an exposure outside the camera's limits with ``ValueError``. A
            camera without this flag clamps the exposure into its bounds.
        exposure_step_s: Round every exposure up to a whole number of these steps,
            or None to apply exposures as asked.
        sensor_resolution: The ``(height, width)`` of the sensor.
    """

    def __init__(
        self,
        scene: Callable[[], NDArray],
        *,
        exposure_bounds: tuple[float, float] = (1e-4, 1.0),
        exposure_s: float = 1e-3,
        strict: bool = False,
        exposure_step_s: float | None = None,
        sensor_resolution: tuple[int, int] = SPOT_SENSOR_RESOLUTION,
    ) -> None:
        super().__init__(
            sensor_resolution, exposure_bounds=exposure_bounds, exposure_s=exposure_s
        )
        self._scene = scene
        self._strict = strict
        self._exposure_step_s = exposure_step_s
        self.requested_exposures: list[float] = []
        self.frame_exposures: list[float] = []

    @property
    def pixel_size(self) -> NDArray[np.float64]:
        return np.array([SPOT_PIXEL_PITCH, SPOT_PIXEL_PITCH])

    def set_exposure(self, exposure_s: float) -> None:
        self.requested_exposures.append(float(exposure_s))
        low, high = self.exposure_search_bounds
        if self._strict and not low <= exposure_s <= high:
            raise ValueError(
                f"An exposure of {exposure_s} s is outside the search bounds "
                f"{(low, high)} s."
            )
        if self._exposure_step_s is not None:
            steps = np.ceil(exposure_s / self._exposure_step_s - 1e-9)
            exposure_s = float(steps * self._exposure_step_s)
        super().set_exposure(exposure_s)

    def render_sensor_frame(self) -> NDArray[np.uint16]:
        self.frame_exposures.append(self.get_exposure())
        counts = np.floor(self._scene() * self.get_exposure())
        return np.clip(counts, 0, self.max_pixel_value).astype(np.uint16)


def _gaussian_spot(
    center: tuple[float, float], peak_rate: float, sigma_px: float = 2.5
) -> NDArray:
    """A Gaussian spot of ``peak_rate`` counts per second at ``center``, ``(row, col)``,
    on the spot camera's sensor.
    """
    rows, columns = np.mgrid[
        0 : SPOT_SENSOR_RESOLUTION[0], 0 : SPOT_SENSOR_RESOLUTION[1]
    ]
    squared_distance = (rows - center[0]) ** 2 + (columns - center[1]) ** 2
    return peak_rate * np.exp(-squared_distance / (2 * sigma_px**2))


def _flat_disc(center: tuple[float, float], radius_px: float, rate: float) -> NDArray:
    """A disc of ``rate`` counts per second on the spot camera's sensor."""
    disc = disc_mask(SPOT_SENSOR_RESOLUTION, (center[1], center[0]), radius_px)
    return rate * disc.astype(float)


class _RecordingSlm(SLM):
    """A native SLM that shows nothing, so a spot camera's scene is its own."""

    pixel_size = np.array([12.5e-6, 12.5e-6])
    resolution = (32, 32)
    wavelength = 0.63e-6

    def set_phase(self, phase, *arguments, **options) -> None:
        pass


# The focal length that sets the size of the starting spot of the fit. For the
# aperture of the recording SLM, the spot has a 1/e^2 radius of about 5 pixels of the
# spot camera.
SPOT_FOCAL_LENGTH = 0.1


# --- expose_until_spot ---------------------------------------------------------------


def test_expose_until_spot_stays_within_the_exposure_bounds() -> None:
    """The frame is saturated at every exposure, so the exposure is cut down to the
    lower bound. The search stops there without asking for less.
    """
    camera = _SpotCamera(
        lambda: np.full(SPOT_SENSOR_RESOLUTION, 1e12), exposure_s=1e-2, strict=True
    )

    assert expose_until_spot(camera, 1e-4, max_steps=10) is None

    assert camera.get_exposure() == 1e-4
    # 10 ms, then 500 us, then the bound, which leaves nothing further to cut.
    assert camera.frame_exposures == pytest.approx([1e-2, 5e-4, 1e-4])


def test_expose_until_spot_steps_from_the_applied_exposure() -> None:
    """The camera rounds every exposure up to a whole millisecond, and each step starts
    from the applied exposure.
    """
    camera = _SpotCamera(
        lambda: np.zeros(SPOT_SENSOR_RESOLUTION),
        exposure_s=1e-3,
        exposure_step_s=1e-3,
    )

    expose_until_spot(camera, 1e-4, max_steps=4, saturation_step_fraction=0.3)

    applied = camera.frame_exposures
    assert len(camera.requested_exposures) == 3
    for exposure, requested in zip(applied, camera.requested_exposures):
        assert requested == pytest.approx(exposure / 0.3)
    assert applied[1:] == pytest.approx([4e-3, 14e-3, 47e-3])


def test_expose_until_spot_sets_only_an_exposure_it_captures_at() -> None:
    """Two dark captures take one step between them, and the camera is left at the
    exposure of the last.
    """
    camera = _SpotCamera(lambda: np.zeros(SPOT_SENSOR_RESOLUTION), exposure_s=1e-3)

    assert expose_until_spot(camera, 1e-4, max_steps=2) is None

    assert len(camera.requested_exposures) == 1
    assert camera.get_exposure() == camera.frame_exposures[-1]


# --- The main order of CoarseMapper --------------------------------------------------

FOUND_TILT = (1e-3, 5e-4)
MIRRORED_TILT = (-1e-3, -5e-4)


def _main_order_mapper(
    scene_by_tilt: dict[tuple[float, float], NDArray],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[CoarseMapper, _SpotCamera]:
    """A coarse mapper whose camera shows the scene of the tilt last displayed."""
    shown: list[tuple[float, float]] = [FOUND_TILT]
    camera = _SpotCamera(
        lambda: scene_by_tilt[shown[-1]], exposure_s=1e-2, strict=True
    )
    geometry = FieldGeometry(
        resolution=(32, 32),
        pixel_size=torch.tensor([12.5e-6, 12.5e-6]),
        wavelength=torch.tensor(0.63e-6),
    )
    model = SLMFFT(
        input_geometry=geometry,
        virtual_slm=VirtualSLM(full_scale_cycles=1.0),
        slm_field=PixelwiseSLMField(),
        focal_length=0.25,
        padded_resolution=(64, 64),
    )
    mapper = CoarseMapper(_RecordingSlm(), camera, model)
    monkeypatch.setattr(
        mapper, "_display_tilt", lambda tilt, focal_length: shown.append(tilt)
    )
    return mapper, camera


@pytest.mark.parametrize(
    ("found_scene", "mirrored_scene", "preferred"),
    [
        (
            _gaussian_spot((30, 40), 5e3),
            _gaussian_spot((30, 56), 5e5),
            MIRRORED_TILT,
        ),
        (
            _gaussian_spot((30, 40), 1e5),
            _gaussian_spot((30, 56), 1e3),
            FOUND_TILT,
        ),
        (
            _flat_disc((30, 40), 2.0, 1e12),
            _flat_disc((30, 56), 4.0, 1e12),
            FOUND_TILT,
        ),
    ],
    ids=["mirrored-overexposed", "mirrored-dim", "both-overexposed-at-the-bound"],
)
def test_prefer_main_order_restores_the_exposure_either_way(
    found_scene: NDArray,
    mirrored_scene: NDArray,
    preferred: tuple[float, float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each tilt is autoexposed, never below the lower bound. The mapper keeps the tilt
    whose spot needs the clearly shorter exposure. When both spots are overexposed
    even at the lower bound, the found tilt is kept.
    """
    mapper, camera = _main_order_mapper(
        {FOUND_TILT: found_scene, MIRRORED_TILT: mirrored_scene}, monkeypatch
    )

    tilt = mapper._prefer_main_order(FOUND_TILT, focal_length=0.25)

    assert tilt == preferred
    assert min(camera.requested_exposures, default=1e-2) >= 1e-4
    assert camera.get_exposure() == 1e-2


# --- get_diffraction_spot_position ---------------------------------------------------

ZEROTH_ORDER = (20.0, 24.0)
PROBE = (40.0, 66.0)


def _probe_and_zeroth_order(probe_rate: float = 1e5) -> NDArray:
    """A probe spot, and a zeroth order three times brighter elsewhere on the sensor."""
    return _gaussian_spot(PROBE, probe_rate) + _gaussian_spot(
        ZEROTH_ORDER, 3 * probe_rate
    )


def _position_in_pixels(position: tuple[float, float]) -> tuple[float, float]:
    """A spot position in metres from the sensor's centre, as ``(row, col)`` pixels."""
    x_metres, y_metres = position
    return (
        y_metres / SPOT_PIXEL_PITCH + SPOT_SENSOR_RESOLUTION[0] / 2,
        x_metres / SPOT_PIXEL_PITCH + SPOT_SENSOR_RESOLUTION[1] / 2,
    )


def _locate(camera: _SpotCamera, **options) -> tuple[tuple[float, float], ROI]:
    position, _, _, roi = get_diffraction_spot_position(
        _RecordingSlm(),
        camera,
        (0.0, 0.0),
        SPOT_FOCAL_LENGTH,
        verbose=False,
        **options,
    )
    return _position_in_pixels(position), roi


def _zeroth_order_disc() -> NDArray[np.bool_]:
    return disc_mask(SPOT_SENSOR_RESOLUTION, (ZEROTH_ORDER[1], ZEROTH_ORDER[0]), 10.0)


def test_diffraction_spot_ignores_a_masked_zeroth_order() -> None:
    scene = _probe_and_zeroth_order()

    unmasked, _ = _locate(_SpotCamera(lambda: scene))
    masked, _ = _locate(_SpotCamera(lambda: scene), mask=~_zeroth_order_disc())

    assert unmasked == pytest.approx(ZEROTH_ORDER, abs=1.0)
    assert masked == pytest.approx(PROBE, abs=1.0)


class _WideRecordingSlm(_RecordingSlm):
    """The recording SLM with a wider aperture, so the starting spot of the fit is
    about one pixel of the spot camera.
    """

    resolution = (256, 256)


@pytest.mark.parametrize("gap", [3.0, 6.0])
def test_a_spot_beside_the_mask_is_fitted_at_its_own_peak(gap: float) -> None:
    """The fit starts at the peak of the frame blurred over one spot radius. A small
    probe ``gap`` pixels from the edge of the masked zeroth order is then fitted where
    it is.
    """
    mask_radius = 10.0
    zeroth_order = (PROBE[0], PROBE[1] - mask_radius - gap)
    scene = _gaussian_spot(PROBE, 1e5, sigma_px=0.85) + _gaussian_spot(
        zeroth_order, 3e5, sigma_px=0.85
    )
    kept = ~disc_mask(
        SPOT_SENSOR_RESOLUTION, (zeroth_order[1], zeroth_order[0]), mask_radius
    )

    position, _, _, _ = get_diffraction_spot_position(
        _WideRecordingSlm(),
        _SpotCamera(lambda: scene),
        (0.0, 0.0),
        SPOT_FOCAL_LENGTH,
        verbose=False,
        mask=kept,
    )

    assert _position_in_pixels(position) == pytest.approx(PROBE, abs=1.0)


def test_diffraction_spot_is_searched_inside_the_search_region() -> None:
    """A search region around the probe finds the probe and not the brighter zeroth
    order. The part of the region off the sensor is dropped. The fitted region is
    given in whole-sensor pixels, inside the search region.
    """
    search = ROI(24, 48, 60, 60)
    camera = _SpotCamera(_probe_and_zeroth_order)

    position, roi = _locate(camera, search_roi=search)

    assert position == pytest.approx(PROBE, abs=1.0)
    top, bottom, left, right = roi.to_bounds()
    assert top >= 24 and left >= 48
    assert bottom <= SPOT_SENSOR_RESOLUTION[0] and right <= SPOT_SENSOR_RESOLUTION[1]


def test_diffraction_spot_takes_a_mask_and_a_search_region_together() -> None:
    """The search region holds both spots. The mask keeps the zeroth order out of the
    exposure and the fit, so the probe is found at its full exposure.
    """
    search = ROI(10, 10, 44, 70)
    camera = _SpotCamera(_probe_and_zeroth_order)

    position, _, cropped, roi = get_diffraction_spot_position(
        _RecordingSlm(),
        camera,
        (0.0, 0.0),
        SPOT_FOCAL_LENGTH,
        verbose=False,
        mask=~_zeroth_order_disc(),
        search_roi=search,
    )

    assert _position_in_pixels(position) == pytest.approx(PROBE, abs=1.0)
    # The crop is as captured, and the zeroth order inside it is left out of the fit.
    fitted_pixels = cropped[roi.crop(~_zeroth_order_disc())]
    assert float(np.max(fitted_pixels)) == pytest.approx(0.8 * 255, abs=0.05 * 255)
    top, bottom, left, right = roi.to_bounds()
    assert top >= 10 and left >= 10 and bottom <= 54 and right <= 80


def test_a_search_region_off_the_sensor_is_refused() -> None:
    camera = _SpotCamera(_probe_and_zeroth_order)
    with pytest.raises(ValueError, match="No part of"):
        _locate(camera, search_roi=ROI(70, 100, 10, 10))


def test_diffraction_spot_without_a_spot_raises_after_autoexposure() -> None:
    """A spot too dim to stand out even at the longest exposure is no spot once the
    camera has autoexposed. At an exposure fixed by the caller, the spot is fitted as
    it is.
    """
    scene = _gaussian_spot(PROBE, 20.0)

    with pytest.raises(RuntimeError, match="No spot"):
        _locate(_SpotCamera(lambda: scene))

    position, _ = _locate(_SpotCamera(lambda: scene), exposure_time=1.0)
    assert position == pytest.approx(PROBE, abs=1.0)


def test_autoexposure_converges_when_the_overexposure_cut_passes_the_lower_bound() -> (
    None
):
    """The spot reaches 0.8 of full scale at 100 us. The first cut from an overexposed
    1 ms asks for 10 us, below the 40 us minimum. The search measures the minimum and
    converges from there, without asking for less.
    """
    camera = _SpotCamera(
        lambda: _gaussian_spot(PROBE, 0.8 * 255 / 1e-4),
        exposure_bounds=(4e-5, 1.0),
        exposure_s=1e-3,
        strict=True,
    )

    position, _, cropped, _ = get_diffraction_spot_position(
        _RecordingSlm(), camera, (0.0, 0.0), SPOT_FOCAL_LENGTH, verbose=False
    )

    assert float(np.max(cropped)) == pytest.approx(0.8 * 255, abs=0.05 * 255)
    assert min(camera.requested_exposures) >= 4e-5
    assert _position_in_pixels(position) == pytest.approx(PROBE, abs=1.0)
    assert camera.get_exposure() == 1e-3


# --- The simulated bench held to the camera contract ---------------------------------

BENCH_SLM_RESOLUTION = (256, 320)
BENCH_SENSOR_RESOLUTION = (240, 320)
BENCH_PIXEL_PITCH = 30e-6
BENCH_FOCAL_LENGTH = 0.25
# The exposures accepted by the bench's camera. A probe of the coarse mapping needs
# about 4 ns, and a request outside these raises.
BENCH_EXPOSURE_BOUNDS = (2e-9, 1.0)
# A region and an exposure left on the camera before a routine runs.
PRESET_REGION = ROI(30, 40, 100, 120)
PRESET_EXPOSURE = 2e-3
# A tilt that puts a probe spot on the aligned bench's sensor, away from the zeroth
# order.
BENCH_PROBE_TILT = (1e-3, 5e-4)


class _Bench:
    """The coarse mapping bench of ``tests/test_coarse_mapper.py``, with a camera held
    to the camera contract.

    Args:
        camera_type: The simulated camera, :class:`StrictSimulatedCamera` by default.
        camera_angle: The rotation of the sensor in degrees.
        camera_shift: The shift of the sensor in pixels.
        sensor_resolution: The ``(height, width)`` of the sensor.
        **camera_options: Passed on to the camera.
    """

    def __init__(
        self,
        camera_type: type[SimulatedCameraTorch] = StrictSimulatedCamera,
        camera_angle: float = 0.0,
        camera_shift: tuple[float, float] = (0.0, 0.0),
        sensor_resolution: tuple[int, int] = BENCH_SENSOR_RESOLUTION,
        **camera_options,
    ) -> None:
        torch.manual_seed(0)
        self.geometry = FieldGeometry(
            resolution=BENCH_SLM_RESOLUTION,
            pixel_size=torch.tensor([12.5e-6, 12.5e-6]),
            wavelength=torch.tensor(0.630e-6),
        )
        self.slm = SimulatedSLMTorch(input_geometry=self.geometry, bitdepth=8)
        intensity = gaussian_beam_intensity(
            *self.geometry.get_spatial_grid(), beam_radius=1e-3
        )
        self.beam = ComplexAmplitude(
            intensity.sqrt() + 0j,
            wavelength=self.geometry.wavelength,
            pixel_size=self.geometry.pixel_size,
        )
        hardware = SLMCZT(
            input_geometry=self.geometry,
            virtual_slm=self.slm.virtual_slm,
            camera_resolution=sensor_resolution,
            camera_pixel_size=(BENCH_PIXEL_PITCH, BENCH_PIXEL_PITCH),
            focal_length=BENCH_FOCAL_LENGTH,
            slm_field=PixelwiseSLMField(self.beam),
            camera_angle=camera_angle,
            camera_shift=tuple(shift * BENCH_PIXEL_PITCH for shift in camera_shift),
        )
        self.camera = camera_type(
            hardware, exposure_bounds=BENCH_EXPOSURE_BOUNDS, **camera_options
        )
        self.camera.set_exposure(1e-3)
        self.camera.get_image()

    def mapping_model(self) -> SLMFFT:
        """The ideal reference model for a coarse mapping of the camera."""
        return SLMFFT(
            input_geometry=self.geometry,
            virtual_slm=VirtualSLM(full_scale_cycles=1.0),
            slm_field=PixelwiseSLMField(self.beam),
            focal_length=BENCH_FOCAL_LENGTH,
            padded_resolution=(512, 512),
        )

    def sensor_model(self) -> SLMCZT:
        """The model of the aligned bench on the camera's grid, for feedback."""
        return SLMCZT(
            input_geometry=self.geometry,
            virtual_slm=VirtualSLM.from_slm(self.slm),
            camera_resolution=BENCH_SENSOR_RESOLUTION,
            camera_pixel_size=(BENCH_PIXEL_PITCH, BENCH_PIXEL_PITCH),
            focal_length=BENCH_FOCAL_LENGTH,
            slm_field=PixelwiseSLMField(self.beam),
        )


def _aligned_bench_mapping() -> CameraMapping:
    """The identity mapping of the aligned bench, the zeroth order in the middle."""
    return _mapping_centered_on(
        (BENCH_SENSOR_RESOLUTION[0] / 2, BENCH_SENSOR_RESOLUTION[1] / 2),
        2 * BENCH_PIXEL_PITCH,
        BENCH_PIXEL_PITCH,
        BENCH_SENSOR_RESOLUTION,
    )


def _locate_the_probe(bench: _Bench, tmp_path) -> None:
    get_diffraction_spot_position(
        bench.slm, bench.camera, BENCH_PROBE_TILT, BENCH_FOCAL_LENGTH, verbose=False
    )


def _capture_the_focal_spot(bench: _Bench, tmp_path) -> None:
    capture_focal_spot(
        bench.slm, bench.camera, _aligned_bench_mapping(), BENCH_FOCAL_LENGTH, 15
    )


def _map_the_camera(bench: _Bench, tmp_path) -> None:
    CoarseMapper(bench.slm, bench.camera, bench.mapping_model()).map_camera()


def _capture_speckle(bench: _Bench, tmp_path) -> None:
    DatasetGenerator(
        bench.slm,
        bench.camera,
        _aligned_bench_mapping(),
        BENCH_FOCAL_LENGTH,
        tmp_path / "dataset.asdf",
        number_of_random_patterns=2,
    ).generate_dataset((1.5e-3, 1.5e-3), seed=0)


def _run_camera_feedback(bench: _Bench, tmp_path) -> None:
    SimpleFeedbackCorrector(
        slm=bench.slm,
        camera=bench.camera,
        slm_camera_model=bench.sensor_model(),
        target=torch.ones(8, 8),
        target_position=(1.5e-3, 0.0),
        camera_mapping=_aligned_bench_mapping(),
    ).run(retriever_iterations=[3], averages=1, verbose=False)


ENTRY_POINTS: dict[str, Callable[[_Bench, object], None]] = {
    "get_diffraction_spot_position": _locate_the_probe,
    "capture_focal_spot": _capture_the_focal_spot,
    "CoarseMapper.map_camera": _map_the_camera,
    "DatasetGenerator.generate_dataset": _capture_speckle,
    "SimpleFeedbackCorrector.run": _run_camera_feedback,
}


@pytest.mark.parametrize(
    "entry_point", ENTRY_POINTS.values(), ids=ENTRY_POINTS.keys()
)
def test_consumers_leave_the_region_and_exposure_as_found(
    entry_point: Callable[[_Bench, object], None], tmp_path
) -> None:
    """The camera refuses an exposure outside its limits. Each routine reads out the
    whole sensor while it runs, and then puts back the initial region and exposure.
    """
    bench = _Bench()
    bench.camera.set_roi(PRESET_REGION)
    bench.camera.set_exposure(PRESET_EXPOSURE)

    entry_point(bench, tmp_path)

    assert bench.camera.roi == PRESET_REGION
    assert bench.camera.get_exposure() == PRESET_EXPOSURE


def test_get_diffraction_spot_position_leaves_the_camera_as_it_found_it() -> None:
    """The probe is located on the whole sensor for any window held by the camera. It
    is therefore found at the same position as without a window. The window and the
    exposure are put back.
    """
    bench = _Bench(add_noise=False)
    bench.camera.set_exposure(PRESET_EXPOSURE)
    expected = get_diffraction_spot_position(
        bench.slm, bench.camera, BENCH_PROBE_TILT, BENCH_FOCAL_LENGTH, verbose=False
    )[0]
    bench.camera.set_roi(PRESET_REGION)

    position = get_diffraction_spot_position(
        bench.slm, bench.camera, BENCH_PROBE_TILT, BENCH_FOCAL_LENGTH, verbose=False
    )[0]

    assert position == pytest.approx(expected, rel=1e-9)
    assert bench.camera.roi == PRESET_REGION
    assert bench.camera.get_exposure() == PRESET_EXPOSURE


def test_exposure_requests_stay_inside_the_cameras_bounds() -> None:
    """A mapping of a sensor away from the zeroth order cuts the exposure to find the
    spots and the main order. It never asks for an exposure refused by the camera. A
    probe found on the sensor by the mapping is then located the same way.
    """
    bench = _Bench(
        camera_angle=10.0, camera_shift=(60, 100), sensor_resolution=(120, 160)
    )
    model = bench.mapping_model()

    mapping = CoarseMapper(bench.slm, bench.camera, model).map_camera()
    # The middle of the sensor in the model's plane, (x, y), and the tilt that puts a
    # spot there.
    middle = mapping.affine.transform_points([(80.0, 60.0)])[0]
    tilt = pixel_to_metres(
        (float(middle[0]), float(middle[1])),
        model[-1].pixel_size_out.tolist()[0],
        tuple(model[-1].resolution_out),
    )
    column, row = get_diffraction_spot_position(
        bench.slm,
        bench.camera,
        tilt,
        BENCH_FOCAL_LENGTH,
        units="pixels",
        verbose=False,
    )[0]

    assert mapping.rotation_degrees == pytest.approx(-10.0, abs=0.5)
    assert (column, row) == pytest.approx((80, 60), abs=2)


def test_the_focal_spot_is_captured_with_a_region_preset() -> None:
    """The zeroth order of the aligned bench sits in the middle of the sensor, outside
    the region left on the camera. The kernel still has the zeroth order in its middle.
    """
    bench = _Bench()
    bench.camera.set_roi(ROI(0, 0, 60, 80))

    kernel = capture_focal_spot(
        bench.slm, bench.camera, _aligned_bench_mapping(), BENCH_FOCAL_LENGTH, 15
    )

    assert np.unravel_index(int(np.argmax(kernel)), kernel.shape) == (7, 7)
    assert bench.camera.roi == ROI(0, 0, 60, 80)


def test_integer_frames_flow_through_the_consumers() -> None:
    """The unsigned integer frames of the contract give the same positions and kernels
    as the float frames of the simulator, so nothing wraps below zero on the way.
    """
    results = []
    for camera_type in (SimulatedCameraTorch, StrictSimulatedCamera):
        bench = _Bench(camera_type=camera_type)
        position, radius, _, _ = get_diffraction_spot_position(
            bench.slm,
            bench.camera,
            BENCH_PROBE_TILT,
            BENCH_FOCAL_LENGTH,
            verbose=False,
        )
        kernel = capture_focal_spot(
            bench.slm, bench.camera, _aligned_bench_mapping(), BENCH_FOCAL_LENGTH, 15
        )
        results.append((position, radius, kernel))

    (float_position, float_radius, float_kernel), (position, radius, kernel) = results
    assert bench.camera.get_image().dtype == np.uint16
    assert position == pytest.approx(float_position, rel=1e-9)
    assert radius == pytest.approx(float_radius, rel=1e-9)
    np.testing.assert_array_equal(kernel, float_kernel)
