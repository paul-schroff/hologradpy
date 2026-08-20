"""Tests for saving a simulated camera and reopening it elsewhere.

A simulated camera mounts its own modules onto the model it is handed: the crosstalk
kernel, the power instability, the stray-light background and the sensor. The model's
own constructor arguments therefore stop describing the thing whose weights get saved,
which is why saving the model alone used to come back with nowhere to put
``background.background``. These pin the round trip for every combination of noise
source.
"""

from __future__ import annotations

import pytest
import torch

from hologradpy.hardware import SimulatedCameraTorch
from hologradpy.optics.complex_amplitude import ComplexAmplitude, FieldGeometry
from hologradpy.optics.modules.slm_fields import PixelwiseSLMField
from hologradpy.optics.modules.virtual_slms import VirtualSLM
from hologradpy.optics.systems import SLMCZT, SLMFFT
from hologradpy.profiles.amplitude import gaussian_beam_intensity
from hologradpy.roi import ROI

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

RESOLUTION = (32, 32)
PIXEL = 12.5e-6
WAVELENGTH = 670e-9


def _geometry() -> FieldGeometry:
    return FieldGeometry(
        resolution=RESOLUTION,
        pixel_size=torch.tensor([PIXEL, PIXEL]),
        wavelength=torch.tensor(WAVELENGTH),
    )


def _beam(geometry: FieldGeometry) -> PixelwiseSLMField:
    x, y = geometry.get_spatial_grid()
    intensity = gaussian_beam_intensity(x, y, beam_radius=1e-4)
    return PixelwiseSLMField(
        ComplexAmplitude.from_geometry(geometry, data=intensity.sqrt() + 0j)
    )


def _czt() -> SLMCZT:
    """A model that knows its output pixel size without being run."""
    geometry = _geometry()
    return SLMCZT(
        input_geometry=geometry,
        virtual_slm=VirtualSLM(phase_scaling=1.0),
        slm_field=_beam(geometry),
        focal_length=0.1,
        camera_resolution=(32, 32),
        camera_pixel_size=(3e-6, 3e-6),
        padded_resolution=(64, 64),
    )


def _fft() -> SLMFFT:
    """A model whose output pixel size only exists once it has been run."""
    geometry = _geometry()
    return SLMFFT(
        input_geometry=geometry,
        virtual_slm=VirtualSLM(phase_scaling=1.0),
        slm_field=_beam(geometry),
        focal_length=0.1,
        padded_resolution=(64, 64),
    )


NOISE_SOURCES = {
    "nothing": {},
    "sensor only": {"noise_level": 4},
    # No seed, so nothing but the saved weights can bring this speckle field back.
    "unseeded background": {"background_scatter_power": 2e-7, "noise_level": 4},
    "seeded background": {
        "background_scatter_power": 2e-7,
        "background_scatter_seed": 3,
    },
    "power instability": {"power_std": 0.05, "power_seed": 7, "noise_level": 3},
    "crosstalk": {"crosstalk_upscale_factor": 3, "noise_level": 2},
    "everything at once": {
        "crosstalk_upscale_factor": 3,
        "background_scatter_power": 1e-7,
        "power_std": 0.03,
        "noise_level": 4,
    },
}


def _saved_and_reloaded(tmp_path, model, **kwargs):
    """A camera, and the same camera written out and read back."""
    camera = SimulatedCameraTorch(slm_camera_model=model, **kwargs)
    # Mount the lazy modules so there are weights to compare.
    camera.slm_camera_model()

    path = tmp_path / "camera.pt"
    camera.save(path)
    return camera, SimulatedCameraTorch.load(path)


@pytest.mark.parametrize("name", sorted(NOISE_SOURCES))
def test_every_noise_source_survives_the_round_trip(tmp_path, name) -> None:
    """The reopened model must carry the same modules holding the same weights."""
    camera, reloaded = _saved_and_reloaded(tmp_path, _czt(), **NOISE_SOURCES[name])

    original = camera.slm_camera_model.state_dict()
    restored = reloaded.slm_camera_model.state_dict()

    assert sorted(restored) == sorted(original)
    for key, value in original.items():
        assert torch.equal(restored[key], value), key

    assert list(reloaded.slm_camera_model._order) == list(
        camera.slm_camera_model._order
    )


def test_an_unseeded_background_comes_back_rather_than_being_redrawn(tmp_path) -> None:
    """The saved speckle field must win over a fresh draw.

    Without a seed the background is random at construction, so a reopened camera that
    rebuilt it would get a different one and quietly stop matching the bench it was
    saved from.
    """
    camera, reloaded = _saved_and_reloaded(
        tmp_path, _czt(), background_scatter_power=2e-7, noise_level=4
    )

    saved = camera.slm_camera_model.state_dict()["background.background"]
    restored = reloaded.slm_camera_model.state_dict()["background.background"]

    assert torch.equal(restored, saved)
    # A second camera built the same way draws its own, which is what makes the
    # comparison above meaningful.
    other = SimulatedCameraTorch(slm_camera_model=_czt(), background_scatter_power=2e-7)
    other.slm_camera_model()
    assert not torch.equal(
        other.slm_camera_model.state_dict()["background.background"], saved
    )


def test_a_model_that_only_knows_its_geometry_once_run_round_trips(tmp_path) -> None:
    """An FFT model has no output pixel size until it has been run once."""
    model = _fft()
    model()
    camera, reloaded = _saved_and_reloaded(
        tmp_path, model, background_scatter_power=1e-7, power_std=0.02, noise_level=4
    )

    assert reloaded.get_image().shape == camera.get_image().shape
    assert reloaded.pixel_size == pytest.approx(camera.pixel_size)


def test_the_exposure_and_roi_come_back(tmp_path) -> None:
    """Both are part of how the camera was set up, so both are saved."""
    camera = SimulatedCameraTorch(slm_camera_model=_czt(), noise_level=4)
    camera.set_exposure(0.037)
    camera.set_roi(ROI(4, 6, 20, 24))
    camera.slm_camera_model()

    path = tmp_path / "camera.pt"
    camera.save(path)
    reloaded = SimulatedCameraTorch.load(path)

    assert reloaded.get_exposure() == pytest.approx(0.037)
    assert reloaded.roi == camera.roi


def test_the_sensor_and_noise_settings_come_back(tmp_path) -> None:
    """The constructor arguments describing the imperfections are stored too."""
    _, reloaded = _saved_and_reloaded(
        tmp_path, _czt(), noise_level=4, bitdepth=12, power_std=0.05, power_seed=7
    )

    assert reloaded.sensor.noise_level == pytest.approx(4)
    assert reloaded.bitdepth == 12
    assert float(reloaded.power_instability.power_std) == pytest.approx(0.05)


def test_the_crosstalk_kernel_comes_back(tmp_path) -> None:
    """Crosstalk is mounted before the SLM stage is built, so it has to be replayed."""
    camera, reloaded = _saved_and_reloaded(
        tmp_path, _czt(), crosstalk_upscale_factor=3, noise_level=2
    )

    assert reloaded.static_crosstalk_kernel is not None
    assert reloaded.static_crosstalk_kernel == pytest.approx(
        camera.static_crosstalk_kernel
    )


def test_a_keyword_overrides_the_saved_one(tmp_path) -> None:
    """Reopening with an override changes that setting and leaves the rest alone."""
    camera = SimulatedCameraTorch(slm_camera_model=_czt(), noise_level=4, bitdepth=12)
    camera.slm_camera_model()
    path = tmp_path / "camera.pt"
    camera.save(path)

    reloaded = SimulatedCameraTorch.load(path, noise_level=0.0)

    assert reloaded.sensor.noise_level == pytest.approx(0.0)
    assert reloaded.bitdepth == 12


@pytest.mark.parametrize("name", ["unseeded background", "everything at once"])
def test_a_reopened_camera_can_be_saved_again(tmp_path, name) -> None:
    """Reopening must leave the camera as saveable as the one it came from.

    The camera records how to rebuild its model before mounting anything, and a
    reopened camera has to record the same thing, or the second file would describe a
    model that already carries the mounted modules.
    """
    camera = SimulatedCameraTorch(slm_camera_model=_czt(), **NOISE_SOURCES[name])
    camera.slm_camera_model()

    first = tmp_path / "first.pt"
    camera.save(first)
    once = SimulatedCameraTorch.load(first)

    second = tmp_path / "second.pt"
    once.save(second)
    twice = SimulatedCameraTorch.load(second)

    original = camera.slm_camera_model.state_dict()
    restored = twice.slm_camera_model.state_dict()

    assert sorted(restored) == sorted(original)
    for key, value in original.items():
        assert torch.equal(restored[key], value), key


def test_a_checkpoint_from_another_class_is_refused(tmp_path) -> None:
    """A file names what wrote it, so it cannot be read back into the wrong camera."""
    camera = SimulatedCameraTorch(slm_camera_model=_czt())
    camera.slm_camera_model()
    path = tmp_path / "camera.pt"
    camera.save(path)

    checkpoint = torch.load(path, weights_only=False)
    checkpoint.class_name = "SomeOtherCamera"
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="SomeOtherCamera"):
        SimulatedCameraTorch.load(path)


def test_a_checkpoint_naming_an_unknown_model_is_refused(tmp_path) -> None:
    """The model class has to exist in this build for the camera to be rebuilt."""
    camera = SimulatedCameraTorch(slm_camera_model=_czt())
    camera.slm_camera_model()
    path = tmp_path / "camera.pt"
    camera.save(path)

    checkpoint = torch.load(path, weights_only=False)
    checkpoint.model_class_name = "SLMNoSuchThing"
    torch.save(checkpoint, path)

    with pytest.raises(KeyError, match="SLMNoSuchThing"):
        SimulatedCameraTorch.load(path)
