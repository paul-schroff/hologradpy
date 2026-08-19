"""Contract tests for OpticalSystem implementations (SLM / Fourier-lens models).

An OpticalSystem is a named, ordered chain of OpticsModules. Called with no
input it runs from its built-in uniform field and returns a ComplexAmplitude in
the final plane. These tests cover the chain contract (forward, output
geometry, named/ordered layer access) and the checkpoint **save/load**
round-trip via ``get_checkpoint_spec`` / ``from_checkpoint_spec``.
"""
from __future__ import annotations

import pytest
import torch

from hologradpy.optics.complex_amplitude import (
    ComplexAmplitude,
    FieldGeometry,
)
from hologradpy.optics.systems import (
    SLMFFT,
    SLMFFTAffine,
    SLMNUFFTAffine,
    SLMCZT,
)
from hologradpy.optics.modules.slm_fields import PixelwiseSLMField
from hologradpy.optics.modules.virtual_slms.abstract import VirtualSLM


pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

SLM_RESOLUTION = (32, 32)
PADDED_RESOLUTION = (64, 64)
CAMERA_RESOLUTION = (24, 24)
CAMERA_PIXEL_SIZE = (20e-6, 20e-6)


def _input_geometry() -> FieldGeometry:
    return FieldGeometry(
        wavelength=torch.tensor([800e-9]),
        pixel_size=torch.tensor([[10e-6, 10e-6]]),
        resolution=SLM_RESOLUTION,
    )


def _constant_field() -> ComplexAmplitude:
    geometry = _input_geometry()
    data = torch.ones(SLM_RESOLUTION, dtype=torch.complex64)
    return ComplexAmplitude(data, geometry.wavelength, geometry.pixel_size)


def _slm_field() -> PixelwiseSLMField:
    return PixelwiseSLMField(_constant_field())


def _make_slm_fft(pointing_focal_shift_std=None) ->SLMFFT:
    return SLMFFT(
        input_geometry=_input_geometry(),
        virtual_slm=VirtualSLM(phase_scaling=1.0),
        slm_field=_slm_field(),
        focal_length=0.1,
        padded_resolution=PADDED_RESOLUTION,
        pointing_focal_shift_std=pointing_focal_shift_std,
    )


def _make_slm_fft_affine(pointing_focal_shift_std=None) ->SLMFFTAffine:
    return SLMFFTAffine(
        input_geometry=_input_geometry(),
        virtual_slm=VirtualSLM(phase_scaling=1.0),
        camera_resolution=CAMERA_RESOLUTION,
        camera_pixel_size=CAMERA_PIXEL_SIZE,
        focal_length=0.1,
        slm_field=_slm_field(),
        padded_resolution=PADDED_RESOLUTION,
        pointing_focal_shift_std=pointing_focal_shift_std,
    )


def _make_slm_nufft_affine(pointing_focal_shift_std=None) ->SLMNUFFTAffine:
    return SLMNUFFTAffine(
        input_geometry=_input_geometry(),
        virtual_slm=VirtualSLM(phase_scaling=1.0),
        camera_resolution=CAMERA_RESOLUTION,
        camera_pixel_size=CAMERA_PIXEL_SIZE,
        focal_length=0.1,
        slm_field=_slm_field(),
        camera_angle=5.0,
        camera_shift=(1.0, 2.0),
        pointing_focal_shift_std=pointing_focal_shift_std,
    )


def _make_slm_czt(pointing_focal_shift_std=None) ->SLMCZT:
    return SLMCZT(
        input_geometry=_input_geometry(),
        virtual_slm=VirtualSLM(phase_scaling=1.0),
        camera_resolution=CAMERA_RESOLUTION,
        camera_pixel_size=CAMERA_PIXEL_SIZE,
        focal_length=0.1,
        slm_field=_slm_field(),
        camera_angle=5.0,
        camera_shift=(1.0, 2.0),
        pointing_focal_shift_std=pointing_focal_shift_std,
    )


SYSTEM_FACTORIES = {
    "SLMFFT": _make_slm_fft,
    "SLMFFTAffine": _make_slm_fft_affine,
    "SLMNUFFTAffine": _make_slm_nufft_affine,
    "SLMCZT": _make_slm_czt,
}
SYSTEM_IDS = list(SYSTEM_FACTORIES)


@pytest.mark.parametrize("name", SYSTEM_IDS)
def test_forward_returns_complex_amplitude(name: str) -> None:
    output = SYSTEM_FACTORIES[name]()()
    assert isinstance(output, ComplexAmplitude)


@pytest.mark.parametrize("name", SYSTEM_IDS)
def test_output_matches_final_module_geometry(name: str) -> None:
    """The system's output plane is the output plane of its last module."""
    model = SYSTEM_FACTORIES[name]()
    output = model()
    final_module = model[-1]

    assert output.resolution == tuple(final_module.resolution_out)
    torch.testing.assert_close(
        output.pixel_size.reshape(-1)[:2],
        final_module.pixel_size_out.reshape(-1)[:2],
    )


@pytest.mark.parametrize("name", SYSTEM_IDS)
def test_layers_named_and_ordered(name: str) -> None:
    model = SYSTEM_FACTORIES[name]()
    layers = model.layers()

    assert len(model) == len(layers)
    # The SLM plane runs beam, then grid, then displayed phase. The beam and the phase
    # are both diagonal multiplies and commute, and this order lets the SLM stage work
    # on a grid finer than the one the beam is measured on.
    assert list(layers)[:3] == ["slm_field", "grid_adapter", "virtual_slm"]
    assert model[0] is layers["slm_field"]
    assert model["virtual_slm"] is layers["virtual_slm"]
    assert isinstance(model.get(VirtualSLM), VirtualSLM)


@pytest.mark.parametrize("name", SYSTEM_IDS)
def test_forward_repeatable(name: str) -> None:
    model = SYSTEM_FACTORIES[name]()
    torch.testing.assert_close(model()._data, model()._data)


def test_device_and_init_field_track_the_system() -> None:
    model = _make_slm_czt()
    assert isinstance(model.device, torch.device)
    # The no-input field lives on the system's device and is cached.
    assert model.init_field.device == model.device
    assert model.init_field is model.init_field
    assert isinstance(model(), ComplexAmplitude)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_to_cuda_moves_device_and_init_field() -> None:
    model = _make_slm_czt().to("cuda")
    assert model.device.type == "cuda"
    assert model.init_field.device.type == "cuda"
    assert model().device.type == "cuda"  # no-input forward on the moved model


def test_layer_name_collisions_are_rejected_clearly() -> None:
    from hologradpy.optics.modules.hardware_models import (
        PointingInstability,
    )

    model = _make_slm_czt()
    # A name already used by a layer.
    with pytest.raises(ValueError, match="already exists"):
        model.add("virtual_slm", PointingInstability(1e-4))
    # A name that would shadow a reserved nn.Module attribute.
    with pytest.raises(ValueError, match="reserved attribute"):
        model.add("forward", PointingInstability(1e-4))


def test_insert_after_places_module_in_chain() -> None:
    from hologradpy.optics.modules.hardware_models import (
        PointingInstability,
    )

    model = _make_slm_czt()
    model.insert_after(PixelwiseSLMField, "jitter", PointingInstability(1e-4))
    order = list(model.layers())
    assert order[order.index("slm_field") + 1] == "jitter"
    assert model["jitter"] is model.get(PointingInstability)
    assert isinstance(model(), ComplexAmplitude)  # forward still runs


@pytest.mark.parametrize("name", SYSTEM_IDS)
def test_pointing_instability_inserted_after_static_field(name: str) -> None:
    """Each SLM model builds a PointingInstability from pointing_focal_shift_std
    (using its own focal_length) and inserts it right after its PixelwiseSLMField
    stage (whatever that layer happens to be named).
    """
    from hologradpy.optics.modules.hardware_models import (
        PointingInstability,
    )

    # Factories all use focal_length=0.1, so tilt_std = focal_shift_std / 0.1.
    model = SYSTEM_FACTORIES[name](pointing_focal_shift_std=2e-6)

    assert model.has(PointingInstability)
    order = list(model.layers())
    static_name = next(n for n in order if isinstance(model[n], PixelwiseSLMField))
    assert order[order.index(static_name) + 1] == "pointing_instability"
    assert model.get(PointingInstability).tilt_std == (2e-6 / 0.1, 2e-6 / 0.1)
    assert isinstance(model(), ComplexAmplitude)  # forward still runs


@pytest.mark.parametrize("name", SYSTEM_IDS)
def test_checkpoint_spec_reconstructs_system(name: str) -> None:
    """get_checkpoint_spec -> from_checkpoint_spec yields an equivalent system
    (in-memory, no file).
    """
    model = SYSTEM_FACTORIES[name]()
    expected = model()._data.clone()

    rebuilt = type(model).from_checkpoint_spec(model.get_checkpoint_spec())

    torch.testing.assert_close(rebuilt()._data, expected)


@pytest.mark.parametrize("name", SYSTEM_IDS)
def test_checkpoint_save_load_round_trip(name: str, tmp_path) -> None:
    """save() then load() reproduces the same output and parameters."""
    model = SYSTEM_FACTORIES[name]()
    expected = model()._data.clone()

    path = str(tmp_path / f"{name}.pt")
    model.save(path)
    restored = type(model).load(path)

    torch.testing.assert_close(restored()._data, expected)

    original_params = dict(model.named_parameters())
    restored_params = dict(restored.named_parameters())
    assert original_params.keys() == restored_params.keys()
    for key in original_params:
        torch.testing.assert_close(restored_params[key], original_params[key])


@pytest.mark.parametrize("name", SYSTEM_IDS)
def test_load_rejects_wrong_class(name: str, tmp_path) -> None:
    """A checkpoint saved from one system can't be loaded by another."""
    model = SYSTEM_FACTORIES[name]()
    path = str(tmp_path / f"{name}.pt")
    model.save(path)

    other = SLMFFTAffine if name == "SLMFFT" else SLMFFT
    with pytest.raises(ValueError, match="saved from"):
        other.load(path)


def test_checkpoint_preserves_pointing_instability(tmp_path) -> None:
    """A model built with pointing (and a seed) round-trips through save/load: the
    PointingInstability and its seed survive, and params match.
    """
    from hologradpy.optics.modules.hardware_models import (
        PointingInstability,
    )

    model = SLMCZT(
        input_geometry=_input_geometry(),
        virtual_slm=VirtualSLM(phase_scaling=1.0),
        camera_resolution=CAMERA_RESOLUTION,
        camera_pixel_size=CAMERA_PIXEL_SIZE,
        focal_length=0.1,
        slm_field=_slm_field(),
        camera_angle=5.0,
        camera_shift=(1.0, 2.0),
        pointing_focal_shift_std=2e-6,
        pointing_seed=7,
    )
    assert model.has(PointingInstability)
    _ = model()  # lazily create parameters
    expected_params = {k: v.clone() for k, v in model.named_parameters()}

    path = str(tmp_path / "czt_pointing.pt")
    model.save(path)
    restored = SLMCZT.load(path)

    assert restored.has(PointingInstability)
    # The seed survived (a plain int in the spec).
    assert restored.get(PointingInstability).seed == 7
    restored_params = dict(restored.named_parameters())
    assert restored_params.keys() == expected_params.keys()
    for key in expected_params:
        torch.testing.assert_close(restored_params[key], expected_params[key])
