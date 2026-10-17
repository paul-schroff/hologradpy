"""Tests for the PSF-parameterized SLM-plane field and its trainer.

The camera field for a phase pattern is ``FT(A * exp(i * phase))``, which by the
convolution theorem is ``FT(A)`` convolved with ``FT(exp(i * phase))``, so the
camera-plane point spread function ``FT(A)`` fixes the SLM-plane beam ``A``.
:class:`PSFSLMField` carries that PSF as a compact kernel and maps it to the SLM
grid, and :class:`WavefrontFitter` fits it against captured speckle.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hologradpy.optics.complex_amplitude import (
    ComplexAmplitude,
    FieldGeometry,
)
from hologradpy.grids import odd_sample_count
from hologradpy.optics.modules.slm_fields import PSFSLMField
from hologradpy.calibration.wavefront.speckle_calibration import (
    PSFSpeckleCalibrator,
    PixelwiseSpeckleCalibrator,
)
from hologradpy.loss_functions import SumOfLosses
from hologradpy.calibration.wavefront.abstract import (
    WavefrontCalibrationData,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

SLM_RESOLUTION = (256, 256)
SLM_PIXEL_SIZE = 12.5e-6
WAVELENGTH = 630e-9
FOCAL_LENGTH = 0.25
# The kernel's pitch, the intensity Nyquist pitch of the square SLM.
KERNEL_PITCH = WAVELENGTH * FOCAL_LENGTH / (2 * SLM_RESOLUTION[1] * SLM_PIXEL_SIZE)


def _extent_of(samples: int) -> float:
    """The image-plane extent of a kernel with an odd number of ``samples`` per axis."""
    return (samples - 0.5) * KERNEL_PITCH


def _uniform_field(resolution=SLM_RESOLUTION) -> ComplexAmplitude:
    geometry = FieldGeometry(
        resolution=resolution,
        pixel_size=torch.tensor([SLM_PIXEL_SIZE, SLM_PIXEL_SIZE]),
        wavelength=torch.tensor(WAVELENGTH),
    )
    return ComplexAmplitude(
        torch.ones(resolution, dtype=torch.complex64),
        wavelength=geometry.wavelength,
        pixel_size=geometry.pixel_size,
    )


@pytest.mark.parametrize("psf_waist", [40e-6, 60e-6, 90e-6])
def test_gaussian_psf_maps_to_the_conjugate_gaussian_beam(psf_waist) -> None:
    """A Gaussian PSF of waist w must give an SLM beam of waist lambda*F/(pi*w).

    This is the Fourier pair the whole parameterization rests on, so it pins the
    kernel to SLM magnification. A wrong magnification would still look like a
    plausible beam, just the wrong size, and nothing else in the fit would catch
    it.
    """
    module = PSFSLMField(
        focal_length=FOCAL_LENGTH,
        psf_extent=_extent_of(39),
        psf_gaussian_waist=psf_waist,
    )
    amplitude = (
        module(_uniform_field()).as_tensor().detach().abs().squeeze().to(torch.float64)
    )

    # Waist from the intensity second moment: for amplitude exp(-r^2 / w^2) the
    # per-axis variance is w^2 / 4, so w = 2 * sqrt(variance).
    axis_y = (torch.arange(SLM_RESOLUTION[0], dtype=torch.float64)
              - (SLM_RESOLUTION[0] - 1) / 2) * SLM_PIXEL_SIZE
    axis_x = (torch.arange(SLM_RESOLUTION[1], dtype=torch.float64)
              - (SLM_RESOLUTION[1] - 1) / 2) * SLM_PIXEL_SIZE
    grid_y, grid_x = torch.meshgrid(axis_y, axis_x, indexing="ij")
    intensity = amplitude**2
    intensity = intensity / intensity.sum()
    variance = float((intensity * grid_x**2).sum())
    measured_waist = 2 * math.sqrt(variance)

    predicted_waist = WAVELENGTH * FOCAL_LENGTH / (math.pi * psf_waist)
    assert measured_waist == pytest.approx(predicted_waist, rel=0.06)


def test_the_kernel_is_a_single_complex_parameter() -> None:
    """One complex parameter, not a pair of real ones.

    Real and imaginary parts were carried separately at first, to stay clear of
    complex autograd after a dropped conjugation was found in the
    ComplexAmplitude dispatch. Adam views a complex parameter as two independent
    reals, so the two are the same optimization to the bit, and one parameter is
    the simpler interface.
    """
    module = PSFSLMField(focal_length=FOCAL_LENGTH, psf_extent=_extent_of(15))
    module(_uniform_field())

    assert module.psf_kernel.is_complex()
    assert module.get_psf_kernel().shape == (15, 15)
    assert sum(1 for _ in module.parameters()) == 1


def test_kernel_gradients_reach_the_whole_slm_field() -> None:
    """Every kernel pixel must move the whole field, which is why it converges
    at a different step size from a per-pixel field.
    """
    module = PSFSLMField(focal_length=FOCAL_LENGTH, psf_extent=_extent_of(15))
    module(_uniform_field())
    module.psf_kernel.requires_grad_(True)

    module.get_wavefront().abs().sum().backward()
    assert module.psf_kernel.grad is not None
    assert torch.isfinite(module.psf_kernel.grad).all()
    assert float(module.psf_kernel.grad.abs().sum()) > 0.0


def test_multiple_wavelengths_are_refused_clearly() -> None:
    """The kernel to SLM magnification scales with wavelength, so one kernel
    cannot serve several at once.
    """
    geometry = FieldGeometry(
        resolution=(32, 32),
        pixel_size=torch.tensor([SLM_PIXEL_SIZE, SLM_PIXEL_SIZE]),
        wavelength=torch.tensor([630e-9, 520e-9]),
    )
    field = ComplexAmplitude(
        torch.ones((2, 32, 32), dtype=torch.complex64),
        wavelength=geometry.wavelength,
        pixel_size=geometry.pixel_size,
    )
    module = PSFSLMField(focal_length=FOCAL_LENGTH, psf_extent=_extent_of(15))
    with pytest.raises(NotImplementedError, match="single wavelength"):
        module(field)


def test_the_sample_count_tracks_the_extent() -> None:
    """A wider extent takes more samples, always an odd number of at least 3, so the
    grid has a centre sample.
    """
    small = odd_sample_count(10 * 5e-6, 3.45e-6)
    large = odd_sample_count(10 * 15e-6, 3.45e-6)
    assert large > small
    assert small % 2 == 1 and large % 2 == 1
    assert odd_sample_count(1e-9, 3.45e-6) >= 3


def test_a_static_field_model_is_not_treated_as_a_psf_one(tmp_path) -> None:
    """A PixelwiseSLMField model must not pick up the PSF parameterization's
    settings.
    """
    from .test_speckle_calibrator import (
        FOCAL_LENGTH as SMALL_FOCAL_LENGTH,
        _build_hardware,
        _build_model,
        _synthetic_mapping,
    )

    slm, camera = _build_hardware()
    calibrator = PixelwiseSpeckleCalibrator(
        slm=slm,
        camera=camera,
        camera_mapping=_synthetic_mapping(),
        slm_camera_model=_build_model(
            slm, camera, SMALL_FOCAL_LENGTH,
        ),
        dataset_path=tmp_path / "dataset.asdf",
        number_of_random_patterns=3,
    )

    field = calibrator.slm_camera_model.slm_field
    # Nothing seeded it, since only a PSF kernel is measured from the camera, and there
    # is no kernel to reach for.
    assert not isinstance(field, PSFSLMField)
    assert not hasattr(field, "get_psf_kernel")

    # It gets the unconstrained field's Adam fit, prior included, not the PSF's.
    settings: dict = {}

    def adam(fitter, loss, learning_rate, *arguments):
        settings.update(loss=loss, learning_rate=learning_rate)
        return []

    calibrator._fit_by_adam = adam
    frames = SimpleNamespace(
        region=torch.ones(4, 4, dtype=torch.bool), dtype=torch.float32
    )
    calibrator._fit(SimpleNamespace(frames=frames), None, False)
    assert isinstance(settings["loss"], SumOfLosses)
    assert settings["learning_rate"] == pytest.approx(1e-2)


def test_psf_calibration_runs_end_to_end() -> None:
    """The PSF path returns a valid calibration and fits far fewer parameters."""
    from .test_speckle_calibrator import (
        FOCAL_LENGTH as SMALL_FOCAL_LENGTH,
        SLM_RESOLUTION as SMALL_SLM_RESOLUTION,
        _build_hardware,
        _build_model,
        _synthetic_mapping,
    )

    slm, camera = _build_hardware()
    calibrator = PSFSpeckleCalibrator(
        slm=slm,
        camera=camera,
        camera_mapping=_synthetic_mapping(),
        slm_camera_model=_build_model(
            slm, camera, SMALL_FOCAL_LENGTH,
            slm_field=PSFSLMField.from_camera_mapping(
                _synthetic_mapping(), focal_length=SMALL_FOCAL_LENGTH
            ),
        ),
        dataset_path=Path(tempfile.mkdtemp()) / "dataset.asdf",
        number_of_random_patterns=4,
    )
    assert isinstance(calibrator.slm_camera_model.slm_field, PSFSLMField)

    trainable = sum(
        parameter.numel()
        for parameter in calibrator.slm_camera_model.parameters()
        if parameter.requires_grad
    )
    field_pixels = 2 * SMALL_SLM_RESOLUTION[0] * SMALL_SLM_RESOLUTION[1]
    assert trainable < field_pixels

    result = calibrator.calibrate(
        speckle_pattern_extent=(5e-4, 5e-4),
        steps=3,
        chunk_size=2,
        seed=0,
        verbose=False,
    )

    assert isinstance(result, WavefrontCalibrationData)
    assert result.complex_amplitude.resolution == SMALL_SLM_RESOLUTION
    assert torch.isfinite(result.complex_amplitude.as_tensor()).all()
    # The fit moved: the loss went down and the kernel is no longer its seed.
    assert calibrator.loss_history[-1] < calibrator.loss_history[0]
    kernel = calibrator.slm_camera_model.slm_field.get_psf_kernel()
    assert float(kernel.imag.abs().max()) > 0.0
    assert np.isfinite(kernel.detach().cpu().numpy()).all()


def _implied_focal_shift(field: torch.Tensor) -> tuple[float, float]:
    """The focal-plane displacement, in kernel samples, implied by the beam tilt.

    A linear phase across the SLM shifts the whole focal pattern, so the fitted tilt
    fixes where a mapping has put the beam.
    """
    phase = np.angle(field.detach().cpu().numpy())
    unwrapped = np.unwrap(np.unwrap(phase, axis=0), axis=1).ravel()
    rows, columns = np.indices(phase.shape)
    x = (columns - phase.shape[1] / 2).ravel().astype(float)
    y = (rows - phase.shape[0] / 2).ravel().astype(float)
    design = np.stack([np.ones_like(x), x, y], axis=1)
    piston, tilt_x, tilt_y = np.linalg.lstsq(design, unwrapped, rcond=None)[0]
    scale = WAVELENGTH * FOCAL_LENGTH / (SLM_PIXEL_SIZE * KERNEL_PITCH)
    return (
        float(tilt_x / (2 * math.pi) * scale),
        float(tilt_y / (2 * math.pi) * scale),
    )


def _delta_kernel_field(kernel_size: int, shift_pixels: int) -> torch.Tensor:
    module = PSFSLMField(focal_length=FOCAL_LENGTH, psf_extent=_extent_of(kernel_size))
    module(_uniform_field())
    center = kernel_size // 2
    seed = torch.zeros(kernel_size, kernel_size)
    seed[center, center + shift_pixels] = 1.0
    with torch.no_grad():
        module.psf_kernel.copy_(seed.to(module.psf_kernel.dtype))
    return module.get_wavefront().detach()


def test_a_centered_kernel_puts_no_tilt_on_the_beam() -> None:
    """A kernel on the origin must give an untilted beam.

    The Gaussian pair test above only checks the waist of the amplitude, so a
    mapping carrying a spurious tilt would sail through it while displacing the
    entire focal pattern. That is not a small error: it decorrelates the
    predicted speckle from the measurement and the fit then goes nowhere.
    Mapping the kernel through the model's lens adjoint instead of this chirp-z
    did exactly that, displacing the pattern by about five camera pixels
    against a speckle grain of under three, and fitted worse than applying no
    correction at all.
    """
    shift_x, shift_y = _implied_focal_shift(_delta_kernel_field(33, 0))
    assert abs(shift_x) < 0.05
    assert abs(shift_y) < 0.05


@pytest.mark.parametrize("shift_pixels", [1, 2, 3])
def test_a_shifted_kernel_tilts_the_beam_by_the_matching_amount(shift_pixels) -> None:
    """Shifting the kernel by n samples displaces the focal pattern by n samples of
    the kernel's pitch.

    The amplitude does not show the origin convention, so this test pins it. The
    mapping runs the transform in the negative direction, so the implied displacement
    has the opposite sign to the kernel shift. The magnitude fixes the beam's position.
    """
    reference = _delta_kernel_field(33, 0)
    shifted = _delta_kernel_field(33, shift_pixels)

    difference = shifted * reference.conj()
    step = float(
        torch.angle((difference[:, 1:] * difference[:, :-1].conj()).sum())
    )
    predicted = (
        2
        * math.pi
        * shift_pixels
        * KERNEL_PITCH
        * SLM_PIXEL_SIZE
        / (WAVELENGTH * FOCAL_LENGTH)
    )
    assert abs(step) == pytest.approx(abs(predicted), rel=1e-3)


@pytest.mark.parametrize("kernel_size", [9, 11, 15], ids=["9", "11", "15"])
def test_the_gaussian_seed_is_centred_on_the_sample_the_lens_uses(kernel_size) -> None:
    """An off-centre kernel is a linear phase across the SLM plane.

    ``odd_sample_count`` always returns an odd size, so the kernel has a centre sample.
    The chirp-z that maps the kernel to the SLM plane centres it at ``size // 2``. The
    seed is built on ``get_spatial_grid``, which has to centre it on the same sample. A
    freshly built model then starts without a tilt.
    """
    module = PSFSLMField(focal_length=FOCAL_LENGTH, psf_extent=_extent_of(kernel_size))
    module(_uniform_field())

    kernel = module.get_psf_kernel().detach().abs()
    peak = np.unravel_index(int(kernel.argmax()), kernel.shape)

    assert peak == (kernel_size // 2, kernel_size // 2)
    # Symmetric to the bit, so the kernel carries no tilt of its own.
    assert float((kernel - torch.flip(kernel, dims=(-2, -1))).abs().max()) == 0.0


@pytest.mark.parametrize("camera_pitch", [10e-6, 20e-6], ids=["10um", "20um"])
def test_the_kernel_is_sampled_at_the_intensity_nyquist_pitch_of_the_slm(
    camera_pitch,
) -> None:
    """The pitch is ``lambda f / (2 L)`` along each axis, whatever the camera pitch of
    the seed. The side is the odd number of samples that covers the extent. The SLM is
    not square, so the two axes differ.
    """
    resolution = (192, 256)
    waist = 60e-6
    spot = _gaussian_spot(waist, camera_pitch)
    module = PSFSLMField(
        focal_length=FOCAL_LENGTH,
        psf_extent=10 * waist,
        init_psf_kernel=spot,
        init_psf_kernel_pixel_size=(camera_pitch, camera_pitch),
    )
    module(_uniform_field(resolution))

    pitch = tuple(
        WAVELENGTH * FOCAL_LENGTH / (2 * count * SLM_PIXEL_SIZE) for count in resolution
    )
    assert module.psf_pixel_size == pytest.approx(pitch, rel=1e-6)
    # 600 um over pitches of 32.8 and 24.6 um, 18.3 and 24.4 samples, rounded up to odd.
    assert module.psf_kernel_size == (19, 25)
    assert tuple(module.get_psf_kernel().shape) == (19, 25)


def _gaussian_spot(waist: float, pitch: float) -> torch.Tensor:
    """The amplitude of a focal spot of ``waist`` on a camera crop of ten waists."""
    side = odd_sample_count(10 * waist, pitch)
    axis = (torch.arange(side, dtype=torch.float64) - side // 2) * pitch
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
    return torch.exp(-(grid_x**2 + grid_y**2) / waist**2).float()


@pytest.mark.parametrize("camera_pitch", [10e-6, 20e-6], ids=["10um", "20um"])
def test_a_spot_at_the_camera_pitch_seeds_the_field_it_makes(camera_pitch) -> None:
    """The spot is Gaussian, so its field is the conjugate Gaussian beam made by a
    Gaussian kernel of the same waist.
    """
    waist = 60e-6
    seeded = PSFSLMField(
        focal_length=FOCAL_LENGTH,
        psf_extent=10 * waist,
        init_psf_kernel=_gaussian_spot(waist, camera_pitch),
        init_psf_kernel_pixel_size=(camera_pitch, camera_pitch),
    )
    gaussian = PSFSLMField(
        focal_length=FOCAL_LENGTH, psf_extent=10 * waist, psf_gaussian_waist=waist
    )
    seeded(_uniform_field())
    gaussian(_uniform_field())

    field = seeded.get_wavefront().detach()
    expected = gaussian.get_wavefront().detach()
    scale = (field.conj() * expected).sum() / (field.abs() ** 2).sum()
    error = float((scale * field - expected).norm() / expected.norm())
    assert seeded.wavefront_fit_residual < 1e-3
    assert error < 1e-3


def test_a_spot_without_its_pitch_is_refused() -> None:
    with pytest.raises(ValueError, match="init_psf_kernel_pixel_size"):
        PSFSLMField(
            focal_length=FOCAL_LENGTH,
            psf_extent=_extent_of(15),
            init_psf_kernel=torch.ones(5, 5),
        )
