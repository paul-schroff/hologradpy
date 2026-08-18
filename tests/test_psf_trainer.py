"""Tests for the PSF-parameterized SLM-plane field and its trainer.

The camera field for a phase pattern is ``FT(A * exp(i * phase))``, which by the
convolution theorem is ``FT(A)`` convolved with ``FT(exp(i * phase))``, so the
camera-plane point spread function ``FT(A)`` fixes the SLM-plane beam ``A``.
:class:`PSFSLMField` carries that PSF as a compact kernel and maps it to the SLM
grid, and :class:`WavefrontFitter` fits it against captured speckle.
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from hologradpy.optics.complex_amplitude import (  # noqa: E402
    ComplexAmplitude,
    FieldGeometry,
)
from hologradpy.optics.modules.slm_fields import (  # noqa: E402
    PSFSLMField,
    kernel_size_from_waist,
)
from hologradpy.calibration.wavefront.speckle_calibration import (  # noqa: E402
    PSFSpeckleCalibrator,
    PixelwiseSpeckleCalibrator,
)
from hologradpy.loss_functions import SumOfLosses  # noqa: E402
from hologradpy.calibration.wavefront.abstract import (  # noqa: E402
    WavefrontCalibrationData,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

SLM_RESOLUTION = (256, 256)
SLM_PIXEL_SIZE = 12.5e-6
WAVELENGTH = 630e-9
FOCAL_LENGTH = 0.25
CAMERA_PIXEL_SIZE = (20e-6, 20e-6)


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
        camera_pixel_size=CAMERA_PIXEL_SIZE,
        psf_kernel_size=48,
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
    module = PSFSLMField(
        focal_length=FOCAL_LENGTH,
        camera_pixel_size=CAMERA_PIXEL_SIZE,
        psf_kernel_size=16,
    )
    module(_uniform_field())

    assert module.psf_kernel.is_complex()
    assert module.get_psf_kernel().shape == (16, 16)
    assert sum(1 for _ in module.parameters()) == 1


def test_kernel_gradients_reach_the_whole_slm_field() -> None:
    """Every kernel pixel must move the whole field, which is why it converges
    at a different step size from a per-pixel field."""
    module = PSFSLMField(
        focal_length=FOCAL_LENGTH,
        camera_pixel_size=CAMERA_PIXEL_SIZE,
        psf_kernel_size=16,
    )
    module(_uniform_field())
    module.psf_kernel.requires_grad_(True)

    module.get_wavefront().abs().sum().backward()
    assert module.psf_kernel.grad is not None
    assert torch.isfinite(module.psf_kernel.grad).all()
    assert float(module.psf_kernel.grad.abs().sum()) > 0.0


def test_multiple_wavelengths_are_refused_clearly() -> None:
    """The kernel to SLM magnification scales with wavelength, so one kernel
    cannot serve several at once."""
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
    module = PSFSLMField(
        focal_length=FOCAL_LENGTH,
        camera_pixel_size=CAMERA_PIXEL_SIZE,
        psf_kernel_size=16,
    )
    with pytest.raises(NotImplementedError, match="single wavelength"):
        module(field)


def test_kernel_size_tracks_the_fitted_waist() -> None:
    """A broader measured spot buys a larger kernel, which is the point: the
    waist is fitted to the aberrated spot, so the kernel grows with the
    aberration."""
    small = kernel_size_from_waist(5e-6, 3.45e-6)
    large = kernel_size_from_waist(15e-6, 3.45e-6)
    assert large > small
    assert small % 2 == 1 and large % 2 == 1  # centered kernels
    assert kernel_size_from_waist(1e-9, 3.45e-6) >= 3  # never degenerate


def test_a_static_field_model_is_not_treated_as_a_psf_one(tmp_path) -> None:
    """A PixelwiseSLMField model must not pick up the PSF parameterization's
    settings."""
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from test_speckle_calibrator import (  # noqa: E402
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

    # It gets the unconstrained field's settings, prior included, not the PSF's.
    settings = calibrator._fit_settings(torch.ones(4, 4))
    assert isinstance(settings.loss, SumOfLosses)
    assert settings.learning_rate == pytest.approx(1e-2)


def test_psf_calibration_runs_end_to_end() -> None:
    """The PSF path returns a valid calibration and fits far fewer parameters."""
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from test_speckle_calibrator import (  # noqa: E402
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
                _synthetic_mapping(),
                focal_length=SMALL_FOCAL_LENGTH,
                camera_pixel_size=tuple(camera.pixel_size),
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

    # Enough epochs to be past Adam's opening transient. At 4 the first steps can
    # still overshoot and leave the loss above where it started, which made the
    # assertion below a coin flip on the dataset. By 10 the fit is clearly
    # descending, and by 30 it reaches 0.004 from 0.090.
    result = calibrator.calibrate(
        speckle_pattern_extent=(5e-4, 5e-4),
        number_of_epochs=10,
        batch_size=2,
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
    """The focal-plane displacement, in camera pixels, implied by the beam tilt.

    A linear phase across the SLM shifts the whole focal pattern, so fitting the
    tilt and converting it is the direct way to ask whether a mapping has put
    the beam where it belongs.
    """
    phase = np.angle(field.detach().cpu().numpy())
    unwrapped = np.unwrap(np.unwrap(phase, axis=0), axis=1).ravel()
    rows, columns = np.indices(phase.shape)
    x = (columns - phase.shape[1] / 2).ravel().astype(float)
    y = (rows - phase.shape[0] / 2).ravel().astype(float)
    design = np.stack([np.ones_like(x), x, y], axis=1)
    piston, tilt_x, tilt_y = np.linalg.lstsq(design, unwrapped, rcond=None)[0]
    scale = WAVELENGTH * FOCAL_LENGTH / (SLM_PIXEL_SIZE * CAMERA_PIXEL_SIZE[1])
    return (
        float(tilt_x / (2 * math.pi) * scale),
        float(tilt_y / (2 * math.pi) * scale),
    )


def _delta_kernel_field(kernel_size: int, shift_pixels: int) -> torch.Tensor:
    module = PSFSLMField(
        focal_length=FOCAL_LENGTH,
        camera_pixel_size=CAMERA_PIXEL_SIZE,
        psf_kernel_size=kernel_size,
    )
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
    """Shifting the kernel by n camera pixels must displace the focal pattern by n.

    Pins the origin convention in a way the amplitude cannot hide. The mapping
    runs the transform in the negative direction, so the implied displacement
    comes back with the opposite sign to the kernel shift, which is a
    convention rather than an error: what matters is that the magnitude is
    exact, since that is what fixes the beam's position.
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
        * CAMERA_PIXEL_SIZE[1]
        * SLM_PIXEL_SIZE
        / (WAVELENGTH * FOCAL_LENGTH)
    )
    assert abs(step) == pytest.approx(abs(predicted), rel=1e-3)
