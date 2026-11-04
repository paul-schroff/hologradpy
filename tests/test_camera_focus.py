"""Tests for find_camera_focus on a small simulated bench.

A curved beam on the SLM puts the camera off the focus by a known distance. A beam whose
focus lies ``off`` before the camera has the phase of a lens that moves the focus by
``-off``, which is what ``_bench`` puts on it.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from mpl_toolkits.mplot3d import Axes3D

from hologradpy.calibration import (
    CameraFocus,
    CameraFocusVisualizer,
    find_camera_focus,
)
from hologradpy.fourier_optics import get_focal_spot_radius
from hologradpy.hardware import SimulatedCameraTorch, SimulatedSLMTorch
from hologradpy.optics.complex_amplitude import ComplexAmplitude, FieldGeometry
from hologradpy.optics.modules.slm_fields import PixelwiseSLMField
from hologradpy.optics.systems import SLMCZT
from hologradpy.profiles.amplitude import gaussian_beam_intensity
from hologradpy.profiles.gaussian import Gaussian, GaussianBeam, radius_along_beam
from hologradpy.profiles.phase import defocus_phase, linear_phase
from hologradpy.roi import ROI

WAVELENGTH = 780e-9
FOCAL_LENGTH = 0.2
BEAM_RADIUS = 0.8e-3
CAMERA_PITCH = 7.5e-6
CAMERA_RESOLUTION = (512, 512)
# The spot lies 160 pixels from the zeroth order, at the sensor's centre.
TILT = (-1.2e-3, 0.0)
WAVENUMBER = 2 * np.pi / WAVELENGTH
WAIST = float(get_focal_spot_radius(BEAM_RADIUS, WAVELENGTH, FOCAL_LENGTH))
RAYLEIGH_RANGE = math.pi * WAIST**2 / WAVELENGTH


class CameraWithZerothOrder(SimulatedCameraTorch):
    """Adds a fixed spot at the sensor's centre, ``zeroth_order`` counts per second at
    its peak, which the lens phases don't move.
    """

    zeroth_order = 0.0

    def _capture_frame(self) -> torch.Tensor:
        frame = super()._capture_frame()
        grid = torch.arange(CAMERA_RESOLUTION[0]) - (CAMERA_RESOLUTION[0] - 1) / 2
        radius = WAIST / CAMERA_PITCH
        shape = torch.exp(-2 * (grid[None, :] ** 2 + grid[:, None] ** 2) / radius**2)
        counts = self.zeroth_order * self.get_exposure() * shape
        return (frame + counts.to(frame)).clamp(max=self.max_pixel_value)


def _bench(off=0.0, off_x=None):
    """The SLM and camera of a beam whose focus lies ``off`` before the camera, or
    ``off_x`` along x if it is given.
    """
    torch.manual_seed(0)
    geometry = FieldGeometry(
        resolution=(256, 320),
        pixel_size=torch.tensor([12.5e-6, 12.5e-6]),
        wavelength=torch.tensor(WAVELENGTH),
    )
    slm = SimulatedSLMTorch(input_geometry=geometry, bitdepth=8)
    x, y = geometry.get_spatial_grid()
    curvature = defocus_phase(x, y, -off, FOCAL_LENGTH, WAVENUMBER)
    if off_x is not None:
        cylinder = WAVENUMBER * (off - off_x) * x**2 / (2 * FOCAL_LENGTH**2)
        curvature = curvature + cylinder
    amplitude = gaussian_beam_intensity(x, y, BEAM_RADIUS).sqrt()
    beam = ComplexAmplitude(
        amplitude * torch.exp(1j * curvature),
        wavelength=geometry.wavelength,
        pixel_size=geometry.pixel_size,
    )
    model = SLMCZT(
        input_geometry=geometry,
        virtual_slm=slm.virtual_slm,
        camera_resolution=CAMERA_RESOLUTION,
        camera_pixel_size=(CAMERA_PITCH, CAMERA_PITCH),
        focal_length=FOCAL_LENGTH,
        slm_field=PixelwiseSLMField(beam),
    )
    camera = CameraWithZerothOrder(model, add_noise=False)
    camera.set_exposure(1e-3)
    camera.get_image()
    return slm, camera


def _find(off=0.0, off_x=None, **options):
    slm, camera = _bench(off, off_x)
    return find_camera_focus(slm, camera, FOCAL_LENGTH, TILT, verbose=False, **options)


@pytest.fixture(scope="module")
def focus() -> CameraFocus:
    """A scan with the focus 5 mm before the camera."""
    return _find(5e-3)


def test_a_camera_beyond_the_focus_moves_towards_the_lens(focus):
    assert focus.camera_move == pytest.approx(-5e-3, abs=0.02 * RAYLEIGH_RANGE)
    assert "5.0" in focus.instruction
    assert "towards the lens before it" in focus.instruction
    assert focus.instruction in str(focus)


def test_a_camera_before_the_focus_moves_away_from_the_lens():
    focus = _find(-8e-3)

    assert focus.camera_move == pytest.approx(8e-3, abs=0.02 * RAYLEIGH_RANGE)
    assert "away from the lens before it" in focus.instruction


def test_the_fit_finds_the_beams_waist_and_rayleigh_range(focus):
    for fit in (focus.fit_x, focus.fit_y):
        assert fit.waist_radius == pytest.approx(WAIST, rel=0.05)
        assert fit.rayleigh_range == pytest.approx(RAYLEIGH_RANGE, rel=0.05)


def test_the_radii_are_read_from_the_fitted_beams(focus):
    np.testing.assert_allclose(
        focus.radii_x, [beam.beam_radius_x for beam in focus.beams]
    )
    np.testing.assert_allclose(
        focus.radii_y, [beam.beam_radius_y for beam in focus.beams]
    )
    assert focus.camera_move_error < 0.02 * RAYLEIGH_RANGE


def test_a_round_beam_is_not_astigmatic(focus):
    assert not focus.is_astigmatic
    assert "astigmatic" not in str(focus)


def test_a_cylindrical_curvature_gives_two_waists():
    focus = _find(2e-3, off_x=8e-3)

    assert focus.fit_x.waist_position == pytest.approx(8e-3, abs=0.05 * RAYLEIGH_RANGE)
    assert focus.fit_y.waist_position == pytest.approx(2e-3, abs=0.05 * RAYLEIGH_RANGE)
    assert focus.camera_move == pytest.approx(-5e-3, abs=0.05 * RAYLEIGH_RANGE)
    assert focus.is_astigmatic
    assert "astigmatic" in str(focus)


def test_a_camera_far_off_the_focus_still_gets_the_move():
    """The crop outgrows the sensor, so it is trimmed to it. The waist's fit is rough
    this far off, and a second scan after the move is the sharp one.
    """
    off = 4 * RAYLEIGH_RANGE

    focus = _find(off)

    assert focus.camera_move == pytest.approx(-off, rel=0.05)
    assert max(max(spot.shape) for spot in focus.spots) == max(CAMERA_RESOLUTION)


def test_a_zeroth_order_brighter_than_the_defocused_spot_is_ignored():
    """The zeroth order peaks at 30 % of the focused spot, and the spot falls to 10 % at
    the scan's ends. The answer stays, and every crop holds the spot.
    """
    slm, camera = _bench(5e-3)
    x, y = slm.get_spatial_grid()
    slm.set_phase(
        linear_phase(x, y, *TILT, wavenumber=WAVENUMBER, focal_length=FOCAL_LENGTH)
    )
    camera.autoexpose(raise_on_rail=False, set_fraction=0.8)
    camera.zeroth_order = 0.3 * camera.get_image().max() / camera.get_exposure()

    focus = find_camera_focus(slm, camera, FOCAL_LENGTH, TILT, verbose=False)

    assert focus.camera_move == pytest.approx(-5e-3, abs=0.05 * RAYLEIGH_RANGE)
    expected = radius_along_beam(focus.defocus, WAIST, 5e-3, RAYLEIGH_RANGE)
    np.testing.assert_allclose(focus.radii_x, expected, rtol=0.1)


def test_the_slm_ends_on_the_spot_and_the_camera_is_put_back():
    slm, camera = _bench(5e-3)
    x, y = slm.get_spatial_grid()
    shown = []
    set_phase = slm.set_phase

    def record(phase):
        shown.append(phase)
        set_phase(phase)

    slm.set_phase = record
    region = ROI(100, 120, 200, 240)
    camera.set_roi(region)
    camera.set_exposure(2e-3)
    exposure, gain = camera.get_exposure(), camera.get_gain()

    find_camera_focus(slm, camera, FOCAL_LENGTH, TILT, steps=3, verbose=False)

    spot = linear_phase(x, y, *TILT, wavenumber=WAVENUMBER, focal_length=FOCAL_LENGTH)
    np.testing.assert_allclose(shown[-1], spot)
    assert not np.allclose(shown[-2], spot)
    assert camera.get_exposure() == exposure
    assert camera.get_gain() == gain
    assert camera.get_image().shape == (region.height, region.width)


def test_a_zeroth_order_alone_is_no_focus():
    """No camera mapping says where the spot lands. If it misses the sensor, the scan
    fits the zeroth order, whose radius the lens phases don't change. So it gives no
    move, and the figure shows a flat curve with no waists.
    """
    slm, camera = _bench()
    x, y = slm.get_spatial_grid()
    slm.set_phase(
        linear_phase(x, y, *TILT, wavenumber=WAVENUMBER, focal_length=FOCAL_LENGTH)
    )
    camera.autoexpose(raise_on_rail=False, set_fraction=0.8)
    camera.zeroth_order = 0.01 * camera.get_image().max() / camera.get_exposure()

    focus = find_camera_focus(slm, camera, FOCAL_LENGTH, (-5e-3, 0.0), verbose=False)

    np.testing.assert_allclose(focus.radii_x, WAIST, rtol=0.01)
    assert not focus.focus_found
    assert focus.instruction.startswith("No focus found")
    assert "Move the camera" not in str(focus)
    assert "astigmatic" not in str(focus)
    visualizer = CameraFocusVisualizer(focus)
    layout = visualizer.default_layout()
    figure = visualizer.render(layout=layout)
    radii = layout.axes["radii"]
    assert radii.get_title() == focus.instruction
    # Two sets of points and two curves, with no waists.
    assert len(radii.get_lines()) == 4
    plt.close(figure)


def test_nothing_on_the_sensor_is_no_focus():
    """The noise-free bench still shows a faint ghost of the grating, whose radii
    scatter rather than follow a focus.
    """
    slm, camera = _bench()

    focus = find_camera_focus(slm, camera, FOCAL_LENGTH, (-5e-3, 0.0), verbose=False)

    assert not focus.focus_found
    assert focus.instruction.startswith("No focus found")


def test_a_scan_too_short_to_see_the_radius_grow_is_no_focus():
    focus = _find(rayleigh_ranges=0.5)

    assert not focus.focus_found


@pytest.mark.parametrize(
    ("off", "off_x"),
    [(0.0, None), (-8e-3, None), (2e-3, 8e-3), (4 * RAYLEIGH_RANGE, None)],
)
def test_a_real_focus_is_found(off, off_x):
    """In focus, on either side of it, astigmatic, and far off it."""
    focus = _find(off, off_x)

    assert focus.focus_found
    assert focus.instruction.startswith("Move the camera")


def test_too_few_steps_are_refused():
    slm, camera = _bench()

    with pytest.raises(ValueError, match="3 or more steps"):
        find_camera_focus(slm, camera, FOCAL_LENGTH, TILT, steps=2)


def test_verbose_prints_each_steps_radii(capsys):
    slm, camera = _bench()

    find_camera_focus(slm, camera, FOCAL_LENGTH, TILT, steps=3)

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3
    assert all("um along x" in line and "um along y" in line for line in lines)


def test_the_visualizer_draws_the_radii_above_the_3d_stack(focus):
    visualizer = CameraFocusVisualizer(focus)
    layout = visualizer.default_layout()

    figure = visualizer.render(layout=layout)
    figure.canvas.draw()

    assert len(figure.axes) == 2
    radii, stack = layout.axes["radii"], layout.axes["stack"]
    assert radii.get_title() == focus.instruction
    assert radii.get_legend() is not None
    assert radii.get_ylim()[0] == 0.0
    assert stack.name == "3d"
    assert radii.get_position().y0 > stack.get_position().y1
    plt.close(figure)


def test_the_stack_draws_each_profile_at_its_defocus(focus, monkeypatch):
    """Each profile sits at its defocus along z, which runs left to right along the
    plot's x axis. It is centred on the beam, scaled to its peak, and shows the
    intensity above 1/e^2. The square 3D axes don't clip the wide stack.
    """
    drawn = []
    contourf = Axes3D.contourf

    def record(self, profile, x, y, **options):
        drawn.append((x, y, profile, options))
        return contourf(self, profile, x, y, **options)

    monkeypatch.setattr(Axes3D, "contourf", record)
    visualizer = CameraFocusVisualizer(focus)
    layout = visualizer.default_layout()

    figure = visualizer.render(layout=layout)

    axes = layout.axes["stack"]
    assert axes.get_xlabel().startswith("z")
    np.testing.assert_allclose(
        axes.get_xlim(), (focus.defocus[0] * 1e3, focus.defocus[-1] * 1e3)
    )
    assert not axes.get_lines()
    assert len(axes.collections) >= len(focus.spots)
    assert not any(profile.get_clip_on() for profile in axes.collections)
    assert len(drawn) == len(focus.spots)
    for (x, y, profile, options), distance, radius in zip(
        drawn, focus.defocus, focus.radii_x
    ):
        assert options["zdir"] == "x"
        assert options["offset"] == pytest.approx(distance * 1e3)
        assert options["levels"][0] == pytest.approx(math.exp(-2))
        assert profile.max() == pytest.approx(1.0, abs=0.1)
        peak = np.unravel_index(np.argmax(profile), profile.shape)
        assert abs(x[peak]) < 0.25 * radius * 1e6
        assert abs(y[peak]) < 0.25 * radius * 1e6
    plt.close(figure)


def test_the_stack_centres_a_profile_and_thins_a_large_crop(focus, monkeypatch):
    """A beam off the crop's centre is drawn at x = 0. A crop as wide as a sensor is
    thinned to a few hundred samples, so it stays quick to draw.
    """
    drawn = []
    contourf = Axes3D.contourf

    def record(self, profile, x, y, **options):
        drawn.append((x, y, profile))
        return contourf(self, profile, x, y, **options)

    monkeypatch.setattr(Axes3D, "contourf", record)
    beam = GaussianBeam(1e-3, shift_x=1e-3, intensity=500.0)
    gaussian = Gaussian((1440, 1080), focus.pixel_size)
    spot = np.asarray(beam.evaluate(gaussian.x, gaussian.y), dtype=np.float64)
    large = CameraFocus(
        focus.defocus[:2],
        [spot, spot],
        [beam, beam],
        focus.pixel_size,
        focus.fit_x,
        focus.fit_y,
    )

    figure = CameraFocusVisualizer(large).render()

    x, y, profile = drawn[0]
    assert max(profile.shape) <= 2 * 300
    peak = np.unravel_index(np.argmax(profile), profile.shape)
    assert abs(x[peak]) < 100.0
    assert abs(y[peak]) < 100.0
    plt.close(figure)
