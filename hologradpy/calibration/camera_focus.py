"""Finding how far to move a camera, by profiling the beam through its focus.

:func:`find_camera_focus` scans the focus along the beam with lens phases on the SLM,
and returns a :class:`CameraFocus`. :class:`CameraFocusVisualizer` shows the scan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from ..analysis.uncertainties import standard_deviation
from ..hardware import SLM, Camera, as_camera, as_slm
from ..profiles.gaussian import (
    FocusFit,
    Gaussian,
    GaussianBeam,
    fit_focus,
    radius_along_beam,
)
from ..profiles.phase import defocus_phase, linear_phase
from ..roi import ROI
from ..visualizer import INTENSITY_CMAP, BaseVisualizer, GridCell, Panel, PlotLayout

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from mpl_toolkits.mplot3d import Axes3D

# The crop of the spot without a lens reaches this many pixels from the spot.
_FIRST_CROP_PIXELS = 150
# The fit of the spot without a lens starts at this radius, in pixels.
_FIRST_RADIUS_PIXELS = 10
# Later crops reach this many expected radii from the spot.
_CROP_RADII = 4.0
# The smallest reach of a crop from the spot, in pixels.
_SMALLEST_CROP_PIXELS = 10
# The contour levels of each profile in the 3D stack.
_STACK_LEVELS = 10
# The stack thins a profile to about this many samples across.
_STACK_SAMPLES = 300
# The zoom that draws the stack across the visualizer's bottom panel, and where the
# stack sits along the panel, from 0 at the left to 1 at the right.
_STACK_ZOOM = 1.65
_STACK_ANCHOR = 0.33


def _waist_error(fit: FocusFit) -> float:
    """The standard deviation of the waist position of ``fit``."""
    return standard_deviation(fit, lambda fit: fit.waist_position)


@dataclass(frozen=True, eq=False)
class CameraFocus:
    """What :func:`find_camera_focus` measured, and how far to move the camera.

    ``print`` shows the waists along x and y, and how far to move the camera.

    Attributes:
        defocus: How far each lens phase moved the focus, in metres, away from the SLM
            if positive, ``(S,)``.
        spots: The crop of the camera frame fitted at each step.
        beams: The Gaussian beam fitted to each crop, in metres from the crop's centre.
        pixel_size: The ``(y, x)`` pitch of the crops, in metres.
        fit_x: The focus fitted to the radii along x.
        fit_y: The focus fitted to the radii along y.
    """

    defocus: NDArray[np.float64]
    spots: list[NDArray[np.float64]]
    beams: list[GaussianBeam]
    pixel_size: tuple[float, float]
    fit_x: FocusFit
    fit_y: FocusFit

    @property
    def radii_x(self) -> NDArray[np.float64]:
        """The 1/e^2 intensity radius along x at each step, in metres."""
        return np.array([beam.beam_radius_x for beam in self.beams])

    @property
    def radii_y(self) -> NDArray[np.float64]:
        """The 1/e^2 intensity radius along y at each step, in metres."""
        return np.array([beam.beam_radius_y for beam in self.beams])

    @property
    def camera_move(self) -> float:
        """How far to move the camera along the beam, in metres, away from the SLM if
        positive.

        A waist at a positive defocus means the focus had to move away from the SLM to
        reach the camera, so the focus sits before the camera. The move is minus the
        mean of the waists along x and y. It means nothing unless :attr:`focus_found`.
        """
        return -(self.fit_x.waist_position + self.fit_y.waist_position) / 2

    @property
    def camera_move_error(self) -> float:
        """The standard deviation of :attr:`camera_move`, in metres."""
        return math.hypot(_waist_error(self.fit_x), _waist_error(self.fit_y)) / 2

    @property
    def is_astigmatic(self) -> bool:
        """Whether the waists along x and y differ by more than twice their combined
        standard deviation.
        """
        difference = abs(self.fit_x.waist_position - self.fit_y.waist_position)
        error = math.hypot(_waist_error(self.fit_x), _waist_error(self.fit_y))
        return difference > 2 * error

    @property
    def focus_found(self) -> bool:
        """Whether both fitted Rayleigh ranges lie inside the scan's reach.

        The scan reaches at least three Rayleigh ranges of the beam at its default
        ``rayleigh_ranges``. A spot the lens phases don't change, such as the zeroth
        order, has a fitted Rayleigh range far beyond the scan.
        """
        reach = float(np.max(np.abs(self.defocus)))
        return max(self.fit_x.rayleigh_range, self.fit_y.rayleigh_range) < reach

    @property
    def instruction(self) -> str:
        """How far to move the camera, as a sentence, or that no focus was found."""
        if not self.focus_found:
            return (
                "No focus found: the spot barely changes through the scan. Check that "
                "the SLM's spot lands on the sensor."
            )
        direction = "away from" if self.camera_move > 0 else "towards"
        return (
            f"Move the camera {abs(self.camera_move) * 1e3:.2f} mm along the beam, "
            f"{direction} the lens before it."
        )

    def __str__(self) -> str:
        lines = [
            f"Waist along {name}: {fit.waist_position * 1e3:+.3f} ± "
            f"{_waist_error(fit) * 1e3:.3f} mm, "
            f"radius {fit.waist_radius * 1e6:.1f} um, "
            f"Rayleigh range {fit.rayleigh_range * 1e3:.2f} mm"
            for name, fit in (("x", self.fit_x), ("y", self.fit_y))
        ]
        if self.focus_found and self.is_astigmatic:
            lines.append(
                "The beam focuses at different places along x and y: it is astigmatic."
            )
        lines.append(self.instruction)
        return "\n".join(lines)


def _measure_spot(
    slm: SLM,
    camera: Camera,
    phase: NDArray,
    half_side: int,
    radius_guess: float,
) -> tuple[NDArray[np.float64], GaussianBeam]:
    """Show ``phase``, and fit a Gaussian beam to the spot on the camera.

    The crop reaches ``half_side`` pixels from the spot, and is trimmed to the sensor.
    The spot is the peak of the frame, blurred over half the radius the crop expects. So
    the search compares spots by their power, and a small spot doesn't outweigh a wide
    one. The fit starts at ``radius_guess``.

    Returns:
        The crop, and the fitted beam in metres from the crop's centre.
    """
    slm.set_phase(phase)
    camera.autoexpose(raise_on_rail=False, set_fraction=0.8)
    frame = np.asarray(camera.get_image(), dtype=np.float64)
    blur = half_side / (2 * _CROP_RADII)
    row, column = np.unravel_index(np.argmax(gaussian_filter(frame, blur)), frame.shape)
    side = 2 * half_side + 1
    crop = ROI.centered((column, row), (side, side)).trimmed_to(frame.shape).crop(frame)
    gaussian = Gaussian(crop.shape, tuple(camera.pixel_size))
    return crop, gaussian.fit(crop, radius_guess, blur_sigma=blur).beam


def find_camera_focus(
    slm: SLM,
    camera: Camera,
    focal_length: float,
    tilt: tuple[float, float] = (0.0, 0.0),
    *,
    base_phase: NDArray | None = None,
    rayleigh_ranges: float = 3.0,
    steps: int = 13,
    verbose: bool = True,
) -> CameraFocus:
    """Find how far to move ``camera`` to put it in the focus of the spot at ``tilt``.

    Lens phases on the SLM move the focus along the beam by known distances. A Gaussian
    fitted to the spot gives the beam's radius at each distance, and
    :func:`~hologradpy.profiles.gaussian.fit_focus` fitted to the radii puts the waist.

    The SLM shows ``base_phase`` under the whole scan. The defocus counts from the focus
    the spot has with ``base_phase``, so the move puts the camera in that focus.

    The scan needs no camera mapping or model, but the spot has to land on the sensor.
    If it doesn't, the fit can fail, or it can fit another spot, such as the zeroth
    order. The lens phases don't change that spot's radius, so
    :attr:`CameraFocus.focus_found` is False. A spot the lens phases do change, such as
    another order of the grating, can't be told from the one at ``tilt``.

    The spot without a lens sets the scan's reach, through its Rayleigh range
    ``pi w^2 / wavelength``. A camera off the focus makes that spot wider, so the scan
    reaches further and its steps grow. A camera far off the focus therefore gets the
    move, but a poor waist radius and Rayleigh range. Move the camera and scan again.

    The whole sensor is read out, and the camera's exposure, gain and region of interest
    are put back afterwards
    (:meth:`~hologradpy.hardware.camera.Camera.preserve_exposure_gain_and_roi`). The SLM
    ends on the spot with ``base_phase``, without a lens of the scan.

    Args:
        slm: The SLM.
        camera: The camera.
        focal_length: The focal length of the Fourier lens onto the camera, in metres.
        tilt: The displacement ``(x, y)`` of the spot in the focal plane, in metres from
            the zeroth order.
        base_phase: The phase under the scan in radians, such as a lens or a wavefront
            correction. If None, the SLM adds no phase under the scan.
        rayleigh_ranges: How many Rayleigh ranges the scan reaches either side.
        steps: The number of lens phases. An odd number includes the spot without a
            lens.
        verbose: If True, prints the radii at each step.

    Returns:
        CameraFocus: The scan, and how far to move the camera.

    Raises:
        ValueError: ``steps`` is below 3, too few to fit a focus.
        RuntimeError: A Gaussian fit fails, as it can when no spot lands on the sensor.
    """
    if steps < 3:
        raise ValueError(f"A focus needs 3 or more steps, got {steps}.")
    slm = as_slm(slm)
    camera = as_camera(camera)
    x, y = slm.get_spatial_grid()
    wavenumber = 2 * np.pi / slm.wavelength
    spot_phase = linear_phase(
        x, y, *tilt, wavenumber=wavenumber, focal_length=focal_length
    )
    if base_phase is not None:
        spot_phase = spot_phase + base_phase
    pixel = float(np.mean(camera.pixel_size))

    spots, beams = [], []
    with camera.preserve_exposure_gain_and_roi(full_sensor=True):
        _, focused = _measure_spot(
            slm, camera, spot_phase, _FIRST_CROP_PIXELS, _FIRST_RADIUS_PIXELS * pixel
        )
        rayleigh_range = math.pi * focused.mean_radius**2 / slm.wavelength
        defocus = np.linspace(-1.0, 1.0, steps) * rayleigh_ranges * rayleigh_range
        for distance in defocus:
            expected = float(
                radius_along_beam(distance, focused.mean_radius, 0.0, rayleigh_range)
            )
            half_side = max(int(_CROP_RADII * expected / pixel), _SMALLEST_CROP_PIXELS)
            lens = defocus_phase(x, y, float(distance), focal_length, wavenumber)
            spot, beam = _measure_spot(
                slm, camera, spot_phase + lens, half_side, expected
            )
            spots.append(spot)
            beams.append(beam)
            if verbose:
                print(
                    f"{distance * 1e3:+8.3f} mm: radius "
                    f"{beam.beam_radius_x * 1e6:6.1f} um along x, "
                    f"{beam.beam_radius_y * 1e6:6.1f} um along y"
                )
        slm.set_phase(spot_phase)

    radii_x = [beam.beam_radius_x for beam in beams]
    radii_y = [beam.beam_radius_y for beam in beams]
    return CameraFocus(
        defocus=defocus,
        spots=spots,
        beams=beams,
        pixel_size=tuple(float(pitch) for pitch in camera.pixel_size),
        fit_x=fit_focus(defocus, radii_x),
        fit_y=fit_focus(defocus, radii_y),
    )


class CameraFocusVisualizer(BaseVisualizer):
    """Render a :class:`CameraFocus`, to show the result and whether to trust it.

    The top panel shows the beam's radius against the defocus, with the fitted curves
    and the waists. Its title says how far to move the camera.

    The bottom panel stacks the beam's profiles in 3D along z, the defocus, from left to
    right. Each profile is centred on its fitted beam, and scaled to its fitted peak,
    because the camera autoexposes on each. Each shows the intensity above 1/e^2 of its
    peak, so the stack's outline is the beam's radius.

    >>> focus = find_camera_focus(slm, camera, focal_length)
    >>> CameraFocusVisualizer(focus).render()
    """

    def __init__(self, focus: CameraFocus) -> None:
        self.focus = focus

    def default_layout(self) -> PlotLayout:
        layout = PlotLayout(
            column_width=9.0, row_gap=0.5, margins=(0.75, 0.2, 0.4, 0.5)
        )
        layout.add_row([GridCell("radii", aspect="auto", height=3.0)])
        layout.add_row([GridCell("stack", aspect="auto", height=4.9, projection="3d")])
        return layout

    def panels(self) -> dict[str, Panel]:
        return {"radii": self._radii_panel, "stack": self._stack_panel}

    def _radii_panel(self, axs: Axes) -> None:
        focus = self.focus
        fine = np.linspace(focus.defocus.min(), focus.defocus.max(), 200)
        curves = (
            ("x", focus.radii_x, focus.fit_x, "C0"),
            ("y", focus.radii_y, focus.fit_y, "C1"),
        )
        for name, radii, fit, color in curves:
            axs.plot(focus.defocus * 1e3, radii * 1e6, "o", color=color, label=name)
            axs.plot(fine * 1e3, fit.radius(fine) * 1e6, color=color)
            if focus.focus_found:
                axs.axvline(fit.waist_position * 1e3, color=color, linestyle=":")
        axs.set_xlabel("Defocus: the focus moved away from the SLM [mm]")
        axs.set_ylabel("1/e^2 radius [um]")
        # From zero, so radii that barely change look flat.
        axs.set_ylim(bottom=0.0)
        axs.set_title(focus.instruction)
        axs.legend(title="Along")

    def _stack_panel(self, axs: Axes3D) -> None:
        focus = self.focus
        levels = np.linspace(math.exp(-2), 1.0, _STACK_LEVELS)
        for distance, spot, beam in zip(focus.defocus, focus.spots, focus.beams):
            gaussian = Gaussian(spot.shape, focus.pixel_size)
            x = (np.asarray(gaussian.x, dtype=np.float64) - beam.shift_x) * 1e6
            y = (np.asarray(gaussian.y, dtype=np.float64) - beam.shift_y) * 1e6
            profile = (spot - beam.offset) / beam.intensity
            stride = max(max(spot.shape) // _STACK_SAMPLES, 1)
            # The plot's x axis is z, the beam's axis. With zdir="x", the first array
            # holds the values, drawn over the plot's y and z axes.
            axs.contourf(
                profile[::stride, ::stride],
                x[::stride, ::stride],
                y[::stride, ::stride],
                levels=levels,
                zdir="x",
                offset=distance * 1e3,
                cmap=INTENSITY_CMAP,
                extend="max",
                alpha=0.9,
            )
        # Matplotlib keeps 3D axes square, as high as the cell. The zoom draws the stack
        # as wide as the cell, past those axes, so they must not clip the profiles.
        for profile in axs.collections:
            profile.set_clip_on(False)
        limit = 1.1 * max(focus.radii_x.max(), focus.radii_y.max()) * 1e6
        axs.set_xlim(focus.defocus.min() * 1e3, focus.defocus.max() * 1e3)
        axs.set_ylim(-limit, limit)
        axs.set_zlim(-limit, limit)
        axs.set_box_aspect((2.6, 1.0, 1.0), zoom=_STACK_ZOOM)
        # Left of the cell's centre, since the axis labels only need room on the right.
        axs.set_anchor((_STACK_ANCHOR, 0.5))
        axs.view_init(elev=18, azim=-62)
        axs.yaxis.set_major_locator(MaxNLocator(nbins=4))
        axs.zaxis.set_major_locator(MaxNLocator(nbins=4))
        axs.set_xlabel("z, the defocus [mm]", labelpad=10)
        axs.set_ylabel("x [um]")
        axs.set_zlabel("y [um]")
