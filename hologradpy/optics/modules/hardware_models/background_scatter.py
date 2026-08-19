from __future__ import annotations

import torch

from ..abstract import OpticsModule
from ...complex_amplitude import ComplexAmplitude, pixel_area
from ....profiles.amplitude import laser_speckle_intensity


class BackgroundScatter(OpticsModule):
    """Adds a static incoherent laser-speckle background intensity to the field.

    Generates its own fully-developed speckle pattern (via
    :func:`~hologradpy.profiles.amplitude.laser_speckle_intensity`) at the
    field resolution during :meth:`lazy_init`. ``power`` [W] is the total optical
    background power added over the sensor. The unit-mean speckle is scaled so
    ``sum(I_bg) * pixel_area == power`` (mean intensity ``power / sensor_area``), using
    the field's pixel area.

    The background is added in intensity, ``|E_out|^2 = |E_in|^2 + I_bg``, with the
    field phase preserved: an :class:`OpticsModule` passes a complex field, so the field
    is rebuilt as ``torch.polar(sqrt(|E|^2 + I_bg), arg(E))``. Rebuilding (not scaling
    ``E``) is what lets the background appear on dark pixels where ``|E| = 0``. Only
    ``|E|^2`` changes, so the module is only meaningful right before a terminal
    intensity sensor (:class:`CameraSensor`).
    """

    def __init__(
        self, power: float, grain_radius: float = 5e-6, seed: int | None = None
    ) -> None:
        """
        Args:
            power: Total added background optical power in watts.
            grain_radius: Speckle grain radius in metres.
            seed: Optional RNG seed for a reproducible speckle pattern.
        """
        super().__init__()
        self.power = float(power)
        self.grain_radius = float(grain_radius)
        self.seed = seed

    def lazy_init(self: BackgroundScatter, complex_amplitude: ComplexAmplitude) -> None:
        generator = None
        if self.seed is not None:
            generator = torch.Generator(
                device=complex_amplitude.device
            ).manual_seed(self.seed)
        # Unit-mean fully-developed speckle at the field resolution (square pixels).
        pattern = laser_speckle_intensity(
            self.resolution_in,
            float(self.pixel_size_in[0, 0]),
            self.grain_radius,
            device=complex_amplitude.device,
            dtype=complex_amplitude.dtype_r,
            generator=generator,
        )
        area = pixel_area(self.pixel_size_in)[0]
        sensor_area = area * pattern.numel()
        # Scale so sum(I_bg) * pixel_area == power (mean intensity power / area).
        self.register_buffer("background", pattern * (self.power / sensor_area))

    def forward(
        self: BackgroundScatter, complex_amplitude: ComplexAmplitude
    ) -> ComplexAmplitude:
        field = complex_amplitude.as_tensor()
        intensity = field.real**2 + field.imag**2 + self.background
        out = ComplexAmplitude(
            torch.polar(intensity.sqrt(), field.angle()),
            complex_amplitude.wavelength,
            complex_amplitude.pixel_size,
        )
        return out.with_geometry(
            wavelength=complex_amplitude.wavelength,
            pixel_size=self.pixel_size_out,
        )
