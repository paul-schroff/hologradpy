from __future__ import annotations

import torch
from torch import Tensor

from scipy.constants import Planck, speed_of_light

from ..abstract import OpticsModule, capture_init
from ...complex_amplitude import ComplexAmplitude, pixel_area


class CameraSensor(OpticsModule):
    """Terminal OpticsModule modeling a camera sensor: it converts the optical
    intensity of the incident field into digital pixel values (ADU).

    The conversion follows the standard sensor chain (modeled on stryq's
    ``MockCamera``, without the point-spread convolution): the per-pixel optical power
    ``|E|^2 * pixel_area`` is turned into a photon count via the photon energy (from the
    field's wavelength) and the exposure time, scaled by the quantum efficiency and an
    optional neutral-density filter, optionally drawn with shot noise, dark current and
    read noise, multiplied by the gain, clipped to the full-well capacity (saturation),
    scaled to the bit-depth full scale, and optionally floored to integer counts.

    ``forward`` returns a *real* pixel image ``(*batch, H, W)``, with the wavelength
    axis summed (a monochrome sensor). It is therefore terminal: it does not return a
    :class:`ComplexAmplitude` and cannot be chained further.

    The noise is drawn while ``add_noise`` is set, in electrons:

    - With ``shot_noise``, the expected electrons are drawn as a Poisson count, so their
      variance equals the signal and grows with the exposure time.
    - The dark current, ``dark_current`` electrons per second, accumulates over the
      exposure and is drawn as a Poisson count of its own. It adds a floor that grows
      with the exposure time.
    - The read noise is a zero-mean Gaussian of standard deviation ``read_noise``
      electrons, the same at every exposure time.

    The noisy electrons carry the gradient of the expected electrons.

    Two modes:

    - Default (``add_noise=True, quantize=True``): realistic capture, with shot noise,
      dark current, read noise and an integer bit-depth floor. Stochastic and not
      differentiable.
    - ``add_noise=False, quantize=False``: the deterministic expected ADU, fully
      differentiable (the only non-smooth step is the full-well clip, like a ReLU). Use
      this for gradient-based calibration / optimization.
    """

    @capture_init
    def __init__(
        self,
        quantum_efficiency: float = 1.0,
        full_well_capacity: float = 1e4,
        exposure_time: float = 1e-3,
        gain: float = 1.0,
        nd_filter_optical_density: float = 0.0,
        bitdepth: int = 8,
        quantize: bool = True,
        add_noise: bool = True,
        shot_noise: bool = True,
        dark_current: float = 0.0,
        read_noise: float = 0.0,
    ) -> None:
        super().__init__()

        self.quantum_efficiency = quantum_efficiency
        self.full_well_capacity = full_well_capacity
        self.exposure_time = exposure_time
        self.gain = gain
        self.nd_filter_optical_density = nd_filter_optical_density
        self.bitdepth = bitdepth
        self.max_pixel_value = 2**bitdepth - 1
        self.quantize = quantize
        self.add_noise = add_noise
        self.shot_noise = shot_noise
        self.dark_current = dark_current
        self.read_noise = read_noise

    @property
    def is_stochastic(self) -> bool:
        """True while the noise is on and there is noise to draw, from the shot noise,
        a dark current or read noise.
        """
        return self.add_noise and (
            self.shot_noise or self.dark_current != 0.0 or self.read_noise != 0.0
        )

    def forward(self, complex_amplitude: ComplexAmplitude) -> Tensor:
        intensity = complex_amplitude.intensity

        area = pixel_area(complex_amplitude.pixel_size)  # (n_wl,)
        photon_energy = (
            Planck * speed_of_light / complex_amplitude.wavelength
        ).reshape(-1)  # (n_wl,)

        # Per-wavelength factor: intensity * pixel_area / photon_energy is the
        # photon rate per pixel; summing over wavelengths gives a monochrome
        # sensor. Back to the field's precision, since a frame is compared against
        # camera counts.
        photon_factor = (area / photon_energy).to(intensity.dtype)  # (n_wl,)
        
        photons = (intensity * photon_factor.reshape(-1, 1, 1)).sum(dim=-3)

        # The ND filter attenuates the signal (not the dark current or the read noise).
        photons = (
            photons * self.exposure_time * 10 ** (-self.nd_filter_optical_density)
        )

        electrons = photons * self.quantum_efficiency
        if self.add_noise:
            # torch.poisson passes no gradient, so the expected electrons carry it
            # through the draw.
            expected = electrons.detach()
            dark = torch.full_like(expected, self.dark_current * self.exposure_time)
            if self.shot_noise:
                # The signal and the dark electrons as one Poisson count at their
                # summed rate.
                counted = torch.poisson(expected + dark)
            else:
                counted = expected + torch.poisson(dark)
            electrons = (
                counted
                + self.read_noise * torch.randn_like(expected)
                + (electrons - expected)
            )
        electrons = electrons * self.gain
        electrons = electrons.clamp(0.0, self.full_well_capacity)

        adu = electrons / self.full_well_capacity * self.max_pixel_value
        if self.quantize:
            adu = adu.floor()
        return adu
