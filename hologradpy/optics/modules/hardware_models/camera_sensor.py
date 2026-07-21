from __future__ import annotations

import torch
from torch import Tensor

from scipy.constants import Planck, speed_of_light

from ..abstract import OpticsModule
from ...complex_amplitude import ComplexAmplitude


class CameraSensor(OpticsModule):
    """Terminal OpticsModule modelling a camera sensor: it converts the optical
    intensity of the incident field into digital pixel values (ADU).

    The conversion follows the standard sensor chain (modelled on stryq's
    ``MockCamera``, without the point-spread convolution): the per-pixel optical power
    ``|E|^2 * pixel_area`` is turned into a photon count via the photon energy (from the
    field's wavelength) and the exposure time, scaled by the quantum efficiency and an
    optional neutral-density filter, optionally given a Poisson read-noise floor,
    multiplied by the gain, clipped to the full-well capacity (saturation), scaled to
    the bit-depth full scale, and optionally floored to integer counts.

    ``forward`` returns a *real* pixel image ``(*batch, H, W)`` -- the wavelength axis
    is summed (a monochrome sensor). It is therefore terminal: it does not return a
    :class:`ComplexAmplitude` and cannot be chained further.

    Two modes:
    - Default (``add_noise=True, quantize=True``): realistic capture -- Poisson
      read noise plus an integer bit-depth floor. Stochastic and not differentiable.
    - ``add_noise=False, quantize=False``: the deterministic expected ADU, fully
      differentiable (the only non-smooth step is the full-well clip, like a ReLU). Use
      this for gradient-based calibration / optimization.
    """

    def __init__(
        self,
        quantum_efficiency: float,
        full_well_capacity: float,
        exposure_time: float = 1e-3,
        gain: float = 1.0,
        noise_level: float = 0.0,
        nd_filter_optical_density: float = 0.0,
        bitdepth: int = 8,
        add_noise: bool = True,
        quantize: bool = True,
    ) -> None:
        super().__init__()

        self.quantum_efficiency = quantum_efficiency
        self.full_well_capacity = full_well_capacity
        self.exposure_time = exposure_time
        self.gain = gain
        self.noise_level = noise_level
        self.nd_filter_optical_density = nd_filter_optical_density
        self.bitdepth = bitdepth
        self.max_pixel_value = 2**bitdepth - 1
        self.add_noise = add_noise
        self.quantize = quantize

    def forward(self, complex_amplitude: ComplexAmplitude) -> Tensor:
        intensity = complex_amplitude.intensity  # |E|^2, real, on-graph

        pixel_area = (
            complex_amplitude.pixel_size[:, 0] * complex_amplitude.pixel_size[:, 1]
        ).reshape(-1)  # (n_wl,)
        photon_energy = (
            Planck * speed_of_light / complex_amplitude.wavelength
        ).reshape(-1)  # (n_wl,)

        # Per-wavelength factor: intensity * pixel_area / photon_energy is the
        # photon rate per pixel; summing over wavelengths gives a monochrome
        # sensor.
        photon_factor = pixel_area / photon_energy  # (n_wl,)

        if intensity.ndim == 2:
            photons = intensity * photon_factor[0]
        else:
            photons = (intensity * photon_factor.reshape(-1, 1, 1)).sum(dim=-3)

        # The ND filter attenuates the signal (not the read noise).
        photons = (
            photons * self.exposure_time * 10 ** (-self.nd_filter_optical_density)
        )

        electrons = photons * self.quantum_efficiency
        if self.add_noise:
            electrons = electrons + torch.poisson(
                torch.full_like(electrons, self.noise_level**2)
            )
        electrons = electrons * self.gain
        electrons = electrons.clamp(0.0, self.full_well_capacity)

        adu = electrons / self.full_well_capacity * self.max_pixel_value
        if self.quantize:
            adu = adu.floor()
        return adu
