"""A field carried onto a larger plane of zeros at the same pitch."""

from __future__ import annotations

from ...utils import to_canvas
from .abstract import OpticsModule, capture_init
from ..complex_amplitude import ComplexAmplitude


class ZeroPadding(OpticsModule):
    """Place the field at the centre of a larger plane of zeros, at the same pitch."""

    @capture_init
    def __init__(self, extent: tuple[float, float]) -> None:
        """
        Args:
            extent: The plane's ``(height, width)`` in metres. Each axis holds the whole
                number of pixels nearest to it.
        """
        super().__init__()
        self.extent = extent

    def lazy_init(self, complex_amplitude: ComplexAmplitude) -> None:
        pitch = self.pixel_size_in.reshape(-1, 2)[0]
        resolution = tuple(
            round(self.extent[axis] / float(pitch[axis])) for axis in range(2)
        )
        if any(padded < own for padded, own in zip(resolution, self.resolution_in)):
            raise ValueError(
                f"A plane of {resolution[0]} x {resolution[1]} pixels is smaller than "
                f"the incoming field of {self.resolution_in[0]} x "
                f"{self.resolution_in[1]}, and ZeroPadding only enlarges a field."
            )
        self.set_output_geometry(resolution=resolution)

    def forward(self, complex_amplitude: ComplexAmplitude) -> ComplexAmplitude:
        return to_canvas(complex_amplitude, self.resolution_out).with_geometry(
            wavelength=complex_amplitude.wavelength, pixel_size=self.pixel_size_out
        )

    def adjoint(self, complex_amplitude: ComplexAmplitude) -> ComplexAmplitude:
        """Crop the plane back to the field, the conjugate transpose of the padding."""
        return to_canvas(complex_amplitude, self.resolution_in).with_geometry(
            wavelength=complex_amplitude.wavelength, pixel_size=self.pixel_size_in
        )
