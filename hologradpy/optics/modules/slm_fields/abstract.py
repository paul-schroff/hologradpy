"""The SLM-plane field an optical system carries, and a wavefront fit recovers."""

from __future__ import annotations

from torch import Tensor

from ..diagonal_elements import DiagonalElement


class SLMField(DiagonalElement):
    """The complex field at the SLM plane, however it happens to be parameterised.

    An :class:`~hologradpy.optics.systems.SLMFourierLensModel` carries one of these in
    its ``slm_field`` slot, and a wavefront calibration recovers it by optimising
    whatever parameters it registers. Subclasses differ in what those parameters are: a
    :class:`~hologradpy.optics.modules.slm_fields.PixelwiseSLMField` stores the field
    directly, one value per SLM pixel, while a
    :class:`~hologradpy.optics.modules.slm_fields.PSFSLMField` stores a compact
    camera-plane kernel that maps to the whole SLM plane.

    The contract a fit relies on is :meth:`get_wavefront`. As a
    :class:`~hologradpy.optics.modules.diagonal_elements.DiagonalElement` a subclass
    also supplies ``get_transmission``, which is the same field laid out for the
    per-pixel multiply the forward model applies.
    """

    def get_wavefront(self: SLMField) -> Tensor:
        """The SLM-plane complex field, ``(height, width)`` on the SLM grid."""
        raise NotImplementedError(
            f"get_wavefront() has not been implemented for {type(self).__name__}."
        )
