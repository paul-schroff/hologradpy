"""OpticsModules that model real apparatus behavior and imperfections.

Unlike the deterministic, idealized optical elements, these inject the physical
realism of an actual setup: the detector response (:class:`CameraSensor`), stray
laser-speckle background (:class:`BackgroundScatter`), and frame-to-frame beam
pointing / power fluctuations (:class:`PointingInstability`, :class:`PowerInstability`).
They are used to build the simulated camera in :mod:`hologradpy.hardware`.
"""

from .background_scatter import BackgroundScatter
from .pointing_instability import PointingInstability
from .power_instability import PowerInstability
from .camera_sensor import CameraSensor

__all__ = [
    "BackgroundScatter",
    "PointingInstability",
    "PowerInstability",
    "CameraSensor",
]
