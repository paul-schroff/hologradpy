"""The OpticsModule family: the abstract base and every concrete element type.

An :class:`OpticsModule` transforms a
:class:`~hologradpy.optics.complex_amplitude.ComplexAmplitude` within the
optical pipeline. This package gathers the whole class hierarchy: the abstract
base (:mod:`abstract`) and its recording mix-in (:mod:`recording`), the field
sources (:mod:`virtual_slms`), the plane-to-plane propagators (:mod:`propagators`),
the per-pixel and geometric field operators (:mod:`diagonal_elements`,
:mod:`geometric_transforms`), and the apparatus/imperfection models
(:mod:`hardware_models`).

Distinct from :mod:`hologradpy.optics.systems`, whose ``OpticalSystem``
is a plain ``nn.Module`` composing these OpticsModules into a full SLM-to-camera
system.
"""

from .abstract import OpticsModule, SaveDict
from .recording import RecordingMixin
from .diagonal_elements import (
    DiagonalElement,
    SimpleLens,
    DoubletLens,
    ZernikePhase,
)
from .slm_fields import PSFSLMField, SLMField, PixelwiseSLMField
from .geometric_transforms import GeometricWarp
from .virtual_slms import VirtualSLM, ZernikeSLM
from .propagators import (
    AngularSpectrumMethod,
    FourierLensFFT,
    FourierLensNUFFT,
    FourierLensCZT,
)
from .hardware_models import (
    BackgroundScatter,
    PointingInstability,
    PowerInstability,
    CameraSensor,
)

__all__ = [
    "OpticsModule",
    "SaveDict",
    "RecordingMixin",
    "DiagonalElement",
    "SLMField",
    "PSFSLMField",
    "PixelwiseSLMField",
    "SimpleLens",
    "DoubletLens",
    "ZernikePhase",
    "GeometricWarp",
    "VirtualSLM",
    "ZernikeSLM",
    "AngularSpectrumMethod",
    "FourierLensFFT",
    "FourierLensNUFFT",
    "FourierLensCZT",
    "BackgroundScatter",
    "PointingInstability",
    "PowerInstability",
    "CameraSensor",
]
