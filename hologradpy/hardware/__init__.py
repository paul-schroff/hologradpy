from .camera import Camera, SimulatedCameraTorch
from .slm import SLM, SimulatedSLMTorch
from ..roi import ROI
from .as_native import as_camera, as_slm
from .factory import (
    open_camera,
    open_slm,
    register_camera_backend,
    register_slm_backend,
    available_camera_backends,
    available_slm_backends,
)

from .slmsuite import (
    SLMSuiteCameraAdapter,
    SLMSuiteSLMAdapter,
    register_slmsuite_backends,
)

__all__ = [
    "SimulatedSLMTorch",
    "SimulatedCameraTorch",
    "Camera",
    "SLM",
    "ROI",
    "SLMSuiteCameraAdapter",
    "SLMSuiteSLMAdapter",
    "as_camera",
    "as_slm",
    "open_camera",
    "open_slm",
    "register_camera_backend",
    "register_slm_backend",
    "register_slmsuite_backends",
    "available_camera_backends",
    "available_slm_backends",
]
