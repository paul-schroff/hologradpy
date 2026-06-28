from slmsuite.hardware.cameras.camera import Camera

from ..utils import roi_bounds


def set_camera_woi(
    camera: Camera,
    center: tuple[int, int],
    roi_size: tuple[int, int],
) -> None:
    """Set ``camera``'s window of interest to an ``roi_size`` (height, width)
    region centred on ``center`` (x, y) pixels."""
    x0, _, y0, _ = roi_bounds(center, roi_size)
    camera.set_woi([x0, roi_size[1], y0, roi_size[0]])
