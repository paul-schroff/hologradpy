from __future__ import annotations

from ...grids import metres_to_pixels
from ..modules.propagators import FourierLensNUFFT
from ..modules.slm_fields import SLMField
from ..modules.virtual_slms.abstract import VirtualSLM
from ..complex_amplitude import FieldGeometry

from .abstract import SLMFourierLensModel, slm_stages
from ..modules.abstract import capture_init


class SLMNUFFT(SLMFourierLensModel):
    """SLM -> NUFFT Fourier lens with learnable focal-plane geometry.

    The :attr:`~.SLMFourierLensModel.focal_plane_partial_affine` of the
    :class:`~.FourierLensNUFFT` places the focal plane on the output plane, the camera
    grid of ``camera_resolution`` at ``camera_pixel_size``. ``camera_angle`` and
    ``camera_shift`` seed it, from the model to the camera.
    """

    virtual_slm: VirtualSLM
    slm_field: SLMField
    fourier_lens: FourierLensNUFFT

    @capture_init
    def __init__(
        self,
        input_geometry: FieldGeometry,
        virtual_slm: VirtualSLM,
        camera_resolution: tuple[int, int],
        camera_pixel_size: tuple[float, float],
        focal_length: float,
        slm_field: SLMField,
        camera_angle: float = 0.0,
        camera_shift: tuple[float, float] = (0.0, 0.0),
        power_normalized: bool = True,
        nufft_kwargs: dict | None = None,
        pointing_focal_shift_std: float | tuple[float, float] | None = None,
        pointing_seed: int | None = None,
        grid_cache: bool = False,
    ) -> None:
        """
        Args:
            input_geometry: The sampling of the SLM plane.
            virtual_slm: The SLM stage.
            camera_resolution: The ``(height, width)`` of the output plane, the camera's
                whole sensor in its mounting or a window of it.
            camera_pixel_size: The ``(y, x)`` pitch of the output plane in metres.
            focal_length: The focal length of the Fourier lens in metres.
            slm_field: The SLM-plane beam.
            camera_angle: The turn of the focal plane as the camera shows it, in degrees
                from +x towards +y. A positive angle turns the displayed image
                counterclockwise. The NUFFT lens turns in radians per SLM sample, so the
                turn acts in metres for square SLM pixels.
            camera_shift: Where the zeroth order lands on the camera, as ``(x, y)``
                metres from the centre pixel of the output plane,
                :func:`~hologradpy.grids.plane_center` of ``camera_resolution``. A
                positive y lies above the centre pixel.
                :func:`~hologradpy.grids.metres_to_pixels` converts it at
                ``camera_pixel_size`` for the partial affine.
            power_normalized: Whether the output carries the optical power of the input.
            nufft_kwargs: Keyword arguments for FINUFFT, such as eps.
            pointing_focal_shift_std: The standard deviation of the beam-pointing shift
                in the focal plane, for
                :meth:`~.PointingInstability.from_focal_shift`, or None for a steady
                beam.
            pointing_seed: The seed of the pointing samples.
            grid_cache: Hold the resampled beam (:func:`~.slm_stages`).
        """
        super().__init__(
            input_geometry=input_geometry,
            focal_length=focal_length,
            pointing_focal_shift_std=pointing_focal_shift_std,
            pointing_seed=pointing_seed,
            **slm_stages(virtual_slm, slm_field, grid_cache),
            fourier_lens=FourierLensNUFFT(
                focal_length,
                resolution_out=camera_resolution,
                pixel_size_out=camera_pixel_size,
                shift=metres_to_pixels(camera_shift, camera_pixel_size),
                angle=camera_angle,
                learnable=True,
                power_normalized=power_normalized,
                nufft_kwargs={} if nufft_kwargs is None else nufft_kwargs,
            ),
        )
