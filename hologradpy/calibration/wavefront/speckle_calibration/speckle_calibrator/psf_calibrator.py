from __future__ import annotations

import os
import warnings
from typing import Any, Sequence

import numpy as np
import torch
from numpy.typing import NDArray

from .abstract import WavefrontSpeckleCalibrator
from ....camera_mapping import CameraMapping
from ....exposure import capture_autoexposed_frame
from ....model_fit import ModelFitter
from ....spot_detection import brightest_pixel, has_prominent_peak
from .....grids import odd_sample_count
from .....hardware import Camera, SLM, as_camera, as_slm
from .....optics import SLMFourierLensModel
from .....roi import ROI
from .....utils import gpu_to_numpy

from ..visualizer import PSFSpeckleVisualizationData

from .....optics.modules.slm_fields import PSFSLMField


class PSFSpeckleCalibrator(WavefrontSpeckleCalibrator):
    """Recover the SLM-plane field through a compact image-plane point spread function.

    The fitted kernel (:class:`~hologradpy.optics.modules.slm_fields.PSFSLMField`) has
    far fewer parameters than the SLM plane and is band limited by construction. A
    Levenberg-Marquardt fit steps the kernel and a translation per frame
    (:meth:`~hologradpy.calibration.model_fit.ModelFitter.fit_levenberg_marquardt`),
    so the kernel is free of the pointing jitter of the beam between frames. The
    ``steps`` and ``chunk_size`` of the fit are passed to :meth:`fit` or
    :meth:`calibrate`.
    """

    slm_field_type = PSFSLMField
    visualization_data_type = PSFSpeckleVisualizationData

    def __init__(
        self,
        slm: SLM,
        camera: Camera,
        slm_camera_model: SLMFourierLensModel,
        dataset_path: str | os.PathLike,
        camera_mapping: CameraMapping | None = None,
        number_of_random_patterns: int = 10,
        zeroth_order_mask_waists: float = 4.0,
        psf_kernel_waists: float = 10.0,
    ) -> None:
        """
        Args:
            slm: The SLM to drive.
            camera: The camera watching its focal plane.
            slm_camera_model: The differentiable model of this setup.
            dataset_path: The dataset file, holding the captured samples and their
                description.
            camera_mapping: The camera mapping to calibrate the model's affine
                transform from. If None, a
                :class:`~hologradpy.calibration.camera_mapping.SpotArrayMapper`
                measures one with the SLM and camera.
            number_of_random_patterns: How many speckle patterns to capture.
            zeroth_order_mask_waists: Radius of the disc around the zeroth order that
                is left out of the fit, in fitted focal-spot waists.
            psf_kernel_waists: The width of the image plane the fitted PSF kernel
                spans, in fitted focal-spot waists
                (:meth:`~hologradpy.optics.modules.slm_fields.PSFSLMField.from_camera_mapping`).
                A kernel much wider than the spot fits noise around it, and one too
                narrow clips the aberrated wings. Ignored when the model already holds
                a :class:`PSFSLMField`, since its kernel is kept.

        Raises:
            ValueError: ``psf_kernel_waists`` is not positive.
        """
        if psf_kernel_waists <= 0:
            raise ValueError(
                f"psf_kernel_waists is a size, so it must be positive, got "
                f"{psf_kernel_waists}."
            )
        # Set before the base constructor, which builds the kernel.
        self.psf_kernel_waists: float = float(psf_kernel_waists)
        super().__init__(
            slm=slm,
            camera=camera,
            slm_camera_model=slm_camera_model,
            dataset_path=dataset_path,
            camera_mapping=camera_mapping,
            number_of_random_patterns=number_of_random_patterns,
            zeroth_order_mask_waists=zeroth_order_mask_waists,
        )

    def _build_slm_field(self) -> PSFSLMField:
        """A kernel sized and seeded from the measured focal spot.

        The spot is captured on a crop of the camera as wide as the kernel's extent,
        and the kernel is fitted to the spot's SLM-plane field.
        """
        camera_pixel_size = tuple(float(pitch) for pitch in self.camera.pixel_size)
        crop_side = odd_sample_count(
            self.psf_kernel_waists * float(self.camera_mapping.spot_fit.waist),
            camera_pixel_size[1],
        )
        spot = capture_focal_spot(
            self.slm, self.camera, self.camera_mapping, self.slm_camera_model, crop_side
        )
        return PSFSLMField.from_camera_mapping(
            self.camera_mapping,
            focal_length=self.focal_length,
            extent_in_waists=self.psf_kernel_waists,
            init_psf_kernel=torch.as_tensor(spot, dtype=torch.float32),
            init_psf_kernel_pixel_size=camera_pixel_size,
        )

    def _fit(
        self,
        fitter: ModelFitter,
        subset_indices: Sequence[int] | None,
        verbose: bool,
        **fit_settings: Any,
    ) -> list[float]:
        """Fit the kernel and a translation per frame by Levenberg-Marquardt
        (:meth:`~hologradpy.calibration.model_fit.ModelFitter.fit_levenberg_marquardt`),
        and return the cost before the first step and after each step.

        Args:
            fitter: The fitter of the dataset.
            subset_indices: Which dataset samples to fit against, or None for all.
            verbose: Show a progress bar.
            **fit_settings: The ``steps`` and ``chunk_size`` of the fit.
        """
        return fitter.fit_levenberg_marquardt(
            subset_indices=subset_indices,
            fit_translations=True,
            verbose=verbose,
            **fit_settings,
        )

    def _calibration_metadata_extras(self) -> dict:
        """The translations fitted to the frames of the dataset, ``(x, y)`` in metres of
        the image plane relative to their mean. A frame left out of the fit holds NaN.
        """
        translations = self.fitter.frame_translations
        return {
            "frame_translations": (
                None if translations is None else np.asarray(translations).tolist()
            )
        }

    def _fit_visualization_extras(self) -> dict:
        """The fitted kernel and its pitch."""
        slm_field = self.slm_camera_model.slm_field
        return {
            "psf_kernel": slm_field.get_psf_kernel().detach().cpu().numpy(),
            "psf_pixel_size": slm_field.psf_pixel_size,
        }


def capture_focal_spot(
    slm: SLM,
    camera: Camera,
    camera_mapping: CameraMapping,
    slm_camera_model: SLMFourierLensModel,
    kernel_size: int | tuple[int, int],
    set_fraction: float = 0.8,
    search_factor: float = 3.0,
) -> NDArray[np.float64]:
    """Capture the amplitude of the focal spot, steered to the sensor centre.

    The captured spot is the point spread function itself, so it is the natural seed
    for a :class:`~hologradpy.optics.modules.slm_fields.PSFSLMField`. It carries
    whatever aberration is present. A Gaussian of the fitted waist carries only the
    width.

    The whole sensor is read out while the spot is captured, and the camera's exposure,
    gain and region of interest are put back afterwards
    (:meth:`~hologradpy.hardware.camera.Camera.preserve_exposure_gain_and_roi`).

    Args:
        slm: The SLM, or a driver that :func:`~hologradpy.hardware.as_native.as_slm`
            wraps. It displays the steering tilt.
        camera: The camera to capture from, or a driver that
            :func:`~hologradpy.hardware.as_native.as_camera` wraps.
        camera_mapping: The fitted mapping, which sets the steering tilt
            (:meth:`~hologradpy.calibration.camera_mapping.CameraMapping.tilt_to_sensor_center`).
        slm_camera_model: The model of the setup, which gives the steering phase.
        kernel_size: Crop side in camera pixels, as an int or ``(height, width)``.
        set_fraction: Fraction of full scale to autoexpose the spot to.
        search_factor: The width of the autoexposed and searched region, in kernel
            widths.
            The whole sensor is searched when the spot is not found in this region.

    Returns:
        NDArray: The cropped amplitude, shaped ``kernel_size``. It is the square root
        of the background-subtracted counts.

    Raises:
        RuntimeError: If no spot is found anywhere on the sensor.
    """
    slm = as_slm(slm)
    camera = as_camera(camera)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    height, width = int(kernel_size[0]), int(kernel_size[1])

    with camera.preserve_exposure_gain_and_roi(full_sensor=True):
        sensor = tuple(camera.sensor_resolution)
        center = (sensor[0] / 2.0, sensor[1] / 2.0)

        tilt = camera_mapping.tilt_to_sensor_center(camera)
        # The tilt is displayed over the full aperture, since the seed is the point
        # spread function of the whole SLM.
        slm.set_phase(gpu_to_numpy(slm_camera_model.steering_phase(tilt)))

        search = ROI.centered(
            center,
            (
                min(int(search_factor * height), sensor[0]),
                min(int(search_factor * width), sensor[1]),
            ),
        ).moved_inside(sensor)

        image = capture_autoexposed_frame(camera, search, set_fraction)

        if has_prominent_peak(search.crop(image), camera):
            found = brightest_pixel(search.crop(image))
            found = (search.top_row + found[0], search.left_column + found[1])
        else:
            zeroth_row, zeroth_column = camera_mapping.zeroth_order_position
            separation = float(
                np.hypot(zeroth_row - center[0], zeroth_column - center[1])
            )
            found = brightest_pixel(
                image,
                mask=~camera_mapping.zeroth_order_mask(image.shape, separation / 2),
            )

            offset = np.hypot(found[0] - center[0], found[1] - center[1])
            # The first autoexposure measured the background in a window without the
            # spot, so the spot can be overexposed. The frame is autoexposed again
            # around the spot.
            search = ROI.centered(
                found, (search.height, search.width)
            ).moved_inside(sensor)
            image = capture_autoexposed_frame(camera, search, set_fraction)
            if not has_prominent_peak(search.crop(image), camera):
                raise RuntimeError(
                    "No focal spot found anywhere on the sensor after steering by "
                    f"{tilt[0] * 1e3:.2f} x {tilt[1] * 1e3:.2f} mm. Check the camera "
                    "mapping, and that the requested tilt is within the SLM's "
                    "diffraction angle."
                )
            warnings.warn(
                f"The steered focal spot landed {offset:.0f} px from the sensor "
                f"center, outside the {search.height} x {search.width} px search "
                "window, so the whole sensor was searched. That offset is the camera "
                "mapping's error: the seed is still good, but the mapping is worth "
                "refitting.",
                stacklevel=2,
            )

    spot_roi = ROI.centered(found, (height, width)).moved_inside(sensor)

    crop: NDArray[np.float64] = spot_roi.crop(image)
    # A robust floor, so read-out background does not become part of the seed.
    background: float = float(np.percentile(crop, 10.0))
    return np.sqrt(np.clip(crop - background, 0.0, None))
