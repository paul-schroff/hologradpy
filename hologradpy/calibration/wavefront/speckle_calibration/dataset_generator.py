from __future__ import annotations
from typing import Dict, List

import os
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import torch

from .calibration_dataset import (
    DATASET_MANIFEST_NAME,
    DatasetDescriptor,
    TrainingSampleFilenames,
)

from ....hardware import Camera, SLM
from ....hardware.camera import CameraData
from ....hardware.slm import SLMData

from ..abstract import WavefrontCalibrationData

from ...camera_mapping import CameraMapping

from ....profiles.masks import circular_mask, elliptical_mask
from ....profiles.phase import band_limited_random_phase
from ....roi import ROI
from ....fourier_optics import fourier_lens_pixel_size
from ....grids import get_pixel_grid, get_spatial_grid


class DatasetGenerator:
    """Generate random SLM phase patterns and capture their camera speckle images."""

    def __init__(
        self,
        slm: SLM,
        camera: Camera,
        camera_mapping: CameraMapping,
        focal_length: float,
        dataset_directory: str | os.PathLike,
        number_of_random_patterns: int = 1,
    ) -> None:
        self.slm: SLM = slm
        self.camera: Camera = camera
        self.camera_mapping: CameraMapping = camera_mapping
        self.focal_length: float = focal_length
        self.dataset_directory: Path = Path(dataset_directory)
        self.number_of_random_patterns: int = number_of_random_patterns
        self.benchmark_calibration: WavefrontCalibrationData | None = None

        self.camera_background_image: NDArray[np.float_] = np.zeros(
            self.camera.resolution
        )

        self.data_filenames: List[TrainingSampleFilenames] = []

        self.phase_pattern_type: str = "band_limited_random"
        self.metadata: Dict[str, tuple[float, float] | float | int | None] = {
            "band_radius_bins": None,
            "speckle_extent": None,
            "seed": None,
            "exposure_time": None,
        }
        self.roi_mask: NDArray[np.bool_] | None = None

    def generate_dataset(
        self,
        extent: tuple[float, float] | None = None,
        capture_background_image: bool = False,
        benchmark_calibration: WavefrontCalibrationData | None = None,
        seed: int | None = None,
    ) -> DatasetDescriptor:
        """Generate the patterns, capture the frames, and save the manifest beside them.

        The whole capture in one call, which is what a caller normally wants.
        :meth:`generate_phase_patterns` and :meth:`capture_camera_images` remain
        separately callable for the cases that need to do something between the two, for
        example inspecting the patterns before they reach the SLM.

        Args:
            extent: Full width ``(y, x)`` of the speckle at the camera, in metres,
                setting both the pattern band limit and the region of interest. A width
                rather than a radius, so it compares directly against the sensor size.
                Defaults to the largest speckle that fits on the sensor.
            capture_background_image: Capture a dark frame after the patterns. Block the
                beam first, since the frame is taken immediately.
            benchmark_calibration: An existing calibration to add to every pattern, for
                measuring the residual of a previous fit.
            seed: Seed for the pattern noise. Leave as None to seed from the system
                entropy, which makes the dataset irreproducible.

        Returns:
            DatasetDescriptor: The manifest for the captured dataset, already saved to
            ``dataset_directory`` so the dataset can be reloaded without it.
        """
        self.generate_phase_patterns(
            extent, benchmark_calibration=benchmark_calibration, seed=seed
        )
        descriptor = self.capture_camera_images(capture_background_image)
        descriptor.save(self.dataset_directory / DATASET_MANIFEST_NAME)
        return descriptor

    def largest_extent_on_sensor(self) -> tuple[float, float]:
        """The widest speckle ``(y, x)``, in metres, that still fits on the sensor.

        Returns:
            tuple[float, float]: Full width per axis, twice the distance from the zeroth
            order to the nearest sensor edge along that axis.

        Raises:
            ValueError: If the zeroth order lies off the sensor, where no speckle
                centred on it fits and an extent has to be chosen deliberately.
        """
        # In camera pixels, stored (y, x). Coarse mapping extrapolates it through the
        # affine transform, so it can land off the sensor entirely.
        zeroth = self.camera_mapping.zeroth_order_position

        margins = tuple(
            min(float(zeroth[i]), self.camera.resolution[i] - float(zeroth[i]))
            for i in range(2)
        )
        if any(margin <= 0 for margin in margins):
            raise ValueError(
                f"The zeroth order sits at {tuple(float(z) for z in zeroth)} on a "
                f"{tuple(self.camera.resolution)} sensor, so no speckle centred on it "
                "fits. Pass an extent explicitly."
            )

        return tuple(
            2 * margins[i] * self.camera.pixel_size[i] for i in range(2)
        )

    def generate_phase_patterns(
        self,
        extent: tuple[float, float] | None = None,
        benchmark_calibration: WavefrontCalibrationData | None = None,
        seed: int | None = None,
    ) -> None:
        """Generate a set of smooth random phase patterns.

        Each pattern is white noise band limited to the requested camera-plane extent,
        by :func:`~hologradpy.profiles.phase.band_limited_random_phase`. The Fourier
        lens transforms the SLM plane into the camera plane, so the SLM's own FFT plane
        is the camera plane up to a scale, and the band limit can be sized directly in
        camera metres.

        Args:
            extent: Full width ``(y, x)`` of the speckle at the camera, in metres, which
                sets both the band limit and the region of interest. It is a width, not
                a radius, so it can be compared directly against the sensor size.
                Defaults to the largest speckle that fits on the sensor.
            benchmark_calibration: An existing calibration to add to every pattern, for
                measuring the residual of a previous fit.
            seed: Seed for the pattern noise. One generator produces every pattern, so
                they differ from one another and the whole set is reproducible. Leave
                as None to seed from the system entropy, which makes the dataset
                irreproducible.
        """
        if extent is None:
            extent = self.largest_extent_on_sensor()

        self.metadata["speckle_extent"] = extent
        self.metadata["seed"] = seed

        half_extent = tuple(size / 2 for size in extent)

        self.benchmark_calibration = benchmark_calibration
        if self.benchmark_calibration is None:
            benchmark_phase = np.zeros(self.slm.resolution)
        else:
            benchmark_phase = np.angle(
                self.benchmark_calibration.complex_amplitude.as_tensor()
                .detach()
                .cpu()
                .numpy()
            )

        radius_fft_pixels = tuple(
            float(
                half_extent[i]
                / fourier_lens_pixel_size(
                    self.slm.wavelength,
                    self.focal_length,
                    self.slm.pixel_size[i],
                    self.slm.resolution[i],
                )
            )
            for i in range(2)
        )
        self.metadata["band_radius_fft_pixels"] = radius_fft_pixels

        pixel_grid = get_pixel_grid(tuple(self.slm.resolution))
        band_mask = elliptical_mask(
            *pixel_grid, radius_x=radius_fft_pixels[1], radius_y=radius_fft_pixels[0]
        )

        generator = torch.Generator(device=band_mask.device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        for i in range(self.number_of_random_patterns):
            print(
                f"Generating phase pattern {i + 1} of {self.number_of_random_patterns}."
            )

            phase = band_limited_random_phase(band_mask, generator=generator)
            phase = np.remainder(
                phase.cpu().numpy() + benchmark_phase, 2 * np.pi
            )

            phase_filename = f"phase_pattern_{i}.npy"
            self.data_filenames.append(
                TrainingSampleFilenames(phase_pattern=phase_filename)
            )
            np.save(self.dataset_directory / phase_filename, phase)

        camera_grid = get_spatial_grid(self.camera.resolution, self.camera.pixel_size)

        zeroth = self.camera_mapping.zeroth_order_position
        center_y = self.camera.resolution[0] / 2
        center_x = self.camera.resolution[1] / 2
        shift_y = (zeroth[0] - center_y) * self.camera.pixel_size[0]
        shift_x = (zeroth[1] - center_x) * self.camera.pixel_size[1]

        speckle_mask = elliptical_mask(
            *camera_grid,
            radius_x=half_extent[1],
            radius_y=half_extent[0],
            shift_x=shift_x,
            shift_y=shift_y,
        )

        zeroth_order_mask = circular_mask(
            *camera_grid,
            4 * self.camera_mapping.focal_spot_radius,
            shift_x=shift_x,
            shift_y=shift_y,
        )

        self.roi_mask = (speckle_mask & ~zeroth_order_mask).cpu().numpy()

    def capture_camera_images(
        self, capture_background_image: bool = False
    ) -> DatasetDescriptor:
        """Display every generated phase pattern and capture the camera speckle.

        Call :meth:`generate_phase_patterns` first: it writes the patterns and
        builds the region-of-interest mask for the autoexposure.

        Args:
            capture_background_image: Capture a background frame after the
                patterns, to be subtracted from every image during training. The frame
                is taken immediately, so **block the beam before calling** if this is
                set. Defaults to False, which leaves the background at zero. This used
                to prompt on stdin, which made the method unusable from a notebook or a
                headless run.

        Returns:
            DatasetDescriptor: The manifest for the captured dataset. It names the
            sample files relative to this generator's ``dataset_directory``.
        """
        if self.roi_mask is None:
            raise RuntimeError(
                "No region-of-interest mask yet. Call generate_phase_patterns() "
                "before capture_camera_images()."
            )

        for i, filename in enumerate(self.data_filenames):
            print(
                f"Displaying phase pattern {i + 1} of "
                f"{self.number_of_random_patterns}"
            )

            phase_pattern = np.load(self.dataset_directory / filename["phase_pattern"])

            self.slm.set_phase(phase_pattern)

            if i == 0:
                roi = ROI.detect(self.roi_mask, pad=0)
                self.metadata["exposure_time"] = self.camera.autoexpose(
                    set_fraction=0.95, roi=roi, mask=roi.crop(self.roi_mask)
                )

            camera_image_filename = f"camera_image_{i}.npy"

            print(
                f"Capturing camera image {i + 1} of "
                f"{self.number_of_random_patterns}"
            )

            camera_image = self.camera.get_image()

            # np.save serialises synchronously, so the frame needs no copy here.
            np.save(self.dataset_directory / camera_image_filename, camera_image)

            self.data_filenames[i]["camera_image"] = camera_image_filename

        if capture_background_image:
            self.camera_background_image = np.asarray(
                self.camera.get_image(), dtype=float
            ).copy()

        return DatasetDescriptor(
            timestamp=datetime.now(),
            phase_pattern_type=self.phase_pattern_type,
            number_of_patterns=self.number_of_random_patterns,
            slm_data=SLMData.from_slm(self.slm),
            camera_data=CameraData.from_camera(self.camera),
            camera_mapping=self.camera_mapping,
            roi_mask=self.roi_mask,
            data_filenames=self.data_filenames,
            camera_background_image=self.camera_background_image,
            benchmark_calibration=self.benchmark_calibration,
            metadata=dict(self.metadata),
        )
