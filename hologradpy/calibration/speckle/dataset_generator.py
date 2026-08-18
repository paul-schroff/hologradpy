from __future__ import annotations
from typing import Dict, List, Literal

import os
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import torch

from .records import SpeckleCaptureData

from ...hardware import Camera, SLM
from ...hardware.camera import CameraData
from ...hardware.slm import SLMData

from ..wavefront.abstract import WavefrontCalibrationData

from ..camera_mapping import CameraMapping

from ...datasets import CaptureStore
from ...profiles.masks import circular_mask, elliptical_mask
from ...profiles.phase import band_limited_random_phase
from ...roi import ROI
from ...fourier_optics import fourier_lens_pixel_size
from ...grids import get_pixel_grid, get_spatial_grid, pixel_to_metres
from ...utils import progress


class DatasetGenerator:
    """Generate random SLM phase patterns and capture their camera speckle images."""

    def __init__(
        self,
        slm: SLM,
        camera: Camera,
        camera_mapping: CameraMapping,
        focal_length: float,
        dataset_path: str | os.PathLike,
        number_of_random_patterns: int = 1,
    ) -> None:
        self.slm: SLM = slm
        self.camera: Camera = camera
        self.camera_mapping: CameraMapping = camera_mapping
        self.focal_length: float = focal_length
        self.dataset_path: Path = Path(dataset_path)
        self.number_of_random_patterns: int = number_of_random_patterns
        self.benchmark_calibration: WavefrontCalibrationData | None = None

        self.phase_patterns: List[NDArray[np.float_]] = []

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
        benchmark_calibration: WavefrontCalibrationData | None = None,
        seed: int | None = None,
        pattern: Literal["band_limited", "uniform"] = "band_limited",
    ) -> SpeckleCaptureData:
        """Generate the patterns and capture their frames into one dataset file.

        The whole capture in one call. :meth:`generate_phase_patterns` and
        :meth:`capture_camera_images` remain separately callable for the cases that need
        to do something between the two, for example inspecting the patterns before they
        reach the SLM.

        Args:
            extent: Full width ``(y, x)`` of the speckle at the camera, in metres,
                setting both the pattern band limit and the region of interest. A width
                rather than a radius, so it compares directly against the sensor size.
                Defaults to the largest speckle that fits on the sensor.
            benchmark_calibration: An existing calibration to add to every pattern, for
                measuring the residual of a previous fit.
            seed: Seed for the pattern noise. Leave as None to seed from the system
                entropy, which makes the dataset irreproducible.
            pattern: How each pattern is drawn, passed to
                :meth:`generate_phase_patterns`.

        Returns:
            SpeckleCaptureData: What describes the capture, which is also written inside
            the dataset file so it can be reopened without it.
        """
        self.generate_phase_patterns(
            extent,
            benchmark_calibration=benchmark_calibration,
            seed=seed,
            pattern=pattern,
        )
        return self.capture_camera_images()

    def largest_extent_on_sensor(self) -> tuple[float, float]:
        """The widest speckle ``(y, x)``, in metres, that still fits on the sensor.

        Returns:
            tuple[float, float]: Full width per axis, twice the distance from the zeroth
            order to the nearest sensor edge along that axis.

        Raises:
            ValueError: If the zeroth order lies off the sensor, where no speckle
                centered on it fits and an extent has to be chosen deliberately.
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
                f"{tuple(self.camera.resolution)} sensor, so no speckle centered on it "
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
        verbose: bool = True,
        pattern: Literal["band_limited", "uniform"] = "band_limited",
    ) -> None:
        """Generate a set of random phase patterns.

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
            pattern: ``"band_limited"`` draws the smooth speckle a wavefront fit wants.
                ``"uniform"`` draws every SLM pixel independently, so neighbouring
                pixels differ as much as they can, which is what excites pixel
                crosstalk. A uniform pattern fills the whole addressable area, so
                ``extent`` says nothing about where its light goes and the region of
                interest becomes the sensor minus the zeroth order.
        """
        uniform = pattern == "uniform"
        if extent is None and not uniform:
            extent = self.largest_extent_on_sensor()

        self.metadata["speckle_extent"] = extent
        self.metadata["seed"] = seed
        self.phase_pattern_type = pattern

        half_extent = None if extent is None else tuple(size / 2 for size in extent)

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

        if half_extent is None:
            # A uniform pattern is not band limited, so there is no band to size.
            radius_fft_pixels = None
            band_mask = torch.ones(tuple(self.slm.resolution), dtype=torch.bool)
        else:
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
            pixel_grid = get_pixel_grid(tuple(self.slm.resolution))
            band_mask = elliptical_mask(
                *pixel_grid,
                radius_x=radius_fft_pixels[1],
                radius_y=radius_fft_pixels[0],
            )
        self.metadata["band_radius_fft_pixels"] = radius_fft_pixels

        generator = torch.Generator(device=band_mask.device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        self.phase_patterns = []
        for _ in progress(
            range(self.number_of_random_patterns),
            description="Generating phase patterns",
            verbose=verbose,
        ):
            if uniform:
                phase = (
                    torch.rand(
                        tuple(self.slm.resolution),
                        generator=generator,
                        device=band_mask.device,
                    )
                    * 2
                    * torch.pi
                )
            else:
                phase = band_limited_random_phase(band_mask, generator=generator)
            phase = np.remainder(
                phase.cpu().numpy() + benchmark_phase, 2 * np.pi
            )

            self.phase_patterns.append(phase)

        camera_grid = get_spatial_grid(self.camera.resolution, self.camera.pixel_size)

        zeroth = self.camera_mapping.zeroth_order_position
        # (row, col) to the (x, y) the conversion takes, and back to (y, x) shifts.
        shift_x, shift_y = pixel_to_metres(
            (zeroth[1], zeroth[0]), self.camera.pixel_size, self.camera.resolution
        )

        if uniform:
            # The light fills everything the SLM can reach, so the only thing to keep
            # out of the region is the undiffracted spot.
            speckle_mask = torch.ones(
                tuple(self.camera.resolution),
                dtype=torch.bool,
                device=camera_grid[0].device,
            )
        else:
            speckle_mask = elliptical_mask(
                *camera_grid,
                radius_x=half_extent[1],
                radius_y=half_extent[0],
                shift_x=shift_x,
                shift_y=shift_y,
            )

        zeroth_order_mask = circular_mask(
            *camera_grid,
            4 * self.camera_mapping.spot_fit.waist,
            shift_x=shift_x,
            shift_y=shift_y,
        )

        self.roi_mask = (speckle_mask & ~zeroth_order_mask).cpu().numpy()

    def capture_camera_images(self, verbose: bool = True) -> SpeckleCaptureData:
        """Display every generated phase pattern and capture the camera speckle.

        Call :meth:`generate_phase_patterns` first: it generates the patterns and the
        region-of-interest needed for the autoexposure.

        The frames stream into the dataset file as they are captured, so a run that dies
        partway leaves the frames it took. The exposure is set before the file is
        opened, because everything but the streamed frames goes into the tree first.

        Returns:
            SpeckleCaptureData: What describes the capture, also written inside
            the file.
        """
        if self.roi_mask is None:
            raise RuntimeError(
                "No region-of-interest mask yet. Call generate_phase_patterns() "
                "before capture_camera_images()."
            )

        # Exposed on the first pattern, ahead of the capture proper: the exposure goes
        # into the record, and the record is written before the first frame.
        self.slm.set_phase(self.phase_patterns[0])
        roi = ROI.detect(self.roi_mask, pad=0)
        self.metadata["exposure_time"] = self.camera.autoexpose(
            set_fraction=0.95, roi=roi, mask=roi.crop(self.roi_mask)
        )

        bitdepth = self.slm.bitdepth
        patterns = [
            self.slm.phase_to_levels(pattern) for pattern in self.phase_patterns
        ]

        capture_data = self._capture_data()
        with CaptureStore.capture(
            self.dataset_path,
            capture_data,
            frame_shape=tuple(self.camera.resolution),
            slm_levels=patterns,
            phase_bitdepth=bitdepth,
        ) as store:
            for pattern in progress(
                self.phase_patterns,
                description="Capturing camera images",
                verbose=verbose,
            ):
                self.slm.set_phase(pattern)
                store.append(self.camera.get_image())

        return capture_data

    def _capture_data(self) -> SpeckleCaptureData:
        return SpeckleCaptureData(
            timestamp=datetime.now(),
            phase_pattern_type=self.phase_pattern_type,
            slm_data=SLMData.from_slm(self.slm),
            camera_data=CameraData.from_camera(self.camera),
            # Lean: the mapping's diagnostic frames are its own business, not
            # something every dataset that references it should carry.
            camera_mapping=self.camera_mapping.lean(),
            roi_mask=self.roi_mask,
            benchmark_calibration=self.benchmark_calibration,
            metadata=dict(self.metadata),
        )
