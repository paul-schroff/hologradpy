from __future__ import annotations
from typing import Literal

import os
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import torch

from .records import SpeckleCaptureData

from ...hardware import Camera, SLM, as_camera, as_slm
from ...hardware.camera import CameraData
from ...hardware.slm import SLMData

from ..wavefront.abstract import WavefrontCalibrationData

from ..camera_mapping import CameraMapping
from ..spot_detection import tilt_to_sensor_center

from ...datasets import CaptureStore
from ...profiles.masks import circular_mask, elliptical_mask
from ...profiles.phase import band_limited_random_phase, linear_phase
from ...roi import ROI
from ...fourier_optics import addressable_half_extent, fourier_lens_pixel_size
from ...grids import get_pixel_grid, get_spatial_grid, pixel_to_metres
from ...utils import as_image, progress

# The largest number of exposure steps when autoexposing on the first pattern.
_AUTOEXPOSURE_MAX_ITERATIONS = 10


class DatasetGenerator:
    """Generate random SLM phase patterns and capture their camera speckle images.

    The patterns and the region of interest are defined on the whole sensor,
    independent of the camera's region of interest. Every frame is captured from the
    whole sensor. A linear phase, the tilt, can move the speckle away from the zeroth
    order, and the camera mapping carries the speckle region onto the sensor.
    """

    def __init__(
        self,
        slm: SLM,
        camera: Camera,
        camera_mapping: CameraMapping,
        focal_length: float,
        dataset_path: str | os.PathLike,
        number_of_random_patterns: int = 1,
        zeroth_order_mask_waists: float = 4.0,
    ) -> None:
        """
        Args:
            slm: The SLM displaying the patterns, or a driver that
                :func:`~hologradpy.hardware.as_native.as_slm` wraps.
            camera: The camera capturing the speckle, or a driver that
                :func:`~hologradpy.hardware.as_native.as_camera` wraps.
            camera_mapping: How the camera sits relative to the model. It places the
                zeroth order on the sensor, and is checked against ``camera``
                (:meth:`~hologradpy.calibration.camera_mapping.CameraMapping.check_camera`).
            focal_length: The focal length of the Fourier lens in metres.
            dataset_path: The dataset file holding the frames.
            number_of_random_patterns: How many patterns to generate and capture.
            zeroth_order_mask_waists: Radius of the disc kept out of the region of
                interest around the zeroth order, in fitted focal-spot waists
                (``camera_mapping.spot_fit.waist``). The model does not predict the
                undiffracted light, so the disc has to cover its wings as well as its
                core. A tilted speckle loses a disc of the same radius around its
                centre. Defaults to 4.

        Raises:
            ValueError: ``zeroth_order_mask_waists`` is negative.
        """
        if zeroth_order_mask_waists < 0:
            raise ValueError(
                "zeroth_order_mask_waists is a radius, so it cannot be negative, got "
                f"{zeroth_order_mask_waists}."
            )
        self.slm: SLM = as_slm(slm)
        self.camera: Camera = as_camera(camera)
        camera_mapping.check_camera(self.camera)
        self.camera_mapping: CameraMapping = camera_mapping
        self.focal_length: float = focal_length
        self.dataset_path: Path = Path(dataset_path)
        self.number_of_random_patterns: int = number_of_random_patterns
        self.zeroth_order_mask_waists: float = float(zeroth_order_mask_waists)
        self.benchmark_calibration: WavefrontCalibrationData | None = None

        self.phase_patterns: list[NDArray[np.float64]] = []

        self.phase_pattern_type: str = "band_limited_random"
        self.metadata: dict[str, tuple[float, float] | float | int | None] = {
            "band_radius_bins": None,
            "speckle_extent": None,
            "speckle_tilt": None,
            "seed": None,
            "exposure_time": None,
        }
        self.roi_mask: NDArray[np.bool_] | None = None

    def generate_dataset(
        self,
        extent: tuple[float, float] | None = None,
        tilt: tuple[float, float] | None = None,
        benchmark_calibration: WavefrontCalibrationData | None = None,
        seed: int | None = None,
        pattern: Literal["band_limited", "uniform"] = "band_limited",
        set_fraction: float = 0.75,
    ) -> SpeckleCaptureData:
        """Generate the patterns and capture their frames into one dataset file.

        The whole capture in one call. :meth:`generate_phase_patterns` and
        :meth:`capture_camera_images` remain separately callable, so a step can run
        between the two. For example, the patterns can be inspected before they reach
        the SLM.

        Args:
            extent: Full width ``(y, x)`` of the speckle in the image plane, in metres.
                It sets both the pattern band limit and the region of interest.
                Defaults to the largest speckle whose image fits on the sensor.
            tilt: Image-plane position ``(x, y)`` of the speckle centre in metres,
                measured from the zeroth order, passed to
                :meth:`generate_phase_patterns`. None, the default, steers the speckle
                to the sensor centre, and ``(0.0, 0.0)`` keeps it on the zeroth order.
            benchmark_calibration: An existing calibration whose correction, the
                negative of its phase, is added to every pattern, for measuring the
                residual of a previous fit.
            seed: Seed for the pattern noise. Leave as None to seed from the system
                entropy, which makes the dataset irreproducible.
            pattern: How each pattern is drawn, passed to
                :meth:`generate_phase_patterns`.
            set_fraction: The brightest speckle of the first pattern is exposed to this
                fraction of full scale. Passed to :meth:`capture_camera_images`.

        Returns:
            SpeckleCaptureData: The record of the capture. A copy is written inside the
            dataset file, so the file can be reopened on its own.
        """
        self.generate_phase_patterns(
            extent,
            tilt=tilt,
            benchmark_calibration=benchmark_calibration,
            seed=seed,
            pattern=pattern,
        )
        return self.capture_camera_images(set_fraction=set_fraction)

    def largest_extent_on_sensor(
        self, tilt: tuple[float, float] | None = None
    ) -> tuple[float, float]:
        """The widest speckle ``(y, x)`` around ``tilt``, in image-plane metres, whose
        image fits on the sensor.

        Args:
            tilt: Image-plane position ``(x, y)`` of the speckle centre in metres,
                measured from the zeroth order. None, the default, is the position of
                the sensor centre.

        Returns:
            tuple[float, float]: Full width per axis.

        Raises:
            ValueError: The speckle centre lies off the sensor, where no speckle centred
                on it fits, or the tilt leaves no room inside the addressable field.
        """
        if tilt is None:
            tilt = tilt_to_sensor_center(self.camera, self.camera_mapping)
        sensor_resolution = np.asarray(self.camera.sensor_resolution, dtype=np.float64)
        tilt = np.asarray(tilt, dtype=np.float64)
        # The sensor (row, column) of the speckle centre, and of a step of one metre
        # along the image-plane x and y, which gives the linear part of the mapping.
        center, step_x, step_y = self.camera_mapping.image_plane_to_sensor(
            np.stack([tilt, tilt + (1.0, 0.0), tilt + (0.0, 1.0)])
        )

        margins = np.minimum(center, sensor_resolution - center)
        if np.any(margins <= 0):
            raise ValueError(
                f"The speckle centre sits at {tuple(float(c) for c in center)} on a "
                f"{tuple(int(n) for n in sensor_resolution)} sensor, so no speckle "
                "centred on it fits. Pass an extent explicitly, or a tilt that lands "
                "on the sensor."
            )

        # Sensor (row, column) pixels per image-plane metre, along x and along y.
        pixels_per_metre_x = step_x - center
        pixels_per_metre_y = step_y - center
        semi_axes = np.array(
            [
                margins[1] / np.linalg.norm(pixels_per_metre_x),
                margins[0] / np.linalg.norm(pixels_per_metre_y),
            ]
        )
        # How far the image of the ellipse reaches along the sensor rows and columns.
        reach = np.hypot(
            pixels_per_metre_x * semi_axes[0], pixels_per_metre_y * semi_axes[1]
        )
        semi_x, semi_y = semi_axes * np.min(margins / reach)

        addressable_x, addressable_y = addressable_half_extent(
            self.slm.wavelength, self.focal_length, self.slm.pixel_size
        )
        semi_x = min(semi_x, addressable_x - abs(tilt[0]))
        semi_y = min(semi_y, addressable_y - abs(tilt[1]))
        if semi_x <= 0 or semi_y <= 0:
            raise ValueError(
                f"The tilt {tuple(float(t) for t in tilt)} m leaves no room inside the "
                f"field the SLM addresses, {addressable_x:.3g} m and "
                f"{addressable_y:.3g} m from the zeroth order along x and y."
            )
        return (2 * float(semi_y), 2 * float(semi_x))

    def generate_phase_patterns(
        self,
        extent: tuple[float, float] | None = None,
        tilt: tuple[float, float] | None = None,
        benchmark_calibration: WavefrontCalibrationData | None = None,
        seed: int | None = None,
        verbose: bool = True,
        pattern: Literal["band_limited", "uniform"] = "band_limited",
    ) -> None:
        """Generate a set of random phase patterns.

        Each pattern is white noise band limited to the requested image-plane extent,
        by :func:`~hologradpy.profiles.phase.band_limited_random_phase`. The Fourier
        lens transforms the SLM plane into the image plane, so the SLM's own FFT plane
        is the image plane up to a scale, and the band limit can be sized directly in
        image-plane metres. The linear phase of the tilt moves the speckle away from
        the zeroth order.

        Args:
            extent: Full width ``(y, x)`` of the speckle in the image plane, in metres,
                which sets both the band limit and the region of interest. It is a
                width, not a radius, so at unit magnification between the image plane
                and the camera it compares directly against the sensor size. Defaults
                to the largest speckle that fits on the sensor
                (:meth:`largest_extent_on_sensor`).
            tilt: Image-plane position ``(x, y)`` of the speckle centre in metres,
                measured from the zeroth order, as
                :func:`~hologradpy.profiles.phase.linear_phase` takes it. None, the
                default, steers the speckle to the sensor centre, and ``(0.0, 0.0)``
                keeps it on the zeroth order.
            benchmark_calibration: An existing calibration whose correction, the
                negative of its phase, is added to every pattern, for measuring the
                residual of a previous fit.
            seed: Seed for the pattern noise. One generator produces every pattern, so
                they differ from one another and the whole set is reproducible. Leave
                as None to seed from the system entropy, which makes the dataset
                irreproducible.
            verbose: Show a progress bar while the patterns are generated. Defaults
                to True.
            pattern: ``"band_limited"`` draws the smooth speckle a wavefront fit wants.
                ``"uniform"`` draws every SLM pixel independently, so neighbouring
                pixels differ as much as they can, which is what excites pixel
                crosstalk. A uniform pattern fills the whole addressable area, so
                ``extent`` says nothing about where its light goes and the region of
                interest becomes the sensor minus the zeroth order.

        Raises:
            ValueError: The requested speckle region reaches past the field the SLM
                addresses, ``wavelength * focal_length / (2 * pixel_size)`` from the
                zeroth order, where the band of the patterns aliases.
        """
        uniform = pattern == "uniform"
        if uniform:
            tilt = (0.0, 0.0)
        elif tilt is None:
            tilt = tilt_to_sensor_center(self.camera, self.camera_mapping)
        tilt = (float(tilt[0]), float(tilt[1]))

        if extent is None and not uniform:
            extent = self.largest_extent_on_sensor(tilt)
        elif extent is not None and not uniform:
            addressable_x, addressable_y = addressable_half_extent(
                self.slm.wavelength, self.focal_length, self.slm.pixel_size
            )
            if (
                abs(tilt[0]) + extent[1] / 2 > addressable_x
                or abs(tilt[1]) + extent[0] / 2 > addressable_y
            ):
                raise ValueError(
                    f"A speckle {tuple(extent)} m wide around {tilt} m reaches past "
                    f"the field the SLM addresses, {addressable_x:.3g} m and "
                    f"{addressable_y:.3g} m from the zeroth order along x and y, where "
                    "the band of the patterns aliases. Pass a smaller extent or tilt."
                )

        self.metadata["speckle_extent"] = extent
        self.metadata["speckle_tilt"] = None if uniform else tilt
        self.metadata["seed"] = seed
        self.phase_pattern_type = pattern

        half_extent = None if extent is None else tuple(size / 2 for size in extent)

        self.benchmark_calibration = benchmark_calibration
        if self.benchmark_calibration is None:
            benchmark_phase = np.zeros(self.slm.resolution)
        else:
            # The record holds the incident field, so its correction is the negative
            # of its phase.
            benchmark_phase = -np.angle(
                as_image(self.benchmark_calibration.complex_amplitude)
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

        slm_grid = get_spatial_grid(
            tuple(self.slm.resolution),
            torch.tensor(tuple(self.slm.pixel_size), dtype=torch.float64),
        )
        tilt_phase = linear_phase(
            *slm_grid,
            *tilt,
            tilt_units="metres",
            wavenumber=2 * np.pi / self.slm.wavelength,
            focal_length=self.focal_length,
        ).numpy()

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
                phase.cpu().numpy() + tilt_phase + benchmark_phase, 2 * np.pi
            )

            self.phase_patterns.append(phase)

        # Every frame is taken from the whole sensor, so the region is laid out on it.
        sensor_resolution = tuple(self.camera.sensor_resolution)
        camera_grid = get_spatial_grid(sensor_resolution, self.camera.pixel_size)

        zeroth = self.camera_mapping.zeroth_order_position
        # (row, col) to the (x, y) the conversion takes, and back to (y, x) shifts.
        shift_x, shift_y = pixel_to_metres(
            (zeroth[1], zeroth[0]), self.camera.pixel_size, sensor_resolution
        )

        if uniform:
            # The light fills everything the SLM can reach, so the only thing to keep
            # out of the region is the undiffracted spot.
            speckle_mask = torch.ones(
                sensor_resolution,
                dtype=torch.bool,
                device=camera_grid[0].device,
            )
        else:
            # Each sensor pixel is placed in the image plane through the camera
            # mapping, so the region follows the camera's rotation, magnification and
            # offset.
            rows, columns = np.indices(sensor_resolution)
            positions = self.camera_mapping.sensor_to_image_plane(
                np.stack([rows.ravel(), columns.ravel()], axis=1)
            )
            speckle_mask = torch.as_tensor(
                elliptical_mask(
                    positions[:, 0].reshape(sensor_resolution),
                    positions[:, 1].reshape(sensor_resolution),
                    radius_x=half_extent[1],
                    radius_y=half_extent[0],
                    shift_x=tilt[0],
                    shift_y=tilt[1],
                ),
                device=camera_grid[0].device,
            )

        zeroth_order_mask_radius = (
            self.zeroth_order_mask_waists * self.camera_mapping.spot_fit.waist
        )
        self.metadata["zeroth_order_mask_radius"] = zeroth_order_mask_radius
        excluded = circular_mask(
            *camera_grid,
            zeroth_order_mask_radius,
            shift_x=shift_x,
            shift_y=shift_y,
        )
        if not uniform and any(tilt):
            centre_row, centre_column = self.camera_mapping.image_plane_to_sensor(
                tilt
            )[0]
            centre_x, centre_y = pixel_to_metres(
                (centre_column, centre_row), self.camera.pixel_size, sensor_resolution
            )
            excluded = excluded | circular_mask(
                *camera_grid,
                zeroth_order_mask_radius,
                shift_x=centre_x,
                shift_y=centre_y,
            )

        self.roi_mask = (speckle_mask & ~excluded).cpu().numpy()
        if not self.roi_mask.any():
            raise ValueError(
                "The region of interest is empty, since the speckle region lies inside "
                "the discs kept out around the zeroth order and the speckle centre. "
                "Pass a larger extent or a smaller zeroth_order_mask_waists."
            )

    def capture_camera_images(
        self,
        verbose: bool = True,
        set_fraction: float = 0.75,
    ) -> SpeckleCaptureData:
        """Display every generated phase pattern and capture the camera speckle.

        Call :meth:`generate_phase_patterns` first. It generates the patterns and the
        region of interest for the autoexposure.

        The exposure and the gain are metered on the first pattern
        (:meth:`_expose_for_patterns`) and held for the whole capture. The frames stream
        into the dataset file as they are captured, so an interrupted run keeps its
        captured frames. The exposure is set before the file is opened, because
        everything but the streamed frames goes into the tree first.

        The whole sensor is read out for the capture, and the camera's exposure, gain
        and region of interest are put back afterwards
        (:meth:`~hologradpy.hardware.camera.Camera.preserve_exposure_gain_and_roi`).
        Frames are stored raw. The metadata records the exposure and how it was metered,
        and the camera record holds the gain, so a fit can subtract a background
        measured at that exposure.

        Args:
            verbose: Show a progress bar while the frames are captured.
            set_fraction: The brightest speckle of the first pattern is exposed to this
                fraction of full scale, leaving headroom for brighter speckle in later
                patterns.

        Returns:
            SpeckleCaptureData: The record of the capture. A copy is also written inside
            the file.

        Raises:
            RuntimeError: The patterns have not been generated.
        """
        if self.roi_mask is None:
            raise RuntimeError(
                "No region-of-interest mask yet. Call generate_phase_patterns() "
                "before capture_camera_images()."
            )

        bitdepth = self.slm.bitdepth
        patterns = [
            self.slm.phase_to_levels(pattern) for pattern in self.phase_patterns
        ]

        with self.camera.preserve_exposure_gain_and_roi(full_sensor=True):
            # The exposure is metered before the capture, since it goes into the
            # record. The record is written before the first frame.
            self.metadata["exposure_time"] = self._expose_for_patterns(set_fraction)
            self.metadata["exposure_set_fraction"] = float(set_fraction)

            capture_data = self._capture_data()
            with CaptureStore.capture(
                self.dataset_path,
                capture_data,
                frame_shape=tuple(self.camera.sensor_resolution),
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

    def _expose_for_patterns(self, set_fraction: float) -> float:
        """Meter the exposure on the first pattern.

        The camera autoexposes on the region of interest of the first pattern, in up
        to ``_AUTOEXPOSURE_MAX_ITERATIONS`` steps, and stays at the final exposure and
        gain.

        Args:
            set_fraction: The peak of the first pattern is exposed to this fraction of
                full scale.

        Returns:
            float: The exposure in seconds, as read back from the camera.
        """
        roi = ROI.detect(self.roi_mask, pad=0)
        mask = roi.crop(self.roi_mask)

        self.slm.set_phase(self.phase_patterns[0])
        return self.camera.autoexpose(
            set_fraction=set_fraction,
            roi=roi,
            mask=mask,
            max_iterations=_AUTOEXPOSURE_MAX_ITERATIONS,
        )

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
