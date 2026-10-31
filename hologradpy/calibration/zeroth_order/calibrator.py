"""The zeroth order a hologram leaves, measured with replicas of the beam."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from ..abstract import CalibratorBase
from ..camera_mapping import CameraMapping
from ..model_fit import field_on_frames
from ...analysis.fitting import fit_translation
from ...datasets import CaptureStore
from ...fourier_optics import intensity_nyquist_pixel_size
from ...fourier_transforms import translate_real
from ...grids import odd_sample_count
from ...hardware import Camera, SLM
from ...hardware.camera import CameraData
from ...hardware.slm import SLMData
from ...optics import SLMFourierLensModel
from ...optics.complex_amplitude import ComplexAmplitude
from ...optics.modules.zeroth_order import ZEROTH_ORDER_LAYER, ZerothOrderField
from ...phase_levels import LinearResponse
from ...roi import ROI
from ...utils import as_image, gpu_to_numpy, progress
from .fitting import fit_kernel_and_wrap_weight, fit_missed_field, probe_phase
from .records import (
    BACKPLANE_REPLICA,
    FLAT_REPLICA,
    ZerothOrderCalibrationData,
    ZerothOrderCaptureData,
)
from .visualizer import ZerothOrderVisualizationData

# The largest number of exposure steps of each autoexposure.
_AUTOEXPOSURE_MAX_ITERATIONS = 10
# The registrations of a fit. The first registers on the model without a zeroth order,
# and the second on the model with the zeroth order the first fitted.
_REGISTRATIONS = 2


class ZerothOrderCalibrator(CalibratorBase):
    """Measure the zeroth order a hologram leaves, as
    :class:`~hologradpy.optics.modules.zeroth_order.ZerothOrderField` adds it to the
    model.

    The SLM displays blazed gratings, each summed with a replica of the beam at
    ``number_of_pistons`` pistons, as Eq. 9 of Cerioni et al. (arXiv:2601.20376) sums a
    hologram and its corrective field (:func:`~.fitting.probe_phase`). Two replicas are
    used. The flat replica has the beam's phase taken out and focuses to the tightest
    spot. The backplane replica keeps the phase of the backplane, the beam's phase
    without the SLM's vendor correction, and focuses as the light at the wraps does.
    On every pixel of the zeroth order's disc, the frames of each grating
    give the field the model misses (:func:`~.fitting.fit_missed_field`). The kernel of
    the field no hologram changes and the weight of the light at the wraps are then
    fitted to the fields of all gratings (:func:`~.fitting.fit_kernel_and_wrap_weight`).

    Every probe is captured twice, back to back. The first frame is exposed on the
    disc, where the grating's spot is overexposed. The second is exposed on the whole
    sensor, and :func:`~hologradpy.analysis.fitting.fit_translation` registers it on
    the model's intensity outside the disc. The first frame takes the counts of the
    second on its overexposed pixels, scaled by the ratio of the exposures, and is then
    moved by the same translation. The scale of the frames comes from the flat replica
    at one and at two probe amplitudes on the first grating.

    The calibrator inserts a :class:`ZerothOrderField` into the model after the virtual
    SLM, named :data:`ZEROTH_ORDER_LAYER`. It adds nothing until the fit, and holds the
    fitted kernel and weight afterwards. The replicas are built from the model's beam,
    so the beam is calibrated first. On a simulated SLM, a phase error of the beam of
    0.01 waves RMS left 0.7 to 1.4 % of the missed power unpredicted.
    """

    def __init__(
        self,
        slm: SLM,
        camera: Camera,
        slm_camera_model: SLMFourierLensModel,
        camera_mapping: CameraMapping,
        dataset_path: str | os.PathLike,
        grating_tilts: Sequence[tuple[float, float]],
        zeroth_order_mask_waists: float = 4.0,
        probe_amplitude: float | None = None,
        number_of_pistons: int = 4,
    ) -> None:
        """
        Args:
            slm: The SLM to drive.
            camera: The camera imaging its focal plane.
            slm_camera_model: The model of this setup, with its beam calibrated and the
                whole sensor as its output plane. It carries no zeroth-order field yet.
            camera_mapping: The camera mapping the model's partial affine is calibrated
                from. It also places the zeroth order on the sensor.
            dataset_path: The dataset file, holding the captured frames and their
                description.
            grating_tilts: The ``(x, y)`` position in metres, from the zeroth order,
                that each grating steers its spot to. A blazed grating wraps more often
                the further its spot lies from the zeroth order, so the positions set
                the wrap counts. These are chosen to span the wrap counts of the
                holograms the calibration serves. The fit tells the light at the wraps
                from the fixed field by the different wrap counts, so it needs at least
                two gratings. Each spot lies far enough from the disc to leave it dark,
                since the light at the wraps around a spot changes from probe to probe.
            zeroth_order_mask_waists: The radius of the zeroth order's disc in fitted
                focal-spot waists (``camera_mapping.spot_fit.waist``), the disc the
                speckle calibration leaves out of its region of interest.
            probe_amplitude: The amplitude of the replica, as a multiple of the
                hologram's, or None to measure it (:meth:`capture`).
            number_of_pistons: The piston steps of a full cycle of each replica.

        Raises:
            ValueError: Fewer than two gratings or three pistons, a disc that is not
                positive, a model that already carries a zeroth-order field or outputs
                less than the whole sensor, or a grating with its spot off the sensor,
                inside the disc or beyond the field the SLM addresses.
        """
        if len(grating_tilts) < 2:
            raise ValueError(
                "The fit tells the light at the wraps from the fixed field by the "
                "different wrap counts of the gratings, so it needs at least two, got "
                f"{len(grating_tilts)}."
            )
        if number_of_pistons < 3:
            raise ValueError(
                "Three piston steps are the fewest that resolve both quadratures of "
                f"the missed field, got {number_of_pistons}."
            )
        if zeroth_order_mask_waists <= 0:
            raise ValueError(
                "zeroth_order_mask_waists is the radius of the measured disc, so it "
                f"has to be positive, got {zeroth_order_mask_waists}."
            )
        if slm_camera_model.has(ZerothOrderField):
            raise ValueError(
                "The model already carries a ZerothOrderField. The calibration "
                "measures the field the model misses, so it starts from a model "
                "without one."
            )
        super().__init__(slm, camera, slm_camera_model.device)
        camera_mapping.check_camera(self.camera)

        self.slm_camera_model: SLMFourierLensModel = slm_camera_model
        self.camera_mapping: CameraMapping = camera_mapping
        self.dataset_path: Path = Path(dataset_path)
        self.grating_tilts: NDArray[np.float64] = np.asarray(
            grating_tilts, dtype=np.float64
        ).reshape(-1, 2)
        self.zeroth_order_mask_waists: float = float(zeroth_order_mask_waists)
        self.probe_amplitude: float | None = (
            None if probe_amplitude is None else float(probe_amplitude)
        )
        self.number_of_pistons: int = int(number_of_pistons)
        self.focal_length: float = slm_camera_model.focal_length

        # The radius of the disc in camera-plane metres, as DatasetGenerator sizes it.
        self.region_radius: float = (
            self.zeroth_order_mask_waists * camera_mapping.spot_fit.waist
        )
        radius_pixels = self.region_radius / float(np.min(self.camera.pixel_size))
        # The zeroth order's disc, True on the sensor pixels it covers.
        self.region: NDArray[np.bool_] = camera_mapping.zeroth_order_mask(
            tuple(self.camera.sensor_resolution), radius_pixels
        )

        kernel_shape = self._kernel_shape(radius_pixels)
        # True on the kernel's samples inside the disc, the only ones the fit sets.
        self.kernel_support: Tensor = self._kernel_support(kernel_shape, radius_pixels)
        self.zeroth_order_field: ZerothOrderField = ZerothOrderField(
            self.focal_length,
            kernel=torch.zeros(kernel_shape, dtype=torch.complex64),
            added_phase=self._vendor_phase(),
        )
        slm_camera_model.insert_after(
            "virtual_slm", ZEROTH_ORDER_LAYER, self.zeroth_order_field
        )
        slm_camera_model.build()
        slm_camera_model.calibrate_from_mapping(camera_mapping)
        self._check_output_plane()
        self._check_gratings(radius_pixels)

        self.capture_data: ZerothOrderCaptureData | None = None
        self.calibration: ZerothOrderCalibrationData | None = None

    def _kernel_pixel_size(self) -> tuple[float, float]:
        """The ``(y, x)`` intensity Nyquist pitch of the SLM in metres, the kernel's
        pitch.
        """
        return tuple(
            float(
                intensity_nyquist_pixel_size(
                    self.slm.wavelength,
                    self.focal_length,
                    float(self.slm.pixel_size[axis]),
                    int(self.slm.resolution[axis]),
                )
            )
            for axis in range(2)
        )

    def _kernel_shape(self, radius_pixels: float) -> tuple[int, int]:
        """The ``(rows, columns)`` of a kernel at the intensity Nyquist pitch that
        covers the disc in the model's focal plane.

        The disc of ``radius_pixels`` sensor pixels holds the focal-plane points ``p``
        with ``|A p| <= r``, for ``A`` the camera mapping's
        :attr:`~.CameraMapping.sensor_pixels_per_metre`. Along each focal-plane axis it
        reaches ``r sqrt(((A^T A)^-1)_ii)``.
        """
        per_metre = self.camera_mapping.sensor_pixels_per_metre
        reach_x, reach_y = radius_pixels * np.sqrt(
            np.diag(np.linalg.inv(per_metre.T @ per_metre))
        )
        pitch_y, pitch_x = self._kernel_pixel_size()
        return (
            odd_sample_count(2 * float(reach_y), pitch_y),
            odd_sample_count(2 * float(reach_x), pitch_x),
        )

    def _kernel_support(
        self, shape: tuple[int, int], radius_pixels: float
    ) -> Tensor:
        """True on the kernel's samples that lie inside the disc.

        A sample outside the disc places its light where no probe measures it, so the
        fit leaves it at zero. The kernel's rows run down the model's focal plane, along
        minus y.
        """
        pitch_y, pitch_x = self._kernel_pixel_size()
        rows, columns = np.indices(shape)
        x = (columns - shape[1] // 2) * pitch_x
        y = -(rows - shape[0] // 2) * pitch_y
        on_sensor = np.einsum(
            "ij,jrc->irc", self.camera_mapping.sensor_pixels_per_metre, np.stack([x, y])
        )
        inside = np.hypot(on_sensor[0], on_sensor[1]) <= radius_pixels
        return torch.as_tensor(inside, device=self.device)

    def _vendor_phase(self) -> NDArray | None:
        """The phase the SLM adds to every hologram for its vendor correction, or None
        without one, as :meth:`~hologradpy.hardware.slm.SLM.set_phase` adds it.
        """
        levels = self.slm.vendor_correction
        if levels is None:
            return None
        return LinearResponse(bitdepth=self.slm.bitdepth).to_phase(levels)

    def _check_output_plane(self) -> None:
        """Raise unless the model outputs the whole sensor, as the frames cover it."""
        _, resolution = self.slm_camera_model.output_plane
        sensor_resolution = tuple(int(size) for size in self.camera.sensor_resolution)
        if tuple(resolution) != sensor_resolution:
            raise ValueError(
                f"The model's output is {tuple(resolution)} but the camera's sensor is "
                f"{sensor_resolution}. The frames are compared with the model pixel by "
                "pixel, so build the model with camera_resolution taken from "
                "camera.sensor_resolution."
            )

    def _check_gratings(self, radius_pixels: float) -> None:
        """Raise unless every grating's spot lies on the sensor, outside the disc and
        inside the field the SLM addresses.
        """
        height, width = (int(size) for size in self.camera.sensor_resolution)
        zeroth_order = np.asarray(self.camera_mapping.zeroth_order_position)
        reach_x, reach_y = self.slm_camera_model.addressable_half_extent()
        spots = self.camera_mapping.image_plane_to_sensor(self.grating_tilts)
        for tilt, spot in zip(self.grating_tilts, spots):
            on_sensor = 0 <= spot[0] <= width - 1 and 0 <= spot[1] <= height - 1
            outside = float(np.hypot(*(spot - zeroth_order))) > radius_pixels
            addressed = abs(tilt[0]) < reach_x and abs(tilt[1]) < reach_y
            if not (on_sensor and outside and addressed):
                raise ValueError(
                    f"The grating steering to {tuple(float(t) for t in tilt)} m puts "
                    f"its spot at (column, row) {tuple(float(s) for s in spot)}. The "
                    "spot has to lie on the sensor, outside the zeroth order's disc of "
                    f"{radius_pixels:.3g} pixels and inside the field the SLM "
                    f"addresses, {reach_x:.3g} m and {reach_y:.3g} m from the zeroth "
                    "order along x and y."
                )

    def _probe_table(self) -> NDArray[np.int64]:
        """One row per probe, as :attr:`ZerothOrderCaptureData.probes` holds it.

        Every grating takes both replicas at every piston step at one probe amplitude.
        The flat replica on the first grating follows at two probe amplitudes, for the
        scale of the frames.
        """
        rows = [
            (grating, replica, step, 1)
            for grating in range(len(self.grating_tilts))
            for replica in (FLAT_REPLICA, BACKPLANE_REPLICA)
            for step in range(self.number_of_pistons)
        ]
        rows += [(0, FLAT_REPLICA, step, 2) for step in range(self.number_of_pistons)]
        return np.asarray(rows, dtype=np.int64)

    def _grating_phase(self, grating: int) -> Tensor:
        """The phase of the blazed grating ``grating``, on the SLM's grid."""
        return self.slm_camera_model.steering_phase(
            tuple(float(t) for t in self.grating_tilts[grating])
        )

    def _probe_phases(
        self, table: NDArray[np.int64], amplitude: float
    ) -> list[NDArray]:
        """The phase in radians each probe of ``table`` displays.

        The SLM adds its vendor correction to every probe. The backplane replica takes
        the correction out, so it carries the phase the light at the wraps carries.
        """
        beam = as_image(self.slm_camera_model.slm_field.get_wavefront().detach())
        vendor_phase = self._vendor_phase()
        correctives = {
            FLAT_REPLICA: -torch.angle(beam),
            BACKPLANE_REPLICA: (
                torch.zeros_like(beam.real)
                if vendor_phase is None
                else -torch.as_tensor(vendor_phase, device=beam.device).to(
                    beam.real.dtype
                )
            ),
        }
        gratings = [
            self._grating_phase(index) for index in range(len(self.grating_tilts))
        ]
        return [
            gpu_to_numpy(
                probe_phase(
                    gratings[grating],
                    correctives[replica],
                    multiple * amplitude,
                    2 * math.pi * step / self.number_of_pistons,
                )
            )
            for grating, replica, step, multiple in table
        ]

    def _model_fields(self, levels: NDArray | Tensor) -> Tensor:
        """The model's field on the whole sensor for holograms of ``levels``,
        ``(N, H, W)``.
        """
        model = self.slm_camera_model
        levels = torch.as_tensor(np.asarray(levels)).reshape(-1, *self.slm.resolution)
        model.virtual_slm.set_levels(levels, self.slm.bitdepth)
        self.zeroth_order_field.update_wraps(model)
        with torch.no_grad():
            return field_on_frames(model())

    def _dark_field(self) -> ComplexAmplitude:
        """A zero incident field. Every stage ahead of the zeroth-order field multiplies
        the incident field, so the model's output for it is the zeroth-order field
        alone, carried through the lens.
        """
        incident = self.slm_camera_model.init_field
        return ComplexAmplitude(
            torch.zeros_like(incident.as_tensor()),
            incident.wavelength,
            incident.pixel_size,
        )

    def capture(
        self, set_fraction: float = 0.75, verbose: bool = True
    ) -> ZerothOrderCaptureData:
        """Display every probe and capture its two frames into the dataset file.

        The whole sensor is read out, and the camera's exposure, gain and region of
        interest are put back afterwards. Two equivalent exposures are set first and
        held through the capture
        (:meth:`~hologradpy.hardware.camera.Camera.set_equivalent_exposure`). The short
        one is an autoexposure on the whole sensor with the first grating alone. The
        long one is the shortest of the autoexposures on the disc with each probe of
        the flat replica at two probe amplitudes, the brightest probes there.

        Without a ``probe_amplitude``, the amplitude is measured on a frame of the
        first grating alone at the short exposure. It is ``2 sqrt(P_0 / P)``, Eq. 8 of
        Cerioni et al. with the factor of two that the phase of a sum costs. ``P_0`` is
        the power in the disc less the grating's own light there, and ``P`` the power
        of the grating's light on the sensor. The registration of the frame on the
        model's intensity (:func:`~hologradpy.analysis.fitting.fit_translation`) fits
        the counts per unit of model intensity and the background, which gives both.

        Args:
            set_fraction: The peak of each autoexposure is set to this fraction of full
                scale.
            verbose: Show a progress bar while the frames are captured.

        Returns:
            ZerothOrderCaptureData: The record of the capture, also written inside the
            dataset file.

        Raises:
            RuntimeError: No zeroth order rises above the background on the frame the
                amplitude is measured on.
        """
        roi = ROI.detect(self.region, pad=0)
        with (
            self.camera.preserve_exposure_gain_and_roi(full_sensor=True),
            self.slm_camera_model.virtual_slm.preserve_levels(),
        ):
            grating = gpu_to_numpy(self._grating_phase(0))
            self.slm.set_phase(grating)
            self.camera.autoexpose(
                set_fraction=set_fraction, max_iterations=_AUTOEXPOSURE_MAX_ITERATIONS
            )
            short_exposure = self.camera.get_equivalent_exposure()
            amplitude = self.probe_amplitude
            if amplitude is None:
                amplitude = self._measured_probe_amplitude(grating)

            table = self._probe_table()
            phases = self._probe_phases(table, amplitude)
            exposures = []
            for index in np.flatnonzero(table[:, 3] == 2):
                self.slm.set_phase(phases[index])
                self.camera.autoexpose(
                    set_fraction=set_fraction,
                    roi=roi,
                    mask=roi.crop(self.region),
                    max_iterations=_AUTOEXPOSURE_MAX_ITERATIONS,
                )
                exposures.append(self.camera.get_equivalent_exposure())
            long_exposure = min(exposures)
            self.camera.set_equivalent_exposure(long_exposure)

            capture_data = ZerothOrderCaptureData(
                timestamp=datetime.now(),
                slm_data=SLMData.from_slm(self.slm),
                camera_data=CameraData.from_camera(self.camera),
                camera_mapping=self.camera_mapping.lean(),
                region=self.region,
                grating_tilts=self.grating_tilts,
                probes=table,
                probe_amplitude=float(amplitude),
                number_of_pistons=self.number_of_pistons,
                metadata={
                    "long_exposure": float(long_exposure),
                    "short_exposure": float(short_exposure),
                    "exposure_set_fraction": float(set_fraction),
                    "saturation_level": float(self.camera.saturation_level),
                    "zeroth_order_mask_radius": float(self.region_radius),
                },
            )
            levels = [self.slm.phase_to_levels(phase) for phase in phases]
            with CaptureStore.capture(
                self.dataset_path,
                capture_data,
                frame_shape=tuple(self.camera.sensor_resolution),
                slm_levels=[level for level in levels for _ in range(2)],
                phase_bitdepth=self.slm.bitdepth,
            ) as store:
                for phase in progress(
                    phases, description="Capturing probes", verbose=verbose
                ):
                    self.slm.set_phase(phase)
                    self.camera.set_equivalent_exposure(long_exposure)
                    store.append(self.camera.get_image())
                    self.camera.set_equivalent_exposure(short_exposure)
                    store.append(self.camera.get_image())

        self.capture_data = capture_data
        return capture_data

    def _measured_probe_amplitude(self, grating: NDArray) -> float:
        """The probe amplitude from a frame of ``grating`` alone, which the SLM
        displays, at the camera's exposure (:meth:`capture`).
        """
        frame = torch.as_tensor(
            np.asarray(self.camera.get_image(), dtype=np.float32), device=self.device
        )
        intensity = self._model_fields(self.slm.phase_to_levels(grating))[0]
        intensity = intensity.abs().square().to(frame.dtype)
        region = torch.as_tensor(self.region, device=self.device)
        unsaturated = frame < self.camera.saturation_level
        fit = fit_translation(frame, intensity, ~region & unsaturated, fit_offset=True)
        # The zeroth order is the light in the disc less the grating's own, which the
        # model predicts.
        expected = translate_real(intensity, fit.shift)
        zeroth_order_power = float(
            (((frame - fit.offset) / fit.scale - expected) * region).sum()
        )
        if not zeroth_order_power > 0.0:
            raise RuntimeError(
                "No zeroth order rises above the background in the zeroth order's "
                "disc on a frame of the first grating. Pass probe_amplitude."
            )
        return 2.0 * math.sqrt(zeroth_order_power / float(intensity.sum()))

    def fit(
        self,
        capture_data: ZerothOrderCaptureData | None = None,
        contrast: float = 1e-3,
        verbose: bool = True,
    ) -> ZerothOrderCalibrationData:
        """Fit the kernel and the weight of the light at the wraps to a captured
        dataset, and give both to the model's zeroth-order field.

        Every pair of frames is registered (:meth:`capture`), and the frames are scaled
        into the model's units (:meth:`_frame_scale`). The missed field of every grating
        is solved on every pixel of the disc. An overexposed pixel of a frame is left
        out of the solve of that pixel. The kernel and the weight are then fitted to
        the fields of all gratings. The frames are registered and fitted a second time
        on the model with the zeroth order of the first fit
        (:meth:`_registered_frames`).

        Args:
            capture_data: The dataset to fit, or None for the one from the last call.
            contrast: The smallest variation of the probes' model fields that resolves a
                pixel, as a fraction of the largest in the disc
                (:func:`~.fitting.fit_missed_field`). The default comes from a
                simulated SLM.
            verbose: Show a progress bar while the frames are registered.

        Returns:
            ZerothOrderCalibrationData: The kernel, the weight and the figures of the
            fit, also kept on :attr:`calibration`.

        Raises:
            RuntimeError: No dataset has been captured or passed.
        """
        if capture_data is not None:
            self.capture_data = capture_data
        if self.capture_data is None:
            raise RuntimeError(
                "No dataset to fit. Capture one with calibrator.capture() and pass it "
                "as capture_data, or call calibrate() to do both."
            )
        data = self.capture_data
        model = self.slm_camera_model
        field = self.zeroth_order_field
        with torch.no_grad():
            field.kernel.zero_()
            field.wrap_weight.zero_()
        field.beam_at_wraps = None

        region = torch.as_tensor(np.asarray(data.region), device=self.device)
        box = ROI.detect(np.asarray(data.region), pad=0)
        region_in_box = box.crop(region)
        table = np.asarray(data.probes)

        with model.virtual_slm.preserve_levels():
            for registration in range(_REGISTRATIONS):
                frames, fields, valid, levels, shifts = self._registered_frames(
                    data, box, verbose
                )
                scale = self._frame_scale(table, frames, fields, valid, region_in_box)

                measured_fields, wrap_fields, masks, floors = [], [], [], []
                for grating in range(len(data.grating_tilts)):
                    chosen = np.flatnonzero(
                        (table[:, 0] == grating) & (table[:, 3] == 1)
                    )
                    missed = fit_missed_field(
                        frames[chosen] / scale,
                        fields[chosen],
                        region_in_box,
                        valid[chosen],
                        contrast,
                    )
                    measured_fields.append(missed.field)
                    masks.append(missed.resolved)
                    floors.append(float(missed.floor[missed.resolved].mean()))
                    wrap_fields.append(
                        self._wrap_fields(levels[chosen], box).mean(dim=0)
                    )

                # The levels on display play no part in the kernel's field, so one
                # hologram without its light at the wraps keeps every pass of the fit
                # to a single field.
                model.virtual_slm.set_phase(torch.zeros(tuple(self.slm.resolution)))
                field.beam_at_wraps = None
                forward, adjoint = self._kernel_field_on(box)
                measured_fields = torch.stack(measured_fields)
                wrap_fields = torch.stack(wrap_fields)
                masks = torch.stack(masks)
                result = fit_kernel_and_wrap_weight(
                    measured_fields, wrap_fields, masks, forward, adjoint
                )
                kernel = result.kernel * self.kernel_support
                with torch.no_grad():
                    fitted = (
                        forward(kernel)[None] + result.wrap_weight * wrap_fields
                    )
                    field.kernel.copy_(kernel)
                    field.wrap_weight.copy_(torch.tensor(result.wrap_weight))
        # The model holds no light at the wraps until the next hologram's are found.
        field.beam_at_wraps = None

        weights = masks.to(fitted.dtype)
        self.calibration = ZerothOrderCalibrationData(
            timestamp=datetime.now(),
            name="zeroth_order",
            focal_length=self.focal_length,
            kernel=gpu_to_numpy(kernel),
            kernel_pixel_size=tuple(field.kernel_pixel_size),
            wrap_weight=complex(result.wrap_weight),
            region=np.asarray(data.region, dtype=bool),
            added_phase=self._vendor_phase(),
            metadata={
                "scale": scale,
                "relative_residual": result.relative_residual,
                "floor": floors,
                "resolved_pixels": [int(mask.sum()) for mask in masks],
                "overexposed_measurements": int(
                    (~valid & region_in_box).sum()
                ),
                "registration_shifts": np.asarray(shifts, dtype=np.float64),
                "probe_amplitude": float(data.probe_amplitude),
                "grating_tilts": np.asarray(data.grating_tilts),
                "contrast": float(contrast),
            },
            visualization_data=ZerothOrderVisualizationData(
                measured_intensity=gpu_to_numpy(
                    (measured_fields * weights).abs().square()
                ),
                fitted_intensity=gpu_to_numpy((fitted * weights).abs().square()),
                kernel_intensity=gpu_to_numpy(kernel.abs().square()),
                wrap_weight=complex(result.wrap_weight),
                relative_residual=result.relative_residual,
            ),
        )
        return self.calibration

    def _registered_frames(
        self, data: ZerothOrderCaptureData, box: ROI, verbose: bool
    ) -> tuple[Tensor, Tensor, Tensor, NDArray, list[tuple[float, float]]]:
        """The first frame of every probe, registered and cropped to ``box``, with the
        model's field of the probe, the measurements to use, the levels and the fitted
        translation.

        The second frame is registered on the model's intensity outside the disc
        (:func:`~hologradpy.analysis.fitting.fit_translation`). The model's intensity
        holds the zeroth-order field as the model carries it, and the light at the
        wraps of that field lands on the grating's spot too. The model's field
        returned is the field without it, the field the zeroth order is measured
        against.

        Returns:
            tuple: The frames ``(P, h, w)`` in counts, the model's fields ``(P, h, w)``,
            True on the measurements to use ``(P, h, w)``, the levels ``(P, H, W)``,
            and the translations ``(columns, rows)`` in sensor pixels.
        """
        region = torch.as_tensor(np.asarray(data.region), device=self.device)
        saturation = float(data.metadata["saturation_level"])
        exposure_ratio = float(data.metadata["long_exposure"]) / float(
            data.metadata["short_exposure"]
        )
        zeroth_order = self.zeroth_order_field
        carries_zeroth_order = bool(zeroth_order.wrap_weight != 0) or bool(
            zeroth_order.kernel.any()
        )
        frames, fields, valid, levels, shifts = [], [], [], [], []
        with CaptureStore.open(self.dataset_path) as store:
            for index in progress(
                range(len(data.probes)),
                description="Registering probes",
                verbose=verbose,
            ):
                measured = store.read(2 * index)
                reference = store.read(2 * index + 1)
                long = torch.as_tensor(
                    measured["camera_image"], dtype=torch.float32, device=self.device
                )
                short = torch.as_tensor(
                    reference["camera_image"], dtype=torch.float32, device=self.device
                )
                with self._zeroth_order_silenced():
                    model_field = self._model_fields(measured["slm_levels"])[0]
                with_zeroth_order = (
                    self._model_fields(measured["slm_levels"])[0]
                    if carries_zeroth_order
                    else model_field
                )
                fit = fit_translation(
                    short,
                    with_zeroth_order.abs().square(),
                    ~region & (short < saturation),
                    fit_offset=True,
                )
                back = (-fit.shift[0], -fit.shift[1])
                # The overexposed pixels take the counts of the short exposure, scaled
                # to the long one, so the translation spreads the light they received.
                # Each of them is left out of the solve where the translation moves it,
                # rounded to whole pixels.
                overexposed = long >= saturation
                merged = torch.where(
                    overexposed,
                    (short - fit.offset) * exposure_ratio + fit.offset,
                    long,
                )
                moved_overexposed = torch.roll(
                    overexposed,
                    shifts=(round(back[1]), round(back[0])),
                    dims=(0, 1),
                )
                frames.append(box.crop(translate_real(merged, back)))
                valid.append(box.crop(~moved_overexposed))
                fields.append(box.crop(model_field))
                levels.append(measured["slm_levels"])
                shifts.append(fit.shift)
        return (
            torch.stack(frames),
            torch.stack(fields),
            torch.stack(valid),
            np.stack(levels),
            shifts,
        )

    @contextmanager
    def _zeroth_order_silenced(self) -> Generator[None, None, None]:
        """Hold the zeroth-order field at no kernel and no light at the wraps for the
        duration of a ``with`` block, and restore it on exit.
        """
        field = self.zeroth_order_field
        held_kernel = field.kernel.detach().clone()
        held_weight = field.wrap_weight.detach().clone()
        with torch.no_grad():
            field.kernel.zero_()
            field.wrap_weight.zero_()
        try:
            yield
        finally:
            with torch.no_grad():
                field.kernel.copy_(held_kernel)
                field.wrap_weight.copy_(held_weight)

    @staticmethod
    def _frame_scale(
        table: NDArray[np.int64],
        frames: Tensor,
        fields: Tensor,
        valid: Tensor,
        region: Tensor,
    ) -> float:
        """The counts per unit of model intensity, from the flat replica on the first
        grating at one and at two probe amplitudes.

        The change of the mean frame over the pistons is compared with the change of
        the model's mean intensity, summed over the pixels of the disc that are valid
        in every one of these frames. The missed field and the floor enter the change
        only through the small change of the model's mean field.
        """
        flat = (table[:, 0] == 0) & (table[:, 1] == FLAT_REPLICA)
        once = np.flatnonzero(flat & (table[:, 3] == 1))
        twice = np.flatnonzero(flat & (table[:, 3] == 2))
        usable = region & valid[np.concatenate([once, twice])].all(dim=0)
        measured = frames[twice].mean(dim=0) - frames[once].mean(dim=0)
        modelled = fields[twice].abs().square().mean(dim=0) - fields[
            once
        ].abs().square().mean(dim=0)
        return float(measured[usable].sum() / modelled[usable].sum())

    def _wrap_fields(self, levels: NDArray, box: ROI) -> Tensor:
        """The model's field on ``box`` of the beam on the wrapped edges of each
        hologram of ``levels``, at unit weight, ``(N, h, w)``.
        """
        model = self.slm_camera_model
        field = self.zeroth_order_field
        model.virtual_slm.set_levels(torch.as_tensor(levels), self.slm.bitdepth)
        field.update_wraps(model)
        with self._zeroth_order_silenced():
            with torch.no_grad(), model.restrict_output_to(box):
                field.wrap_weight.fill_(1.0)
                return field_on_frames(model(self._dark_field()))

    def _kernel_field_on(
        self, box: ROI
    ) -> tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
        """The model's field on ``box`` of a kernel alone, and its adjoint.

        The kernel's samples outside the disc are held at zero
        (:attr:`kernel_support`). The kernel replaces the zeroth-order field's own, and
        a weight of zero its light at the wraps, in a functional call on a dark incident
        field. The map is linear, so its adjoint is the gradient of the field at a zero
        kernel, taken with ``grad_outputs``.
        """
        model = self.slm_camera_model
        dark = self._dark_field()
        field = self.zeroth_order_field
        kernel_name = f"{ZEROTH_ORDER_LAYER}.kernel"
        weight_name = f"{ZEROTH_ORDER_LAYER}.wrap_weight"
        no_weight = torch.zeros_like(field.wrap_weight)
        shape = tuple(field.kernel.shape)
        support = self.kernel_support

        def forward(kernel: Tensor) -> Tensor:
            with model.restrict_output_to(box):
                output = torch.func.functional_call(
                    model,
                    {kernel_name: kernel * support, weight_name: no_weight},
                    (dark,),
                )
            return field_on_frames(output)[0]

        def adjoint(field: Tensor) -> Tensor:
            with torch.enable_grad():
                kernel = torch.zeros(
                    shape, dtype=field.dtype, device=field.device, requires_grad=True
                )
                (gradient,) = torch.autograd.grad(
                    forward(kernel), kernel, grad_outputs=field
                )
            return gradient

        return forward, adjoint

    def calibrate(
        self, set_fraction: float = 0.75, contrast: float = 1e-3, verbose: bool = True
    ) -> ZerothOrderCalibrationData:
        """Capture the probes and fit the zeroth order to them (:meth:`capture` and
        :meth:`fit`).

        Args:
            set_fraction: The peak of each autoexposure, as a fraction of full scale.
            contrast: The smallest variation that resolves a pixel, for :meth:`fit`.
            verbose: Show the progress bars.

        Returns:
            ZerothOrderCalibrationData: The kernel, the weight and the figures of the
            fit.
        """
        capture_data = self.capture(set_fraction=set_fraction, verbose=verbose)
        return self.fit(capture_data, contrast=contrast, verbose=verbose)
