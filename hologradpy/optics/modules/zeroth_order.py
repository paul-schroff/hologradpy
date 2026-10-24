"""The zeroth order a hologram leaves, added to the SLM-plane field before the lens."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor
from torch.nn import Parameter

from .abstract import OpticsModule, capture_init
from .slm_fields.psf import image_plane_to_slm_chirp_z
from ..complex_amplitude import ComplexAmplitude
from ...fourier_optics import intensity_nyquist_pixel_size
from ...fourier_transforms import fft_resample
from ...vector_fields import forward_difference

if TYPE_CHECKING:
    from ...calibration.zeroth_order import ZerothOrderCalibrationData
    from ...fourier_transforms import ChirpZPartialAffine
    from ..systems.abstract import SLMFourierLensModel

#: The name of the layer a :class:`ZerothOrderField` takes in a model, after the
#: virtual SLM.
ZEROTH_ORDER_LAYER = "zeroth_order_field"


class ZerothOrderField(OpticsModule):
    """The zeroth order a hologram leaves, added to the SLM-plane field ahead of the
    Fourier lens.

    The module follows the virtual SLM in a model, inserted with
    ``model.insert_after("virtual_slm", "zeroth_order_field", field)``. It adds two
    fields to the one the SLM shapes:

    - The light that the fringing fields scatter at the phase wraps. It is
      ``wrap_weight`` times the beam on the pixel edges at which the displayed phase
      wraps, with the phase the SLM adds taken out of the beam. It changes with the
      hologram, and :meth:`update_wraps` finds the edges of the hologram on display.
      The light then stays fixed until the next call, so a design can hold it through
      a pass.
    - A field that no hologram changes, such as the reflection from the cover plate.
      A compact image-plane kernel sets it, as one sets the beam of
      :class:`~.slm_fields.psf.PSFSLMField`. The kernel is sampled at the intensity
      Nyquist pitch, ``lambda f / (2 L)`` for an SLM spanning ``L``, and centred on
      the zeroth order. A chirp-z transform carries it onto the grid of the field
      arriving (:func:`~.slm_fields.psf.image_plane_to_slm_chirp_z`).

    The field arriving can be finer than the SLM, as a crosstalk model needs. The light
    at the wraps is then resampled onto it as :class:`~.grid_adapter.GridAdapter`
    resamples the beam, so its focal field stays the same.

    The module takes a scalar field of one wavelength.
    """

    @capture_init
    def __init__(
        self: ZerothOrderField,
        focal_length: float,
        wrap_weight: complex = 0.0,
        kernel: Tensor | None = None,
        added_phase: Tensor | None = None,
        kernel_pixel_size: tuple[float, float] | None = None,
    ) -> None:
        """
        Args:
            focal_length: The focal length of the Fourier lens in metres.
            wrap_weight: The field scattered at one wrapped edge, as a multiple of the
                beam's field on the pixel that counts the edge.
            kernel: The image-plane kernel of the field no hologram changes,
                ``(rows, columns)`` at the intensity Nyquist pitch, or None for none.
            added_phase: A phase in radians that the SLM adds to every hologram before
                it displays it, such as its vendor correction, at the SLM's
                resolution, or None. The edges are those of the sum.
            kernel_pixel_size: The ``(y, x)`` pitch in metres the kernel was sampled
                at, or None. A given pitch is compared with the intensity Nyquist pitch
                of the field arriving on the first forward pass.
        """
        super().__init__()
        self.focal_length: float = float(focal_length)
        self.wrap_weight = Parameter(
            torch.tensor(complex(wrap_weight), dtype=torch.complex64),
            requires_grad=False,
        )
        self.register_parameter(
            "kernel",
            None
            if kernel is None
            else Parameter(
                torch.as_tensor(kernel).to(torch.complex64).clone(),
                requires_grad=False,
            ),
        )
        self.register_buffer(
            "added_phase",
            None if added_phase is None else torch.as_tensor(added_phase).clone(),
        )
        # The beam on the wrapped edges of the hologram last given to update_wraps, at
        # the SLM's resolution.
        self.register_buffer("beam_at_wraps", None, persistent=False)
        # The (y, x) pitch of the kernel in metres, set from the field's grid.
        self.kernel_pixel_size: tuple[float, float] | None = (
            None
            if kernel_pixel_size is None
            else (float(kernel_pixel_size[0]), float(kernel_pixel_size[1]))
        )
        self._chirp_z: ChirpZPartialAffine | None = None
        self._held_kernel: Tensor | None = None
        self._held_kernel_field: Tensor | None = None

    @classmethod
    def from_calibration_data(
        cls: type[ZerothOrderField], calibration_data: ZerothOrderCalibrationData
    ) -> ZerothOrderField:
        """The zeroth order a calibration measured
        (:class:`~hologradpy.calibration.zeroth_order.ZerothOrderCalibrator`), with the
        phase the SLM added to the probes as :attr:`added_phase`.

        Args:
            calibration_data: The calibration.

        Returns:
            ZerothOrderField: The field. Its first forward pass checks that the model's
            SLM grid samples the kernel at the calibration's pitch.
        """
        added_phase = calibration_data.added_phase
        return cls(
            calibration_data.focal_length,
            wrap_weight=complex(calibration_data.wrap_weight),
            kernel=torch.as_tensor(np.asarray(calibration_data.kernel)),
            added_phase=None if added_phase is None else np.asarray(added_phase),
            kernel_pixel_size=tuple(calibration_data.kernel_pixel_size),
        )

    def lazy_init(
        self: ZerothOrderField, complex_amplitude: ComplexAmplitude
    ) -> None:
        if complex_amplitude.number_of_wavelengths != 1:
            raise NotImplementedError(
                "ZerothOrderField takes one wavelength. The kernel's pitch and the "
                "light at the wraps both scale with the wavelength, so several "
                "wavelengths need one module each."
            )
        if complex_amplitude.number_of_components != 1:
            raise NotImplementedError(
                "ZerothOrderField adds a scalar field, and how it divides between the "
                "components of a field vector is a modelling choice this module does "
                "not make."
            )
        wavelength = float(complex_amplitude.wavelength.reshape(-1)[0])
        pixel_size = complex_amplitude.pixel_size.reshape(-1)
        field_pitch = (float(pixel_size[0]), float(pixel_size[1]))
        nyquist_pitch = tuple(
            float(
                intensity_nyquist_pixel_size(
                    wavelength, self.focal_length, pitch, count
                )
            )
            for pitch, count in zip(field_pitch, self.resolution_in)
        )
        if self.kernel_pixel_size is not None and not np.allclose(
            self.kernel_pixel_size, nyquist_pitch, rtol=1e-6, atol=0.0
        ):
            raise ValueError(
                f"The kernel was sampled at a pitch of {self.kernel_pixel_size} m, "
                "but the intensity Nyquist pitch of this model's SLM grid is "
                f"{nyquist_pitch} m. The kernel belongs to a model of another SLM "
                "grid, wavelength or focal length."
            )
        self.kernel_pixel_size = nyquist_pitch
        if self.kernel is None:
            return
        self.kernel.data = self.kernel.data.to(
            device=complex_amplitude.device, dtype=complex_amplitude.dtype
        )
        self._chirp_z = image_plane_to_slm_chirp_z(
            tuple(self.kernel.shape),
            self.kernel_pixel_size,
            tuple(self.resolution_in),
            field_pitch,
            wavelength,
            self.focal_length,
            complex_amplitude.device,
        )

    def kernel_field(self: ZerothOrderField, kernel: Tensor) -> Tensor:
        """The SLM-plane field of an image-plane kernel the shape of :attr:`kernel`, on
        the grid of the field arriving.

        Args:
            kernel: The kernel, ``(rows, columns)``.

        Returns:
            Tensor: The field ``(H, W)``.
        """
        self._ensure_kernel_transform()
        return self._chirp_z(kernel.unsqueeze(0)).reshape(self.resolution_in)

    def kernel_field_adjoint(self: ZerothOrderField, field: Tensor) -> Tensor:
        """The adjoint of :meth:`kernel_field`, from an SLM-plane field ``(H, W)`` on
        the grid of the field arriving to the kernel's grid.
        """
        self._ensure_kernel_transform()
        return self._chirp_z.adjoint(field.unsqueeze(0)).reshape(self.kernel.shape)

    def _ensure_kernel_transform(self: ZerothOrderField) -> None:
        """Raise unless the module holds a kernel and has seen its grid."""
        self._ensure_initialized()
        if self._chirp_z is None:
            raise RuntimeError(
                "This ZerothOrderField holds no kernel, so it has no transform for one."
            )

    def update_wraps(self: ZerothOrderField, model: SLMFourierLensModel) -> None:
        """Find the edges at which the hologram on display wraps, and hold the beam on
        them for every forward pass until the next call.

        The edges are those the device shows. The model's desired phase plus
        :attr:`added_phase` goes through the model's phase response and is rounded to
        whole levels. An edge wraps where the displayed fraction of full scale jumps by
        more than half of it from one pixel to the next. A wrap along a row is counted
        on the pixel of lower column index, and one along a column on the pixel of
        lower row index, so a pixel counts 0, 1 or 2. A batch of holograms gives one
        set of edges each.

        The beam of the model is calibrated through :attr:`added_phase`, and at a wrap
        the device shows a phase near zero. The added phase is therefore taken out of
        the beam on the edges.

        Args:
            model: The model, for its virtual SLM and its beam.
        """
        virtual_slm = model.virtual_slm
        with torch.no_grad():
            phase = virtual_slm.get_phase()
            beam = model.slm_field.get_wavefront().detach()
            if self.added_phase is not None:
                added_phase = self.added_phase.to(
                    device=phase.device, dtype=phase.dtype
                )
                phase = phase + added_phase
                beam = beam * torch.exp(-1j * added_phase).to(beam.dtype)
            response = virtual_slm.phase_response
            levels = response.response.round_to_levels(response.to_levels(phase))
            along_rows, along_columns = forward_difference(
                levels / response.number_of_levels
            )
            # Each difference is one pixel short along its axis. Padding it at the end
            # puts every edge on the pixel of lower index.
            dtype = beam.real.dtype
            on_rows = torch.nn.functional.pad(
                (along_rows.abs() > 0.5).to(dtype), (0, 1)
            )
            on_columns = torch.nn.functional.pad(
                (along_columns.abs() > 0.5).to(dtype), (0, 0, 0, 1)
            )
            self.beam_at_wraps = beam * (on_rows + on_columns)

    def added_field(self: ZerothOrderField) -> Tensor | None:
        """The field this module adds, on the grid of the field arriving, or None for
        none. It is ``(H, W)``, or ``(N, H, W)`` for a batch of holograms.

        Raises:
            RuntimeError: The module scatters light at the wraps, but
                :meth:`update_wraps` has not been called.
        """
        self._ensure_initialized()
        parts = []
        if self.beam_at_wraps is not None:
            at_wraps = fft_resample(self.beam_at_wraps, tuple(self.resolution_in))
            parts.append(self.wrap_weight * at_wraps)
        elif bool(self.wrap_weight != 0):
            raise RuntimeError(
                "The light at the wraps needs the edges of the hologram on display. "
                "Call update_wraps(model) once the hologram is set."
            )
        if self.kernel is not None:
            parts.append(self._kernel_field_held())
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else parts[0] + parts[1]

    def _kernel_field_held(self: ZerothOrderField) -> Tensor:
        """The field of :attr:`kernel`, held while the kernel takes no gradient and has
        not changed.
        """
        kernel = self.kernel
        if kernel.requires_grad:
            return self.kernel_field(kernel)
        unchanged = (
            self._held_kernel is not None
            and self._held_kernel.device == kernel.device
            and torch.equal(self._held_kernel, kernel.detach())
        )
        if self._held_kernel_field is None or not unchanged:
            with torch.no_grad():
                self._held_kernel_field = self.kernel_field(kernel)
            self._held_kernel = kernel.detach().clone()
        return self._held_kernel_field

    def forward(
        self: ZerothOrderField, complex_amplitude: ComplexAmplitude
    ) -> ComplexAmplitude:
        added = self.added_field()
        if added is None:
            return complex_amplitude
        if added.ndim >= 3:
            # A batch of holograms meets its own fields, (N, 1, 1, H, W).
            added = added[..., None, None, :, :]
        out = complex_amplitude + added
        return out.with_geometry(
            wavelength=complex_amplitude.wavelength,
            pixel_size=self.pixel_size_out,
        )
