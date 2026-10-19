"""Fits of an SLM-camera model to the camera frames of displayed holograms."""

from __future__ import annotations

import operator
from collections.abc import Iterable, Sequence
from functools import reduce
from typing import TYPE_CHECKING

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from ..analysis.fourier_filter import white_noise_rms
from ..datasets import HologramFrames
from ..loss_functions import LossFunction, MaskedIntensityMSE
from ..optics.complex_amplitude import ComplexAmplitude
from ..optimization import LevenbergMarquardt
from ..profiles.phase import linear_phase
from ..utils import ProgressBar

if TYPE_CHECKING:
    from ..optics.systems import SLMFourierLensModel
    from ..roi import ROI


class FrameTranslations(torch.nn.Module):
    """A translation of the pattern of every frame, as the pointing of the beam moves
    it, applied as a tilt of the model's incident field.

    The translations enter relative to their mean over the frames, since the tilt of a
    fitted field moves every pattern alike.
    """

    def __init__(
        self, number_of_frames: int, dtype: torch.dtype, device: torch.device
    ) -> None:
        """
        Args:
            number_of_frames: The frames, one translation each.
            dtype: The real dtype of the translations.
            device: Where the translations are held.
        """
        super().__init__()
        self.translations = torch.nn.Parameter(
            torch.zeros(number_of_frames, 2, dtype=dtype, device=device)
        )

    def relative(self, translations: Tensor | None = None) -> Tensor:
        """The translations relative to their mean, ``(N, 2)`` with ``(x, y)`` in metres
        of the image plane.

        Args:
            translations: Values that replace the parameter, as the trial values of a
                fit do, or None for the parameter.
        """
        values = self.translations if translations is None else translations
        return values - values.mean(dim=0)

    def incident_field(
        self,
        slm_camera_model: SLMFourierLensModel,
        indices: slice | Sequence[int],
        translations: Tensor | None = None,
    ) -> ComplexAmplitude:
        """The model's incident field, tilted by the relative translation of each frame
        at ``indices`` (:func:`translated_incident_field`).

        Args:
            slm_camera_model: The model, for its incident field.
            indices: The frames.
            translations: Values that replace the parameter, or None for the parameter.
        """
        return translated_incident_field(
            slm_camera_model, self.relative(translations)[indices]
        )


class ModelFitter:
    """Fit parameters of an SLM-camera model to the frames of displayed holograms.

    For every hologram of :attr:`frames`, the model predicts the frame, and the
    parameters of :meth:`trainable_parameters` are stepped to match it. :meth:`fit` and
    :meth:`fit_batches` take Adam steps on :attr:`loss`, and
    :meth:`fit_levenberg_marquardt` takes Levenberg-Marquardt steps on a residual of the
    square roots of the intensities. Every fit leaves the virtual SLM with the levels it
    held before.
    """

    # Shown on the progress bar.
    description: str = "Fitting"

    def __init__(
        self,
        slm_camera_model: SLMFourierLensModel,
        frames: HologramFrames,
        parameters: Iterable[torch.nn.Parameter] | None = None,
        loss: LossFunction | None = None,
        learning_rate: float = 1e-2,
    ) -> None:
        """
        Args:
            slm_camera_model: The differentiable model to fit.
            frames: The holograms and the frames measured for them.
            parameters: The parameters of the model to step, or None for those of
                :meth:`trainable_parameters` in a subclass.
            loss: The cost of an Adam step, taking ``(predicted_field, frame)``.
                Defaults to :class:`~hologradpy.loss_functions.MaskedIntensityMSE` over
                the region of ``frames``.
            learning_rate: The Adam step size.
        """
        self.slm_camera_model: SLMFourierLensModel = slm_camera_model
        self.frames: HologramFrames = frames
        self.parameters: list[torch.nn.Parameter] | None = (
            None if parameters is None else list(parameters)
        )
        self.loss: LossFunction = (
            MaskedIntensityMSE(frames.region.to(frames.dtype)) if loss is None else loss
        )
        self.learning_rate: float = learning_rate
        self.component_history: dict[str, list[float]] = {}
        # The translation of every frame from the last fit with translations, NaN for a
        # frame left out of it, or None.
        self.frame_translations: NDArray[np.float64] | None = None
        # The Adam optimizer of the Adam steps, kept from one call of fit_batches to
        # the next.
        self.optimizer: torch.optim.Adam | None = None

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        """The parameters a fit steps, with their gradients switched on.

        Raises:
            NotImplementedError: No parameters were given, and no subclass names them.
        """
        if self.parameters is None:
            raise NotImplementedError(
                "ModelFitter needs the parameters to fit, or a subclass that names "
                "them in trainable_parameters."
            )
        for parameter in self.parameters:
            parameter.requires_grad_(True)
        return list(self.parameters)

    def enabled_parameters(self) -> list[torch.nn.Parameter]:
        """Every parameter of the model left requiring a gradient."""
        return [p for p in self.slm_camera_model.parameters() if p.requires_grad]

    def fit(
        self,
        number_of_epochs: int = 100,
        batch_size: int = 3,
        subset_indices: Sequence[int] | NDArray[np.int_] | None = None,
        shuffle: bool = True,
        verbose: bool = True,
    ) -> list[float]:
        """Fit the model by Adam with a new optimizer, and return the mean loss of each
        epoch.

        Args:
            number_of_epochs: Passes over the holograms.
            batch_size: Holograms per step.
            subset_indices: Which holograms to use. Defaults to all.
            shuffle: Shuffle the holograms between epochs.
            verbose: Show a progress bar.

        Returns:
            list[float]: The mean loss of each epoch.

        Raises:
            ValueError: ``batch_size`` exceeds the number of holograms.
        """
        indices = (
            np.arange(len(self.frames))
            if subset_indices is None
            else np.asarray(subset_indices)
        )
        if len(indices) < batch_size:
            raise ValueError(
                f"batch_size ({batch_size}) exceeds the number of samples "
                f"({len(indices)}), so no batch is filled. Lower batch_size or "
                "capture more phase patterns."
            )
        batches = DataLoader(
            Subset(self.frames, indices.tolist()),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
        )
        self.optimizer = None
        self.frame_translations = None

        number_of_batches = len(batches)
        if verbose:
            print(
                f"Running {number_of_epochs} epochs of {number_of_batches} "
                f"batches (batch size {batch_size}), "
                f"{number_of_epochs * number_of_batches} iterations total."
            )

        history: list[float] = []
        self.component_history = {}
        epochs = ProgressBar(
            total=number_of_epochs, description=self.description, verbose=verbose
        )
        with epochs:
            for _ in range(number_of_epochs):
                losses, components = self._adam_steps(batches)
                history.append(sum(losses) / max(number_of_batches, 1))
                for label, values in components.items():
                    self.component_history.setdefault(label, []).append(
                        sum(values) / max(number_of_batches, 1)
                    )
                epochs.update(loss=history[-1])
        return history

    def fit_batches(self, batches: Iterable[dict]) -> list[float]:
        """Take one Adam step per batch of ``batches``, and return the loss of every
        batch.

        The optimizer is built at the first call and kept, so its moment estimates
        carry over from one call to the next.

        Args:
            batches: Batches of items of :attr:`frames`, as a
                :class:`torch.utils.data.DataLoader` gives them.
        """
        losses, _ = self._adam_steps(batches)
        return losses

    def _adam_steps(
        self, batches: Iterable[dict]
    ) -> tuple[list[float], dict[str, list[float]]]:
        """One Adam step per batch, and the loss and the loss components of each."""
        parameters = self.trainable_parameters()
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                parameters, lr=self.learning_rate, amsgrad=True
            )
        losses: list[float] = []
        components: dict[str, list[float]] = {}
        with self.slm_camera_model.virtual_slm.preserve_levels():
            for batch in batches:
                self.optimizer.zero_grad()
                parts = self.loss.components(
                    self.predicted_fields(batch["levels"]), batch["frame"]
                )
                loss = reduce(operator.add, parts.values())
                loss.backward()
                self.optimizer.step()
                losses.append(float(loss))
                for label, value in parts.items():
                    components.setdefault(label, []).append(float(value))
        return losses, components

    def fit_levenberg_marquardt(
        self,
        steps: int = 3,
        chunk_size: int | None = 5,
        subset_indices: Sequence[int] | NDArray[np.int_] | None = None,
        noise_frames: Sequence[int] | None = None,
        fit_translations: bool = False,
        verbose: bool = True,
    ) -> list[float]:
        """Fit the model by Levenberg-Marquardt
        (:class:`~hologradpy.optimization.LevenbergMarquardt`), and return the cost
        before every step and after the last.

        The residual of a frame is ``sqrt(I_model + floor) - sqrt(I_frame + floor)`` at
        the pixels of the region, with each intensity scaled to mean 1 over the region.
        On the square root, the shot noise has the same variance at every pixel. The
        floor is the white-noise standard deviation of the frames ``noise_frames`` in
        these units. It is measured from the power of their FFT bins outside the band of
        :attr:`frames` (:func:`~hologradpy.analysis.fourier_filter.white_noise_rms`).
        The floor is zero for a set without a band, and when the band holds every
        frequency of the frames.

        Each parameter is a block of the fit. With ``fit_translations``, the translation
        of every frame (:class:`FrameTranslations`) is one more block, and the fitted
        translations are kept on :attr:`frame_translations`.

        Args:
            steps: Levenberg-Marquardt steps. On a simulated 1039 nm bench with 20
                patterns and a PSF kernel, the cost came within 0.1 % of its final
                value after two.
            chunk_size: Frames per chunk, or None for one chunk. The fit holds the graph
                of one chunk, so a smaller chunk uses less memory. On that bench, the
                fit took 2.0 to 2.4 GiB with chunks of 5 frames of 1079 x 1433 pixels.
            subset_indices: Which holograms to fit. Defaults to all.
            noise_frames: The positions, among the fitted holograms, of the frames that
                set the floor, or None for every frame.
            fit_translations: Fit a translation per frame.
            verbose: Show a progress bar.

        Returns:
            list[float]: The cost before every step and after the last.
        """
        model = self.slm_camera_model
        virtual_slm = model.virtual_slm
        indices = (
            np.arange(len(self.frames))
            if subset_indices is None
            else np.asarray(subset_indices)
        )
        frames = self.frames.subset(indices)
        region = frames.region
        pixel_count = int(region.sum())
        number_of_frames = len(frames)

        measured = frames.frames.clamp(min=0.0) * region
        measured = measured / measured.sum(dim=(-2, -1), keepdim=True) * pixel_count
        noise = None
        if frames.band_mask is not None:
            noise = white_noise_rms(
                measured if noise_frames is None else measured[list(noise_frames)],
                region,
                frames.band_mask,
            )
        noise_floor = 0.0 if noise is None else noise
        measured_roots = (measured[:, region] + noise_floor).sqrt()

        wanted = {id(parameter) for parameter in self.trainable_parameters()}
        names, parameters = zip(
            *(
                (name, parameter)
                for name, parameter in model.named_parameters()
                if id(parameter) in wanted
            )
        )
        translations = (
            FrameTranslations(number_of_frames, frames.dtype, region.device)
            if fit_translations
            else None
        )
        fitted = list(parameters)
        if translations is not None:
            fitted.append(translations.translations)

        frames_per_chunk = number_of_frames if chunk_size is None else chunk_size
        chunks = [
            slice(first, min(first + frames_per_chunk, number_of_frames))
            for first in range(0, number_of_frames, frames_per_chunk)
        ]

        def closure(chunk: int, values: list[Tensor] | None) -> Tensor:
            """The residual of ``chunk``, with the fitted parameters replaced by
            ``values``.
            """
            frames_of_chunk = chunks[chunk]
            virtual_slm.levels.data = stored_levels[frames_of_chunk]
            values = fitted if values is None else values
            incident = (
                None
                if translations is None
                else translations.incident_field(model, frames_of_chunk, values[-1])
            )
            field = torch.func.functional_call(
                model, dict(zip(names, values[: len(names)])), (incident,)
            )
            intensity = field_on_frames(field, frames.roi).abs()[:, region] ** 2
            intensity = intensity / intensity.sum(dim=-1, keepdim=True) * pixel_count
            return (intensity + noise_floor).sqrt() - measured_roots[frames_of_chunk]

        optimizer = LevenbergMarquardt(fitted, number_of_chunks=len(chunks))
        progress = ProgressBar(
            total=steps, description=self.description, verbose=verbose
        )
        costs: list[float] = []
        with virtual_slm.preserve_levels(), progress:
            virtual_slm.set_levels(frames.levels, frames.bitdepth)
            stored_levels = virtual_slm.levels.data.clone()
            for _ in range(steps):
                costs.append(optimizer.step(closure))
                progress.update(cost=costs[-1])
            with torch.no_grad():
                costs.append(
                    0.5
                    * sum(
                        float((closure(chunk, None).double() ** 2).sum())
                        for chunk in range(len(chunks))
                    )
                )

        self.frame_translations = None
        if translations is not None:
            self.frame_translations = np.full((len(self.frames), 2), np.nan)
            self.frame_translations[indices] = (
                translations.relative().detach().double().cpu().numpy()
            )
        self.component_history = {}
        return costs

    def predicted_fields(
        self, levels: Tensor, translations: Tensor | None = None
    ) -> Tensor:
        """The model's field on the frames' grid for a batch of holograms.

        The whole batch is imprinted at once, and the model runs one forward pass.

        Args:
            levels: The levels of the holograms, ``(N, H, W)``.
            translations: One ``(x, y)`` per hologram in metres of the image plane,
                ``(N, 2)``, which tilts its incident field
                (:func:`translated_incident_field`), or None.

        Returns:
            Tensor: The complex fields, ``(N, h, w)``.
        """
        model = self.slm_camera_model
        model.virtual_slm.set_levels(levels, self.frames.bitdepth)
        incident = (
            None
            if translations is None
            else translated_incident_field(model, translations)
        )
        return field_on_frames(model(incident), self.frames.roi)

    def measured_and_predicted_roi(
        self, sample_index: int = 0
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """The measured frame and the predicted intensity of one hologram, inside the
        region. The prediction takes the hologram's fitted translation, when the last
        fit gave it one.

        Args:
            sample_index: The hologram.
        """
        sample = self.frames[sample_index]
        translations = None
        if self.frame_translations is not None:
            translation = self.frame_translations[sample_index : sample_index + 1]
            if np.isfinite(translation).all():
                translations = torch.as_tensor(
                    translation,
                    dtype=self.frames.dtype,
                    device=self.frames.region.device,
                )
        region = self.frames.region.to(self.frames.dtype)
        with torch.no_grad(), self.slm_camera_model.virtual_slm.preserve_levels():
            predicted = self.predicted_fields(sample["levels"][None], translations)
            predicted_intensity = (predicted.abs() ** 2) * region
        measured = sample["frame"] * region
        return (
            measured.detach().cpu().numpy(),
            predicted_intensity.squeeze(0).cpu().numpy(),
        )


def translated_incident_field(
    model: SLMFourierLensModel, translations: Tensor
) -> ComplexAmplitude:
    """The model's field incident on the SLM, once per frame.

    The field of a frame is tilted by the linear phase that moves its pattern by its
    translation, as the pointing of the beam does.

    Args:
        model: The model, for its incident field, its input grid and its focal length.
        translations: One ``(x, y)`` per frame, in metres of the image plane, shaped
            ``(N, 2)``.

    Returns:
        ComplexAmplitude: The fields, ``(N, components, wavelengths, H, W)``.
    """
    incident = model.init_field
    grid_x, grid_y = model.input_geometry.get_spatial_grid()
    tilt = linear_phase(
        grid_x,
        grid_y,
        translations[:, 0, None, None],
        translations[:, 1, None, None],
        wavenumber=2 * np.pi / float(incident.wavelength.reshape(-1)[0]),
        focal_length=model.focal_length,
    ).to(incident.dtype_r)
    tilted = incident.as_tensor() * torch.exp(1j * tilt)[:, None, None]
    return ComplexAmplitude.from_tensor(
        tilted, incident.wavelength, incident.pixel_size
    )


def field_on_frames(field: ComplexAmplitude, roi: ROI | None = None) -> Tensor:
    """The model's camera-plane field of every hologram, ``(N, h, w)``, cropped to
    ``roi``, or whole for None.

    Raises:
        ValueError: The model carries more than one wavelength.
    """
    if field.number_of_wavelengths != 1:
        raise ValueError(
            "A fit to camera frames compares one frame per hologram, so the model must "
            f"carry a single wavelength, not {field.number_of_wavelengths}."
        )
    flat, _ = field.flatten_batch()  # (N, 1, H, W)
    return flat[:, 0] if roi is None else roi.crop(flat[:, 0])
