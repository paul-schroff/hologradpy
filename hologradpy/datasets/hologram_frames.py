"""The levels of displayed holograms and the camera frames measured for them, prepared
for a fit of the model.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset

from ..analysis.fourier_filter import intensity_band_mask
from ..hardware.camera import Background, background_at
from ..roi import ROI
from .stores import CaptureStore

if TYPE_CHECKING:
    from ..calibration.camera_mapping import CameraMapping
    from ..optics.systems import SLMFourierLensModel


class HologramFrames(Dataset):
    """The levels of displayed holograms and the camera frame measured for each, on one
    grid, held in memory for a fit of the model.

    Every frame enters the set background subtracted and cropped to :attr:`roi`, and the
    set scales it to mean 1 over :attr:`region`. The levels are kept in the real dtype
    of the frames, which holds integers of up to 24 bits exactly. Item ``i`` is
    ``{"levels": ..., "frame": ..., "index": i}``, so a
    :class:`torch.utils.data.DataLoader` batches the set.
    """

    def __init__(
        self,
        region: Tensor,
        bitdepth: int | None = None,
        roi: ROI | None = None,
        band_mask: Tensor | None = None,
        background_level: float | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """An empty set, filled by :meth:`append`.

        Args:
            region: True at the pixels of a frame that a fit compares, ``(h, w)``. The
                set is held on its device.
            bitdepth: The bit depth of the levels, checked against the virtual SLM's, or
                None.
            roi: The crop of the model's output that the frames cover, or None when the
                output is the frames' grid.
            band_mask: The band of the optics on the frames' grid
                (:func:`~hologradpy.analysis.fourier_filter.intensity_band_mask`), or
                None.
            background_level: The mean over the region of the background subtracted
                from the frames, or None when none was subtracted.
            dtype: The real dtype of the frames and the levels.
        """
        self.region: Tensor = region.to(torch.bool)
        self.bitdepth: int | None = bitdepth
        self.roi: ROI | None = roi
        self.band_mask: Tensor | None = band_mask
        self.background_level: float | None = background_level
        self.dtype: torch.dtype = dtype
        self.levels: Tensor | None = None
        self.frames: Tensor | None = None

    @classmethod
    def from_capture(
        cls,
        dataset_path: str | os.PathLike,
        slm_camera_model: SLMFourierLensModel,
        roi_mask: NDArray[np.bool_],
        camera_mapping: CameraMapping | None = None,
        background: Background | None = None,
        equivalent_exposure: float | None = None,
    ) -> HologramFrames:
        """The samples of a :class:`~hologradpy.datasets.CaptureStore` file, prepared
        for a fit of ``slm_camera_model``. The file is read once and closed.

        The frames are cropped to the bounding box of ``roi_mask``, and the region is
        the mask on it.

        Args:
            dataset_path: The dataset file.
            slm_camera_model: The model the frames are fitted with, which sets the
                device and the real dtype of the set.
            roi_mask: True at the pixels of the whole sensor that a fit compares.
            camera_mapping: The map from camera pixels to the model's output pixels,
                which sets the band, or None for no band.
            background: The counts subtracted from every frame, in any form
                :func:`~hologradpy.hardware.camera.background_at` takes, or None to
                subtract nothing.
            equivalent_exposure: The equivalent exposure of the frames in seconds, at
                which a background function is evaluated.

        Returns:
            HologramFrames: The set.
        """
        device = slm_camera_model.device
        dtype = slm_camera_model.init_field.dtype_r
        roi = ROI.detect(roi_mask, pad=0)
        region_on_sensor = np.asarray(roi_mask, dtype=bool)
        region = torch.as_tensor(roi.crop(region_on_sensor), device=device)

        with CaptureStore.open(dataset_path) as store:
            bitdepth = store.phase_bitdepth
            samples = [store.read(index) for index in range(len(store))]

        subtracted: float | NDArray | None = None
        background_level = None
        if background is not None:
            subtracted = background_at(
                background, equivalent_exposure, np.shape(samples[0]["camera_image"])
            )
            background_level = (
                float(subtracted)
                if np.ndim(subtracted) == 0
                else float(np.mean(roi.crop(subtracted)[roi.crop(region_on_sensor)]))
            )

        frames = np.stack(
            [
                roi.crop(
                    sample["camera_image"]
                    if subtracted is None
                    else sample["camera_image"] - subtracted
                )
                for sample in samples
            ]
        )
        levels = np.stack([sample["slm_levels"] for sample in samples])
        band = (
            None
            if camera_mapping is None
            else intensity_band_mask(
                slm_camera_model, camera_mapping, tuple(region.shape)
            )
        )

        hologram_frames = cls(
            region,
            bitdepth=bitdepth,
            roi=roi,
            band_mask=band,
            background_level=background_level,
            dtype=dtype,
        )
        hologram_frames._extend(levels, frames)
        return hologram_frames

    def __len__(self) -> int:
        return 0 if self.frames is None else len(self.frames)

    def __getitem__(self, index: int) -> dict[str, Tensor | int]:
        return {
            "levels": self.levels[index],
            "frame": self.frames[index],
            "index": index,
        }

    def append(self, levels: NDArray | Tensor, frame: Tensor) -> None:
        """Add a hologram and the frame measured for it.

        Args:
            levels: The displayed levels, ``(H, W)``.
            frame: The frame on the set's grid, ``(h, w)``, background subtracted and
                cropped.
        """
        self._extend(torch.as_tensor(levels)[None], torch.as_tensor(frame)[None])

    def subset(self, indices: Sequence[int] | NDArray[np.int_]) -> HologramFrames:
        """The holograms and frames at ``indices``, in their order.

        Args:
            indices: The positions in this set.

        Returns:
            HologramFrames: A set with the same grid, region and band.
        """
        chosen = torch.as_tensor(np.asarray(indices), dtype=torch.long)
        subset = HologramFrames(
            self.region,
            bitdepth=self.bitdepth,
            roi=self.roi,
            band_mask=self.band_mask,
            background_level=self.background_level,
            dtype=self.dtype,
        )
        subset.levels = self.levels[chosen.to(self.levels.device)]
        subset.frames = self.frames[chosen.to(self.frames.device)]
        return subset

    def _extend(self, levels: NDArray | Tensor, frames: NDArray | Tensor) -> None:
        """Add holograms ``(N, H, W)`` and their frames ``(N, h, w)``, each frame scaled
        to mean 1 over the region.
        """
        device = self.region.device
        levels = torch.as_tensor(levels, dtype=self.dtype, device=device)
        frames = torch.as_tensor(frames, dtype=self.dtype, device=device)
        region = self.region.to(self.dtype)
        means = (frames * region).sum(dim=(-2, -1), keepdim=True) / region.sum()
        frames = frames / means.clamp_min(torch.finfo(self.dtype).tiny)
        if self.levels is not None:
            levels = torch.cat([self.levels, levels])
            frames = torch.cat([self.frames, frames])
        self.levels = levels
        self.frames = frames
