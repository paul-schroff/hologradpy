from __future__ import annotations

import torch
from torch.utils.data import Dataset

from .stores import CapturedSample, RetrievalSample, _SampleStore


class SampleDataset(Dataset):
    """Torch dataset over the samples of a store. Takes an open store rather than a 
    path.

    Args:
        store: An open :class:`CaptureStore` or :class:`RetrievalStepStore`.
        transform: Applied to each sample as it is loaded.
        cache: Keep loaded samples in memory, which is worth it for a set that is walked
            repeatedly and fits.
    """

    def __init__(
        self,
        store: _SampleStore,
        transform=None,
        cache: bool = True,
    ) -> None:
        self.store = store
        self.transform = transform
        self.cache = cache
        self._cached: dict[int, CapturedSample | RetrievalSample] = {}

    def __len__(self) -> int:
        return len(self.store)

    def __getitem__(self, sample_index: int) -> CapturedSample | RetrievalSample:
        if torch.is_tensor(sample_index):
            sample_index = sample_index.tolist()

        if self.cache and sample_index in self._cached:
            return self._cached[sample_index]

        sample = self.store.read(sample_index)
        if self.transform:
            sample = self.transform(sample)
        if self.cache:
            self._cached[sample_index] = sample
        return sample
