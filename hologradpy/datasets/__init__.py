"""Captured sets and retrieval steps: what is written, and how it is read back.

A **capture** is camera frames and the SLM levels that produced them, from a bench, and
is irreproducible. A **retrieval's steps** are the optimiser's own parameter every nth
iteration, from a search, and the image each one predicts is rebuilt rather than stored.
They are separate stores because they are separate things.
"""

from .dataset import SampleDataset
from .stores import (
    PHASE_BITDEPTH_KEY,
    SAMPLE_DTYPE,
    SAMPLE_STORE_SUFFIX,
    SERIES,
    CaptureStore,
    CapturedSample,
    RetrievalSample,
    RetrievalStepStore,
)

__all__ = [
    "PHASE_BITDEPTH_KEY",
    "SAMPLE_DTYPE",
    "SAMPLE_STORE_SUFFIX",
    "SERIES",
    "CaptureStore",
    "CapturedSample",
    "RetrievalSample",
    "RetrievalStepStore",
    "SampleDataset",
]
