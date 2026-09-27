from __future__ import annotations

from typing import TypeVar
import numpy as np
from numpy.typing import NDArray
import torch

from scipy.ndimage import label

from ...analysis.unwrapping import unwrap_phase_1D
from ...optics.complex_amplitude import ComplexAmplitude
from ...grids import coordinates_to_indices, get_spatial_grid
from ...profiles.phase import vortex_field
from ...utils import as_image, gpu_to_numpy

ArrayLike = TypeVar("ArrayLike", torch.Tensor, NDArray)


# TODO: Docstrings
class VortexDetector:
    def __init__(self, shape: tuple[int, int], device: str = "cpu") -> None:
        self.labels: torch.Tensor
        self.center_coordinates: torch.Tensor
        self.center_indices: torch.Tensor
        self.charges: torch.Tensor
        self.number_of_vortices: int
        self.zero_crossing_mask: torch.Tensor

        self.pixel_grid = get_spatial_grid(
            shape,
            pixel_size=(1.0, 1.0),
            device=device,
        )

    def detect_vortices(
        self,
        complex_amplitude: ComplexAmplitude,
        target_intensity: torch.Tensor,
        threshold: float = 0.2,
        pad: int = 1,
    ) -> None:
        self.zero_crossing_mask = find_zero_crossing_intersections(
            complex_amplitude, target_intensity, threshold
        )

        self.labels, self.number_of_vortices = label_connected_components(
            self.zero_crossing_mask
        )

        self.center_coordinates = find_label_centers(*self.pixel_grid, self.labels)
        self.center_indices = coordinates_to_indices(
            *self.pixel_grid, self.center_coordinates
        )
        self.charges = find_vortex_charge(
            complex_amplitude, *self.pixel_grid, self.center_indices, pad
        )

    def generate_anti_vortex_field(self) -> torch.Tensor:
        return vortex_field(*self.pixel_grid, self.center_coordinates, -self.charges)


def find_zero_crossings(input: torch.Tensor) -> torch.Tensor:
    """Find zero crossings in a 2D array.

    Args:
        input: Input 2D array.

    Returns:
        torch.Tensor: Boolean array indicating the positions of zero
            crossings.
    """
    zero_crossings_x = (input[:-1, :-1] * input[1:, 1:] < 0) | (
        input[:-1, 1:] * input[1:, :-1] < 0
    )
    zero_crossings_y = (input[:-1, :-1] * input[1:, :-1] < 0) | (
        input[:-1, 1:] * input[1:, :-1] < 0
    )
    padded_mask = torch.nn.functional.pad(
        zero_crossings_x & zero_crossings_y, (0, 1, 0, 1)
    )
    return padded_mask


def find_zero_crossing_intersections(
    complex_amplitude: ComplexAmplitude,
    target_intensity: ArrayLike,
    threshold: float = 0.2,
) -> ArrayLike:
    """Find the intersections of zero crossings in the real and imaginary
    parts of the ``complex_amplitude``. Only considers intersections where
    the ``target_intensity`` is above a given ``threshold``.

    Args:
        complex_amplitude: The complex electric field.
        target_intensity: The target intensity to threshold the zero
            crossings.
        threshold: The intensity threshold to consider a zero crossing
            valid. Defaults to 0.2.

    Returns:
        ArrayLike: Boolean array marking the zero crossing intersections
            that pass the intensity threshold.
    """
    field = as_image(complex_amplitude)
    zero_crossings_real = find_zero_crossings(field.real)
    zero_crossings_imag = find_zero_crossings(field.imag)
    zero_crossings = zero_crossings_real & zero_crossings_imag
    return zero_crossings * (target_intensity > threshold)


def label_connected_components(
    boolean_mask: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Label connected components in a boolean mask. Uses
    ``scipy.ndimage.label`` for labeling.

    Args:
        boolean_mask: Boolean mask to label.

    Returns:
        tuple[torch.Tensor, int]: A tuple containing the labeled mask and
            the number of labels.
    """
    labels, number_of_labels = label(gpu_to_numpy(boolean_mask))
    return (
        torch.tensor(labels, dtype=torch.int, device=boolean_mask.device),
        number_of_labels,
    )


def find_label_centers(
    x: torch.Tensor, y: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Finds the centers of regions in a labeled mask.

    Args:
        x: The x-coordinates of the spatial grid.
        y: The y-coordinates of the spatial grid.
        labels: The labeled mask.

    Returns:
        torch.Tensor: The coordinates of the centers of the labeled
            regions. x-coordinates are in ``label_centers[:, 0]`` and
            y-coordinates are in ``label_centers[:, 1]``.
    """
    number_of_labels = int(labels.max().item())
    label_centers = torch.zeros(number_of_labels, 2, device=labels.device)

    for i in range(number_of_labels):
        label_mask = labels == (i + 1)
        # pixel_coordinates = label_mask.argwhere()
        # average_coordinates = pixel_coordinates.float().mean(dim=0)

        label_centers[i, 0] = x[label_mask].mean()
        label_centers[i, 1] = y[label_mask].mean()
    return label_centers


def find_vortex_charge(
    complex_amplitude: ComplexAmplitude,
    x: torch.Tensor,
    y: torch.Tensor,
    center_indices: torch.Tensor,
    pad: int = 1,
) -> torch.Tensor:
    """Finds the charge of vortices given their centers.

    Args:
        complex_amplitude: The complex electric field.
        x: The x-coordinates of the spatial grid.
        y: The y-coordinates of the spatial grid.
        center_indices: The coordinates of the vortex centers.
            x-coordinates are in ``center_indices[:, 1]`` and y-coordinates
            are in ``center_indices[:, 0]``.
        pad: The padding around the center to consider for charge
            calculation. Defaults to 1.

    Returns:
        torch.Tensor: The charges of the vortices. The charge is +1 for a
            clockwise vortex, -1 for a counter-clockwise vortex, and 0 if
            the phase difference is smaller than pi in magnitude.
    """
    charges = torch.zeros(
        len(center_indices), dtype=torch.int, device=complex_amplitude.device
    )
    field = as_image(complex_amplitude)

    for i in range(len(center_indices)):
        center_index = center_indices[i]

        roi = field[
            center_index[0] - pad : center_index[0] + pad + 1,
            center_index[1] - pad : center_index[1] + pad + 1,
        ]

        roi_phase = torch.angle(roi)

        phase_square_path = torch.cat(
            (
                roi_phase[0, :].flatten(),
                roi_phase[1:, -1].flatten(),
                torch.flip(roi_phase[-1, :-1].flatten(), dims=(0,)),
                torch.flip(roi_phase[1:-1, 0].flatten(), dims=(0,)),
            )
        )

        unwrapped_phase = unwrap_phase_1D(phase_square_path)
        phase_difference = unwrapped_phase[-1] - unwrapped_phase[0]

        if phase_difference > np.pi:
            charges[i] = 1
        elif phase_difference < -np.pi:
            charges[i] = -1
        else:
            charges[i] = 0
    return charges
