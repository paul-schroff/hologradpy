from typing import TypeVar
import numpy as np
from numpy.typing import NDArray
import torch

from scipy.ndimage import label

from ...optics.complex_amplitude import ComplexAmplitude
from ...grids import get_spatial_grid
from ...utils import gpu_to_numpy

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
        input (torch.Tensor): Input 2D array.
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

# TODO: This might already exist in grids.py
def coordinates_to_indices(
    x: torch.Tensor,
    y: torch.Tensor,
    coordinates: torch.Tensor,
) -> list[tuple[int, int]]:
    """Convert coordinates to pixel indices.
    Args:
        x (torch.Tensor): The x-coordinates of the spatial grid.
        y (torch.Tensor): The y-coordinates of the spatial grid.
        coordinates (torch.Tensor): The coordinates to convert. x-coordinates
            are in `coordinates[:, 0]` and y-coordinates are in
            `coordinates[:, 1]`.
    Returns:
        list[tuple[int, int]]: The indices of the coordinates.
    """
    indices: list[tuple[int, int]] = []
    for i in range(coordinates.shape[0]):
        indices.append(
            torch.unravel_index(
                torch.argmin(
                    (x - coordinates[i, 0]).abs() + (y - coordinates[i, 1]).abs()
                ),
                x.shape,
            )
        )
    return indices


def find_zero_crossing_intersections(
    complex_amplitude: ComplexAmplitude,
    target_intensity: ArrayLike,
    threshold: float = 0.2,
) -> ArrayLike:
    """Find the intersections of zero crossings in the real and imaginary
    parts of the `electric_field`. Only considers intersections where the
    `target_intensity` is above a given `threshold`.

    Args:
        complex_amplitude (ComplexAmplitude): The complex electric field.
        target_intensity (ArrayLike): The target intensity to threshold the
            zero crossings.
        threshold (float, optional): The intensity threshold to consider a
            zero crossing valid. Defaults to 0.2.
    """
    zero_crossings_real = find_zero_crossings(complex_amplitude.real)
    zero_crossings_imag = find_zero_crossings(complex_amplitude.imag)
    zero_crossings = zero_crossings_real & zero_crossings_imag
    return zero_crossings * (target_intensity > threshold)


def label_connected_components(
    boolean_mask: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Label connected components in a boolean mask. Uses
    `scipy.ndimage.label` for labeling.

    Args:
        boolean_mask (torch.Tensor): Boolean mask to label.
    Returns:
        tuple[torch.Tensor, int]: A tuple containing the labeled mask and the
        number of labels.
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
        x (torch.Tensor): The x-coordinates of the spatial grid.
        y (torch.Tensor): The y-coordinates of the spatial grid.
        labels (torch.Tensor): The labeled mask.
    Returns:
        torch.Tensor: The coordinates of the centers of the labeled regions.
            x-coordinates are in `centers[:, 0]` and y-coordinates are in
            `centers[:, 1]`.
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
        complex_amplitude (ComplexAmplitude): The complex electric field.
        x (torch.Tensor): The x-coordinates of the spatial grid.
        y (torch.Tensor): The y-coordinates of the spatial grid.
        center_indices (torch.Tensor): The coordinates of the vortex centers.
            x-coordinates are in `center_indices[:, 1]` and y-coordinates are in
            `center_indices[:, 0]`.
        pad (int, optional): The padding around the center to consider for
            charge calculation. Defaults to 1.
    Returns:
        torch.Tensor: The charges of the vortices. The charge is +1 for a
            clockwise vortex, -1 for a counter-clockwise vortex, and 0 if the
            phase difference is smaller than pi.
    """
    charges = torch.zeros(
        len(center_indices), dtype=torch.int, device=complex_amplitude.device
    )

    for i in range(len(center_indices)):
        center_index = center_indices[i]

        roi: ComplexAmplitude = complex_amplitude[
            center_index[0] - pad : center_index[0] + pad + 1,
            center_index[1] - pad : center_index[1] + pad + 1,
        ]

        roi_phase = roi.phase

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

# TODO: Move to analysis/unwrapping.py
def unwrap_phase_1D(phase: torch.Tensor) -> torch.Tensor:
    """Unwrap a 1D phase array.
    Args:
        phase (torch.Tensor): 1D phase tensor.

    Returns:
        torch.Tensor: Unwrapped 1D phase tensor.
    """
    unwrapped_phase = phase.clone()
    for i in range(1, phase.shape[0]):
        delta = phase[i] - phase[i - 1]
        if delta > torch.pi:
            unwrapped_phase[i:] -= 2 * torch.pi
        elif delta < -torch.pi:
            unwrapped_phase[i:] += 2 * torch.pi
    return unwrapped_phase

# TODO: Move to profiles/phase.py
def vortex_phase(
    x: torch.Tensor,
    y: torch.Tensor,
    center_coordinates: torch.Tensor,
    charge: torch.Tensor,
) -> torch.Tensor:
    """Calculate the phase of an optical vortex at a given `center` with a
    given `charge`.

    Args:
        x (torch.Tensor): x-coordinates of the spatial grid.
        y (torch.Tensor): y-coordinates of the spatial grid.
        center_coordinates (torch.Tensor): Coordinates of the vortex center.
            x-coordinate is in `center_coordinates[0]` and y-coordinate is in
            `center_coordinates[1]`.
        charge (torch.Tensor): Charge of the vortex.

    Returns:
        torch.Tensor: The phase of the vortex.
    """
    phase = charge * torch.angle(
        x - center_coordinates[0] + 1j * (y - center_coordinates[1])
    )
    return phase

# TODO: Move to profiles/phase.py
def vortex_field(
    x: torch.Tensor,
    y: torch.Tensor,
    vortex_coordinates: torch.Tensor,
    charges: torch.Tensor,
) -> torch.Tensor:
    """
    Calculated the electric field of multiple vortices at given `centers` with
    given `charges`.

    Args:
        x (torch.Tensor): x-coordinates of the spatial grid.
        y (torch.Tensor): y-coordinates of the spatial grid.
        vortex_coordinates (torch.Tensor): Coordinates of the vortex centers.
            x-coordinates are in `vortex_coordinates[:, 0]` and y-coordinates
            are in `vortex_coordinates[:, 1]`.
        charges (torch.Tensor): Charges of the vortices.
    Returns:
        torch.Tensor: The resulting electric field with vortices.
    """
    for i in range(vortex_coordinates.shape[0]):
        charge = charges[i]
        center = vortex_coordinates[i, :]
        if i == 0:
            field = torch.exp(1j * vortex_phase(x, y, center, charge))
        else:
            field *= torch.exp(1j * vortex_phase(x, y, center, charge))
    return field
