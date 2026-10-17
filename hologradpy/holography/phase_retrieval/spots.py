"""The field of a Fourier lens at a set of focal spots, as a sum over the SLM pixels.

The sum follows R. Di Leonardo et al., Opt. Express 15, 1913 (2007), so each spot lands
exactly at its position. It is separable along the rows and the columns of the SLM, so
``N`` spots take ``N * (height + width)`` phase factors and two matrix products.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from numpy.typing import ArrayLike
from torch import Tensor

from ...grids import centered_indices
from ...profiles.phase import tilt_to_angle

if TYPE_CHECKING:
    from ...optics.systems import SLMFourierLensModel


def spot_phasors(
    slm_camera_model: SLMFourierLensModel,
    positions: ArrayLike,
    resolution: tuple[int, int],
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """The phase factors that carry the SLM field to focal spots, ``(N, height)``
    along the rows of the SLM and ``(N, width)`` along its columns.

    The field at spot ``n`` is the sum over the SLM of the field times
    ``row_phasors[n, :, None] * column_phasors[n, None, :]``, which is the Fourier
    transform of the lens evaluated at the position of the spot. The phases are
    computed in float64 at the first wavelength of the model.

    Args:
        slm_camera_model: The model, for the pitch and the wavelength of its SLM plane
            and the focal length of its lens.
        positions: ``(N, 2)`` focal-plane positions, ``(x, y)`` in metres from the
            zeroth order.
        resolution: The ``(height, width)`` of the SLM field.
        dtype: The complex dtype of the phase factors.

    Returns:
        tuple[Tensor, Tensor]: The row and the column phase factors, on the device of
        the model.
    """
    geometry = slm_camera_model.input_geometry
    pitch_y, pitch_x = geometry.pixel_size.reshape(-1, 2)[0].tolist()
    wavenumber = float(geometry.wavenumber.reshape(-1)[0])
    device = slm_camera_model.device
    positions = torch.as_tensor(positions, dtype=torch.float64, device=device)
    # The deflection angles that steer the beam onto the spots.
    angles = tilt_to_angle(positions, focal_length=slm_camera_model.focal_length)
    rows = centered_indices(resolution[0], device, torch.float64) * pitch_y
    columns = centered_indices(resolution[1], device, torch.float64) * pitch_x
    row_phasors = torch.exp(-1j * wavenumber * angles[:, 1:] * rows)
    column_phasors = torch.exp(-1j * wavenumber * angles[:, :1] * columns)
    return row_phasors.to(dtype), column_phasors.to(dtype)


def propagate_to_spots(
    field: Tensor, row_phasors: Tensor, column_phasors: Tensor
) -> Tensor:
    """The field at the spots of :func:`spot_phasors`.

    Args:
        field: The ``(height, width)`` SLM field.
        row_phasors: The ``(N, height)`` phase factors along the rows.
        column_phasors: The ``(N, width)`` phase factors along the columns.

    Returns:
        Tensor: The ``(N,)`` field at the spots.
    """
    return (row_phasors * (field @ column_phasors.T).T).sum(-1)


def propagate_from_spots(
    spot_field: Tensor, row_phasors: Tensor, column_phasors: Tensor
) -> Tensor:
    """The SLM field that ``spot_field`` sends back, by the adjoint of
    :func:`propagate_to_spots`.

    Args:
        spot_field: The ``(N,)`` field at the spots, in the dtype of the phase factors.
        row_phasors: The ``(N, height)`` phase factors along the rows.
        column_phasors: The ``(N, width)`` phase factors along the columns.

    Returns:
        Tensor: The ``(height, width)`` SLM field.
    """
    return (row_phasors.conj().T * spot_field) @ column_phasors.conj()
