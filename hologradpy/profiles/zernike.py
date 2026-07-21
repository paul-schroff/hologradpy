from typing import Literal
import torch

from ..grids import get_pixel_grid

Conventions = Literal["OSA", "ANSI", "Noll", "Fringe", "Arizona", "Wyant"]


class ZernikeConventionHandler:
    _Canonical = Literal["osa_ansi", "noll", "fringe_arizona", "wyant"]

    convention_mapping: dict[Conventions, _Canonical] = {
        "OSA": "osa_ansi",
        "ANSI": "osa_ansi",
        "Noll": "noll",
        "Fringe": "fringe_arizona",
        "Arizona": "fringe_arizona",
        "Wyant": "wyant",
    }

    def __init__(self, convention: Conventions) -> None:
        self._canonical_convention = self.convention_mapping[convention]

    @staticmethod
    def osa_ansi_indices_to_nm(j: int) -> tuple[int, int]:
        n = torch.floor(torch.tensor(((8 * j + 1) ** 0.5 - 1) / 2)).item()
        m = 2 * j - n * (n + 2)
        return int(n), int(m)

    @staticmethod
    def noll_indices_to_nm(j: int) -> tuple[int, int]:
        n = 0
        while (n + 1) * (n + 2) // 2 < j:
            n += 1

        # Build the ordered m sequence for this radial group.
        # Noll rule: even j → m ≥ 0, odd j → m ≤ 0.
        # Within the group, modes are ordered by |m| ascending; for each |m|
        # pair the sign is determined by the running j parity at that position.
        j_start = n * (n + 1) // 2 + 1
        ordered_ms: list[int] = []
        if n % 2 == 0:
            ordered_ms.append(0)
        for abs_m in range(n % 2 or 2, n + 1, 2):
            current_j = j_start + len(ordered_ms)
            if current_j % 2 == 0:
                ordered_ms.extend([abs_m, -abs_m])
            else:
                ordered_ms.extend([-abs_m, abs_m])

        return n, ordered_ms[j - j_start]

    @staticmethod
    def fringe_arizona_indices_to_nm(j: int) -> tuple[int, int]:
        group = int((j - 1) ** 0.5 // 1)
        position_in_group = j - group * group - 1
        azimuthal_abs = group - position_in_group // 2
        n = 2 * group - azimuthal_abs
        if azimuthal_abs == 0 or position_in_group % 2 == 0:
            m = azimuthal_abs
        else:
            m = -azimuthal_abs
        return n, m

    def get_normalization_factor(
        self,
        radial_order_n: int,
        azimuthal_frequency_m: int,
    ) -> torch.Tensor:
        match self._canonical_convention:
            case "osa_ansi" | "noll":
                if azimuthal_frequency_m == 0:
                    factor = (radial_order_n + 1) ** 0.5
                else:
                    factor = (2 * (radial_order_n + 1)) ** 0.5
            case "fringe_arizona" | "wyant":
                factor = 1
        return factor

    def j_to_nm(self, j: int) -> tuple[int, int]:
        match self._canonical_convention:
            case "osa_ansi":
                return self.osa_ansi_indices_to_nm(j)
            case "noll":
                return self.noll_indices_to_nm(j)
            case "fringe_arizona":
                return self.fringe_arizona_indices_to_nm(j)
            case "wyant":
                # Wyant convention is the same as the Fringe/Arizona,
                # starting at j=0 instead of j=1.
                return self.fringe_arizona_indices_to_nm(j + 1)


class Zernike:
    def __init__(
        self,
        resolution: tuple[int, int],
        unit_disk_mode: Literal["fill", "fit"] = "fit",
        unit_disk_radius: float | None = None,
        number_of_radial_orders: int | None = None,
        indices: list[int | tuple[int, int]] | None = None,
        convention: Conventions = "ANSI",
        device: torch.device = "cpu",
    ) -> None:
        self.resolution = resolution
        self.convention_handler = ZernikeConventionHandler(convention)
        self.device = device
        self.radial_coordinate, self.angular_coordinate, self.mask = (
            self.get_unit_disk_coordinates(
                resolution,
                unit_disk_mode=unit_disk_mode,
                unit_disk_radius=unit_disk_radius,
                device=self.device,
            )
        )

        self._zernike_array, self._indices = self.get_zernikes(
            number_of_radial_orders=number_of_radial_orders, indices=indices
        )

    def raise_init_error(self):
        if self._zernike_array is None:
            raise ValueError(
                "Zernike polynomials have not been generated yet. Call "
                "get_zernikes() to generate the Zernike polynomials before "
                "accessing them."
            )

    @property
    def zernike_array(self) -> torch.Tensor:
        return self._zernike_array

    @property
    def number_of_zernikes(self) -> int:
        return len(self._indices)

    @property
    def indices(self) -> list[int] | None:
        self.raise_init_error()
        return self._indices

    @staticmethod
    def log_factorial(n: int, device: torch.device) -> torch.Tensor:
        n = torch.tensor(n, device=device)
        return torch.special.gammaln(n + 1)

    @staticmethod
    def sanitize_zernike_indices(
        radial_order_n: int, azimuthal_frequency_m: int
    ) -> None:
        if radial_order_n < 0 or not isinstance(radial_order_n, int):
            raise ValueError("Radial order n must be a non-negative integer.")

        if (
            abs(azimuthal_frequency_m) > radial_order_n
            or (radial_order_n - abs(azimuthal_frequency_m)) % 2 != 0
            or not isinstance(azimuthal_frequency_m, int)
        ):
            raise ValueError(
                "Invalid azimuthal frequency m for the given radial order n."
                " Must satisfy abs(m) <= n and (n - abs(m)) % 2 == 0."
            )

    def get_radial_polynomial(
        self,
        radial_order_n: int,
        azimuthal_frequency_m: int,
        radial_coordinate: torch.Tensor,
    ) -> torch.Tensor:
        # Implementation for radial polynomial calculation
        k_max = (radial_order_n - abs(azimuthal_frequency_m)) // 2

        radial_polynomial = torch.zeros_like(radial_coordinate)
        for k in range(k_max + 1):
            factorial_ratio = (
                self.log_factorial(radial_order_n - k, device=self.device)
                - self.log_factorial(k, device=self.device)
                - self.log_factorial(
                    (radial_order_n + abs(azimuthal_frequency_m)) // 2 - k,
                    device=self.device,
                )
                - self.log_factorial(
                    (radial_order_n - abs(azimuthal_frequency_m)) // 2 - k,
                    device=self.device,
                )
            ).exp()

            radial_polynomial += (
                (-1) ** k
                * factorial_ratio
                * radial_coordinate ** (radial_order_n - 2 * k)
            )
        return radial_polynomial

    def get_zernike_basis_function(
        self,
        radial_order_n: int,
        azimuthal_frequency_m: int,
        radial_coordinate: torch.Tensor,
        angular_coordinate: torch.Tensor,
    ) -> torch.Tensor:

        radial_polynomial = self.get_radial_polynomial(
            radial_order_n, azimuthal_frequency_m, radial_coordinate
        )

        normalization_factor = self.convention_handler.get_normalization_factor(
            radial_order_n, azimuthal_frequency_m
        )

        if azimuthal_frequency_m >= 0:
            # Even functions
            factor = torch.cos(azimuthal_frequency_m * angular_coordinate)
        else:
            # Odd functions
            factor = torch.sin(abs(azimuthal_frequency_m) * angular_coordinate)
        return normalization_factor * radial_polynomial * factor

    def get_unit_disk_coordinates(
        self,
        resolution: tuple[int, int],
        unit_disk_mode: Literal["fill", "fit"] = "fill",
        unit_disk_radius: float | None = None,
        device: torch.device = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if unit_disk_radius is None:
            if unit_disk_mode == "fit":
                unit_disk_radius = min(resolution) / 2
            elif unit_disk_mode == "fill":
                unit_disk_radius = (
                    0.5 * (resolution[0] ** 2 + resolution[1] ** 2) ** 0.5
                )
            else:
                raise ValueError("Invalid unit_disk_mode. Must be 'fill' or 'fit'.")

        x, y = get_pixel_grid(resolution, device)

        # unit_disk_radius = torch.tensor(unit_disk_radius, device=device)

        radial_coordinate = (x**2 + y**2) ** 0.5 / unit_disk_radius
        angular_coordinate = torch.atan2(y, x)
        mask = radial_coordinate <= 1

        return radial_coordinate, angular_coordinate, mask

    def evaluate_zernike_polynomials(
        self, radial_orders_n: list[int], azimuthal_frequencies_m: list[int]
    ) -> torch.Tensor:
        if len(radial_orders_n) != len(azimuthal_frequencies_m):
            raise ValueError(
                "Length of radial_orders_n and azimuthal_frequencies_m must "
                "be the same."
            )

        zernike_polynomials: list[torch.Tensor] = []

        for n, m in zip(radial_orders_n, azimuthal_frequencies_m):
            self.sanitize_zernike_indices(n, m)
            zernike_polynomial = self.get_zernike_basis_function(
                n, m, self.radial_coordinate, self.angular_coordinate
            )
            zernike_polynomials.append(zernike_polynomial * self.mask)

        return torch.stack(zernike_polynomials)

    def get_zernikes(
        self,
        number_of_radial_orders: int | None = None,
        indices: list[int | tuple[int, int]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        if indices is not None and number_of_radial_orders is not None:
            raise ValueError("Cannot specify both indices and number_of_radial_orders.")
        elif indices is not None:
            radial_orders_n = []
            azimuthal_frequencies_m = []

            for index in indices:
                if isinstance(index, int):
                    n, m = self.convention_handler.j_to_nm(index)
                else:
                    n, m = index

                radial_orders_n.append(n)
                azimuthal_frequencies_m.append(m)
        elif number_of_radial_orders is not None:
            radial_orders_n = []
            azimuthal_frequencies_m = []
            for n in range(number_of_radial_orders):
                for m in range(-n, n + 1, 2):
                    radial_orders_n.append(n)
                    azimuthal_frequencies_m.append(m)
        else:
            raise ValueError("Must specify either indices or number_of_radial_orders.")

        indices = [(n, m) for n, m in zip(radial_orders_n, azimuthal_frequencies_m)]

        return (
            self.evaluate_zernike_polynomials(radial_orders_n, azimuthal_frequencies_m),
            indices,
        )

    def get_phase(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Reconstruct a phase from Zernike coefficients (inverse of
        :meth:`fit`).

        Args:
            coefficients (torch.Tensor): Coefficients of shape
                ``(*batch, number_of_zernikes)``.

        Returns:
            torch.Tensor: Phase of shape ``(*batch, H, W)``.
        """
        if coefficients.shape[-1] != self.number_of_zernikes:
            raise ValueError(
                f"Number of coefficients must match the number of Zernike "
                f"polynomials. Expected {self.number_of_zernikes}, got "
                f"{coefficients.shape[-1]}."
            )
        return torch.einsum(
            "...c,chw->...hw",
            coefficients,
            self.zernike_array.to(coefficients.dtype),
        )

    def fit(
        self,
        phase: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fit Zernike coefficients to a measured phase (inverse of
        :meth:`get_phase`).

        Solves a masked, weighted linear least-squares problem (via the normal
        equations) for the coefficients that best reconstruct ``phase`` from
        the Zernike basis. Any leading dimensions of ``phase`` (e.g. batch and
        wavelength) are fitted independently. Because the basis is sampled on a
        discrete, masked grid it is not perfectly orthonormal, so the fit is a
        least-squares solve rather than a direct projection.

        Args:
            phase (torch.Tensor): Measured phase of shape ``(*batch, H, W)``.
            mask (torch.Tensor | None, optional): Mask of the pixels to fit
                over, broadcastable to ``phase`` (e.g. ``(H, W)`` shared,
                ``(n_wavelengths, H, W)`` per wavelength, or ``(*batch, H, W)``
                per sample). Combined with the unit-disk mask. Defaults to the
                unit disk only.

        Returns:
            torch.Tensor: Coefficients of shape
            ``(*batch, number_of_zernikes)``.
        """
        number_of_pixels = self.resolution[0] * self.resolution[1]
        basis = self.zernike_array.reshape(
            self.number_of_zernikes, number_of_pixels
        ).to(torch.float64)

        phase_flat = phase.reshape(*phase.shape[:-2], number_of_pixels).to(
            torch.float64
        )

        weight = self.mask.reshape(number_of_pixels).to(torch.float64)
        if mask is not None:
            weight = weight * mask.reshape(
                *mask.shape[:-2], number_of_pixels
            ).to(torch.float64)

        # Weighted normal equations (A^T W A) c = A^T W b, W = diag(weight).
        weighted_basis = weight[..., None, :] * basis
        normal_matrix = weighted_basis @ basis.transpose(-1, -2)
        rhs = torch.einsum("...mp,...p->...m", weighted_basis, phase_flat)

        coefficients = torch.linalg.solve(
            normal_matrix, rhs.unsqueeze(-1)
        ).squeeze(-1)
        return coefficients.to(phase.dtype)


def make_per_wavelength_coefficients(
    initial_coefficients: torch.Tensor | None,
    number_of_wavelengths: int,
    number_of_coefficients: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build a ``(n_wavelengths, n_coefficients)`` Zernike coefficient tensor.

    ``initial_coefficients`` may be:
    - ``None`` -> small random values,
    - a 1D tensor ``(n_coefficients,)`` -> broadcast across all wavelengths,
    - a 2D tensor ``(n_wavelengths, n_coefficients)`` -> used as-is.
    """
    target_shape = (number_of_wavelengths, number_of_coefficients)

    if initial_coefficients is None:
        return 0.1 * torch.rand(target_shape, dtype=dtype, device=device)

    coefficients = torch.as_tensor(
        initial_coefficients, dtype=dtype, device=device
    )
    # A 1D set of coefficients is shared (broadcast) across wavelengths.
    if coefficients.ndim == 1:
        coefficients = coefficients.unsqueeze(0).repeat(
            number_of_wavelengths, 1
        )
    if tuple(coefficients.shape) != target_shape:
        raise ValueError(
            "initial_coefficients must have shape "
            f"({number_of_coefficients},) or {target_shape}, but got "
            f"{tuple(coefficients.shape)}."
        )
    return coefficients
