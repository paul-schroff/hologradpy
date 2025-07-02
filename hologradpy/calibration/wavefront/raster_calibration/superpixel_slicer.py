import numpy as np
from numpy.typing import NDArray

class SuperpixelSlicer:
    def __init__(
        self,
        slm_shape: tuple[int, int],
        number_of_superpixels_x: int,
        number_of_superpixels_y: int,
        superpixel_width: int,
        superpixel_height: int,
        start_index_x: int = 0,
        start_index_y: int = 0,
        end_index_x: int = None,
        end_index_y: int = None,
        intensity: NDArray[np.float_] | None = None,
        max_correction_factor: int = 4,
    ) -> None:
        # TODO: Make this function less convoluted: The same can be achieved
        # with fewer lines of code.
        self.slm_shape = slm_shape
        self.number_of_superpixels_x = number_of_superpixels_x
        self.number_of_superpixels_y = number_of_superpixels_y

        self._number_of_superpixels = (
            self.number_of_superpixels_x * self.number_of_superpixels_y
        )

        self.superpixel_width = superpixel_width
        self.superpixel_height = superpixel_height

        self.superpixel_aspect = self.superpixel_width / self.superpixel_height
        self.superpixel_area = self.superpixel_width * self.superpixel_height

        if intensity is None:
            self.intensity_compensation = False
        else:
            self.intensity_compensation = True
            self.intensity = intensity

        start_indices_x = np.floor(
            np.linspace(
                start_index_x, 
                end_index_x - superpixel_width,
                number_of_superpixels_x
            )
        ).astype("int")

        end_indices_x = start_indices_x + superpixel_width

        self.start_indices_x =(
            np.tile(start_indices_x, number_of_superpixels_y)
        )
        self.end_indices_x = np.tile(end_indices_x, number_of_superpixels_y)

        start_indices_y = np.floor(
            np.linspace(
                start_index_y,
                end_index_y - superpixel_height,
                number_of_superpixels_y
            )
        ).astype("int")

        end_indices_y = start_indices_y + superpixel_height

        self.start_indices_y = (
            np.repeat(start_indices_y, number_of_superpixels_x)
        )
        self.end_indices_y = (
            np.repeat(end_indices_y, number_of_superpixels_x)
        )

        self.superpixel_indices = list(range(self._number_of_superpixels))

        self.slices = []
        if self.intensity_compensation:
            for i in self.superpixel_indices:
                self.slices.append(
                    self.get_intensity_adjusted_slice(i, max_correction_factor)
                )
            self.remove_invalid_slices()
        else:
            for i in self.superpixel_indices:
                self.slices.append(self.get_slice(i))


    @property
    def number_of_superpixels(self) -> int:
        self._number_of_superpixels = len(self.slices)
        return self._number_of_superpixels
    

    def get_slice(self, superpixel_index: int) -> tuple[slice, slice]:
        superpixel_slice = (
            slice(
                self.start_indices_y[superpixel_index],
                self.end_indices_y[superpixel_index],
            ),
            slice(
                self.start_indices_x[superpixel_index],
                self.end_indices_x[superpixel_index],
            ),
        )
        return superpixel_slice
    

    def remove_invalid_slices(
        self,
    ) -> list[tuple[slice, slice]]:
        """
        Removes slices that are out of bounds of the SLM shape.
        """
        valid_slices = []
        for slice_ in self.slices:
            if ((0 <= slice_[0].start <= self.slm_shape[0]) and
                (0 <= slice_[0].stop <= self.slm_shape[0]) and
                (0 <= slice_[1].start <= self.slm_shape[1]) and
                (0 <= slice_[1].stop <= self.slm_shape[1])):
                valid_slices.append(slice_)
        self.slices = valid_slices


    @property
    def central_index(self) -> int:
        """
        Index of the central superpixel.
        Returns
        -------
        int
            The index of the central superpixel.
        """
        return (
            self.number_of_superpixels_y // 2 * self.number_of_superpixels_x
            + self.number_of_superpixels_x // 2
        )


    @property
    def central_slice(self) -> tuple[slice, slice]:
        """
        Slice of the central superpixel.
        Returns
        -------
        tuple[slice, slice]
            The slice of the central superpixel.
        """
        return self.get_slice(self.central_index)
    

    def get_superpixel_power(self, superpixel_index: int) -> float:
        superpixel_slice = self.get_slice(superpixel_index)
        return np.sum(self.intensity[superpixel_slice])
    

    @property
    def reference_index(self) -> int:
        """Returns the index of the superpixel with the highest intensity."""
        superpixel_powers = np.asarray([
            self.get_superpixel_power(i) for i in self.superpixel_indices
        ])
        return self.superpixel_indices[np.argmax(superpixel_powers)]
    
    
    @property
    def reference_slice(self) -> tuple[slice, slice]:
        """Returns the slice of the superpixel with the highest intensity."""
        return self.get_slice(self.reference_index)
    

    @property
    def reference_power(self) -> float:
        return self.get_superpixel_power(self.reference_index)
    

    def get_intensity_adjusted_slice(
        self,
        superpixel_index: int,
        max_correction_factor: int
    ) -> tuple[slice, slice]:
        """Returns the slice of a superpixel which accounts for the varying
        intensity profile across the SLM. The size of the superpixel is
        increased in darker regions of the SLM to maintain the same spot
        intensity in the camera image. Note that increasing the size of the 
        superpixel will also decrease the size of the diffraction spot in the
        camera image.

        Args:
            superpixel_index (int): The index of the superpixel to adjust.
        Returns:
            tuple[slice, slice]: The adjusted slice for the superpixel.
        """
        superpixel_slice = self.get_slice(superpixel_index)
        superpixel_power = np.sum(self.intensity[superpixel_slice])

        power_fraction = superpixel_power / self.reference_power

        correction_factor = (1 / power_fraction) ** 0.25
        correction_factor = min(correction_factor, max_correction_factor)

        adjusted_superpixel_width = self.superpixel_width * correction_factor
        adjusted_superpixel_height = self.superpixel_height * correction_factor
        
        pad_width = int(adjusted_superpixel_width - self.superpixel_width) // 2
        pad_height = int(adjusted_superpixel_height - self.superpixel_height) // 2

        adjusted_slice = (
            slice(
                superpixel_slice[0].start - pad_height,
                superpixel_slice[0].stop + pad_height
            ),
            slice(
                superpixel_slice[1].start - pad_width,
                superpixel_slice[1].stop + pad_width
            ),
        )
        return adjusted_slice

