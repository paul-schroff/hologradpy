from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .vortex_detection import VortexDetector
from .visualizer import VortexAnnihilationData

from ..phase_retrieval.abstract import GradientPhaseRetriever


class VortexAnnihilator:
    """Remove phase vortices from a retrieved light potential.

    A vortex is a phase winding in image-plane, which forces the intensity there to
    zero. Conjugate gradient cannot undo one by itself as the winding is topological, so
    no small change to the SLM phase removes it. This detects them instead, multiplies 
    in a field of the opposite charge to cancel the winding, propagates that back to the
    SLM, and restarts the retrieval from there.
    """

    def __init__(self, phase_retriever: GradientPhaseRetriever) -> None:
        """
        Args:
            phase_retriever: The retriever whose model and target are used, and whose
                SLM phase is corrected in place.
        """
        self.phase_retriever = phase_retriever
        self.vortex_detector = VortexDetector(
            self.phase_retriever.slm_camera_model[-1].resolution_out,
            device=self.phase_retriever.device,
        )

    def _state(self) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """The image-plane intensity and phase, and the vortices currently in them."""
        complex_amplitude = self.phase_retriever.slm_camera_model()
        positions, charges = self._charged_vortices()
        return (
            complex_amplitude.intensity.detach().cpu().numpy(),
            complex_amplitude.phase.detach().cpu().numpy(),
            positions,
            charges,
        )

    def _charged_vortices(self) -> tuple[NDArray, NDArray]:
        """The vortices the detector found that actually carry charge."""
        positions = np.asarray(
            [
                (int(row), int(column))
                for row, column in self.vortex_detector.center_indices
            ],
            dtype=int,
        ).reshape(-1, 2)
        charges = self.vortex_detector.charges.detach().cpu().numpy().reshape(-1)
        charged = charges != 0
        return positions[charged], charges[charged]

    def annihilate_vortices(
        self,
        target_intensity_threshold: float = 0.2,
        max_iterations: int = 5,
        cg_iterations: int = 20,
        verbose: bool = True,
    ) -> VortexAnnihilationData:
        """Detect and cancel vortices, retrieving again after each round.

        Args:
            target_intensity_threshold: Fraction of the peak target intensity above
                which a pixel is searched. Vortices outside the signal region do not
                matter, and looking for them there finds noise.
            max_iterations: Rounds to try before giving up.
            cg_iterations: Conjugate gradient iterations after each correction.
            verbose: Print the count found at each round.

        Returns:
            VortexAnnihilationData: The field before and after, the vortices found in
                each, and the count per round. Its last count is zero when the run
                converged.
        """
        target_intensity = self.phase_retriever.target

        counts: list[int] = []
        initial: tuple[NDArray, NDArray, NDArray, NDArray] | None = None
        iteration = 0
        if verbose:
            print("Starting vortex annihilation...")

        while True:
            iteration += 1

            complex_amplitude = self.phase_retriever.slm_camera_model()

            self.vortex_detector.detect_vortices(
                complex_amplitude,
                target_intensity=target_intensity,
                threshold=target_intensity_threshold,
                pad=1,
            )

            number_of_vortices = len(self._charged_vortices()[1])
            counts.append(number_of_vortices)
            if initial is None:
                initial = self._state()

            if verbose:
                print(f"Iteration {iteration}: Detected {number_of_vortices} vortices.")

            if number_of_vortices > 0:
                anti_vortex_field = self.vortex_detector.generate_anti_vortex_field()

                corrected_field = complex_amplitude * anti_vortex_field

                corrected_slm_phase = (
                    self.phase_retriever.slm_camera_model.fourier_lens.adjoint(
                        corrected_field
                    )
                ).phase

                self.phase_retriever.slm_camera_model.virtual_slm.set_phase(
                    corrected_slm_phase
                )

                self.phase_retriever.retrieve_phase(cg_iterations)
            else:
                if verbose:
                    print("No vortices detected to remove, stopping.")
                break

            if iteration >= max_iterations:
                if verbose:
                    print("Reached maximum iterations, stopping vortex removal.")
                break

        signal_region = self.phase_retriever.signal_region

        final = self._state()
        return VortexAnnihilationData(
            counts=counts,
            signal_region=(
                None
                if signal_region is None
                else signal_region.detach().cpu().numpy().astype(bool)
            ),
            initial_intensity=initial[0],
            initial_phase=initial[1],
            initial_positions=initial[2],
            initial_charges=initial[3],
            final_intensity=final[0],
            final_phase=final[1],
            final_positions=final[2],
            final_charges=final[3],
        )
