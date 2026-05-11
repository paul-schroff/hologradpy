import torch

from .vortex_detection import VortexDetector

from ..phase_retrieval.conjugate_gradient import CGPhaseRetriever

# TODO: Docstrings
class VortexAnnihilator:
    def __init__(self, phase_retriever: CGPhaseRetriever):
        self.phase_retriever = phase_retriever
        self.vortex_detector = VortexDetector(
            self.phase_retriever.slm_camera_model[-1].resolution_out,
            device=self.phase_retriever.device
        )
    
    def annihilate_vortices(
        self,
        target_intensity_threshold: float = 0.2,
        max_iterations: int = 5,
        cg_iterations: int = 20,
    ) -> torch.Tensor:
        target_intensity = self.phase_retriever.target

        number_of_vortices = 1
        iteration = 0
        print("Starting vortex annihilation...")

        while True:
            iteration += 1

            complex_amplitude = self.phase_retriever.slm_camera_model()

            self.vortex_detector.detect_vortices(
                complex_amplitude,
                target_intensity=target_intensity,
                threshold=target_intensity_threshold,
                pad=1
            )

            number_of_vortices = self.vortex_detector.number_of_vortices

            print(
                f"Iteration {iteration}: " +
                f"Detected {number_of_vortices} vortices."
            )

            if number_of_vortices > 0:
                anti_vortex_field = (
                    self.vortex_detector.generate_anti_vortex_field()
                )

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
                print("No vortices detected to remove, stopping.")
                break

            if iteration >= max_iterations:
                print("Reached maximum iterations, stopping vortex removal.")
                break