from __future__ import annotations
import time

import numpy as np
from numpy.typing import NDArray

import torch
import torchmin

from ..propagation.optical_systems import VirtualSlm

from ..propagation.utils.tensor_utils import gpu_to_numpy

from .loss_functions import rms, eff


class PhaseRetrieval:
    """
    This function calculates the SLM phase pattern for a given target light
    potential in the image plane using conjugate gradient minimisation or
    stochastic gradient descent (Adam).
    """

    def __init__(
        self,
        slm_obj: VirtualSlm,
        n_iter: int = 10,
        i_tar: NDArray[np.float_] = None,
        phi_tar: NDArray[np.float_] = None,
        signal_region: NDArray[np.float_] = None,
        save: bool = False,
        n_save: int = 10,
        loss_type: str = "amp",
        optim_type: str = "cg",
    ):
        """
        :param slm_obj: Virtual SLM object created by :py:class:`VirtualSlm`.
        :param n_iter: Number of iterations.
        :param i_tar: Target light potential.
        :param phi_tar: Target phase pattern.
        :param signal_region: Binary mask containing signal region.
        :param bool save: Save SLM phase pattern and electric field at the
            image plane after every ``n_save`` th
            iteration.
        :param n_save: See line above.
        :param str loss_type: Which cost function to use.

            -'amp'
                Amplitude-only cost function.
            -'fid'
                Phase and amplitude cost function.

        :param str optim_type: Which gradient-based optimiser to use

            -'cg'
                Conjugate gradient algorithm.
            -'adam'
                Adam optimiser.
        """

        self.slm_obj = slm_obj
        fft_shift = slm_obj.fft_shift

        self.signal = signal_region
        self.i_tar = i_tar
        self.fft_shift = fft_shift

        # Initialise signal region, target intensity and phase patterns
        signal_t = torch.tensor(signal_region, dtype=torch.bool).to(slm_obj.device)
        i_tar_t = torch.tensor(
            i_tar / np.sum(i_tar * signal_region), dtype=self.slm_obj.dtype_r
        ).to(slm_obj.device)
        if fft_shift is False and slm_obj.propagation_type == "fft":
            signal_t = torch.fft.ifftshift(signal_t)
            i_tar_t = torch.fft.ifftshift(i_tar_t)

        if phi_tar is None:
            phi_tar = np.zeros_like(self.i_tar)
        self.phi_tar = phi_tar
        self.phi_tar_t = torch.tensor(phi_tar).to(slm_obj.device)

        self.signal_t = signal_t
        self.i_tar_t = i_tar_t

        self.save = save
        self.n_save = n_save
        self.n_iter = n_iter

        self.loss = 0
        self.closure_counter = 0
        self.callback_counter = 0
        self.phi = []
        self.eta_pred = []
        self.eff_pred = []

        self.loss_type = loss_type

        self.optim_type = optim_type
        self.optimizer = None
        self.set_optimizer()

    def set_target(self, target: NDArray) -> None:
        """
        Sets the target light potential.

        :param target: Target light potential.
        """
        self.i_tar = target / np.sum(target * self.signal)
        self.i_tar_t = torch.tensor(self.i_tar, dtype=self.slm_obj.dtype_r).to(
            self.slm_obj.device
        )
        if self.fft_shift is False and self.slm_obj.propagation_type == "fft":
            self.i_tar_t = torch.fft.ifftshift(self.i_tar_t)

    def set_optimizer(self) -> None:
        """
        Sets the optimisation algorithm based on ``self.optim_type``.
        """
        if self.optim_type == "cg":
            self.optimizer = torchmin.Minimizer(
                self.slm_obj.parameters(),
                method="cg",
                max_iter=self.n_iter,
                disp=1,
                callback=self.callback,
            )
        elif self.optim_type == "adam":
            self.optimizer = torch.optim.Adam(self.slm_obj.parameters(), lr=0.01)

    def loss_fn(self, e_out: torch.Tensor) -> torch.Tensor:
        """
        Defines the loss function based on ``self.loss_type``.

        :param e_out: Electric field at the image plane.
        :return: Loss value.
        """
        if self.loss_type == "amp":
            return loss_function_intensity_mse(e_out, self.i_tar_t, self.signal_t)
        elif self.loss_type == "fid":
            return loss_fn_fid(e_out, self.i_tar_t, self.phi_tar_t, self.signal_t)

    def callback(self, x: torch.Tensor) -> None:
        """
        This function is called after every iteration of the optimisation. It
        saves intermediate SLM phase patterns and the electric field in the
        image plane if ``save=True``. The progress of the optimisation is
        printed after every iteration.
        """
        self.callback_counter += 1
        if self.save is True and self.callback_counter % self.n_save == 0:
            e_out_cb = self.slm_obj()
            self.eta_pred.append(
                gpu_to_numpy(
                    rms(self.signal_t, self.i_tar_t, torch.abs(e_out_cb) ** 2, 0.5)
                )
            )
            self.eff_pred.append(
                gpu_to_numpy(eff(self.signal_t, torch.abs(e_out_cb) ** 2))
            )
            self.phi.append(gpu_to_numpy(self.slm_obj.phi_disp))

        print(
            "CG iteration #",
            self.callback_counter,
            "Cost:",
            self.loss,
            "Cost function evaluations:",
            self.closure_counter,
        )

    def retrieve_phase(
        self,
    ) -> tuple[NDArray[np.float_], tuple[NDArray[np.float_], NDArray[np.float_]]]:
        """
        Performs phase retrieval algorithm.

        :return: Optimised SLM phase(s), (RMS error and efficiency if
            ``save=True``)
        """
        self.callback_counter = 0
        self.closure_counter = 0
        self.phi = []
        self.eta_pred = []
        self.eff_pred = []

        date = time.strftime("%d-%m-%y__%H-%M-%S", time.localtime())
        print("\nMaximum iteration number : {0}".format(self.n_iter))
        print("Calculation start : %s\n" % date)

        if self.slm_obj.device == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            start = time.time()

        if self.optim_type == "cg":

            def closure():
                self.closure_counter += 1
                self.optimizer.zero_grad()
                loss = self.loss_fn(self.slm_obj())
                self.loss = loss.item()
                return loss

            self.optimizer.step(closure)

        elif self.optim_type == "adam":
            for t in range(self.n_iter):
                self.optimizer.zero_grad()
                loss = self.loss_fn(self.slm_obj())
                self.callback(0)
                print(t, loss.item())

                loss.backward(retain_graph=True)
                self.optimizer.step()

        if self.slm_obj.device == "cuda":
            end.record()
            torch.cuda.synchronize()
            runtime = start.elapsed_time(end) / 1e3
        else:
            end = time.time()
            runtime = end - start

        print("Ran for %.3fs" % runtime)
        print("Ran for %.0f min and %.3fs" % (runtime // 60, runtime % 60))

        # Tidy up to save GPU memory
        if self.slm_obj.device == "cuda":
            torch.cuda.empty_cache()
        return self.phi, (np.asarray(self.eta_pred), np.asarray(self.eff_pred))
