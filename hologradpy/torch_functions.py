"""
Module containing PyTorch-specific functions to perform conjugate gradient
minimisation.
"""

import time
import os
import torch
from torch import Tensor as tt
import torch.nn as nn
import torchmin
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from . import patterns as pt
from .analysis import fitting as ft
from .analysis import error_metrics as m
from .hardware import hardware as hw
import cv2 as cv
from checkerboard import detect_checkerboard
from scipy import ndimage

torch.autograd.set_detect_anomaly(True)
torch.use_deterministic_algorithms(True, warn_only=True)

torch.pi = torch.acos(torch.zeros(1)).item() * 2


class ASM:
    """
    This class models the propagation of light from the SLM to the Fourier
    lens and from the Fourier lens to the image plane using the angular
    spectrum method. The lens is modelled as a doublet. The ASM wavefront
    correction is also calculated in this class.
    """

    def __init__(
        self,
        slm_disp_obj: hw.SlmBase,
        pms_obj: hw.ParamsBase,
        pix_res: int,
        npix_tot: int,
        pd1: float,
        pd2: float,
        extent_lens: float,
        xf: float,
        shift: bool = False,
        precision: str = "single",
        device: str = "cuda",
    ):
        """
        The lens function, propagation phase factor for the ASM and the ASM
        wavefront correction are calculated here.

        :param slm_disp_obj: Object created by a subclass of
            :py:class:`hardware.SlmBase`
        :param pms_obj: Object created by a subclass of
            :py:class:`hardware.ParamsBase`
        :param pix_res: Number of computational pixels per SLM pixel.
        :param npix_tot: Total number of pixels in the computational SLM
            plane.
        :param pd1: Propagation distance from SLM to lens [m].
        :param pd2: Propagation distance from lens to camera [m].
        :param extent_lens: Spatial extent of the lens [m].
        :param xf: Position in the image plane where the phase for the
            wavefront measurement was measured [m].
        :param bool shift: Perform FFT shift?
        :param str precision: Computational precision.

            -'single'
                complex64
            -'double'
                complex128
        :param str device: Device to use (GPU or CPU)
        """
        if precision == "single":
            dtype = torch.complex64
        else:
            dtype = torch.complex128

        # Propagation phase
        k_max = np.pi / slm_disp_obj.slm_pitch * pix_res
        kx = np.linspace(-k_max, k_max, npix_tot)
        k_x, k_y = np.meshgrid(kx, kx)
        self.theta1 = torch.tensor(
            np.exp(1j * pd1 * np.sqrt(pms_obj.k**2 - k_x**2 - k_y**2 + 0j)),
            dtype=dtype,
            device=device,
        )
        self.theta2 = torch.tensor(
            np.exp(1j * pd2 * np.sqrt(pms_obj.k**2 - k_x**2 - k_y**2 + 0j)),
            dtype=dtype,
            device=device,
        )
        self.shift = shift

        if self.shift is False:
            self.theta1 = torch.fft.ifftshift(self.theta1)
            self.theta2 = torch.fft.ifftshift(self.theta2)

        # Make lens field
        range_l = extent_lens / 2
        xl = np.arange(-range_l, range_l, 2 * range_l / npix_tot)
        x_l, y_l = np.meshgrid(xl, xl)
        phi_l = pt.doublet(
            x_l,
            y_l,
            pms_obj.k,
            pms_obj.n1,
            pms_obj.n2,
            pms_obj.r1,
            pms_obj.r2,
            pms_obj.r3,
        )
        a_l = pt.circ_mask_xy(x_l, y_l, 0, 0, extent_lens / 2)
        self.e_lens = torch.tensor(a_l * np.exp(1j * phi_l), dtype=dtype, device=device)

        # quadratic phase to cancel focal length mismatch
        range_q = slm_disp_obj.slm_size[0] / 2
        x_q_native = np.arange(-range_q, range_q, slm_disp_obj.pitch)
        X_q_native, Y_q_native = np.meshgrid(x_q_native, x_q_native)
        phi_corr_native = -pt.slm_phase_doublet(
            X_q_native,
            Y_q_native,
            pms_obj.k,
            xf,
            xf,
            pd1,
            pd2,
            pms_obj.fl,
            pms_obj.n1,
            pms_obj.n2,
            pms_obj.r1,
            pms_obj.r2,
            pms_obj.r3,
        )
        self.phi_corr_native = ft.remove_tilt(phi_corr_native)

        xcorr = np.arange(-range_q, range_q, slm_disp_obj.pitch / pix_res)
        x_corr, y_corr = np.meshgrid(xcorr, xcorr)
        phi_corr = -pt.slm_phase_doublet(
            x_corr,
            y_corr,
            xf,
            xf,
            pd1,
            pd2,
            pms_obj.fl,
            pms_obj.n1,
            pms_obj.n2,
            pms_obj.r1,
            pms_obj.r2,
            pms_obj.r3,
        )
        self.phi_corr = ft.remove_tilt(phi_corr)

    def forward(self, e_in: torch.Tensor) -> torch.Tensor:
        """
        This function performs the simulation.

        :param e_in: Electric field at the SLM plane.
        :return: Electric field at the image plane.
        """
        return asm(e_in, self.e_lens, self.theta1, self.theta2, shift=self.shift)
