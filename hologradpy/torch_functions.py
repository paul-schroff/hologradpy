"""
Module containing PyTorch-specific functions to perform conjugate gradient minimisation.
"""

import time
import os
import torch
from torch import Tensor as tt
import torch.nn as nn
import torchmin
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from . import patterns as pt
from . import fitting as ft
from . import error_metrics as m
from . import hardware as hw
import cv2 as cv
from checkerboard import detect_checkerboard
from scipy import ndimage

torch.autograd.set_detect_anomaly(True)
torch.use_deterministic_algorithms(True, warn_only=True)

torch.pi = torch.acos(torch.zeros(1)).item() * 2


def check_device(verbose=None):
    """
    Check if GPU is available.

    :param bool verbose: Verbose output?
    :return: 'cuda' if GPU available, otherwise 'cpu'.
    """
    if verbose is None:
        verbose = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose is True:
        print(f'Using {device} device')
    return device


def gpu_to_numpy(gpu_tensor):
    return gpu_tensor.clone().cpu().detach().numpy()


def fft(e_in, shift=True, norm=None):
    """
    Performs the FFT.

    :param e_in: Input electric field.
    :param bool shift: Perform FFT shift?
    :param norm: Normalisation of FFT.
    :return: FFT of ``e_in``.
    """
    if shift is True:
        e_out = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(e_in), norm=norm))
    else:
        e_out = torch.fft.fft2(e_in, norm=norm)
    return e_out


def ifft(e_in, shift=True, norm=None):
    """
    Performs the IFFT.

    :param e_in: Input electric field.
    :param bool shift: Perform IFFT shift?
    :param norm: Normalisation of IFFT.
    :return: IFFT of ``e_in``.
    """
    if shift is True:
        e_out = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(e_in), norm=norm))
    else:
        e_out = torch.fft.ifft2(e_in, norm=norm)
    return e_out


def asm(e_in, e_lens, theta1, theta2=None, shift=True):
    """
    Performs the angular spectrum method (ASM) twice, from the SLM to the Fourier lens and from the Fourier lens to the
    camera.

    :param e_in: Electric field at the SLM.
    :param e_lens: Electric field of the lens (phase and aperture).
    :param theta1: Propagation phase from SLM to lens.
    :param theta2: Propagation phase from lens to camera. If not provided it is assumed ``theta2 = theta1``.
    :param bool shift: Perform FFT shift?
    :return: Electric field at the camera.
    """
    e_out = ifft(fft(e_in, shift) * theta1, shift) * e_lens
    if theta2 is None:
        e_out = ifft(fft(e_out, shift) * theta1, shift)
    else:
        e_out = ifft(fft(e_out, shift) * theta2, shift)
    return e_out


class ASM:
    """
    This class models the propagation of light from the SLM to the Fourier lens and from the Fourier lens to the image
    plane using the angular spectrum method. The lens is modelled as a doublet. The ASM wavefront correction is also
    calculated in this class.
    """
    def __init__(self, slm_disp_obj, pms_obj, pix_res, npix_tot, pd1, pd2, extent_lens, xf, shift=False,
                 precision='single', device='cuda'):
        """
        The lens function, propagation phase factor for the ASM and the ASM wavefront correction are calculated here.

        :param slm_disp_obj: Object created by a subclass of :py:class:`hardware.SlmBase`
        :param pms_obj: Object created by a subclass of :py:class:`hardware.ParamsBase`
        :param pix_res: Number of computational pixels per SLM pixel.
        :param npix_tot: Total number of pixels in the computational SLM plane.
        :param pd1: Propagation distance from SLM to lens [m].
        :param pd2: Propagation distance from lens to camera [m].
        :param extent_lens: Spatial extent of the lens [m].
        :param xf: Position in the image plane where the phase for the wavefront measurement was measured [m].
        :param bool shift: Perform FFT shift?
        :param str precision: Computational precision.

            -'single'
                complex64
            -'double'
                complex128
        :param str device: Device to use (GPU or CPU)
        """
        if precision == 'single':
            dtype = torch.complex64
        else:
            dtype = torch.complex128

        # Propagation phase
        k_max = np.pi / slm_disp_obj.slm_pitch * pix_res
        kx = np.linspace(-k_max, k_max, npix_tot)
        k_x, k_y = np.meshgrid(kx, kx)
        self.theta1 = torch.tensor(np.exp(1j * pd1 * np.sqrt(pms_obj.k ** 2 - k_x ** 2 - k_y ** 2 + 0j)), dtype=dtype,
                                   device=device)
        self.theta2 = torch.tensor(np.exp(1j * pd2 * np.sqrt(pms_obj.k ** 2 - k_x ** 2 - k_y ** 2 + 0j)), dtype=dtype,
                                   device=device)
        self.shift = shift

        if self.shift is False:
            self.theta1 = torch.fft.ifftshift(self.theta1)
            self.theta2 = torch.fft.ifftshift(self.theta2)

        # Make lens field
        range_l = extent_lens / 2
        xl = np.arange(-range_l, range_l, 2 * range_l / npix_tot)
        x_l, y_l = np.meshgrid(xl, xl)
        phi_l = pt.doublet(x_l, y_l, pms_obj.k, pms_obj.n1, pms_obj.n2, pms_obj.r1, pms_obj.r2, pms_obj.r3)
        a_l = pt.circ_mask_xy(x_l, y_l, 0, 0, extent_lens / 2)
        self.e_lens = torch.tensor(a_l * np.exp(1j * phi_l), dtype=dtype, device=device)

        # quadratic phase to cancel focal length mismatch
        range_q = slm_disp_obj.slm_size[0] / 2
        x_q_native = np.arange(-range_q, range_q, slm_disp_obj.pitch)
        X_q_native, Y_q_native = np.meshgrid(x_q_native, x_q_native)
        phi_corr_native = -pt.slm_phase_doublet(X_q_native, Y_q_native, pms_obj.k, xf, xf, pd1, pd2, pms_obj.fl,
                                                pms_obj.n1, pms_obj.n2, pms_obj.r1, pms_obj.r2, pms_obj.r3)
        self.phi_corr_native = ft.remove_tilt(phi_corr_native)

        xcorr = np.arange(-range_q, range_q, slm_disp_obj.pitch / pix_res)
        x_corr, y_corr = np.meshgrid(xcorr, xcorr)
        phi_corr = -pt.slm_phase_doublet(x_corr, y_corr, xf, xf, pd1, pd2, pms_obj.fl, pms_obj.n1, pms_obj.n2,
                                         pms_obj.r1, pms_obj.r2, pms_obj.r3)
        self.phi_corr = ft.remove_tilt(phi_corr)

    def forward(self, e_in):
        """
        This function performs the simulation.

        :param e_in: Electric field at the SLM plane.
        :return: Electric field at the image plane.
        """
        return asm(e_in, self.e_lens, self.theta1, self.theta2, shift=self.shift)


class VirtualSlm(nn.Module):
    """
    This class models pixel crosstalk on the SLM and the propagation of light from the SLM to the camera.
    """
    def __init__(self, slm_disp_obj, pms_obj, phi, npix_pad, npix=None, e_slm=None, kernel_ct=None, pix_res=None,
                 propagation_type='fft', extent_lens=None, pd1=None, pd2=None, xf=None, device='cpu', slm_mask=None,
                 precision=None, fft_shift=True):
        """
        :param slm_disp_obj: Object created by a subclass of :py:class:`hardware.SlmBase`
        :param pms_obj: Object created by a subclass of :py:class:`hardware.ParamsBase`
        :param phi: SLM phase pattern [rad].
        :param npix_pad: Size of zero-padded SLM plane [px].
        :param npix: Size of SLM [px].
        :param e_slm: Constant electric field at the SLM [px].
        :param kernel_ct: Blurring kernel to model pixel crosstalk.
        :param pix_res: Computational pixels per SLM pixel.
        :param str propagation_type: Propagation type.

            -'fft'
                Uses the FFT to simulate the propagation of light.
            -'asm'
                Uses the ASM to simulate the propagation of light.
        :param extent_lens: Spatial extent of the Fourier lens
        :param pd1: Propagation distance from the SLM to the Fourier lens.
        :param pd2: Propagation distance from the Fourier lens to the camera.
        :param xf: Parameter for the ASM wavefront correction.
        :param device: Device to use (GPU or CPU).
        :param slm_mask: Binary mask to set some SLM pixels to zero.
        :param str precision: Computational precision.

            -'single'
                complex64, float32
            -'double'
                complex128, float64

        :param bool fft_shift: Perform FFT shift?
        """
        super().__init__()

        self.slm_disp_obj = slm_disp_obj
        self.pms_obj = pms_obj

        # Choose computational precision
        if precision == 'single':
            dtype_c = torch.complex64
            dtype_r = torch.float32
        else:
            dtype_c = torch.complex128
            dtype_r = torch.float64
        self.precision = precision
        self.dtype_r = dtype_r
        self.dtype_c = dtype_c

        self.fft_shift = fft_shift

        self.device = device
        self.npix_pad = npix_pad

        if pix_res is None:
            pix_res = 1
        self.pix_res = pix_res
        self.npix_full = self.pix_res * self.npix_pad

        self.propagation_type = propagation_type
        if propagation_type == 'asm':
            self.asm_obj = ASM(slm_disp_obj, pms_obj, self.pix_res, self.npix_full, pd1, pd2, extent_lens, xf,
                               shift=self.fft_shift, precision=self.precision, device=self.device)
            phi -= self.asm_obj.phi_corr_native
            e_slm = e_slm * np.exp(1j * self.asm_obj.phi_corr)

        # Initialise optimisation parameters
        if self.precision == 'double':
            self.phi = nn.Parameter(torch.tensor(phi, dtype=torch.float64).to(device), requires_grad=True)
        else:
            self.phi = nn.Parameter(torch.tensor(phi, dtype=torch.float32).to(device), requires_grad=True)

        if npix is None:
            npix = phi.shape[0]
        self.npix = npix
        self.propagation_type = propagation_type
        if kernel_ct is None:
            self.kernel_ct = None
        else:
            self.kernel_ct = torch.tensor(kernel_ct, dtype=dtype_r).to(device).unsqueeze(0).unsqueeze(0)
        if e_slm is None:
            e_slm = torch.ones((self.npix_pad * self.pix_res, self.npix_pad * self.pix_res))
        self.e_slm = torch.tensor(e_slm, dtype=dtype_c).to(device)

        if slm_mask is None:
            slm_mask = np.ones((self.npix, self.npix))
        self.slm_mask = torch.tensor(slm_mask, dtype=dtype_r).to(device)

        self.pad = self.pix_res * (self.npix_pad - self.npix) // 2
        self.counter = 0
        self.phi_disp = torch.zeros_like(self.phi, dtype=dtype_r)
        torch.cuda.empty_cache()

    def set_phi(self, new_phi):
        """
        Set SLM phase from numpy array.

        :param ndarray new_phi: SLM phase [rad].
        """
        if self.precision == 'double':
            self.phi.data = torch.tensor(new_phi, dtype=torch.float64).to(self.device)
        else:
            self.phi.data = torch.tensor(new_phi, dtype=torch.float32).to(self.device)

    def forward(self):
        """
        Model the SLM and simulate the propagation of light from the SLM plane to the image plane. This method is used
        by gradient-based optimizers.

        :return: Electric field in the image plane.
        """
        # Restrict phase value to lower limit 0 and upper limit 2 * pi when modelling pixel crosstalk. This prevents
        # discontinuities in the cost function. Wrap the phase otherwise.
        if self.kernel_ct is None:
            x = self.phi.remainder(self.slm_disp_obj.max_phase)
        else:
            x = torch.clamp(self.phi, min=0, max=self.slm_disp_obj.max_phase)

        # Set some SLM pixels to zero if desired.
        x = x * self.slm_mask

        # Save a copy of the phase pattern as it would be displayed on the SLM.
        self.phi_disp = x.clone()

        # Upscale SLM phase.
        if self.pix_res > 1:
            x = torch.repeat_interleave(x, self.pix_res, dim=0)
            x = torch.repeat_interleave(x, self.pix_res, dim=1)

        # Convolve upscaled SLM phase with pixel crosstalk kernel.
        if self.kernel_ct is not None:
            x = x.unsqueeze(0).unsqueeze(0)
            x = torch.nn.functional.conv2d(x, self.kernel_ct, padding='same').squeeze()

        # Add displayed SLM phase to the constant electric field at the SLM.
        x = tt.exp(1j * x) * self.e_slm

        # Zero pad SLM.
        x = nn.ZeroPad2d((self.pad, self.pad, self.pad, self.pad))(x)

        # Propagate electric field in the SLM plane to the image plane.
        if self.propagation_type == 'fft':
            x = fft(x, shift=self.fft_shift, norm='ortho')
        elif self.propagation_type == 'asm':
            x = self.asm_obj.forward(x)
        self.counter += 1
        return x


def rms(signal, i_target, i_out, frac):
    """
    Calculate normalised root-mean-squared error between two images inside a region of interest. Only pixels which are
    brighter than ``frac * np.max(i_target_norm)`` are taken into account, where ``i_target_norm`` is the normalised
    target intensity pattern.

    :param signal: Binary mask containing region of interest (signal region).
    :param i_target: Target intensity pattern.
    :param i_out: Intensity pattern of light potential.
    :param frac: Threshold as explained above.
    :return: Normalised rms error.
    """
    # Find non-zero indices of measure region.
    mr_idx = (i_target * signal) > ((1 - frac) * tt.max(i_target * signal))

    # Normalise intensity patterns
    i_target_w_norm = i_target[mr_idx] / tt.sum(i_target[mr_idx])
    i_out_w_norm = i_out[mr_idx] / tt.sum(i_out[mr_idx])

    # Calculate normalised root-mean-squared error
    n = ((i_out_w_norm - i_target_w_norm) / i_target_w_norm) ** 2
    n = tt.sqrt(torch.mean(n))
    return n


def eff(signal, i_out):
    """
    Calculates the predicted efficiency of a light potential by dividing the pixel sum in the signal region by
    the pixel sum in the entire pattern.

    :param signal: Binary mask containing the signal region.
    :param i_out: Intensity pattern of the light potential.
    :return: Predicted efficiency.
    """
    return tt.sum(signal * i_out) / tt.sum(i_out)


def loss_fn_fid(e_out, i_tar, phi_tar, signal):
    """
    Phase and amplitude cost function from https://doi.org/10.1364/OE.25.011692.

    :param e_out: Electric field at the image plane.
    :param i_tar: Target intensity pattern.
    :param phi_tar: Target phase pattern.
    :param signal: Binary mask containing signal region.
    :return: Cost.
    """
    a_out = e_out.abs()
    phi_out = e_out.angle()
    overlap = tt.sum(signal * a_out * tt.sqrt(i_tar) * tt.cos(phi_out - phi_tar))
    overlap = overlap / (tt.sqrt(tt.sum(i_tar) * tt.sum((a_out * signal) ** 2)))
    return 1e12 * (1 - overlap) ** 2


def loss_fn_amp(e_out, i_tar, signal):
    """
    Amplitude-only cost function from https://doi.org/10.1364/OE.22.026548.

    :param e_out: Electric field at the image plane.
    :param i_tar: Target intensity pattern.
    :param signal: Binary mask containing signal region.
    :return: Cost.
    """
    mse = nn.MSELoss(reduction='sum')
    i_out = tt.abs(e_out) ** 2
    return 5e11 * mse(i_out * signal / tt.sum(i_out * signal), i_tar)


class PhaseRetrieval:
    """
    This function calculates the SLM phase pattern for a given target light potential in the image plane using conjugate
    gradient minimisation or stochastic gradient descent (Adam).
    """
    def __init__(self, slm_obj, n_iter=10, i_tar=None, phi_tar=None, signal_region=None, save=False, n_save=10,
                 loss_type='amp', optim_type='cg'):
        """
        :param slm_obj: Virtual SLM object created by :py:class:`VirtualSlm`.
        :param n_iter: Number of iterations.
        :param i_tar: Target light potential.
        :param phi_tar: Target phase pattern.
        :param signal_region: Binary mask containing signal region.
        :param bool save: Save SLM phase pattern and electric field at the image plane after every ``n_save`` th
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
        i_tar_t = torch.tensor(i_tar / np.sum(i_tar * signal_region), dtype=self.slm_obj.dtype_r).to(slm_obj.device)
        if fft_shift is False and slm_obj.propagation_type == 'fft':
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

    def set_target(self, target):
        """
        Sets the target light potential.

        :param target: Target light potential.
        """
        self.i_tar = target / np.sum(target * self.signal)
        self.i_tar_t = torch.tensor(self.i_tar, dtype=self.slm_obj.dtype_r).to(self.slm_obj.device)
        if self.fft_shift is False and self.slm_obj.propagation_type == 'fft':
            self.i_tar_t = torch.fft.ifftshift(self.i_tar_t)

    def set_optimizer(self):
        """
        Sets the optimisation algorithm based on ``self.optim_type``.
        """
        if self.optim_type == 'cg':
            self.optimizer = torchmin.Minimizer(self.slm_obj.parameters(), method='cg', max_iter=self.n_iter, disp=1,
                                                callback=self.callback)
        elif self.optim_type == 'adam':
            self.optimizer = torch.optim.Adam(self.slm_obj.parameters(), lr=0.01)

    def loss_fn(self, e_out):
        """
        Defines the loss function based on ``self.loss_type``.

        :param e_out: Electric field at the image plane.
        :return: Loss value.
        """
        if self.loss_type == 'amp':
            return loss_fn_amp(e_out, self.i_tar_t, self.signal_t)
        elif self.loss_type == 'fid':
            return loss_fn_fid(e_out, self.i_tar_t, self.phi_tar_t, self.signal_t)

    def callback(self, x):
        """
        This function is called after every iteration of the optimisation. It saves intermediate SLM phase patterns and
        the electric field in the image plane if ``save=True``. The progress of the optimisation is printed after every
        iteration.
        """
        self.callback_counter += 1
        if self.save is True and self.callback_counter % self.n_save == 0:
            e_out_cb = self.slm_obj()
            self.eta_pred.append(gpu_to_numpy(rms(self.signal_t, self.i_tar_t, torch.abs(e_out_cb) ** 2, 0.5)))
            self.eff_pred.append(gpu_to_numpy(eff(self.signal_t, torch.abs(e_out_cb) ** 2)))
            self.phi.append(gpu_to_numpy(self.slm_obj.phi_disp))

        print('CG iteration #', self.callback_counter, 'Cost:', self.loss, 'Cost function evaluations:',
              self.closure_counter)

    def retrieve_phase(self):
        """
        Performs phase retrieval algorithm.

        :return: Optimised SLM phase(s), (RMS error and efficiency if ``save=True``)
        """
        self.callback_counter = 0
        self.closure_counter = 0
        self.phi = []
        self.eta_pred = []
        self.eff_pred = []

        date = time.strftime('%d-%m-%y__%H-%M-%S', time.localtime())
        print('\nMaximum iteration number : {0}'.format(self.n_iter))
        print("Calculation start : %s\n" % date)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        if self.optim_type == 'cg':
            def closure():
                self.closure_counter += 1
                self.optimizer.zero_grad()
                loss = self.loss_fn(self.slm_obj())
                self.loss = loss.item()
                return loss
            self.optimizer.step(closure)

        elif self.optim_type == 'adam':
            for t in range(self.n_iter):
                self.optimizer.zero_grad()
                loss = self.loss_fn(self.slm_obj())
                self.callback(0)
                print(t, loss.item())

                loss.backward(retain_graph=True)
                self.optimizer.step()

        end.record()
        torch.cuda.synchronize()
        runtime = start.elapsed_time(end) / 1e3

        print('Ran for %.3fs' % runtime)
        print('Ran for %.0f min and %.3fs' % (runtime // 60, runtime % 60))

        # Tidy up to save GPU memory
        torch.cuda.empty_cache()
        return self.phi, (np.asarray(self.eta_pred), np.asarray(self.eff_pred))


def camera_calibration(slm_obj, slm_disp_obj, cam_obj, pms_obj, save=None, exp_time=None, checkerboard_rows=None,
                       checkerboard_columns=None, checkerboard_square_size=None, linear_phase=None):
    """
    This function performs the camera calibration to obtain the coordinate transform between
    the camera image and the computational image plane. To do this, an SLM phase pattern is calculated for a
    checkerboard-shaped target light potential using CG minimisation and displayed on the SLM. The corners of the
    checkerboard in the resulting camera image are detected and fitted to the corners of the checkerboard in the
    computational image plane using an affine transformation.

    :param slm_obj: Virtual SLM object created by :py:class:`VirtualSlm`.
    :param slm_disp_obj: Object created by a subclass of :py:class:`hardware.SlmBase`.
    :param cam_obj: Object created by a subclass of :py:class:`hardware.CameraBase`.
    :param pms_obj: Object created by a subclass of :py:class:`hardware.ParamsBase`.
    :param bool save: Save data?
    :param exp_time: Exposure time.
    :param checkerboard_rows:
    :param checkerboard_columns:
    :param checkerboard_square_size:
    :param linear_phase:
    :return:
    """
    if save is None:
        save = False
    if checkerboard_rows is None:
        checkerboard_rows = 9
    if checkerboard_columns is None:
        checkerboard_columns = 7
    if checkerboard_square_size is None:
        checkerboard_square_size = int(16)
    checkerboard_square_size = int(checkerboard_square_size)
    if linear_phase is None:
        linear_phase = np.array([-slm_obj.npix // 4, -slm_obj.npix // 4])

    n_iter = 50
    if exp_time is None:
        exp_time = 180
    n_img = 10

    # Initial Guess
    guess_type = 'guess'
    phase_angle = int(-slm_obj.npix // 4)
    quad_phase = np.array([10e-4, 0.4])

    mask_pos = int(phase_angle)
    tar_width = int(slm_obj.npix // 4)
    tar_blur = 1.4

    holo = pt.Hologram(slm_disp_obj, pms_obj, 'checkerboard', slm_obj.npix, npix_pad=slm_obj.npix_pad,
                       pix_res=slm_obj.pix_res, phase_guess_type=guess_type, linear_phase=linear_phase,
                       quadratic_phase=quad_phase, slm_field_type='measured', propagation_type=slm_obj.propagation_type,
                       target_position=mask_pos, target_width=tar_width, target_blur=tar_blur, checkerboard_rows=checkerboard_columns,
                       checkerboard_columns=checkerboard_rows, checkerboard_square_size=checkerboard_square_size)

    if slm_obj.propagation_type == 'asm':
        slm_obj.set_phi(holo.phi_init - slm_obj.asm_obj.phi_corr_native)
    else:
        slm_obj.set_phi(holo.phi_init)

    phase_retrieval_obj = PhaseRetrieval(slm_obj, n_iter=n_iter, i_tar=holo.i_tar, signal_region=holo.sig_mask)
    phi_sv = phase_retrieval_obj.retrieve_phase()[0]

    phiCal = slm_obj.phi_disp.clone().cpu().detach().numpy()
    slm_mask = slm_obj.slm_mask.clone().cpu().detach().numpy()

    n_crop = int(512 * holo.asm_corr)
    centre = int(slm_obj.pix_res * slm_obj.npix_pad // 2)

    IoutCal = holo.i_tar[centre - n_crop:centre + n_crop, centre - n_crop:centre + n_crop]
    IoutCal = IoutCal / np.max(IoutCal) * 255

    #%% Checkerboard detection
    imgCal = np.zeros((cam_obj.res[0], cam_obj.res[1]))
    cam_obj.start(n_img)
    for i in range(n_img):
        phiCal += slm_disp_obj.max_phase / n_img
        phiCal = np.remainder(phiCal, slm_disp_obj.max_phase)
        slm_disp_obj.display(phiCal * slm_mask)

        time.sleep(0.2)

        imgCal += cam_obj.get_image(exp_time)
    cam_obj.stop()
    imgCal = imgCal / n_img

    print('Starting checkerboard detection')
    cb_size = (checkerboard_rows - 1, checkerboard_columns - 1)

    for i in range(3):
        IoutCal_blur = ndimage.gaussian_filter(IoutCal, i)
        cornersT, scoreT = detect_checkerboard(IoutCal_blur, cb_size)
        cornersT = np.squeeze(cornersT)
        if cornersT.ndim == 0:
            print('No target checkerboard detected')
        else:
            print('Target checkerboard detected')
            break

    for i in range(5):
        imgSmooth = ndimage.gaussian_filter(imgCal, i - 1)
        imgSmooth = imgSmooth / np.max(imgSmooth) * 255

        cornersH, scoreH = detect_checkerboard(imgSmooth, cb_size)
        cornersH = np.squeeze(cornersH)
        if cornersH.ndim == 0:
            print('No checkerboard detected')
        else:
            print('Checkerboard detected')
            print('Score:', scoreH)
            break

    fig3, axs3 = plt.subplots(1, 2)
    # im1 = axs3[1].imshow(IoutCal_blur / np.max(IoutCal_blur), cmap='turbo')
    # divider1 = make_axes_locatable(axs3[1])
    # cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    # cbar1 = fig3.colorbar(im1, cax=cax1, orientation='vertical')
    # cbar1.set_label('normalised intensity')
    # axs3[1].plot(cornersT[:, 0], cornersT[:, 1], 'wx', markerfacecolor="None")
    # axs3[1].set_title('Target intensity')

    im0 = axs3[0].imshow(imgSmooth, cmap='turbo')
    axs3[0].plot(cornersH[:, 0], cornersH[:, 1], 'r+')
    divider0 = make_axes_locatable(axs3[0])
    cax0 = divider0.append_axes('right', size='5%', pad=0.05)
    cbar0 = fig3.colorbar(im0, cax=cax0, orientation='vertical')
    cbar0.set_label('pixel value')
    axs3[0].set_title('Smoothened camera image')

    tf, mask = cv.estimateAffine2D(cornersT + slm_obj.npix_pad * slm_obj.pix_res // 2 - n_crop, cornersH)
    itf = cv.invertAffineTransform(tf)

    imgCaltf = cv.warpAffine(imgCal, itf, (slm_obj.npix_pad * slm_obj.pix_res, slm_obj.npix_pad * slm_obj.pix_res))

    im2 = axs3[1].imshow(imgCaltf / np.max(imgCaltf), cmap='turbo')
    divider2 = make_axes_locatable(axs3[1])
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    cbar2 = fig3.colorbar(im2, cax=cax2, orientation='vertical')
    cbar2.set_label('normalised intensity')
    axs3[1].plot(cornersT[:, 0] + slm_obj.npix_pad * slm_obj.pix_res // 2 - n_crop,
                 cornersT[:, 1] + slm_obj.npix_pad * slm_obj.pix_res // 2 - n_crop, 'r+')
    axs3[1].set_title('Transformed Camera Image')

    if save is True:
        date_saved = time.strftime('%y-%m-%d_%H-%M-%S', time.localtime())
        path = pms_obj.data_path + date_saved + '_' + 'torch_camcal'
        os.mkdir(path)

        np.save(path + '//imgCal', imgCal)
        np.save(path + '//cornersT', cornersT)
        np.save(path + '//cornersH', cornersH)
        np.save(path + '//scoreT', scoreT)
        np.save(path + '//scoreH', scoreH)
        np.save(path + '//ITarCal', holo.i_tar)
        np.save(path + '//tf', tf)
        np.save(path + '//itf', itf)
        np.save(path + '//imgCaltf', imgCaltf)

    torch.cuda.empty_cache()
    return tf, itf


def camera_feedback(phase_retrieval_obj, slm_disp_obj, cam_obj, tf, itf, iter_fb=1, iter_cg=None, detect_vortices=False,
                    threshold_vtx=0.2, n_save=10, n_avg=10, exp_time=1000, fb_blur=0, alpha=None, convergence=False,
                    iter_convergence=None, path=None):
    """
    This function implements a camera feedback algorithm to reduce experimental errors in the light potentials
    (see `<https://dx.doi.org/10.1088/0953-4075/48/11/115303>`_).
    Before applying any camera feedback, optical vortices in the light potential are detected using the
    patterns.detect_vortices() function and removed if required.

    After vortices are removed, the optimised phase pattern is displayed on the SLM and a camera image, M, is recorded.
    To create the target light potential for the next feedback iteration, T[..., i + 1], a discrepancy, D, between the
    camera image and the original target light potential, T[..., 0], is calculated and added to the previous light
    potential, T[..., i].

    At the end of each feedback iteration, the root-mean-squared error (RMSE) and the peak signal-to-noise ratio (PSNR)
    of the camera image are calculated and saved. To find the experimental convergence of the CG minimisation,
    intermediate SLM phase patterns are saved and displayed on the SLM. A camera image is taken for each pattern and the
    RMSE is calculated.

    :param phase_retrieval_obj: Instance of the class :py:class:`PhaseRetrieval`.
    :param slm_disp_obj: Object created by a subclass of :py:class:`hardware.SlmBase`.
    :param cam_obj: Object created by a subclass of :py:class:`hardware.CameraBase`.
    :param tf: Affine transform matrix.
    :param itf: Inverse affine transform matrix.
    :param iter_fb: Number of feedback iterations.
    :param iter_cg: Number of conjugate gradient iterations per feedback iteration.
    :param bool detect_vortices: Detect vortices?
    :param threshold_vtx: See ``patterns.detect_vortices()``
    :param n_save: Save data for every ``n_save`` th CG iteration.
    :param n_avg: Number of camera frames to capture and average per feedback iteration.
    :param exp_time: Exposure time.
    :param fb_blur: Width of blurring kernel for camera image [px].
    :param alpha: Feedback gain parameter for each feedback iteration.
    :param bool convergence: Save intermediate phase patterns during CG minimisation?
    :param iter_convergence: During which feedback iterations to save intermediate phase patterns.
    :param path: Save path.
    :return: See code.
    """

    npix_full = phase_retrieval_obj.slm_obj.npix_full
    npix_pad = phase_retrieval_obj.slm_obj.npix_pad
    npix = phase_retrieval_obj.slm_obj.npix

    # Define arrays
    T = np.zeros((cam_obj.res[0], cam_obj.res[1], iter_fb + 1))                 # Target array, camera coordinates
    M = np.zeros_like(T)                                                        # Array for measured light potentials
    D = np.zeros((cam_obj.res[0], cam_obj.res[1]))                              # Array for discrepancy (T - M)
    img = np.zeros((cam_obj.res[0], cam_obj.res[1], iter_fb))                   # Array to store raw camera images
    phi = np.zeros((slm_disp_obj.res[0], slm_disp_obj.res[0], iter_fb + 1))     # Array to store SLM phase patterns
    phi[..., 0] = phase_retrieval_obj.slm_obj.phi_disp.detach().cpu().numpy()   # Define first phase pattern

    # Define root-mean-squared error and peak signal-to-noise ratio
    rmse = np.zeros(iter_fb)
    psnr = np.zeros(iter_fb)

    # Define lists to store convergence data
    eff_conv_sv = []        # Efficiency
    rmse_conv_sv = []       # Experimental RMSE
    rmse_pred_conv_sv = []  # Predicted RMSE
    n_conv_sv = []          # CG iteration number

    # Transform target intensity and signal mask
    i_tar_tf = cv.warpAffine(phase_retrieval_obj.i_tar / np.max(phase_retrieval_obj.i_tar), tf,
                             (cam_obj.res[1], cam_obj.res[0]))
    sig_mask_tf = cv.warpAffine(phase_retrieval_obj.signal, tf, (cam_obj.res[1], cam_obj.res[0]))
    camera_feedback.sig_mask_tf = sig_mask_tf

    # Normalise target intensity
    T[..., 0] = m.normalize(i_tar_tf, sig_mask_tf)

    # Initisalize first measured image
    M[..., 0] = np.copy(T[..., 0])

    # Start camera
    cam_obj.start(2 * n_avg * iter_fb)

    # %% Performa vortex detection and removal if desired
    # ToDO: Make vortex detection compatible with ASM
    if detect_vortices is True:
        n_vtx = 1       # Initialize number of detected vortices
        counter = 0     # Vortex detection iteration number

        # Only re-run vortex detection if there were vortices left after the previous iteration
        while n_vtx > 0:
            # Run CG algorithm
            phase_retrieval_obj.n_iter = 100
            phase_retrieval_obj.save = False
            phi_vtx_sv = phase_retrieval_obj.retrieve_phase()[0]
            e_out_vtx = phase_retrieval_obj.slm_obj().cpu().detach().numpy()

            # Detect vortices
            vtx = pt.detect_vortices(slm_disp_obj.res[0], e_out_vtx,
                                     phase_retrieval_obj.i_tar / np.max(phase_retrieval_obj.i_tar) *
                                     phase_retrieval_obj.signal, threshold=threshold_vtx)
            n_vtx = vtx.shape[0]  # Number of detected vortices

            # Plot vortices after first iteration
            if counter == 0:
                vtx_sv = np.copy(vtx)
                i_vtx = np.abs(e_out_vtx) ** 2  # Calculate intensity pattern in the image plane

                plt.figure()
                plt.imshow(i_vtx[slm_disp_obj.res[0] // 2:3 * slm_disp_obj.res[0] // 2,
                           slm_disp_obj.res[0] // 2:3 * slm_disp_obj.res[0] // 2], cmap='turbo')
                plt.plot(vtx_sv[:, 1][vtx_sv[:, -1] > 0], vtx_sv[:, 0][vtx_sv[:, -1] > 0], c='aquamarine', marker='o',
                         linestyle='None', markerfacecolor='None', label='positive')
                plt.plot(vtx_sv[:, 1][vtx_sv[:, -1] < 0], vtx_sv[:, 0][vtx_sv[:, -1] < 0], c='orchid', marker='o',
                         linestyle='None', markerfacecolor='None', label='negative')
                plt.legend()

            print('Iteration', counter + 1, ' of vortex detection:', n_vtx, 'vortices detected.')

            # Remove vortices
            if n_vtx > 0:
                # Calculate vortex field using detected vortex charges and positions
                e_anti_vtx = pt.vortex_field(phase_retrieval_obj.i_tar, vtx[:, -1],
                                             vtx[:, 1] + npix_full // 2 - npix // 2,
                                             vtx[:, 0] + npix_full // 2 - npix // 2)

                # Calculate corrected vortex field
                e_corr = e_out_vtx * e_anti_vtx
                e_corr = e_corr[(npix_full - npix_pad) // 2: (npix_full + npix_pad) // 2,
                                (npix_full - npix_pad) // 2: (npix_full + npix_pad) // 2]

                # Propagate corrected vortex field from image plane to SLM plane
                e_slm_corr = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(e_corr)))

                # Extract SLM phase
                phi_slm_new = np.angle(e_slm_corr)[(npix_pad - npix) // 2: (npix_pad + npix) // 2,
                                                   (npix_pad - npix) // 2: (npix_pad + npix) // 2]
                phi_slm_new = np.remainder(phi_slm_new, 2 * np.pi)

                # Update slm object using new SLM phase
                phase_retrieval_obj.slm_obj.set_phi(phi_slm_new)
            counter += 1
        phase_retrieval_obj.slm_obj.set_phi(phi[..., 0])

    phase_retrieval_obj.save = convergence
    phase_retrieval_obj.n_save = n_save

    # %% Perform camera feedback algorithm
    for i in range(iter_fb):
        # Calculate target for this iteration (i + 1)
        T[..., i + 1] = sig_mask_tf * (T[..., i] + alpha[i] * D)

        # Set negative intensity values to zero
        t_neg_mask = T[..., i + 1].squeeze() < 0
        T[t_neg_mask, i + 1] = 0

        # Blur target light potential
        T[..., i + 1] = ndimage.gaussian_filter(T[..., i + 1], fb_blur)

        # Transform target light potential
        if i == 0:
            T_tf = phase_retrieval_obj.i_tar
        else:
            T_tf = cv.warpAffine(T[..., i + 1], itf, (npix_full, npix_full))

        # Perform CG algorithm
        phase_retrieval_obj.set_target(T_tf)
        phase_retrieval_obj.n_iter = int(iter_cg[i])
        phi_sv, meas_sv = phase_retrieval_obj.retrieve_phase()

        # phase_retrieval(slm, n_iter=int(iter_cg[i]), i_tar=T_tf, signal=holo.sig_mask,
        #                                 save=convergence, n_save=n_save, loss_fn=loss_fn, optim=optim)

        # Generate and save convergence data
        if convergence is True and i + 1 in iter_convergence:
            n_conv = len(phi_sv)  # Number of saved convergence points

            img_conv = np.zeros((cam_obj.res[0], cam_obj.res[1], n_conv))  # Array to store convergence images
            eta_conv = np.zeros(n_conv)  # Array to store RMSE of convergence images

            # Display intermediate SLM phase patterns and record camera images
            for ii in range(n_conv):
                slm_disp_obj.display(phi_sv[ii])  # Display phase on SLM
                img_conv[..., ii] = hw.get_image_avg(cam_obj, exp_time, n_avg)  # Take camera image

                # Calculate RMSE of camera images
                eta_conv[ii] = m.rms(sig_mask_tf, i_tar_tf, img_conv[..., ii])

            # Save data
            np.save(path + '//img_conv_' + str(i), img_conv)
            np.save(path + '//phi_sv_' + str(i), phi_sv)
            np.save(path + '//meas_sv_' + str(i), meas_sv)

            rmse_conv_sv.append(np.copy(eta_conv))
            rmse_pred_conv_sv.append(np.copy(meas_sv[0]))
            eff_conv_sv.append(np.copy(meas_sv[1]))
            n_conv_sv.append(np.copy(n_conv))

        # Transfer optimised SLM phase to CPU
        phi[..., i + 1] = gpu_to_numpy(phase_retrieval_obj.slm_obj.phi_disp)

        # Display phase on SLM
        slm_disp_obj.display(phi[..., i + 1])

        # Take a picture of the light potential
        img[..., i] = hw.get_image_avg(cam_obj, exp_time, n_avg)

        # Normalize camera image
        M[..., i + 1] = m.normalize(img[..., i], sig_mask_tf)

        # Calculate discrepancy
        D = T[..., 0] - M[..., i + 1]

        # Calculate RMSE and PSNR
        rmse[i] = m.rms(sig_mask_tf, i_tar_tf, img[..., i])
        psnr[i] = m.psnr(sig_mask_tf, i_tar_tf, img[..., i])

        print('Feedback iteration number %.0f' % (i + 1))
        print('New RMS @ 50 %.4f' % rmse[i])
        print('PSNR %.4f' % psnr[i])

        torch.cuda.empty_cache()
    cam_obj.stop()
    return phi, img, M, T, [rmse, psnr], [rmse_conv_sv, rmse_pred_conv_sv, eff_conv_sv, n_conv_sv]
