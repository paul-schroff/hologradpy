import os
import time

import numpy as np
from numpy.typing import NDArray

import matplotlib as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy import ndimage
import cv2 as cv
from checkerboard import detect_checkerboard

import torch

from slmsuite.hardware.cameras.camera import Camera
from slmsuite.hardware.slms.slm import SLM

from ..propagation.utils.tensor_utils import (
    gpu_to_numpy
)
from ..propagation.optical_systems import VirtualSlm
from .phase_retrieval import PhaseRetrieval
from .. import patterns as pt
from ..hardware import hardware as hw
from ..analysis import error_metrics as m


def camera_calibration(
        slm_obj: VirtualSlm,
        slm_disp_obj: hw.SlmBase,
        cam_obj: hw.CameraBase,
        pms_obj: hw.ParamsBase, 
        save: bool = False,
        exp_time: int = 180,
        checkerboard_rows: int = 9,
        checkerboard_columns: int = 7, 
        checkerboard_square_size: int = 16,
        linear_phase: NDArray[np.float_] = None
    ) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
    """
    This function performs the camera calibration to obtain the coordinate 
    transform between the camera image and the computational image plane. To do 
    this, an SLM phase pattern is calculated for a checkerboard-shaped target 
    light potential using CG minimisation and displayed on the SLM. The corners 
    of the checkerboard in the resulting camera image are detected and fitted 
    to the corners of the checkerboard in the computational image plane using 
    an affine transformation.

    :param slm_obj: Virtual SLM object created by :py:class:`VirtualSlm`.
    :param slm_disp_obj: Object created by a subclass of 
        :py:class:`hardware.SlmBase`.
    :param cam_obj: Object created by a subclass of 
        :py:class:`hardware.CameraBase`.
    :param pms_obj: Object created by a subclass of 
        :py:class:`hardware.ParamsBase`.
    :param bool save: Save data?
    :param exp_time: Exposure time.
    :param checkerboard_rows: Number of rows in the checkerboard.
    :param checkerboard_columns: Number of columns in the checkerboard.
    :param checkerboard_square_size: Size of the checkerboard squares.
    :param linear_phase: Linear phase array.
    :return: Affine transform matrix and its inverse.
    """
    if linear_phase is None:
        linear_phase = np.array([-slm_obj.npix // 4, -slm_obj.npix // 4])

    n_iter = 50
    n_img = 10

    # Initial Guess
    guess_type = 'guess'
    phase_angle = int(-slm_obj.npix // 4)
    quad_phase = np.array([10e-4, 0.4])

    mask_pos = int(phase_angle)
    tar_width = int(slm_obj.npix // 4)
    tar_blur = 1.4

    holo = pt.Hologram(
        slm_disp_obj,
        pms_obj,
        'checkerboard',
        slm_obj.npix,
        npix_pad=slm_obj.npix_pad,
        pix_res=slm_obj.pix_res,
        phase_guess_type=guess_type,
        linear_phase=linear_phase,
        quadratic_phase=quad_phase,
        slm_field_type='measured',
        propagation_type=slm_obj.propagation_type,
        target_position=mask_pos,
        target_width=tar_width,
        target_blur=tar_blur,
        checkerboard_rows=checkerboard_columns,
        checkerboard_columns=checkerboard_rows,
        checkerboard_square_size=checkerboard_square_size
    )

    if slm_obj.propagation_type == 'asm':
        slm_obj.set_phi(holo.phi_init - slm_obj.asm_obj.phi_corr_native)
    else:
        slm_obj.set_phi(holo.phi_init)

    phase_retrieval_obj = PhaseRetrieval(
        slm_obj,
        n_iter=n_iter,
        i_tar=holo.i_tar,
        signal_region=holo.sig_mask
    )
    phi_sv = phase_retrieval_obj.retrieve_phase()[0]

    phiCal = slm_obj.phi_disp.clone().cpu().detach().numpy()
    slm_mask = slm_obj.slm_mask.clone().cpu().detach().numpy()

    n_crop = int(512 * holo.asm_corr)
    centre = int(slm_obj.pix_res * slm_obj.npix_pad // 2)

    IoutCal = holo.i_tar[centre - n_crop:centre + n_crop,
                         centre - n_crop:centre + n_crop]
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
    # axs3[1].plot(
    #     cornersT[:, 0],
    #     cornersT[:, 1],
    #     'wx',
    #     markerfacecolor="None"
    # )
    # axs3[1].set_title('Target intensity')

    im0 = axs3[0].imshow(imgSmooth, cmap='turbo')
    axs3[0].plot(cornersH[:, 0], cornersH[:, 1], 'r+')
    divider0 = make_axes_locatable(axs3[0])
    cax0 = divider0.append_axes('right', size='5%', pad=0.05)
    cbar0 = fig3.colorbar(im0, cax=cax0, orientation='vertical')
    cbar0.set_label('pixel value')
    axs3[0].set_title('Smoothened camera image')

    tf, mask = cv.estimateAffine2D(
        cornersT + slm_obj.npix_pad * slm_obj.pix_res // 2 - n_crop, cornersH
    )
    itf = cv.invertAffineTransform(tf)

    imgCaltf = cv.warpAffine(
        imgCal,
        itf,
        (slm_obj.npix_pad * slm_obj.pix_res,
         slm_obj.npix_pad * slm_obj.pix_res)
    )

    im2 = axs3[1].imshow(imgCaltf / np.max(imgCaltf), cmap='turbo')
    divider2 = make_axes_locatable(axs3[1])
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    cbar2 = fig3.colorbar(im2, cax=cax2, orientation='vertical')
    cbar2.set_label('normalised intensity')
    axs3[1].plot(
        cornersT[:, 0] + slm_obj.npix_pad * slm_obj.pix_res // 2 - n_crop,
        cornersT[:, 1] + slm_obj.npix_pad * slm_obj.pix_res // 2 - n_crop,
        'r+'
    )
    axs3[1].set_title('Transformed Camera Image')

    if save:
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

    if slm_obj.device == 'cuda':
        torch.cuda.empty_cache()
    return tf, itf


def camera_feedback(
        phase_retrieval_obj: PhaseRetrieval,
        slm_disp_obj: hw.SlmBase,
        cam_obj: hw.CameraBase, 
        tf: NDArray[np.float_],
        itf: NDArray[np.float_],
        iter_fb: int = 1,
        iter_cg: list[int] | None = None, 
        detect_vortices: bool = False,
        threshold_vtx: float = 0.2,
        n_save: int = 10,
        n_avg: int = 10, 
        exp_time: int = 1000,
        fb_blur: int = 0,
        alpha: list[float] | None = None,
        convergence: bool = False, 
        iter_convergence: list[int] | None = None,
        path: str | None = None
    ) -> tuple[NDArray[np.float_],
               NDArray[np.float_],
               NDArray[np.float_],
               NDArray[np.float_],
               list[NDArray[np.float_]],
               list[list[NDArray[np.float_]]]]:
    """
    This function implements a camera feedback algorithm to reduce experimental
    errors in the light potentials (see 
    `<https://dx.doi.org/10.1088/0953-4075/48/11/115303>`_). Before applying 
    any camera feedback, optical vortices in the light potential are detected 
    using the patterns.detect_vortices() function and removed if required.

    After vortices are removed, the optimised phase pattern is displayed on the 
    SLM and a camera image, M, is recorded. To create the target light 
    potential for the next feedback iteration, T[..., i + 1], a discrepancy, D, 
    between the camera image and the original target light potential, 
    T[..., 0], is calculated and added to the previous light potential, 
    T[..., i].

    At the end of each feedback iteration, the root-mean-squared error (RMSE) 
    and the peak signal-to-noise ratio (PSNR) of the camera image are 
    calculated and saved. To find the experimental convergence of the CG 
    minimisation, intermediate SLM phase patterns are saved and displayed on 
    the SLM. A camera image is taken for each pattern and the RMSE is 
    calculated.

    :param phase_retrieval_obj: Instance of the class 
        :py:class:`PhaseRetrieval`.
    :param slm_disp_obj: Object created by a subclass of 
        :py:class:`hardware.SlmBase`.
    :param cam_obj: Object created by a subclass of 
        :py:class:`hardware.CameraBase`.
    :param tf: Affine transform matrix.
    :param itf: Inverse affine transform matrix.
    :param iter_fb: Number of feedback iterations.
    :param iter_cg: Number of conjugate gradient iterations per feedback 
        iteration.
    :param bool detect_vortices: Detect vortices?
    :param threshold_vtx: See ``patterns.detect_vortices()``
    :param n_save: Save data for every ``n_save`` th CG iteration.
    :param n_avg: Number of camera frames to capture and average per feedback 
        iteration.
    :param exp_time: Exposure time.
    :param fb_blur: Width of blurring kernel for camera image [px].
    :param alpha: Feedback gain parameter for each feedback iteration.
    :param bool convergence: Save intermediate phase patterns during CG 
        minimisation?
    :param iter_convergence: During which feedback iterations to save 
        intermediate phase patterns.
    :param path: Save path.
    :return: See code.
    """

    npix_full = phase_retrieval_obj.slm_obj.npix_full
    npix_pad = phase_retrieval_obj.slm_obj.npix_pad
    npix = phase_retrieval_obj.slm_obj.npix

    # Define arrays
    # Target array, camera coordinates
    T = np.zeros((cam_obj.res[0], cam_obj.res[1], iter_fb + 1))

    # Array for measured light potentials            
    M = np.zeros_like(T)

    # Array for discrepancy (T - M)                                             
    D = np.zeros((cam_obj.res[0], cam_obj.res[1]))

    # Array to store raw camera images                   
    img = np.zeros((cam_obj.res[0], cam_obj.res[1], iter_fb))

    # Array to store SLM phase patterns
    phi = np.zeros((slm_disp_obj.res[0], slm_disp_obj.res[0], iter_fb + 1))

    # Define first phase pattern
    phi[..., 0] = phase_retrieval_obj.slm_obj.phi_disp.detach().cpu().numpy()

    # Define root-mean-squared error and peak signal-to-noise ratio
    rmse = np.zeros(iter_fb)
    psnr = np.zeros(iter_fb)

    # Define lists to store convergence data
    eff_conv_sv = []        # Efficiency
    rmse_conv_sv = []       # Experimental RMSE
    rmse_pred_conv_sv = []  # Predicted RMSE
    n_conv_sv = []          # CG iteration number

    # Transform target intensity and signal mask
    i_tar_tf = cv.warpAffine(
        phase_retrieval_obj.i_tar / np.max(phase_retrieval_obj.i_tar),
        tf,
        (cam_obj.res[1], cam_obj.res[0])
    )

    sig_mask_tf = cv.warpAffine(
        phase_retrieval_obj.signal,
        tf,
        (cam_obj.res[1], cam_obj.res[0])
    )

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

        # Only re-run vortex detection if there were vortices left after the 
        # previous iteration
        while n_vtx > 0:
            # Run CG algorithm
            phase_retrieval_obj.n_iter = 100
            phase_retrieval_obj.save = False
            phi_vtx_sv = phase_retrieval_obj.retrieve_phase()[0]
            e_out_vtx = phase_retrieval_obj.slm_obj().cpu().detach().numpy()

            # Detect vortices
            vtx = pt.detect_vortices(
                slm_disp_obj.res[0],
                e_out_vtx,
                (phase_retrieval_obj.i_tar / np.max(phase_retrieval_obj.i_tar) 
                 * phase_retrieval_obj.signal),
                threshold=threshold_vtx
            )
            n_vtx = vtx.shape[0]  # Number of detected vortices

            # Plot vortices after first iteration
            if counter == 0:
                vtx_sv = np.copy(vtx)

                # Calculate intensity pattern in the image plane
                i_vtx = np.abs(e_out_vtx) ** 2

                plt.figure()
                plt.imshow(
                    i_vtx[
                        slm_disp_obj.res[0] // 2:3 * slm_disp_obj.res[0] // 2,
                        slm_disp_obj.res[0] // 2:3 * slm_disp_obj.res[0] // 2
                        ],
                    cmap='turbo'
                )
                plt.plot(
                    vtx_sv[:, 1][vtx_sv[:, -1] > 0],
                    vtx_sv[:, 0][vtx_sv[:, -1] > 0],
                    c='aquamarine',
                    marker='o',
                    linestyle='None',
                    markerfacecolor='None',
                    label='positive'
                )
                plt.plot(
                    vtx_sv[:, 1][vtx_sv[:, -1] < 0],
                    vtx_sv[:, 0][vtx_sv[:, -1] < 0],
                    c='orchid',
                    marker='o',
                    linestyle='None',
                    markerfacecolor='None',
                    label='negative'
                )
                plt.legend()

            print('Iteration', counter + 1, ' of vortex detection:', 
                  n_vtx, 'vortices detected.')

            # Remove vortices
            if n_vtx > 0:
                # Calculate vortex field using detected vortex charges and 
                # positions
                e_anti_vtx = pt.vortex_field(
                    phase_retrieval_obj.i_tar,
                    vtx[:, -1],
                    vtx[:, 1] + npix_full // 2 - npix // 2,
                    vtx[:, 0] + npix_full // 2 - npix // 2
                )

                # Calculate corrected vortex field
                e_corr = e_out_vtx * e_anti_vtx
                e_corr = e_corr[
                    (npix_full - npix_pad) // 2: (npix_full + npix_pad) // 2,
                    (npix_full - npix_pad) // 2: (npix_full + npix_pad) // 2
                ]

                # Propagate corrected vortex field from image plane to SLM 
                # plane
                e_slm_corr = np.fft.ifftshift(
                    np.fft.ifft2(np.fft.fftshift(e_corr))
                )

                # Extract SLM phase
                phi_slm_new = np.angle(e_slm_corr)[
                    (npix_pad - npix) // 2: (npix_pad + npix) // 2,
                    (npix_pad - npix) // 2: (npix_pad + npix) // 2
                ]
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

        # phase_retrieval(
        #     slm,
        #     n_iter=int(iter_cg[i]),
        #     i_tar=T_tf,
        #     signal=holo.sig_mask,
        #     save=convergence,
        #     n_save=n_save,
        #     loss_fn=loss_fn,
        #     optim=optim
        # )

        # Generate and save convergence data
        if convergence is True and i + 1 in iter_convergence:
            # Number of saved convergence points
            n_conv = len(phi_sv)

            # Array to store convergence images
            img_conv = np.zeros((cam_obj.res[0], cam_obj.res[1], n_conv))

            # Array to store RMSE of convergence images
            eta_conv = np.zeros(n_conv)  

            # Display intermediate SLM phase patterns and record camera images
            for ii in range(n_conv):
                # Display phase on SLM
                slm_disp_obj.display(phi_sv[ii])

                # Take camera image
                img_conv[..., ii] = hw.get_image_avg(cam_obj, exp_time, n_avg)

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
    return (
        phi,
        img,
        M,
        T,
        [rmse, psnr],
        [rmse_conv_sv, rmse_pred_conv_sv, eff_conv_sv, n_conv_sv]
    )