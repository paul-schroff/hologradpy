"""
Module to measure the constant amplitude and phase at the SLM.
"""

import os
import time
import numpy as np
from . import patterns as pt
from . import fitting as ft
from . import hardware as hw
from . import error_metrics as m
import scipy.optimize as opt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def find_camera_position(slm_disp_obj, cam_obj, pms_obj, lin_phase, exp_time=100, aperture_diameter=25, roi=[500, 500]):
    """
    This function generates a spot on the camera by displaying a circular aperture on the SLM containing a linear phase
    gradient. The position of the spot is found by fitting a Gaussian to the camera image.

    :param slm_disp_obj: Instance of your own subclass of ``hardware.SlmBase``
    :param cam_obj:
    :param pms_obj:
    :param npix: Number of used SLM pixels
    :param lin_phase: x and y gradient of the linear phase
    :param cam_name: Name of the camera to be used
    :param exp_time: Exposure time
    :param aperture_diameter: Diameter of the circular aperture
    :param roi: Width and height of the region of interest on the camera to remove the zeroth-order diffraction spot
    :return: x and y coordinates of the spot on the camera
    """
    resolution_y, resolution_x = slm_disp_obj.res
    zeros = np.zeros((resolution_y, resolution_y))
    slm_phase = pt.init_phase(zeros, slm_disp_obj, pms_obj, lin_phase=lin_phase)
    circ_aperture = pt.circ_mask(zeros, 0, 0, aperture_diameter / 2)

    # Display phase pattern on SLM
    slm_disp_obj.display(slm_phase * circ_aperture)

    # Take camera image
    cam_obj.start()
    img = cam_obj.get_image(exp_time)
    cam_obj.stop()

    # Mask to crop camera image (removes the zeroth-order diffraction spot)
    crop_mask = pt.rect_mask(img, 0, 0, roi[0], roi[1])

    # Fit Gaussian to camera image
    p_opt, p_err = ft.fit_gaussian(img * crop_mask)
    return p_opt[:2], img


def get_aperture_indices(nx, ny, x_start, x_stop, y_start, y_stop, aperture_width, aperture_height):
    """
    This function calculates a grid of ``nx * ny`` rectangular regions in an array and returns the start and end indices
    of each region. All units are in pixels.

    :param nx: Number of rectangles along x.
    :param ny: Number of rectangles along y.
    :param x_start: Start index for first rectangle along x.
    :param x_stop: End index for last rectangle along x.
    :param y_start: Start index for first rectangle along y.
    :param y_stop: End index for last rectangle along y.
    :param aperture_width: Width of rectangle.
    :param aperture_height: Height of rectangle.
    :return: List with four entries for the start and end index along x and y:
             [idx_start_y, idx_end_y, idx_start_x, idx_end_x]. Each list entry is a vector of length ``nx * ny``
             containing the start/end index for each rectangle along x/y.
    """
    idx_start_x = np.floor(np.linspace(x_start, x_stop - aperture_width, nx)).astype('int')
    idx_end_x = idx_start_x + aperture_width
    idx_start_x = np.tile(idx_start_x, ny)
    idx_end_x = np.tile(idx_end_x, ny)

    idx_start_y = np.floor(np.linspace(y_start, y_stop - aperture_height, ny)).astype('int')
    idx_end_y = idx_start_y + aperture_height
    idx_start_y = np.repeat(idx_start_y, nx)
    idx_end_y = np.repeat(idx_end_y, nx)
    return [idx_start_y, idx_end_y, idx_start_x, idx_end_x]


def measure_slm_intensity(slm_disp_obj, cam_obj, pms_obj, aperture_number, aperture_width, exp_time, spot_pos, roi_width):
    """
    This function measures the intensity profile of the laser beam incident onto the SLM by displaying a sequence of
    rectangular phase masks on the SLM. The phase mask contains a linear phase which creates a diffraction spot on the
    camera. The position of the phase mask is varied across the entire area of the SLM and the intensity of each
    diffraction spot is measured using the camera. Read the SI of https://doi.org/10.1038/s41598-023-30296-6 for
    details.

    :param slm_disp_obj: Instance of your own subclass of ``hardware.SlmBase``.
    :param cam_obj: Instance of your own subclass of ``hardware.CameraBase``.
    :param aperture_number: Number of square regions along x/ y.
    :param aperture_width: Width of square regions [px].
    :param exp_time: Exposure time.
    :param spot_pos: x/y position of the diffraction spot in th computational Fourier plane [Fourier pixels].
    :param roi_width: Width of the region of interest on the camera [camera pixels].
    :return:
    """
    roi_mem = cam_obj.roi
    date_saved = time.strftime('%y-%m-%d_%H-%M-%S', time.localtime())
    path = pms_obj.data_path + date_saved + '_' + 'measure_slm_intensity'
    os.mkdir(path)

    res_y, res_x = slm_disp_obj.res
    border_x = int(np.abs(res_x - res_y) // 2)
    npix = np.min(slm_disp_obj.res)

    zeros = np.zeros((npix, npix))
    zeros_full = np.zeros((npix, np.max(slm_disp_obj.res)))

    lin_phase = np.array([-spot_pos, -spot_pos])
    slm_phase = pt.init_phase(np.zeros((aperture_width, aperture_width)), slm_disp_obj, pms_obj, lin_phase=lin_phase)
    # slm_phase = np.remainder(slm_phase, 2 * np.pi)
    slm_idx = get_aperture_indices(aperture_number, aperture_number, border_x, npix + border_x, 0, npix, aperture_width,
                                   aperture_width)

    # Display central sub-aperture on SLM and check if camera is over-exposed.
    i = (aperture_number ** 2) // 2 - aperture_number // 2
    phi_centre = np.zeros_like(zeros)
    phi_centre[slm_idx[0][i]:slm_idx[1][i], slm_idx[2][i]:slm_idx[3][i]] = slm_phase

    slm_disp_obj.display(phi_centre)

    # Take camera image
    cam_obj.start()
    measure_slm_intensity.img_exp_check = cam_obj.get_image(exp_time)
    cam_obj.stop()

    # Find Camera position with respect to SLM
    popt_clb, img_cal = find_camera_position(slm_disp_obj, cam_obj, pms_obj, lin_phase, exp_time=exp_time / 10,
                                             aperture_diameter=npix // 20, roi=[400, 400])

    ny, nx = cam_obj.res
    calib_pos_x = int(popt_clb[0] + nx // 2)
    calib_pos_y = int(popt_clb[1] + ny // 2)

    plt.figure()
    plt.imshow(img_cal, cmap='turbo')
    plt.plot(calib_pos_x, calib_pos_y, 'wx')
    plt.title('Camera image and fitted spot position')

    # Take camera images
    roi = [roi_width, roi_width, int((calib_pos_x - roi_width / 2) // 2 * 2),
           int((calib_pos_y - roi_width / 2) // 2 * 2)]
    cam_obj.roi = roi
    cam_obj.start(aperture_number ** 2)

    img = np.zeros((roi[1], roi[0], aperture_number ** 2))
    aperture_power = np.zeros(aperture_number ** 2)

    for i in range(aperture_number ** 2):
        masked_phase = np.copy(zeros_full)
        masked_phase[slm_idx[0][i]:slm_idx[1][i], slm_idx[2][i]:slm_idx[3][i]] = slm_phase

        slm_disp_obj.display(masked_phase)

        img[..., i] = cam_obj.get_image(int(exp_time))
        aperture_power[i] = np.sum(img[..., i]) / (np.size(img[..., i]) * exp_time)
    cam_obj.stop()
    cam_obj.roi = roi_mem

    np.save(path + '//img', img)
    np.save(path + '//aperture_power', aperture_power)

    # Find SLM intensity profile
    i_rec = np.reshape(aperture_power, (aperture_number, aperture_number))

    # Fit Gaussian to measured intensity
    extent_slm = (slm_disp_obj.slm_size[0] + aperture_width * slm_disp_obj.pitch) / 2
    x_fit = np.linspace(-extent_slm, extent_slm, aperture_number)
    x_fit, y_fit = np.meshgrid(x_fit, x_fit)
    sig_x, sig_y = pms_obj.beam_diameter, pms_obj.beam_diameter
    popt_slm, perr_slm = ft.fit_gaussian(i_rec, dx=0, dy=0, sig_x=sig_x, sig_y=sig_y, xy=[x_fit, y_fit])

    i_fit_slm = pt.gaussian(slm_disp_obj.meshgrid_slm[0], slm_disp_obj.meshgrid_slm[1], *popt_slm)

    # Plotting
    extent_slm_mm = extent_slm * 1e3
    extent = [-extent_slm_mm, extent_slm_mm, -extent_slm_mm, extent_slm_mm]

    fig, axs = plt.subplots(1, 2)
    divider = make_axes_locatable(axs[0])
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    im = axs[0].imshow(i_rec / np.max(i_rec), cmap='turbo', extent=extent)
    axs[0].set_title('Intensity at SLM Aperture', fontname='Cambria')
    axs[0].set_xlabel("x [mm]", fontname='Cambria')
    axs[0].set_ylabel("y [mm]", fontname='Cambria')

    divider = make_axes_locatable(axs[1])
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    fig.add_axes(ax_cb)
    im = axs[1].imshow(i_fit_slm / np.max(i_fit_slm), cmap='turbo', extent=extent)
    axs[1].set_title('Fitted Gaussian', fontname='Cambria')
    axs[1].set_xlabel("x [mm]", fontname='Cambria')
    axs[1].set_ylabel("y [mm]", fontname='Cambria')
    cbar = plt.colorbar(im, cax=ax_cb)
    cbar.set_label('normalised intensity', fontname='Cambria')

    plt.figure()
    plt.imshow(img[..., (aperture_number ** 2 - aperture_number) // 2], cmap='turbo')
    plt.title('Camera image of central sub-aperture')

    # Save data
    np.save(path + '//i_rec', i_rec)
    np.save(path + '//i_fit_slm', i_fit_slm)
    np.save(path + '//popt_slm', popt_slm)
    return path


def measure_slm_wavefront(slm_disp_obj, cam_obj, pms_obj, n_aperture, aperture_width, img_size, exp_time, spot_pos,
                          n_avg=10, benchmark=False, phi_load_path=None, roi_min_x=16, roi_min_y=16, roi_n=8):
    roi_mem = cam_obj.roi
    res_y, res_x = slm_disp_obj.res
    npix = np.min(slm_disp_obj.res)
    border_x = int(np.abs(res_x - res_y) // 2)
    zeros_full = np.zeros((npix, np.max(slm_disp_obj.res)))

    fl = pms_obj.fl

    lin_phase = np.array([-spot_pos, -spot_pos])
    slm_phase = pt.init_phase(zeros_full, slm_disp_obj, pms_obj, lin_phase=lin_phase)

    if benchmark is True:
        phi_load = np.load(phi_load_path)
    else:
        phi_load = np.zeros((n_aperture, n_aperture))

    slm_idx = get_aperture_indices(n_aperture, n_aperture, border_x, npix + border_x - 1, 0, npix - 1, aperture_width,
                                   aperture_width)
    n_centre = n_aperture ** 2 // 2 + n_aperture // 2 - 1
    n_centre_ref = n_aperture ** 2 // 2 + n_aperture // 2

    idx = range(n_aperture ** 2)
    idx = np.reshape(idx, (n_aperture, n_aperture))
    idx = idx[roi_min_x:roi_min_x + roi_n, roi_min_y:roi_min_y + roi_n]
    idx = idx.flatten()

    phi_int = np.zeros_like(zeros_full)
    phi_int[slm_idx[0][n_centre]:slm_idx[1][n_centre], slm_idx[2][n_centre]:slm_idx[3][n_centre]] = \
        slm_phase[slm_idx[0][n_centre]:slm_idx[1][n_centre], slm_idx[2][n_centre]:slm_idx[3][n_centre]]
    phi_int[slm_idx[0][n_centre_ref]:slm_idx[1][n_centre_ref], slm_idx[2][n_centre_ref]:slm_idx[3][n_centre_ref]] = \
        slm_phase[slm_idx[0][n_centre_ref]:slm_idx[1][n_centre_ref], slm_idx[2][n_centre_ref]:slm_idx[3][n_centre_ref]]

    slm_disp_obj.display(phi_int)

    cam_obj.start()
    img_exp_check = cam_obj.get_image(exp_time)
    cam_obj.stop()

    # Load measured laser intensity profile
    i_rec = np.load(pms_obj.i_path)
    i_laser = pt.load_filter_upscale(i_rec, npix, 1, filter_size=pms_obj.i_filter_size)
    i_laser = np.pad(i_laser, ((0, 0), (128, 128)))

    # Find Camera position with respect to SLM
    popt_clb, img_cal = find_camera_position(slm_disp_obj, cam_obj, pms_obj, lin_phase, exp_time=exp_time / 20,
                                             aperture_diameter=npix // 20, roi=[400, 400])

    cal_pos_x = popt_clb[0] + cam_obj.res[1] // 2
    cal_pos_y = popt_clb[1] + cam_obj.res[0] // 2

    plt.figure()
    plt.imshow(img_cal, cmap='turbo')
    plt.plot(cal_pos_x, cal_pos_y, 'wx')

    # Determine region of interest on camera
    w_cam = int(img_size) // 2 * 2
    h_cam = int(img_size) // 2 * 2
    offset_x = int((cal_pos_x - w_cam // 2) // 2 * 2)
    offset_y = int((cal_pos_y - h_cam // 2) // 2 * 2)
    roi_cam = [w_cam, h_cam, offset_x, offset_y]

    # Take camera images
    p_max = np.sum(i_laser[slm_idx[0][n_centre]:slm_idx[1][n_centre], slm_idx[2][n_centre]:slm_idx[3][n_centre]])

    norm = np.zeros(roi_n ** 2)
    img = np.zeros((img_size, img_size, roi_n ** 2))
    aperture_power = np.zeros(roi_n ** 2)
    dt = np.zeros(roi_n ** 2)
    aperture_width_adj = np.zeros(roi_n ** 2)

    n_img = int(2 * n_avg * roi_n ** 2)
    cam_obj.roi = roi_cam
    cam_obj.start(n_img)
    test = np.copy(zeros_full)
    for i in range(roi_n ** 2):
        t_start = time.time()
        ii = idx[i]
        idx_0, idx_1 = np.unravel_index(ii, phi_load.shape)

        norm[i] = p_max / np.sum(i_laser[slm_idx[0][ii]:slm_idx[1][ii], slm_idx[2][ii]:slm_idx[3][ii]])
        masked_phase = np.copy(zeros_full)
        test_now = np.copy(zeros_full)
        aperture_width_tar = np.sqrt(aperture_width ** 2 * norm[i])
        pad = int((aperture_width_tar - aperture_width) // 2)
        aperture_width_adj[i] = aperture_width + 2 * pad

        masked_phase[slm_idx[0][ii] - pad:slm_idx[1][ii] + pad, slm_idx[2][ii] - pad:slm_idx[3][ii] + pad] = \
            slm_phase[slm_idx[0][ii] - pad:slm_idx[1][ii] + pad, slm_idx[2][ii] - pad:slm_idx[3][ii] + pad] + \
            phi_load[idx_0, idx_1]
        masked_phase[slm_idx[0][n_centre]:slm_idx[1][n_centre], slm_idx[2][n_centre]:slm_idx[3][n_centre]] = \
            slm_phase[slm_idx[0][n_centre]:slm_idx[1][n_centre], slm_idx[2][n_centre]:slm_idx[3][n_centre]]

        test_now[slm_idx[0][ii] - pad:slm_idx[1][ii] + pad, slm_idx[2][ii] - pad:slm_idx[3][ii] + pad] = 1
        test += test_now

        slm_disp_obj.display(np.remainder(masked_phase, 2 * np.pi))
        if i == 0:
            time.sleep(2 * slm_disp_obj.delay)

        img_avg = hw.get_image_avg(cam_obj, exp_time, n_avg)

        img[:, :, i] = np.copy(img_avg)
        aperture_power[i] = np.mean(img[:, :, i]) * aperture_width ** 2 / aperture_width_adj[i] ** 2

        dt[i] = time.time() - t_start
        print(dt[i])
        print(i)
    cam_obj.stop()
    cam_obj.roi = roi_mem
    t = np.cumsum(dt)

    # Save data
    date_saved = time.strftime('%y-%m-%d_%H-%M-%S', time.localtime())
    path = pms_obj.data_path + date_saved + '_measure_slm_wavefront'
    os.mkdir(path)
    np.save(path + '/img', img)

    plt.figure()
    plt.imshow(test)

    # Fit sine to images
    fit_sine = ft.FitSine(fl, pms_obj.k)

    popt_sv = []
    pcov_sv = []
    perr_sv = []

    x, y = pt.make_grid(img[:, :, 0], scale=cam_obj.pitch)

    for i in range(roi_n ** 2):
        ii = idx[i]
        img_i = img[:, :, i]

        dx = (slm_idx[2][ii] - slm_idx[2][n_centre]) * slm_disp_obj.pitch
        dy = (slm_idx[0][ii] - slm_idx[0][n_centre]) * slm_disp_obj.pitch
        fit_sine.set_dx_dy(dx, dy)

        a_guess = np.sqrt(np.max(img_i)) / 2
        p0 = np.array([0, a_guess, a_guess])
        bounds = ([-np.pi, 0, 0], [np.pi, 2 * a_guess, 2 * a_guess])
        x_data = np.vstack((x.ravel(), y.ravel()))
        popt, pcov = opt.curve_fit(fit_sine.fit_sine, x_data, img_i.ravel(), p0, bounds=bounds, maxfev=50000)

        perr = np.sqrt(np.diag(pcov))
        popt_sv.append(popt)
        pcov_sv.append(pcov)
        perr_sv.append(perr)
        print(i + 1)

    dphi = -np.reshape(np.vstack(popt_sv)[:, 0], (roi_n, roi_n))
    dphi_err = np.reshape(np.vstack(perr_sv)[:, 0], (roi_n, roi_n))
    a = np.reshape(np.vstack(popt_sv)[:, 1], (roi_n, roi_n))
    b = np.reshape(np.vstack(popt_sv)[:, 2], (roi_n, roi_n))

    aperture_area = np.reshape(aperture_width_adj, dphi.shape) ** 2
    i_fit = np.abs(a * b)
    i_fit_adj = i_fit / aperture_area
    i_fit_mask = i_fit > 0.01 * np.max(i_fit)

    # Determine phase
    dphi_uw_nopad = pt.unwrap_2d(dphi)
    dphi_uw_notilt = ft.remove_tilt(dphi_uw_nopad)
    pad_roi = ((roi_min_x, n_aperture - roi_n - roi_min_x), (roi_min_y, n_aperture - roi_n - roi_min_y))
    dphi_uw = np.pad(dphi_uw_nopad, pad_roi)

    if benchmark is True:
        rmse = m.rms_phase(dphi_uw_notilt / 2 / np.pi)
        p2v = np.max(dphi_uw_notilt / 2 / np.pi) - np.min(dphi_uw_notilt / 2 / np.pi)
        print('RMS error: lambda /', 1 / rmse)
        print('Peak-to-valley error: lambda /', 1 / p2v)

    dphi_uw_mask = pt.unwrap_2d_mask(dphi, i_fit_mask)
    dphi_uw_mask = np.pad(dphi_uw_mask, pad_roi)

    plt.figure()
    plt.imshow(dphi_uw / np.pi / 2, cmap='magma')
    plt.colorbar()
    plt.title('Unwrapped measured phase')

    fig1, axs1 = plt.subplots(1, 2, sharex=True, sharey=True)
    axs1[0].imshow(img[:, :, -1], cmap='turbo')
    fit_test = np.reshape(fit_sine.fit_sine(x_data, *popt_sv[-1]), (img_size, img_size))
    axs1[1].imshow(fit_test, cmap='turbo')

    # Save data
    np.save(path + '//dphi', dphi)
    np.save(path + '//dphi_uw', dphi_uw)
    np.save(path + '//cal_pos_x', cal_pos_x)
    np.save(path + '//cal_pos_y', cal_pos_y)
    np.save(path + '//i_fit', i_fit)
    np.save(path + '//dphi_uw_mask', dphi_uw_mask)
    np.save(path + '//i_fit_mask', i_fit_mask)
    np.save(path + '//t', t)
    np.save(path + '//popt_sv', popt_sv)
    np.save(path + '//perr_sv', perr_sv)
    return path
