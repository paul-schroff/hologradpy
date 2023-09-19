"""
This module contains functions to characterise light potentials.
"""

import numpy as np                          # Used for array manipulation


def normalize(img, roi, thres=0.5):
    """
    Normalises an image by dividing it by the pixel sum in a region of interest. Only pixels brighter than
    thres * max(roi * img) are taken into account.

    :param img: Input image.
    :param roi: Binary mask containing region of interest.
    :param thres: Pixel value threshold (see above).
    :return: Normalised image.
    """

    img_roi = img * roi
    theshold = thres * np.max(img_roi)
    mask = img_roi > theshold
    return img / np.mean(img[mask])


def fidelity(signal, a_tar, phi_tar, a_out, phi_out):
    """
    Calculate fidelity between two electric fields in a region of interest (signal region).

    :param signal: Binary mask containing the region of interest.
    :param a_tar: Target amplitude pattern.
    :param phi_tar: Target phase pattern.
    :param a_out: Amplitude of light potential.
    :param phi_out: Phase of light potential.
    :return: Fidelity.
    """
    e_tar_s = (a_tar * np.exp(1j * phi_tar)) * signal
    e_out_s = (a_out * np.exp(1j * phi_out)) * signal

    fid = np.sum(e_tar_s * np.conjugate(e_out_s)) / (np.sum(np.abs(e_tar_s) ** 2) * np.sum(np.abs(e_out_s) ** 2)) ** 0.5
    fid = np.abs(fid) ** 2
    return fid


def rms(signal, i_target, i_out, frac=0.5):
    """
    Calculate normalised root-mean-squared error between two images inside a region of interest. Only pixels which are brighter
    than ``frac * max(i_target_norm)`` are taken into account, where ``i_target_norm`` is the normalised target
    intensity pattern.

    :param signal: Binary mask containing region of interest (signal region).
    :param i_target: Target intensity pattern.
    :param i_out: Intensity pattern of light potential.
    :param frac: Threshold as explained above.
    :return: Normalised rms error.
    """
    i_target = i_target * signal
    i_out = i_out * signal
    mr_idx = i_target > (1 - frac) * np.max(i_target)
    mr_mask = np.zeros_like(signal)
    mr_mask[mr_idx] = 1
    mr = np.count_nonzero(mr_mask)

    i_target_w_norm = i_target * mr_mask / np.sum(i_target * mr_mask)
    i_out_w_norm = i_out * mr_mask / np.sum(i_out * mr_mask)

    n = (mr_mask * (i_out_w_norm - i_target_w_norm) / i_target_w_norm) ** 2

    n = np.sqrt(np.sum(n[mr_mask > 0]) / mr)

    return n


def rms_phase(phi):
    """
    Calculates the root-mean-squared error of an image.

    :param phi: Phase pattern.
    :return: Root-mean-squared error.
    """
    phi_mean = np.mean(phi)
    return np.sqrt(np.mean((phi - phi_mean) ** 2))


def psnr(signal, i_target, i_out):
    """
    Calculates the peak signal-to-noise ratio between two images in a region of interest according to
    https://doi.org/10.1364/OE.24.006249.

    :param signal: Binary mask containing region of interest (signal region).
    :param i_target: Target intensity pattern.
    :param i_out: Intensity pattern of light potential.
    :return: Peak signal-to-noise ratio [dB].
    """
    i_target_w = i_target * signal
    i_out_w = i_out * signal
    
    i_target_w_norm = i_target_w / np.sum(i_target_w)
    i_out_w_norm = i_out_w / np.sum(i_out_w)
    
    mr = np.count_nonzero(signal)
    
    mse = np.sum(signal * (i_out_w_norm - i_target_w_norm) ** 2) / mr
    
    return 20 * np.log10(np.max(i_target_w_norm * signal) / np.sqrt(mse))


def eff(signal, i_out):
    """
    Calculates the predicted efficiency of a light potential by dividing the pixel sum in the signal region by
    the pixel sum in the entire pattern.

    :param signal: Binary mask containing the signal region.
    :param i_out: Intensity pattern of the light potential.
    :return: Efficiency.
    """
    i_out_tot = np.sum(i_out)
    i_out_w_tot = np.sum(i_out * signal)

    return i_out_w_tot / i_out_tot
