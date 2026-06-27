"""Camera helper utilities."""

import numpy as np


def get_image_avg(cam_obj, exp_time, n_avg):
    """
    This function captures multiple camera images and calculates the average.

    :param cam_obj: A camera object exposing ``roi`` and ``get_image``.
    :param exp_time: Exposure time.
    :param n_avg: Number of frames to be averaged.
    :return: Averaged image.
    """
    img = np.zeros((cam_obj.roi[1], cam_obj.roi[0]))
    for j in range(n_avg):
        img += cam_obj.get_image(exp_time)
    img = img / n_avg
    return img
