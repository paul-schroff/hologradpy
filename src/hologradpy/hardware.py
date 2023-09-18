"""
This module provides to interface with the camera and the SLM.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import final


class ParamsBase(ABC):
    """
    Class storing experimental parameters and constant properties.
    """

    @property
    @abstractmethod
    def wavelength(self):
        """ Wavelength [m]."""
        pass

    @property
    @final
    def k(self):
        """ Wavenumber [rad/m]."""
        return 2 * np.pi / self.wavelength

    @property
    @abstractmethod
    def beam_diameter(self):
        """ Diameter of incident Gaussian beam [m]."""
        pass

    # Lens-related parameters
    @property
    @abstractmethod
    def fl(self):
        """ Focal length [m]."""
        pass

    # Doublet lens used with the ASM
    n1 = 1.59847  #: Refractive index of flint
    n2 = 1.76182  #: Refractive index of crown
    r1 = 0.1377  #: Radius of curvature of lens surface 1 [m]
    r2 = -0.1377  #: Radius of curvature of lens surface 2 [m]
    r3 = -0.9304  #: Radius of curvature of lens surface 3 [m]
    lens_aperture = 48.3e-3  #: Diameter of the lens aperture [m]

    # Load paths for measured constant field at the SLM
    @property
    @abstractmethod
    def phi_path(self):
        pass

    @property
    @abstractmethod
    def i_path(self):
        pass

    # Save path
    @property
    @abstractmethod
    def data_path(self):
        """Path to store data."""
        pass

    # Parameters to process measured constant SLM phase and intensity
    crop = 32  #: Number of unused pixels at the edges of the SLM
    phi_filter_size = 5  #: Size of the uniform filter to smoothen the constant SLM phase
    i_filter_size = 3  #: Size of the uniform filter to smoothen the constant SLM intensity


class CameraBase(ABC):
    def __init__(self, res, pitch, roi):
        """
        Abstract camera base class. You will have to implement your own camera driver by writing a subclass of this
        class. If you implement your own ``__init__()`` method in your subclass, call
        ``super().__init__(res, pitch, roi)`` in your ``__init__()`` method first to initialise all required parameters.

        :param res: Resolution of the camera (width, height) [px].
        :param pitch: Pixel pitch of the camera [m].
        :param roi: Region of interest on the camera (width, height, offset x, offset y) [px]. The offset is measured
            with respect to the bottom left corner of the image.
        """
        self.res = res
        self.pitch = pitch
        self.roi = roi

    @property
    def cam_size(self):
        """
        Calculates the physical size of the camera.

        :return: x, y dimensions of the camera [m].
        """
        return self.pitch * self.res

    @abstractmethod
    def start(self, n=1):
        """
        You have to implement this yourself. Starts the acquisition.

        :param n: Number of frames to be captured.
        """
        raise NotImplementedError

    @abstractmethod
    def get_image(self, exp_time):
        """
        You have to implement this yourself. Acquires and returns a camera image of the shape determined by
        ``self.roi``.

        :param exp_time: Exposure time.
        :return: Camera image of shape as defined by ``self.roi``
        """
        raise NotImplementedError

    @abstractmethod
    def stop(self):
        """
        You have to implement this yourself. Stops the acquisition.
        """
        raise NotImplementedError


def get_image_avg(cam_obj, exp_time, n_avg):
    """
    This function captures multiple camera images and calculates the average.

    :param cam_obj: Instance of your own camera class which is a subclass of ``CameraBase``.
    :param exp_time: Exposure time.
    :param n_avg: Number of frames to be averaged.
    :return: Averaged image.
    """
    img = np.zeros((cam_obj.roi[1], cam_obj.roi[0]))
    for j in range(n_avg):
        img += cam_obj.get_image(exp_time)
    img = img / n_avg
    return img


class SlmBase(ABC):
    def __init__(self, res, pitch):
        """
        Abstract SLM base class to display a phase pattern on the device. You will have to implement your own driver by
        writing a subclass which inherits from ``SlmBase``. If you implement your own ``__init__()`` method in your
        subclass, call ``super().__init__(res, pitch)`` in your ``__init__()`` method first to initialise all required
        parameters.

        :param res: Resolution of the SLM (width, height) [px].
        :param pitch: Pixel pitch of the SLM [m].
        """
        self.res = res
        self.pitch = pitch

    @property
    def slm_size(self):
        """
        Calculates the physical size of the SLM.

        :return: x, y dimensions of the SLM [m].
        """
        return self.pitch * self.res

    @property
    def meshgrid_slm(self):
        """
        Calculates an x, y meshgrid using the pixel pitch and the native resolution of the SLM.
        :return: x, y meshgrid [m].
        """
        x = np.arange(-self.slm_size[0] / 2, self.slm_size[0] / 2, self.pitch)
        return np.meshgrid(x, x)

    @abstractmethod
    def display(self, phi):
        """
        This function displays a phase pattern on the SLM. You have to implement this yourself.

        :param phi: SLM phase pattern [radians].
        :return:
        """
        raise NotImplementedError
