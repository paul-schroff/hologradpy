"""
Setting up your hardware
========================

This script sets constant experimental parameters and implements camera and SLM drivers. You will have to do this for
your own hardware.
"""
import time
import numpy as np
from src.hologradpy import hardware as hw

# These modules are only needed for our camera and SLM drivers
import cv2 as cv
from harvesters.core import Harvester
from screeninfo import get_monitors
import imageio

# %%
# Setting experimental parameters and other constant parameters
# -------------------------------------------------------------
#
# To start off, we write our own subclass of ``hardware.ParamsBase`` which sets some experimental parameters and other
# constants needed in this script.


class Params(hw.ParamsBase):
    wavelength = 670e-9         # Wavelength [m]
    beam_diameter = 7.25e-3     # Diameter of incident beam [m]
    fl = 0.25                   # Focal length [m]

    data_path = '../../holography_data/'

    # Path to measured constant SLM phase and intenstiy
    phi_path = data_path + '23-09-13_13-20-49_measure_slm_wavefront/dphi_uw.npy'
    i_path = data_path + '23-09-13_11-47-42_measure_slm_intensity/i_rec.npy'

    phi_filter_size = 2
    crop = 64


pms_obj = Params()

# %%
# Implementing SLM and camera drivers
# -----------------------------------
# This module provides the ``hardware.CameraBase`` class and the ``hardware.SlmDisp`` class to interface with the camera
# and the SLM. You will have to write your own subclasses for the specific devices you are using. Here, we defined the
# subclasses ``Camera`` and ``SlmDisp`` to interface with a MatrixVision BlueFox 3 camera and a Hamamatsu SLM. Make sure
# you implement all abstract methods of ``hw.CameraBase`` and ``hw.SlmBase`` in your own subclasses.


class Camera(hw.CameraBase):
    def __init__(self, res, pitch, name='before', roi=None, gain=0, bayer=False):
        if roi is None:
            roi = [1280, 960, 0, 0]
        self.roi = roi
        super().__init__(res, pitch, self.roi)
        self.count = 0
        self.gain = gain
        self.bayer = bayer
        self.name = name
        self.h = None
        self.ia = None
        self.bayer_slope = 0.010487497932442524
        self.bayer_offset = 2.195178550143578
        self.max_frame_count = 2 ** 16 - 1

    def start(self, n=1):
        if n >= self.max_frame_count:
            n = self.max_frame_count

        self.h = Harvester()

        self.h.add_file('C:/Program Files/MATRIX VISION/mvIMPACT Acquire/bin/x64/mvGenTLProducer.cti')

        self.h.update()

        print("start init ia")

        serial_numbers = []
        for info in self.h.device_info_list:
            serial_numbers.append(info.serial_number)
        if self.name == 'before':
            n_cam = serial_numbers.index('F0600075')
        if self.name == 'after':
            n_cam = serial_numbers.index('F0600086')
        self.ia = self.h.create(n_cam)
        self.ia.remote_device.node_map.ExposureAuto.value = 'Off'
        self.ia.remote_device.node_map.mvLowLight.value = 'Off'
        self.ia.remote_device.node_map.ExposureAuto.value = 'Off'
        self.ia.remote_device.node_map.BlackLevelAuto.value = 'Off'
        self.ia.remote_device.node_map.GainAuto.value = 'Off'
        self.ia.remote_device.node_map.ExposureTime.value = 100
        self.ia.remote_device.node_map.PixelFormat.value = 'Mono16'
        self.ia.remote_device.node_map.AcquisitionMode.value = 'MultiFrame'
        self.ia.remote_device.node_map.AcquisitionFrameRateEnable.value = True
        self.ia.remote_device.node_map.AcquisitionFrameRate.value = 12
        if self.name == 'before':
            self.ia.remote_device.node_map.ReverseX.value = True
            self.ia.remote_device.node_map.ReverseY.value = True
        elif self.name == 'after':
            self.ia.remote_device.node_map.ReverseX.value = False
            self.ia.remote_device.node_map.ReverseY.value = True
        self.ia.remote_device.node_map.TriggerMode.value = 'On'
        self.ia.remote_device.node_map.TriggerSource.value = 'Software'
        self.ia.remote_device.node_map.TriggerSelector.value = 'FrameStart'

        w, h, dx, dy = self.roi

        self.ia.remote_device.node_map.AcquisitionFrameCount.value = n
        self.ia.remote_device.node_map.Gain.value = self.gain

        if dx <= self.ia.remote_device.node_map.OffsetX.value:
            self.ia.remote_device.node_map.OffsetX.value = dx
            self.ia.remote_device.node_map.OffsetY.value = dy
            self.ia.remote_device.node_map.Width.value = w
            self.ia.remote_device.node_map.Height.value = h
        else:
            self.ia.remote_device.node_map.Width.value = w
            self.ia.remote_device.node_map.Height.value = h
            self.ia.remote_device.node_map.OffsetX.value = dx
            self.ia.remote_device.node_map.OffsetY.value = dy

        self.ia.start()
        print("start acquisition")

    def get_image(self, exp_time_):
        if self.count >= self.max_frame_count:
            self.stop()
            self.start(n=self.max_frame_count)

        self.ia.remote_device.node_map.ExposureTime.value = exp_time_

        self.ia.remote_device.node_map.TriggerSoftware.execute()

        with self.ia.fetch() as buffer:
            component = buffer.payload.components[0]
            width = component.width
            height = component.height

            im = np.array(component.data.reshape(height, width)).astype(np.double)

        if self.bayer is True:
            im[0::2, 1::2] = im[0::2, 1::2] * (1 + self.bayer_slope) + self.bayer_offset
            im[1::2, 0::2] = im[1::2, 0::2] * (1 + self.bayer_slope) + self.bayer_offset
        self.count += 1
        return im

    def stop(self):
        self.ia.stop()
        print("stopped acquisition")
        self.ia.destroy()
        self.h.reset()


class SlmDisp(hw.SlmBase):
    def __init__(self, res, pitch, calib=None, delay=0.2, dx=0, dy=0):
        super().__init__(res, pitch)
        self.max_phase = 2 * np.pi  # Largest value for phase wrapping
        self.slm_norm = 128         # Gray level on the SLM corresponding to max_phase
        # Gray level vs phase lookup table
        self.lut = np.load(pms_obj.data_path + '23-02-17_13-49-14_calibrate_grey_values/phase.npy')
        self.idx_lut = np.argmin(np.abs(self.lut - self.max_phase))  # Index of max_phase in lut
        self.lut = self.lut[:self.idx_lut]
        # Path to Hamamatsu SLM correction pattern.
        self.cal_path = pms_obj.data_path + 'deformation_correction_pattern/CAL_LSH0802439_' + '{:.0f}'.\
            format(np.around(pms_obj.wavelength * 1e9, decimals=-1)) + 'nm.bmp'
        self.delay = 0.2  # Time to wait after displaying a phase pattern on the SLM [s]

        if calib == 1 or calib is True:
            self.calib_flag = True
            self.calib = imageio.imread(self.cal_path)
            self.calib = np.pad(self.calib, ((0, 0), (0, 8)))
        elif calib == 0 or calib is False or calib is None:
            self.calib_flag = False
            self.calib = np.zeros((self.res[1], self.res[0]))
        self.delay = delay
        self.dx = dx
        self.dy = dy

        monitor = get_monitors()[-1]

        cv.namedWindow('screen', cv.WINDOW_NORMAL)
        cv.resizeWindow('screen', self.res[1], self.res[0])
        cv.moveWindow('screen', monitor.x, monitor.y)
        cv.setWindowProperty('screen', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        cv.waitKey(100)
        print("SlmDisp initialised")

    def display(self, phi_slm):
        im_res_y, im_res_x = phi_slm.shape
        slm_res_y, slm_res_x = self.res
        slm_pad_x = (slm_res_x - im_res_x) // 2
        slm_pad_y = (slm_res_y - im_res_y) // 2

        slm_norm = self.slm_norm

        if slm_pad_x == 0 and slm_pad_y == 0:
            phi_zeros = slm_norm * phi_slm / (2 * np.pi)
        elif -slm_pad_y - self.dy == 0:
            phi_zeros = np.zeros((slm_res_y, slm_res_x))
            phi_disp = slm_norm * phi_slm / (2 * np.pi)
            phi_zeros[slm_pad_y - self.dy:, slm_pad_x - self.dx:-slm_pad_x - self.dx] = phi_disp
        elif -slm_pad_x - self.dx == 0:
            phi_zeros = np.zeros((slm_res_y, slm_res_x))
            phi_disp = slm_norm * phi_slm / (2 * np.pi)
            phi_zeros[slm_pad_y - self.dy:-slm_pad_y - self.dy, slm_pad_x - self.dx:] = phi_disp
        else:
            phi_zeros = np.zeros((slm_res_y, slm_res_x))
            phi_disp = slm_norm * phi_slm / (2 * np.pi)
            phi_zeros[slm_pad_y - self.dy:-slm_pad_y - self.dy, slm_pad_x - self.dx:-slm_pad_x - self.dx] = phi_disp

        if self.calib_flag is False:
            phi_zeros = phi_zeros.astype('uint8')
        else:
            phi_zeros = np.remainder(phi_zeros + self.calib, slm_norm).astype('uint8')

        cv.imshow('screen', phi_zeros)
        cv.waitKey(1)
        time.sleep(self.delay)


# %%
# We can now use the classes ``Params``, ``Camera`` and ``SlmDisp`` in other scripts.
