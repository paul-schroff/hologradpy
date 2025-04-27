import torch
from torch import Tensor as tt
import torch.nn as nn
import numpy as np
from numpy.typing import NDArray
from .. import hardware as hw

from .utils.fourier_utils import fft_2d as fft
from ..torch_functions import ASM

class VirtualSlm(nn.Module):
    """
    This class models pixel crosstalk on the SLM and the propagation of light 
    from the SLM to the camera.
    """
    def __init__(self, 
                 slm_disp_obj: hw.SlmBase, 
                 pms_obj: hw.ParamsBase, 
                 phi: NDArray[np.float_], 
                 npix_pad: int, 
                 npix: int | None = None, 
                 e_slm: NDArray[np.float_] | None = None, 
                 kernel_ct: NDArray[np.float_] | None = None, 
                 pix_res: int | None = None,
                 propagation_type: str = 'fft', 
                 extent_lens: float | None = None, 
                 pd1: float | None = None, 
                 pd2: float | None = None, 
                 xf: float | None = None, 
                 device: str = 'cpu', 
                 slm_mask: NDArray[np.float_] | None = None,
                 precision: str | None = None, 
                 fft_shift: bool = True):
        """
        :param slm_disp_obj: Object created by a subclass of 
            :py:class:`hardware.SlmBase`
        :param pms_obj: Object created by a subclass of 
            :py:class:`hardware.ParamsBase`
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
            self.asm_obj = ASM(
                slm_disp_obj,
                pms_obj,
                self.pix_res,
                self.npix_full,
                pd1,
                pd2,
                extent_lens,
                xf,
                shift=self.fft_shift,
                precision=self.precision,
                device=self.device
            )
            phi -= self.asm_obj.phi_corr_native
            e_slm = e_slm * np.exp(1j * self.asm_obj.phi_corr)

        # Initialise optimisation parameters
        if self.precision == 'double':
            self.phi = nn.Parameter(
                torch.tensor(phi, dtype=torch.float64).to(device),
                requires_grad=True
            )
        else:
            self.phi = nn.Parameter(
                torch.tensor(phi, dtype=torch.float32).to(device),
                requires_grad=True
            )

        if npix is None:
            npix = phi.shape[0]
        self.npix = npix
        self.propagation_type = propagation_type
        if device == 'cuda':
            torch.cuda.empty_cache()

    def set_phi(self, new_phi: NDArray[np.float_]) -> None:
        """
        Set SLM phase from numpy array.

        :param new_phi: SLM phase [rad].
        """
        if self.precision == 'double':
            self.phi.data = torch.tensor(new_phi,
                                         dtype=torch.float64).to(self.device)
        else:
            self.phi.data = torch.tensor(new_phi,
                                         dtype=torch.float32).to(self.device)

    def forward(self) -> torch.Tensor:
        """
        Model the SLM and simulate the propagation of light from the SLM plane 
        to the image plane. This method is used by gradient-based optimizers.

        :return: Electric field in the image plane.
        """
        # Restrict phase value to lower limit 0 and upper limit 2 * pi when 
        # modelling pixel crosstalk. This prevents discontinuities in the cost 
        # function. Wrap the phase otherwise.
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
            x = torch.nn.functional.conv2d(
                x,
                self.kernel_ct,
                padding='same'
            ).squeeze()

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