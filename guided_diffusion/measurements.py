'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

from abc import ABC, abstractmethod
from functools import partial
import yaml
from torch.nn import functional as F
from torchvision import torch
from motionblur.motionblur import Kernel
import torch.fft as fft

from util.resizer import Resizer
from util.img_utils import Blurkernel, fft2_m


# =================
# Operation classes
# =================

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass
    
    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)

    def get_x_start(self, y):
        # return A^Ty
        return y
    
    def generate_operator(self, **kwargs):
        '''
            A function to generate/regenerate operators and measurement matrices
        '''
        pass

@register_operator(name='noise')
class DenoiseOperator(LinearOperator):
    def __init__(self, device):
        self.device = device
    
    def forward(self, data):
        return data

    def transpose(self, data):
        return data
    
    def ortho_project(self, data):
        return data

    def project(self, data):
        return data_config

@register_operator(name='super_resolution')
class SuperResolutionOperatorV2(LinearOperator):
    def __init__(self, scale_factor,  device, degradation = 'gaussian'):
        self.device = device
        self.scale_factor = scale_factor
        self.degradation = degradation

        self.generate_operator()

    def generate_operator(self, **kwargs):
        if self.degradation == 'bicubic':
            self.conv = Blurkernel(blur_type='bicubic',
                                kernel_size=self.scale_factor,
                                device=self.device).to(self.device)
            self.kernel = self.conv.get_kernel()
            self.conv.update_weights(self.kernel.type(torch.float32))
        elif self.degradation == 'avg_pool':
            self.conv = Blurkernel(blur_type='avg_pool',
                                kernel_size=self.scale_factor,
                                device=self.device).to(self.device)
            self.kernel = self.conv.get_kernel()
            self.conv.update_weights(self.kernel.type(torch.float32))
        else:    
            self.conv = Blurkernel(blur_type='gaussian',
                                kernel_size=9,
                                std=3,
                                device=self.device).to(self.device)
            self.kernel = self.conv.get_kernel()
            self.conv.update_weights(self.kernel.type(torch.float32))
        
        return super().generate_operator(**kwargs)
    
    def get_x_start(self, y):
        # zero pad 
        N,C,H,W = y.shape
        AAT_inv_y = self.A_AT_for_back(y)
        adjoint = torch.zeros(N,C,H*self.scale_factor, W*self.scale_factor, dtype=y.dtype).to(self.device)
        adjoint[:,:,::self.scale_factor,::self.scale_factor] = AAT_inv_y
        # adjoint = AAT_inv_y.repeat_interleave(self.scale_factor, dim=-1).repeat_interleave(self.scale_factor, dim=-2)
        # assume symmetric kernel 
        print("ad shape ",adjoint.shape, y.shape)
        deconv = self.conv.adjoint(adjoint) 

        if adjoint.shape != deconv.shape:
            deconv = deconv[:,:,1:,1:]
        return deconv

    def forward(self, data, **kwargs):

        assert len(data.shape) == 4
        # blur 
        blurred = self.conv(data)

        # Happenes for bicubic because of an even kernel
        if blurred.shape[2:] != data.shape[2:]:
            # blurred = blurred[:,:,2:,2:]
            blurred = blurred[:,:,1:,1:]

        return blurred[:,:,::self.scale_factor, ::self.scale_factor]

    def transpose(self, data, **kwargs):
        return data

    def project(self, data, measurement, **kwargs):
        return data - self.transpose(self.forward(data)) + self.transpose(measurement)   
    
    def calculate_delta_M_v1(self, d,  m, delta):   
        '''
            \Gamma compute SR from \Lambda 
        ''' 
        if len(delta) != m * d:
            raise ValueError(f"Length of delta must be m*d. {len(delta)} != {m} * {d}")
        
        sqrt_d, sqrt_m = int(d**0.5), int(m**0.5)
        i = torch.arange(m)

        r_offset = (i%sqrt_m + sqrt_d * (sqrt_m) * (i // sqrt_m)).long().unsqueeze(1).unsqueeze(2)
        offset = m * sqrt_d * torch.arange(sqrt_d).unsqueeze(0).unsqueeze(2)
        offset2 = sqrt_m * torch.arange(sqrt_d).unsqueeze(0).unsqueeze(1)
        all_offset = (r_offset + offset + offset2).reshape(-1).to(self.device)

        lambda_i = torch.gather(delta,0,all_offset).reshape(m,-1).sum(axis=1) / d
        return lambda_i.reshape(sqrt_m, sqrt_m)

    def A_AT_for_back(self, y):
        # Zero pad kernel 
        # FFT shift kernel 
        n = [y.shape[-1] * self.scale_factor]
        k = self.kernel.shape[0]

        pad_width = pad_height = n[0]-k
        
        if k % 2 == 0:
            padding = (pad_width//2, pad_width//2, pad_height//2, pad_height//2) 
        else:
            padding = (pad_width//2+1, pad_width//2, pad_height//2+1, pad_height//2) 
        
        padded_kernel = (torch.nn.functional.pad(self.kernel, padding))
        padded_kernel = torch.fft.ifftshift(padded_kernel).type(torch.float32).to(self.device)

        delta = torch.fft.fft2(padded_kernel).type(torch.complex64)
        # print("Inside supervised v2: delta.shape: ", delta.shape, n)
        delta = delta.reshape(-1)
        delta_conj = torch.conj(delta)

        # Map from n^^2 diagonal to m^^2 diag where n/m = self.scale_factor
        delta_squared = (delta * delta_conj)
        d, M = self.scale_factor, int(n[0]//self.scale_factor)
        delta_M = self.calculate_delta_M_v1(d**2, M**2, delta_squared)
        aat_inv = 1 / (delta_M + 1e-8)

        y_fft = fft.fft2(y, dim=[-2,-1])
        y_hat = fft.ifft2(aat_inv.unsqueeze(0) * y_fft, dim=[-2,-1]).real

        return y_hat

    def A_AT(self, n, diff, var_x0_xt, sigma_n, **kwargs):
        # Zero pad kernel 
        # FFT shift kernel 
        k = self.kernel.shape[0]

        pad_width = pad_height = n[0]-k
        
        if k % 2 == 0:
            padding = (pad_width//2, pad_width//2, pad_height//2, pad_height//2) 
        else:
            padding = (pad_width//2+1, pad_width//2, pad_height//2+1, pad_height//2) 
        
        padded_kernel = (torch.nn.functional.pad(self.kernel, padding))
        padded_kernel = torch.fft.ifftshift(padded_kernel).type(torch.float32).to(self.device)

        delta = torch.fft.fft2(padded_kernel).type(torch.complex64)
        # print("Inside supervised v2: delta.shape: ", delta.shape, n)
        delta = delta.reshape(-1)
        delta_conj = torch.conj(delta)

        # Map from n^^2 diagonal to m^^2 diag where n/m = self.scale_factor
        delta_squared = (delta * delta_conj)
        d, M = self.scale_factor, int(n[0]//self.scale_factor)
        delta_M = self.calculate_delta_M_v1(d**2, M**2, delta_squared)

        inv_cov = sigma_n**2 + var_x0_xt * delta_M
        cov = 1 / inv_cov

        diff_fft = fft.fft2(diff, dim=[-2,-1])
        loss_right_term = fft.ifft2(cov.unsqueeze(0) * diff_fft, dim=[-2,-1]).real
        
        loss = torch.sum(diff * loss_right_term)
        
        return loss

    def get_kernel(self):
        return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)

@register_operator(name='motion_blur')
class MotionBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.intensity = intensity
        self.generate_operator()

    def generate_operator(self, **kwargs):
        self.conv = Blurkernel(blur_type='motion',
                               kernel_size=self.kernel_size,
                               std=self.intensity,
                               device=self.device).to(self.device)

        self.kernel = Kernel(size=(self.kernel_size, self.kernel_size), intensity=self.intensity)
        kernel = torch.tensor(self.kernel.kernelMatrix, dtype=torch.float32)
        self.conv.update_weights(kernel)

    def forward(self, data, **kwargs):
        # A^T * A 
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        kernel = self.kernel.kernelMatrix.type(torch.float32).to(self.device)
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)

    def A_AT(self, n, diff, var_x0_xt, sigma_n, **kwargs):
        # Zero pad kernel 
        # FFT shift kernel 
        kernel = torch.from_numpy(self.kernel.kernelMatrix).to(self.device)
        k = kernel.shape[0]
        
        pad_width = pad_height = n[0]-k
        padding = (pad_width//2+1, pad_width//2, pad_height//2+1, pad_height//2) 
        padded_kernel = (torch.nn.functional.pad(kernel, padding))
        padded_kernel = torch.fft.ifftshift(padded_kernel).type(torch.float32)

        delta = torch.fft.fft2(padded_kernel).type(torch.complex64)
        delta_conj = torch.conj(delta)

        inv_cov = sigma_n**2 + var_x0_xt * delta * delta_conj
        cov = 1 / inv_cov
        # print(sigma_n**2 , torch.max(torch.real((var_x0_xt * delta * delta_conj))))

        diff_fft = fft.fft2(diff, dim=[-2,-1])
        loss_right_term = fft.ifft2(cov.unsqueeze(0) * diff_fft, dim=[-2,-1]).real
        
        loss = torch.sum(diff * loss_right_term)

        return loss


@register_operator(name='gaussian_blur')
class GaussialBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.intensity = intensity

        self.generate_operator()

    def generate_operator(self, **kwargs):
        # print(" GENERATING gaussian blur ")
        self.conv = Blurkernel(blur_type='gaussian',
                               kernel_size=self.kernel_size,
                               std=self.intensity,
                               device=self.device).to(self.device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))
    
    def forward(self, data, **kwargs):
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def A_AT(self, n, diff, var_x0_xt, sigma_n, **kwargs):
        # Zero pad kernel 
        # FFT shift kernel 
        k = self.kernel.shape[0]
        
        pad_width = pad_height = n[0]-k
        padding = (pad_width//2+1, pad_width//2, pad_height//2+1, pad_height//2) 
        padded_kernel = (torch.nn.functional.pad(self.kernel, padding))
        padded_kernel = torch.fft.ifftshift(padded_kernel).type(torch.float32).to(self.device)

        delta = torch.fft.fft2(padded_kernel).type(torch.complex64)
        delta_conj = torch.conj(delta)

        inv_cov = sigma_n**2 + var_x0_xt * delta * delta_conj
        cov = 1 / inv_cov
        
        diff_fft = fft.fft2(diff, dim=[-2,-1])
        loss_right_term = fft.ifft2(cov.unsqueeze(0) * diff_fft, dim=[-2,-1]).real
        
        loss = torch.sum(diff * loss_right_term)

        return loss

    def get_kernel(self):
        return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)

@register_operator(name='inpainting')
class InpaintingOperator(LinearOperator):
    '''This operator get pre-defined mask and return masked image.'''
    def __init__(self, device):
        self.device = device
    
    def forward(self, data, **kwargs):
        try:
            return data * kwargs.get('mask', None).to(self.device)
        except:
            raise ValueError("Require mask")
    
    def transpose(self, data, **kwargs):
        return data
    
    def ortho_project(self, data, **kwargs):
        return data - self.forward(data, **kwargs)
    
    def project(self, data, measurement, **kwargs):
        '''
            FOR MCG: (I - A^TA) * data + A^T measurement
        '''
        return self.ortho_project(data, **kwargs) + self.transpose(self.forward(measurement, **kwargs))

    def A_AT(self, n, diff, var_x0_xt, sigma_n, mask, **kwargs):
        var_y_xt = (sigma_n)**2 + var_x0_xt * mask
        norm = torch.sum(torch.div(diff**2, var_y_xt))
        
        return norm 

# =============
# Noise classes
# =============


__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper

def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser

class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

@register_noise(name='clean')
class Clean(Noise):
    def forward(self, data):
        return data

@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma