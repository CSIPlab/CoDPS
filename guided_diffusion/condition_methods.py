from abc import ABC, abstractmethod
import torch
from  torch.distributions import multivariate_normal

import matplotlib.pyplot as plt
from util.img_utils import clear_color

__CONDITIONING_METHOD__ = {}

def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)

    
class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser
    
    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)
    
    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        if self.noiser.__name__ == 'gaussian':
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        
        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement-Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError
             
        return norm_grad, norm
   
    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass
 
@register_conditioning_method(name='CoDPS')
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
         # scale, t_div, div_ratio, inv_var 
        
        self.scale = float(kwargs.get('scale', 1.0))
        self.t_div = float(kwargs.get('t_div', 1.0))
        self.div_ratio = float(kwargs.get('div_ratio', 1.0))
        self.inv_var = float(kwargs.get('inv_var', 1.0))
        self.use_A_AT = bool(kwargs.get('use_A_AT'))

        print(f"IMP PosteriorSampling: scale={self.scale}, t_div={self.t_div}, div_ratio={self.div_ratio}, inv_var={self.inv_var}, use_A_AT={self.use_A_AT}")


    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        x_prev = None
        baralphas = kwargs.get('alpha_cum_prod', 1.0)
        t =  kwargs.get('t', 1.0)
        sigma_n =  self.noiser.sigma
        alphbar_ratio = (baralphas[t] / (1-baralphas[t]))
        scale = self.scale #/ alphbar_ratio
        mask = kwargs.get('mask', None)

        if not x_t.requires_grad:
            x_t.requires_grad_(True)

        if t < self.t_div:
            scale /= self.div_ratio
            # return x_t, torch.tensor([0])

        var_x0_xt = 1 / (alphbar_ratio + self.inv_var)
        
        mu_x0_xt_term_1 =  baralphas[t]**0.5 / (1- baralphas[t]) * x_t
        mu_x0_xt_term_2 = x_0_hat * self.inv_var
        mu_x0_xt = (mu_x0_xt_term_1 + mu_x0_xt_term_2 )  * var_x0_xt
        
        mean_y_xt =  self.operator.forward(mu_x0_xt, **kwargs)

        if self.use_A_AT:
            norm = self.operator.A_AT(n = x_t.shape[-2:], diff = measurement - mean_y_xt, sigma_n = sigma_n, var_x0_xt = var_x0_xt, mask = mask )

            loss = 1 / 2 * norm 
        else:
            # For inpaintnig 
            if mask is not None:
                var_y_xt = (sigma_n)**2 + var_x0_xt
                norm = torch.sum(torch.div((measurement - mean_y_xt)**2, var_y_xt))
                loss = 1 / 2 * norm
            else:
                # For general inverse problems , Assume AA^T = I
                var_y_xt = (sigma_n)**2 + var_x0_xt
                norm = torch.linalg.norm((measurement - mean_y_xt) )
                loss = 1 / (2 * var_y_xt)* norm**2

        norm_grad = torch.autograd.grad(outputs=loss, inputs=x_t)[0]
        
        x_t = x_t.detach()
        x_t -= norm_grad * scale

        return x_t, norm.cpu()**0.5

@register_conditioning_method(name='CoDPS+')
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        
        self.scale = float(kwargs.get('scale', 1.0))
        self.t_div = float(kwargs.get('t_div', 1.0))
        self.div_ratio = float(kwargs.get('div_ratio', 1.0))
        self.inv_var = float(kwargs.get('inv_var', 1.0))
        self.use_A_AT = bool(kwargs.get('use_A_AT'))

        print(f"IMP PosteriorSampling: scale={self.scale}, t_div={self.t_div}, div_ratio={self.div_ratio}, inv_var={self.inv_var}, use_A_AT={self.use_A_AT}")


    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        baralphas = kwargs.get('alpha_cum_prod', 1.0)
        t =  kwargs.get('t', 1.0)
        
        if t <= 0:
            return x_t, torch.tensor([0])
        
        # print("start with T ", t)
        sigma_n =  max(self.noiser.sigma, 1e-5) 
        alphbar_ratio = (baralphas[t] / (1-baralphas[t]))
        scale = self.scale #/ alphbar_ratio
        mask = kwargs.get('mask', None)
        alphas = kwargs.get('alphas', None)

        x_0_hat = x_0_hat.detach().requires_grad_()

        if t < self.t_div:
            scale /= self.div_ratio

        var_x0_xt = 1 / (alphbar_ratio + self.inv_var)
        mean_y_xt =  self.operator.forward(x_0_hat, **kwargs)

        # assert torch.allclose(measurement, self.operator.forward(measurement, **kwargs))
        
        if self.use_A_AT:
            norm = self.operator.A_AT(n = x_t.shape[-2:], diff = measurement - mean_y_xt, sigma_n = sigma_n, var_x0_xt = var_x0_xt, mask = mask )

            loss = 1 / 2 * norm 
        else:
            # For inpaintnig 
            if mask is not None:
                norm = torch.sum((measurement - mean_y_xt)**2)
                loss = 1 / 2 * norm
            else:
                # For general inverse problems , Assume AA^T = I
                var_y_xt = (sigma_n)**2 + var_x0_xt
                norm = torch.linalg.norm((measurement - mean_y_xt) )
                loss = 1 / (2 * var_y_xt)* norm**2

        norm_grad = torch.autograd.grad(outputs=loss, inputs=x_0_hat)[0]
        
        d_x0_ht_d_x_t = 1 / (baralphas[t]**0.5)
        
        if self.inv_var != 0:
            var_x0 = 1/ self.inv_var
            var_xt = 1 - baralphas[t] + var_x0 * baralphas[t]
            d_x0_ht_d_x_t =  var_x0 * (baralphas[t]**0.5) / var_xt
        # print("no graoh ", torch.linalg.norm(norm_grad), torch.linalg.norm(measurement - mean_y_xt))
        x_t -= norm_grad * scale * d_x0_ht_d_x_t

        return x_t, norm.cpu()**0.5