import sys 

import time
from functools import partial
import os
import argparse
import yaml

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.utils as tv_utils
from torch.profiler import profile, record_function, ProfilerActivity

import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger

def seed(seed):
    # set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=22)
    parser.add_argument('--save_dir', type=str, default='./results')
    args = parser.parse_args()
   
    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
   
    # Print configurations 
    print("Args: ", args)
    print("task_config ", task_config)

    # Seed 
    seed(args.seed)
    
    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")
   
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)
   
    # Working directory
    out_path = os.path.join(args.save_dir, measure_config['operator']['name'])
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # Exception) In case of inpainting, we need to generate a mask 
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
           **measure_config['mask_opt']
        )
    total_psr = 0    
    psnr_init = 0
    record = False
    num_proc_imgs = len(loader)
    elapsed_times = []
    
    # Do Inference
    for i, ref_img in enumerate(loader):
        if i > 9: break
        # Generate new measurement matrix for each operator
        operator.generate_operator()
        
        logger.info(f"Inference for image {i}")
        fname = str(i).zfill(5) + '.png'
        ref_img = ref_img.to(device)

        # Exception) In case of inpainging,
        if measure_config['operator'] ['name'] == 'inpainting':
            mask = mask_gen(ref_img)
            mask = mask[:, 0, :, :].unsqueeze(dim=0)
            measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
            sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)

            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img, mask=mask)
            y_n = noiser(y)

        else: 
            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img)
            y_n = noiser(y)
         
        # Sampling
        x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
        if "imp" in cond_config['method']:
            x0 = operator.get_x_start(y_n)
            tv_utils.save_image((x0 + 1) / 2, f"pinvs/zzz_x0_{i}_torch_scale.png")    
        
        file_name_only = None
        if record: 
            file_name_only = os.path.splitext(fname)[0]
            os.makedirs(f"{out_path}/progress/{file_name_only}/", exist_ok=True)

        start_time = time.time()

        sample = sample_fn(x_start=x_start, measurement=y_n, record=record, save_root=out_path, fname = file_name_only)

        elapsed_time = time.time() - start_time
        
        recon_psnr = peak_signal_noise_ratio(clear_color(sample), clear_color(ref_img))
        
        total_psr += recon_psnr

        print(f"PSNR: {recon_psnr:4}, Running time: {elapsed_time} secs" )
        elapsed_times.append(elapsed_time)

        plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
        plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
        plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(sample))
        
    print("Final psnr: ", total_psr / num_proc_imgs)
    print(f"AVG elapsed times: count: len: {len(elapsed_times)}, mean: {np.mean(elapsed_times)}, std: {np.std(elapsed_times)}")

if __name__ == '__main__':
    main()
    