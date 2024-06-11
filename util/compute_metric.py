from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio
from tqdm import tqdm

import matplotlib.pyplot as plt
import lpips
import numpy as np
import torch
from skimage.metrics import structural_similarity
from torchmetrics.functional import structural_similarity_index_measure 
import cv2

device = 'cpu'
device = 'cuda:0'
loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

task = 'gaussian_blur'
# task = 'inpainting'
# task = 'super_resolution'
# task = 'motion_blur'

factor = 4
sigma = 0.1
scale = 1.0

resample_exp = False
# resample_exp = True

diff_pir_exp = False
# diff_pir_exp = True

ddrm_exp = False
# ddrm_exp = True

MCG = False
# MCG = True

BAD_PSNR = 10

NUM_IMGS = 1000

# label_root = Path(f'./samples/imagenet256_1k/')


# files from server results 
recon_imgs = ""

psnr_delta_list = []
psnr_normal_list = []

lpips_delta_list = []
lpips_normal_list = []

ssim_delta_list = []
ssim_delta_list_v2 = []

psnrs_less_28 = []
for idx in tqdm(range(NUM_IMGS)):
    if idx % 500 == 0: print(f"Iter {idx}/{1000}")
    fname = str(idx).zfill(5)
    
    label = plt.imread(label_root / f'{fname}.png')[:, :, :3]
    delta_recon = plt.imread(recon_imgs / f'{fname}.png')[:, :, :3]

    psnr_delta = peak_signal_noise_ratio(label, delta_recon)
    mse = np.mean((label - delta_recon)**2)
    my_psnr = 10 * np.log10(1 / (mse + 1e-10))
    
    ssim_delta = structural_similarity(label, delta_recon, channel_axis=-1, data_range = label.max()- label.min())
    ssim_v2 = structural_similarity_index_measure(torch.from_numpy(delta_recon).permute(2,0,1).unsqueeze(0)
        , torch.from_numpy(label).permute(2,0,1).unsqueeze(0), reduction=None)

    if psnr_delta < BAD_PSNR: psnrs_less_28.append((fname, psnr_delta))

    psnr_delta_list.append(psnr_delta)
    ssim_delta_list.append(ssim_delta)
    ssim_delta_list_v2.append(ssim_v2)

    delta_recon = torch.from_numpy(delta_recon).permute(2, 0, 1).to(device)
    
    label = torch.from_numpy(label).permute(2, 0, 1).to(device)

    delta_recon = delta_recon.view(1, 3, 256, 256) * 2. - 1.
    
    label = label.view(1, 3, 256, 256) * 2. - 1.

    delta_d = loss_fn_vgg(delta_recon, label)

    lpips_delta_list.append(delta_d.cpu().item())

    print(f"Index: {idx} | PSNR Delta: {psnr_delta_list[-1]:.4f},{my_psnr:4f} | SSIM Delta: {ssim_delta_list[-1]:.4f} | LPIPS Delta: {lpips_delta_list[-1]:.4f}")

psnr_delta_avg = sum(psnr_delta_list) / len(psnr_delta_list)
lpips_delta_avg = sum(lpips_delta_list) / len(lpips_delta_list)
ssim_avg = sum(ssim_delta_list) / len(ssim_delta_list)
ssim_avg_v2 = sum(ssim_delta_list_v2) / len(ssim_delta_list_v2)

print(recon_imgs)
print(f'Delta PSNR: {psnr_delta_avg}', " std: ", np.array(psnr_delta_list).std())
print(f'Delta LPIPS: {lpips_delta_avg}')
print(f'SSIM: {ssim_avg}')

print("Bad PSNRS: ", len(psnrs_less_28), psnrs_less_28)    