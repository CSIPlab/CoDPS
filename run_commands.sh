#!/bin/bash

# Setting the fields for the FFHQ dataset using DDIM sampling
dataset="FFHQ"
sampling_type="DDIM"
task_config="motion_deblur_config"

# Informing the user of the configuration being used
echo "Running the model with the following configurations:"
echo "Dataset: $dataset"
echo "Sampling Type: $sampling_type"
echo "Task Config: $task_config"


# Running the command
python3 main.py --model_config=configs/${dataset}/${sampling_type}/model_config.yaml \
  --diffusion_config=configs/${dataset}/${sampling_type}/diffusion_config.yaml \
  --task_config=configs/${dataset}/${sampling_type}/${task_config}.yaml \
  --save_dir=results/ffhq/ --seed 225

# Sample output for reference
# : '
# Sample Output:
# Running the model with the following configurations:
# Dataset: FFHQ
# Sampling Type: DDIM
# Task Config: motion_deblur_config
# 2024-08-19 22:14:40,104 [CoDPS] >> Device set to cuda:0.
# Args:  Namespace(model_config='"configs/FFHQ/DDIM/model_config.yaml"', diffusion_config='"configs/FFHQ/DDIM/diffusion_config.yaml"', task_config='"configs/FFHQ/DDIM/motion_deblur_config.yaml"', gpu=0, seed=225, save_dir='"results/ffhq/motion_deblur"')
# task_config  '"{'conditioning': {'method': 'CoDPS+', 'params': {'scale': 0.05, 't_div': 15, 'div_ratio': 20, 'inv_var': 1, 'use_A_AT': True}}, 'data': {'name': 'ffhq', 'root': './samples/ffhq_00000_256/'}, 'measurement': {'operator': {'name': 'motion_blur', 'kernel_size': 61, 'intensity': 0.5}, 'noise': {'name': 'gaussian', 'sigma': 0.05}}}"
# 2024-08-19 22:14:42,401 [CoDPS] >> Operation: motion_blur / Noise: gaussian
# IMP PosteriorSampling: scale=0.05, t_div=15.0, div_ratio=20.0, inv_var=1.0, use_A_AT=True
# 2024-08-19 22:14:42,401 [CoDPS] >> Conditioning method : CoDPS+
# 2024-08-19 22:14:42,416 [CoDPS] >> Inference for image 0

# PSNR: 27.722464361797567
# 2024-08-19 22:14:52,395 [CoDPS] >> Inference for image 1

# PSNR: 28.586990291581706
# 2024-08-19 22:14:57,449 [CoDPS] >> Inference for image 2

# PSNR: 27.650951518236766
# Final psnr:  27.98680205720535
# '