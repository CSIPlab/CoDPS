# Gaussian is All You Need: A Unified Framework for Solving Inverse Problems via Diffusion Posterior Sampling

## Abstract 

Diffusion models can generate a variety of high-quality images by modeling complex data distributions. Trained diffusion models can also be very effective image priors for solving inverse problems. Most of the existing diffusion-based methods integrate data consistency steps within the diffusion reverse sampling process. The data consistency steps rely on an approximate likelihood function. In this paper, we show that the existing approximations are either insufficient or computationally inefficient. 
To address these issues, we propose a unified likelihood approximation method that incorporates a covariance correction term to enhance the performance and avoids propagating gradients through the diffusion model. The correction term, when integrated into the reverse diffusion sampling process, achieves better convergence towards the true data posterior for selected distributions and improves performance on real-world natural image datasets. Furthermore, we present an efficient way to factorize and invert the covariance matrix of the likelihood function for several inverse problems. We present comprehensive experiments to demonstrate the effectiveness of our method over several existing approaches.

# Getting Started

## Environment 

To set up the project environment, follow these steps:

1. Ensure that you have [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) installed on your machine.

2. create a Conda environment with Python version 3.9.18:
   ```bash
   conda create --name codps python=3.9.18
    ```
3. Activate the Environment:
   ```bash
    conda activate codps
    ```

## Requirements    
   Install Required Packages:
   ```bash
   pip install -r requirements.txt
   ```

## Pretrained models 
The Download pretrained model for the FFHQ dataset from [here](https://github.com/DPS2022/diffusion-posterior-sampling?tab=readme-ov-file#3-set-environment) and for the ImageNet dataset from [here](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt). 

Move the downloaded files to their default locations: `ckpts/ffhq_10m.pt` and `ckpts/256x256_diffusion_uncond.pt`, respectively. Alternatively, you can update the model_path configuration in one of the config files, such as `configs/FFHQ/DDPM/model_config.yaml`, to point to the correct location.

## Datasets

For ImageNet, we use the processed 256x256 validation set from [here](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz). For the [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset), following [DPS](https://github.com/DPS2022/diffusion-posterior-sampling), we use the first 1,000 images (folder `00000`) of the FFHQ dataset as validation. We used the `data/data_resizer.py` script to resize and crop the images to `256x256`.


## Evaluation

We provide all the configurations used in the paper inside the `config` folder. To evaluate the models, you can run the provided Bash script `run_commands.sh`. You can modify the evaluation by overriding specific fields in the script. Edit the `run_commands.sh` script to modify the following variables:

- **`dataset`**: Specify the dataset to use (e.g., `FFHQ`, `IMAGENET`).
- **`sampling_type`**: Define the sampling type (e.g., `DDIM`, `DDPM`).
- **`task_config`**: Choose the task configuration file (e.g., `motion_deblur_config.yaml`, `gaussian_deblur_config.yaml`, `inpainting_config.yaml`, `super_resolution_config.yaml`).

### Running the Script

After setting the configurations, execute

```bash
bash run_commands.sh
```

Example usage:

```bash
python3 main.py --model_config=configs/FFHQ/DDIM/model_config.yaml \
  --diffusion_config=configs/FFHQ/DDIM/diffusion_config.yaml \
  --task_config=configs/FFHQ/DDIM/motion_deblur_config.yaml \
  --save_dir=results/ffhq/motion_deblur --seed=225
```

### Batch Evaluation

To evaluate the models with configuration pairs, use the `batch_command.sh` script. This script runs the model with several combination of datasets and task configurations.

```bash
bash batch_command.sh
```

## Acknowledgement
This repo is built upon [DPS](https://github.com/DPS2022/diffusion-posterior-sampling) and utilizes [Guided-diffusion](https://github.com/openai/guided-diffusion) and [MotionBlur](https://github.com/LeviBorodenko/motionblur).