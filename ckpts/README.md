# Model Checkpoints Directory (`ckpts`)

Download and save all model checkpoints in this folder.

## Pretrained models 
The Download pretrained model for the FFHQ dataset from [here](https://github.com/DPS2022/diffusion-posterior-sampling?tab=readme-ov-file#3-set-environment) and for the ImageNet dataset from [here](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt). 

Move the downloaded files to their default locations: `ckpts/ffhq_10m.pt` and `ckpts/256x256_diffusion_uncond.pt`, respectively. Alternatively, you can update the model_path configuration in one of the config files, such as `configs/FFHQ/DDPM/model_config.yaml`, to point to the correct location.
