#!/bin/bash

datasets=("FFHQ" "IMAGENET")        
sampling_types=("DDIM" "DDPM")      
tasks=("motion_deblur_config" "gaussian_deblur_config" "super_resolution_config") 

seed=225
save_base_dir="results"

for dataset in "${datasets[@]}"; do
  for sampling_type in "${sampling_types[@]}"; do
    for task in "${tasks[@]}"; do
      model_config="configs/${dataset}/${sampling_type}/model_config.yaml"
      diffusion_config="configs/${dataset}/${sampling_type}/diffusion_config.yaml"
      task_config="configs/${dataset}/${sampling_type}/${task}.yaml"
      save_dir="${save_base_dir}/${dataset}/${sampling_type}/${task}"

      echo "Running with :"
      echo "Dataset: $dataset"
      echo "Sampling Type: $sampling_type"
      echo "Task Config: $task"
      echo "Save Directory: $save_dir"
      
      # Run the Python script with the current configuration
      python3 main.py --model_config="$model_config" \
        --diffusion_config="$diffusion_config" \
        --task_config="$task_config" \
        --save_dir="$save_dir" --seed="$seed"

      echo "Completed configuration: $dataset/$sampling_type/$task"
    done
  done
done
