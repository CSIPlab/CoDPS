import numpy as np 
import torch
import os
from resizer import resize
from PIL import Image

import torchvision.transforms.functional as TF
import cv2
# Load images from foler

# 
data_path = ""
save_path = ""

image_files = os.listdir(data_path)

def resize_crop(img_in):
    # Check if the image array is provided
    if not isinstance(img_in, np.ndarray):
        raise ValueError("Input must be a NumPy array representing an image.")

    # Get the current dimensions
    height, width, _ = img_in.shape

    # Check if either dimension is less than 256
    if width < 256 or height < 256:
        raise ValueError("Both dimensions must be greater than or equal to 256.")

    # Calculate the minimum dimension
    min_dimension = min(width, height)

    # Crop to the minimum dimension
    left = (width - min_dimension) // 2
    top = (height - min_dimension) // 2
    right = (width + min_dimension) // 2
    bottom = (height + min_dimension) // 2

    img_cropped = img_in[top:bottom, left:right]

    # Resize to 256x256 using interpolation
    img_resized = cv2.resize(img_cropped, (256, 256), interpolation=cv2.INTER_AREA)

    return img_resized

for i, filename in enumerate(image_files):
    image_path = os.path.join(data_path, filename)
    image = torch.from_numpy(np.array(Image.open(image_path)).astype(np.float64)).permute(2,0,1) / 255
    resized_img = resize(image, scale_factors=1/4)#.permute(1,2,0)
    resized_img -= resized_img.min()          # Scale minimum value to 0
    resized_img /= resized_img.max()          # Scale maximum value to 1

    resized_img = TF.to_pil_image(resized_img)  # Convert back to PIL Image
    resized_img.save(save_path + '/' + filename)

    if i % 50 == 0:
        print(f"Progress: {i}/{len(image_files)}")

