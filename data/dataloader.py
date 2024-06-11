import os
from glob import glob
from PIL import Image
from typing import Callable, Optional
from functools import partial
from typing import Any, Tuple

import numpy as np
from torch.utils.data import DataLoader

import torchvision
from torchvision.datasets import VisionDataset

import torchvision.transforms as transforms

__DATASET__ = {}

def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls
    return wrapper


def get_dataset(name: str, root: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    return __DATASET__[name](root=root, **kwargs)


def get_dataloader(dataset: VisionDataset,
                   batch_size: int, 
                   num_workers: int, 
                   train: bool):
    dataloader = DataLoader(dataset, 
                            batch_size, 
                            shuffle=train, 
                            num_workers=num_workers, 
                            drop_last=train)
    return dataloader


@register_dataset(name='ffhq')
class FFHQDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)

        self.fpaths = sorted(glob(root + '/**/*.png', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img


@register_dataset(name='imagenet')
class ImagenetDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)

        self.fpaths = sorted(glob(root + '/**/*.png', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img


from torchvision.datasets.imagenet import (
    META_FILE,
    check_integrity,
    load_meta_file,
    parse_devkit_archive,
    parse_train_archive,
    parse_val_archive,
    verify_str_arg,
)

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp", ".JPEG")


def center_crop_arr(pil_image, image_size=256):
    # Imported from openai/guided-diffusion
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

@register_dataset(name='imagenet_v2')
class ImageNet(torchvision.datasets.ImageFolder):
    def __init__(self, root: str, split: str = "val", subset_txt='', samples_root="", meta_root="", transform = None, **kwargs):
        # if split == "train" or split == "val":
        #     root = os.path.join(root, "imagenet")
        root = self.root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", ("train", "val", "custom"))

        self.parse_archives()
        
        try:
            wnid_to_classes = load_meta_file(self.root)[0]
        except Exception:
            wnid_to_classes = load_meta_file(meta_root)[0]
            
        transform = transforms.Compose([partial(center_crop_arr, image_size=256), transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])    
        kwargs.pop('transforms', None)
        super(ImageNet, self).__init__(self.split_folder, transform= transform, **kwargs)
        self.root = root

        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx for idx, clss in enumerate(self.classes) for cls in clss}

        if len(subset_txt) > 0:
            with open(subset_txt, "r") as f:
                lines = f.readlines()
            self.samples = []
            for line in lines:
                idx = line.split()[0]
                if self.split == "custom":
                    idx = idx[:-5] + '.png'
                path = os.path.join(self.split_folder, idx)
                label = int(line.split()[1])
                self.samples.append((path, label))

            self.targets = [s[1] for s in self.samples]

        if len(samples_root) > 0:
            wnid_exists = [entry.name for entry in os.scandir(samples_root) if entry.is_dir()]
            
            #check fot empty dirs
            wnid_exists = [x for x in wnid_exists if len(os.listdir(os.path.join(samples_root,x)))>0]
            #import pdb; pdb.set_trace()
            
            wnid_to_idx = {wnid: self.wnid_to_idx[wnid] for wnid in wnid_exists}

            samples_done = self.make_dataset(samples_root, wnid_to_idx, extensions=IMG_EXTENSIONS)
            samples_done = [s[0].split("/")[-1].split(".")[-2] for s in samples_done]
            samples = []
            for sample in self.samples:
                k = [s in sample[0] for s in samples_done]
                if not any(k):
                    samples.append(sample)
            self.samples = samples
            self.targets = [s[1] for s in self.samples]

    def parse_archives(self) -> None:
        if not check_integrity(os.path.join(self.root, META_FILE)):
            try:
                parse_devkit_archive(self.root)
            except Exception:
                pass

        if not os.path.isdir(self.split_folder):
            if self.split == "train":
                parse_train_archive(self.root)
            elif self.split == "val":
                parse_val_archive(self.root)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        class_wnid = path.split("/")[-2]
        name = path.split("/")[-1].split(".")[0]

        return sample

    @property
    def split_folder(self) -> str:
        if self.split == "train" or self.split == "val":
            return os.path.join(self.root, self.split)
        else:
            return self.root

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

def get_imagenet_loader(dset, *, batch_size, num_workers, shuffle, drop_last, pin_memory, **kwargs):
    sampler = DistributedSampler(dset, shuffle=shuffle, drop_last=drop_last)
    loader = DataLoader(
        dset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle, sampler=sampler, pin_memory=pin_memory, persistent_workers=True
    )
    return loader
