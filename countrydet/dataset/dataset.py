"""
---------------------------
Dataset & Augmentations
---------------------------
"""

import os
import glob
import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from countrydet.engine.cfg import COUNTRIES, NAME_TO_IDX


class DocDataset(Dataset):
    """Assumes a folder structure: root/COUNTRY_NAME/*.jpg (or png)
    Example: root/RUS/0001.jpg
    """
    def __init__(self, root: str, split: str = "train", image_size: int = 336, val_ratio: float = 0.15, seed: int = 42):
        self.root = root
        self.image_size = image_size
        self.split = split
        random.seed(seed)
        self.samples: list[tuple[str, int]] = []
        for cname in COUNTRIES:
            files = []
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
                files.extend(glob.glob(os.path.join(root, cname, ext)))
            files.sort()
            if not files:
                print("No files detected")
                continue
            # deterministic split per class
            idx = int(len(files) * (1 - val_ratio))
            if split == "train":
                chosen = files[:idx]
            else:
                chosen = files[idx:]
            self.samples += [(f, NAME_TO_IDX[cname]) for f in chosen]
        
        # Augmentations
        if split == "train":
            self.tf = transforms.Compose([
                transforms.Resize(int(image_size * 1.1)),
                transforms.RandomRotation( ( -4, 4 ) ),
                transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.05),
                transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
                transforms.RandomAdjustSharpness(1.5, p=0.2),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),  # CLIP norm
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fp, y = self.samples[idx]
        img = Image.open(fp).convert("RGB")
        x = self.tf(img)
        return x, y, fp