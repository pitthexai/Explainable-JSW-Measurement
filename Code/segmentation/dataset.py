import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from monai.transforms import (
    Compose, Spacingd, Resized, RandFlipd, RandGridDistortiond, 
    RandGaussianNoised, RandShiftIntensityd, EnsureTyped, ToTensord, Lambdad
)

def convert_mask_dtype(mask):
    return mask.squeeze(0).long()


def get_transforms(height, width, mode="train"):
    base_transforms = [
        Spacingd(keys=["image", "mask"], pixdim=(1.0, 1.0), mode=("bilinear", "nearest")),
        Resized(keys=["image", "mask"], spatial_size=(height, width), mode=("bilinear", "nearest")),
    ]

    if mode == "train":
        augmentation_transforms = [
            RandFlipd(keys=["image", "mask"], spatial_axis=1, prob=0.5),
            RandGridDistortiond(keys=["image", "mask"], prob=0.2, distort_limit=0.03),
            RandGaussianNoised(keys="image", prob=0.2, mean=0.0, std=0.1),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        ]
    else:
        augmentation_transforms = []

    return Compose([
        *base_transforms,
        EnsureTyped(keys=["image", "mask"]),
        *augmentation_transforms,
        ToTensord(keys=["image", "mask"]),
        Lambdad(keys="mask", func=convert_mask_dtype)
    ])


class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, device, transform=None):
        self.device = device
        self.transform = transform

        self.images_filenames = sorted(os.listdir(images_dir))
        self.masks_filenames = sorted(os.listdir(masks_dir))
        self.images_dir = images_dir
        self.masks_dir = masks_dir

    def __len__(self):
        return len(self.images_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images_filenames[idx])
        mask_path = os.path.join(self.masks_dir, self.masks_filenames[idx])

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.expand_dims(mask, axis=0).astype(np.int64)

        sample = {"image": image, "mask": mask}
        if self.transform:
            sample = self.transform(sample)

        return torch.tensor(sample["image"]).to(self.device), (torch.tensor(sample["mask"]) / 255).long().to(self.device)