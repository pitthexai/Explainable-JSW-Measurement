import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms

class MaskDataset(Dataset):
    def __init__(self, df, mask_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        side_letter = "R" if row["SIDE"] == 1 else "L"
        mask_filename = f"{int(row['ID'])}_{side_letter}.png"
        mask_path = os.path.join(self.mask_dir, mask_filename)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask {mask_path} not found.")

        mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
        _, binary_mask = cv2.threshold(mask, 128, 1, cv2.THRESH_BINARY)
        binary_mask_rgb = np.stack([binary_mask]*3, axis=-1).astype(np.uint8)
        binary_mask_rgb = Image.fromarray(binary_mask_rgb)

        if self.transform:
            binary_mask_rgb = self.transform(binary_mask_rgb)

        medial_target = row[['V00JSW150','V00JSW175','V00JSW200','V00JSW225',
                             'V00JSW250','V00JSW275','V00JSW300']].values.astype(np.float32)
        lateral_target = row[['V00LJSW700','V00LJSW725','V00LJSW750','V00LJSW775','V00LJSW800',
                              'V00LJSW825','V00LJSW850','V00LJSW875','V00LJSW900']].values.astype(np.float32)

        return {
            'image': binary_mask_rgb,
            'medial_target': medial_target,
            'lateral_target': lateral_target,
            'IMPIXSZ': row["V00IMPIXSZ"],
            "ORIG_H": row["ORIG_H"]
        }

def get_transforms():
    return transforms.Compose([transforms.ToTensor()])