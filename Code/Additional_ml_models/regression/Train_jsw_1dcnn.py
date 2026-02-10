#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dataset import MaskDataset, get_transforms
from deterministic import get_white_pixel_percentage

# ============================================================
# Collate function to skip invalid (None) samples
# ============================================================
def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.default_collate(batch)

# ============================================================
# Constants
# ============================================================
MEDIAL_COLUMNS = [
    "V00JSW150", "V00JSW175", "V00JSW200",
    "V00JSW225", "V00JSW250", "V00JSW275", "V00JSW300"
]

LATERAL_COLUMNS = [
    "V00LJSW700", "V00LJSW725", "V00LJSW750",
    "V00LJSW775", "V00LJSW800", "V00LJSW825",
    "V00LJSW850", "V00LJSW875", "V00LJSW900"
]

KNEE_SIDES = ["Left", "Right"]

# ============================================================
# 1D CNN Model
# ============================================================
class JSW1DCNN(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * seq_len, seq_len)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ============================================================
# Dataset for JSW sequences
# ============================================================
class JSWSequenceDataset(Dataset):
    def __init__(self, df, mask_root, side, compartment):
        self.df = df.reset_index(drop=True)
        self.side = side
        self.compartment = compartment
        self.transforms = get_transforms()
        self.base_ds = MaskDataset(self.df, mask_root, transform=self.transforms)

        self.cols = MEDIAL_COLUMNS if compartment == "medial" else LATERAL_COLUMNS

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        data = self.base_ds[idx]
        img = data["image"].unsqueeze(0).cpu()

        # -------- Input sequence --------
        x = []
        for col in self.cols:
            pos = int(col[-3:]) / 1000
            if self.side == "Right":
                pos = 1 - pos
            x.append(get_white_pixel_percentage(img, pos))

        x = np.array(x, dtype=np.float32)

        if not np.all(np.isfinite(x)):
            return None

        # -------- Target sequence --------
        y = data[f"{self.compartment}_target"]
        if np.any(~np.isfinite(y)):
            return None

        return (
            torch.tensor(x, dtype=torch.float32).unsqueeze(0),  # (1, seq_len)
            torch.tensor(y, dtype=torch.float32)                # (seq_len,)
        )

# ============================================================
# Training function
# ============================================================
def train_model(dataset, seq_len, out_path):
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_skip_none
    )

    model = JSW1DCNN(seq_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(50):
        total_loss = 0.0
        steps = 0

        for batch in loader:
            if batch is None:
                continue

            x, y = batch
            pred = model(x)
            loss = loss_fn(pred, y)

            if torch.isnan(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        if steps > 0 and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss = {total_loss / steps:.4f}")

    torch.save(model.state_dict(), out_path)
    print(f"✅ Saved {out_path}")

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":

    csv_path = "/data_vault/hexai02/Explainable-JSW-Measurement-main/jsw_dataset_final/train/jsw_measurements.csv"
    mask_root = "/data_vault/hexai02/Explainable-JSW-Measurement-main/jsw_dataset_final/train/masks"

    out_dir = "Models/jsw_1dcnn"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    for side in KNEE_SIDES:
        side_code = 1 if side == "Right" else 2
        df_side = df[df["SIDE"] == side_code]

        for compartment in ["medial", "lateral"]:
            seq_len = 7 if compartment == "medial" else 9
            dataset = JSWSequenceDataset(df_side, mask_root, side, compartment)
            out_path = os.path.join(out_dir, f"{side}_{compartment}.pth")

            print(f"\n🔹 Training 1D CNN for {side} {compartment}")
            train_model(dataset, seq_len, out_path)
