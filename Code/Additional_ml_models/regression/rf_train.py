#!/usr/bin/env python3
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import HistGradientBoostingRegressor




from sklearn.ensemble import RandomForestRegressor
from torch.utils.data import DataLoader

from dataset import MaskDataset, get_transforms
from deterministic import get_white_pixel_percentage

# --------------------------------------------------------------------------
# Fixed columns (MUST match evaluate.py)
# --------------------------------------------------------------------------
MEDIAL_COLUMNS = [
    "V00JSW150", "V00JSW175", "V00JSW200",
    "V00JSW225", "V00JSW250", "V00JSW275", "V00JSW300"
]

LATERAL_COLUMNS = [
    "V00LJSW700", "V00LJSW725", "V00LJSW750",
    "V00LJSW775", "V00LJSW800", "V00LJSW825",
    "V00LJSW850", "V00LJSW875", "V00LJSW900"
]


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------
def extract_features(dataset, side_label):
    """
    Extract white-pixel features and JSW targets from MaskDataset
    """
    X_medial, y_medial = [], []
    X_lateral, y_lateral = [], []

    for i in range(len(dataset)):
        data = dataset[i]
        img = data["image"].unsqueeze(0)

        # --------------------
        # MEDIAL
        # --------------------
        medial_feats = []
        medial_targets = []

        for idx, col in enumerate(MEDIAL_COLUMNS):
            pos = int(col[-3:]) / 1000
            if side_label == "Right":
                pos = 1 - pos

            wp = get_white_pixel_percentage(img, pos)
            gt = data["medial_target"][idx]

            if np.isfinite(gt):
                medial_feats.append(wp)
                medial_targets.append(gt)

        if len(medial_targets) == len(MEDIAL_COLUMNS):
            X_medial.append(medial_feats)
            y_medial.append(medial_targets)

        # --------------------
        # LATERAL
        # --------------------
        lateral_feats = []
        lateral_targets = []

        for idx, col in enumerate(LATERAL_COLUMNS):
            pos = int(col[-3:]) / 1000
            if side_label == "Right":
                pos = 1 - pos

            wp = get_white_pixel_percentage(img, pos)
            gt = data["lateral_target"][idx]

            if np.isfinite(gt):
                lateral_feats.append(wp)
                lateral_targets.append(gt)

        if len(lateral_targets) == len(LATERAL_COLUMNS):
            X_lateral.append(lateral_feats)
            y_lateral.append(lateral_targets)

    return (
        np.array(X_medial), np.array(y_medial),
        np.array(X_lateral), np.array(y_lateral)
    )


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train_rf_models(dataset_csv, mask_root, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(dataset_csv)
    transforms = get_transforms()

    for side, side_code in [("Left", 2), ("Right", 1)]:
        print(f"\n🔹 Training Random Forest for {side} knee")

        df_side = df[df["SIDE"] == side_code].reset_index(drop=True)
        dataset = MaskDataset(df_side, mask_root, transform=transforms)

        X_med, y_med, X_lat, y_lat = extract_features(dataset, side)

        # --------------------
        # Random Forest models
        # --------------------
        rf_medial = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )

        rf_lateral = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )

        print(f"  Medial samples: {X_med.shape[0]}")
        print(f"  Lateral samples: {X_lat.shape[0]}")

        rf_medial.fit(X_med, y_med)
        rf_lateral.fit(X_lat, y_lat)

        joblib.dump(rf_medial, os.path.join(out_dir, f"{side[0]}_medial.pkl"))
        joblib.dump(rf_lateral, os.path.join(out_dir, f"{side[0]}_lateral.pkl"))

        print(f"  ✅ Saved {side[0]}_medial.pkl and {side[0]}_lateral.pkl")
        # --------------------
        # Extra Trees models
        # --------------------
        et_medial = ExtraTreesRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        )

        et_lateral = ExtraTreesRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        )

        et_medial.fit(X_med, y_med)
        et_lateral.fit(X_lat, y_lat)

        joblib.dump(et_medial, os.path.join(out_dir, f"{side[0]}_medial_extratrees.pkl"))
        joblib.dump(et_lateral, os.path.join(out_dir, f"{side[0]}_lateral_extratrees.pkl"))

        print(f"  ✅ Saved {side[0]}_medial_extratrees.pkl and {side[0]}_lateral_extratrees.pkl")

        # --------------------
        # HistGradientBoosting
        # --------------------
        hgb_medial = MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_depth=6,
                learning_rate=0.05,
                max_iter=300,
                random_state=42
            )
        )

        hgb_lateral = MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_depth=6,
                learning_rate=0.05,
                max_iter=300,
                random_state=42
            )
        )

        hgb_medial.fit(X_med, y_med)
        hgb_lateral.fit(X_lat, y_lat)

        joblib.dump(hgb_medial, os.path.join(out_dir, f"{side[0]}_medial_hgb.pkl"))
        joblib.dump(hgb_lateral, os.path.join(out_dir, f"{side[0]}_lateral_hgb.pkl"))

        print(f"  ✅ Saved {side[0]}_medial_hgb.pkl and {side[0]}_lateral_hgb.pkl")


        


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
if __name__ == "__main__":

    DATASET_CSV = (
        "/data_vault/hexai02/Explainable-JSW-Measurement-main/"
        "jsw_dataset_final/train/jsw_measurements.csv"
    )

    MASK_ROOT = (
        "/data_vault/hexai02/Explainable-JSW-Measurement-main/"
        "jsw_dataset_final/train/masks"
    )

    OUTPUT_DIR = "Models/rf"

    train_rf_models(
        dataset_csv=DATASET_CSV,
        mask_root=MASK_ROOT,
        out_dir=OUTPUT_DIR
    )
