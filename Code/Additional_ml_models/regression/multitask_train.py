import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from dataset import MaskDataset, get_transforms
from multitask_features import extract_multitask_features, ALL_COLUMNS

DATASET_CSV = "/data_vault/hexai02/Explainable-JSW-Measurement-main/jsw_dataset_final/train/jsw_measurements.csv"
MASK_ROOT = "/data_vault/hexai02/Explainable-JSW-Measurement-main/jsw_dataset_final/train/masks"

out_dir = "Models/multitask"
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(DATASET_CSV)
transforms = get_transforms()
dataset = MaskDataset(df, MASK_ROOT, transform=transforms)

X, y = [], []

for i in range(len(dataset)):
    data = dataset[i]
    img = data["image"].unsqueeze(0).cpu()

    side = "Right" if df.iloc[i]["SIDE"] == 1 else "Left"
    features = extract_multitask_features(img, side)

    targets = np.concatenate([
        data["medial_target"],
        data["lateral_target"]
    ])
    if not np.all(np.isfinite(targets)):
        continue

    X.append(features)
    y.append(targets)

X = np.array(X)
y = np.array(y)

model = RandomForestRegressor(
    n_estimators=400,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

joblib.dump(model, os.path.join(out_dir, "rf_multitask.pkl"))
print("✅ Saved multitask RandomForest model")
