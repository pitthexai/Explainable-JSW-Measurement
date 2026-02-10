#!/usr/bin/env python3
import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from dataset import MaskDataset, get_transforms
from deterministic import get_white_pixel_percentage

# Import model + column lists from your training file
from Train_jsw_1dcnn import JSW1DCNN, MEDIAL_COLUMNS, LATERAL_COLUMNS

KNEE_SIDES = ["Left", "Right"]


def evaluate_sequence_model(model, dataset, side, cols, compartment):
    """
    Evaluate a 1D CNN model on either medial or lateral targets.
    Safely skips samples containing NaNs in inputs or targets.
    """
    model.eval()

    gt_all = []
    pred_all = []

    for i in range(len(dataset)):
        data = dataset[i]
        img = data["image"].unsqueeze(0).cpu()

        # -------- build input sequence --------
        x = []
        for col in cols:
            pos = int(col[-3:]) / 1000
            if side == "Right":
                pos = 1 - pos

            val = get_white_pixel_percentage(img, pos)
            x.append(val)

        x = np.array(x, dtype=np.float32)
        y = data[f"{compartment}_target"]

        # -------- NaN filtering (CRITICAL) --------
        if not np.all(np.isfinite(x)):
            continue
        if not np.all(np.isfinite(y)):
            continue

        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            pred = model(x_t).squeeze().cpu().numpy()

        gt_all.extend(y.tolist())
        pred_all.extend(pred.tolist())

    # Final safety check
    gt_all = np.array(gt_all, dtype=np.float32)
    pred_all = np.array(pred_all, dtype=np.float32)

    mask = np.isfinite(gt_all) & np.isfinite(pred_all)
    gt_all = gt_all[mask]
    pred_all = pred_all[mask]

    if len(gt_all) == 0:
        return {"MSE": np.nan, "MAE": np.nan}

    return {
        "MSE": mean_squared_error(gt_all, pred_all),
        "MAE": mean_absolute_error(gt_all, pred_all),
    }


if __name__ == "__main__":
    # ---- Paths (your test set) ----
    csv_path = "/data_vault/hexai02/Explainable-JSW-Measurement-main/jsw_dataset_final/test/jsw_measurements.csv"
    mask_root = "/data_vault/hexai02/Explainable-JSW-Measurement-main/jsw_dataset_final/test/masks"

    # ---- Model directory ----
    model_dir = "Models/jsw_1dcnn"
    out_dir = "Results"
    os.makedirs(out_dir, exist_ok=True)

    # Load metadata CSV and transforms
    df = pd.read_csv(csv_path)
    transforms = get_transforms()

    summary_rows = []

    for side in KNEE_SIDES:
        side_code = 1 if side == "Right" else 2
        df_side = df[df["SIDE"] == side_code].reset_index(drop=True)

        dataset = MaskDataset(df_side, mask_root, transform=transforms)

        # -------------------------
        # Medial model
        # -------------------------
        medial_model_path = os.path.join(model_dir, f"{side}_medial.pth")
        if not os.path.exists(medial_model_path):
            raise FileNotFoundError(f"Missing model: {medial_model_path}")

        medial_model = JSW1DCNN(seq_len=len(MEDIAL_COLUMNS))
        medial_model.load_state_dict(torch.load(medial_model_path, map_location="cpu"))

        medial_metrics = evaluate_sequence_model(
            medial_model, dataset, side, MEDIAL_COLUMNS, "medial"
        )

        # -------------------------
        # Lateral model
        # -------------------------
        lateral_model_path = os.path.join(model_dir, f"{side}_lateral.pth")
        if not os.path.exists(lateral_model_path):
            raise FileNotFoundError(f"Missing model: {lateral_model_path}")

        lateral_model = JSW1DCNN(seq_len=len(LATERAL_COLUMNS))
        lateral_model.load_state_dict(torch.load(lateral_model_path, map_location="cpu"))

        lateral_metrics = evaluate_sequence_model(
            lateral_model, dataset, side, LATERAL_COLUMNS, "lateral"
        )

        # -------------------------
        # Save summary row
        # -------------------------
        summary_rows.append({
            "Knee": side,
            "Approach": "1D-CNN",
            "Medial_MSE": medial_metrics["MSE"],
            "Medial_MAE": medial_metrics["MAE"],
            "Lateral_MSE": lateral_metrics["MSE"],
            "Lateral_MAE": lateral_metrics["MAE"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "summary_1dcnn.csv")
    summary_df.to_csv(summary_path, index=False)

    print(summary_df)
    print(f"\n✅ Saved: {summary_path}")
