import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from dataset import MaskDataset, get_transforms
from typing import Dict
from model import ResNet50Regressor
import joblib
from deterministic import get_white_pixel_percentage
from multitask_features import extract_multitask_features, ALL_COLUMNS


# --------------------------------------------------------------------------
# Fixed columns
# --------------------------------------------------------------------------
KNEE_SIDES = ["Left", "Right"]

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
# Metric computation (ROBUST TO NaNs)
# --------------------------------------------------------------------------
def compute_metrics_from_df(df: pd.DataFrame) -> Dict[str, float]:
    metrics = {}

    # --------------------
    # MEDIAL
    # --------------------
    medial_gt, medial_pred = [], []
    for col in MEDIAL_COLUMNS:
        gt_col = df[f"{col}_Ground_Truth"].to_numpy()
        pred_col = df[f"{col}_Prediction"].to_numpy()

        for gt, pred in zip(gt_col, pred_col):
            if np.isfinite(gt) and np.isfinite(pred):
                medial_gt.append(gt)
                medial_pred.append(pred)

    metrics["Medial_MSE"] = (
        mean_squared_error(medial_gt, medial_pred)
        if len(medial_gt) > 0 else np.nan
    )
    metrics["Medial_MAE"] = (
        mean_absolute_error(medial_gt, medial_pred)
        if len(medial_gt) > 0 else np.nan
    )

    # --------------------
    # LATERAL
    # --------------------
    lateral_gt, lateral_pred = [], []
    for col in LATERAL_COLUMNS:
        gt_col = df[f"{col}_Ground_Truth"].to_numpy()
        pred_col = df[f"{col}_Prediction"].to_numpy()

        for gt, pred in zip(gt_col, pred_col):
            if np.isfinite(gt) and np.isfinite(pred):
                lateral_gt.append(gt)
                lateral_pred.append(pred)

    metrics["Lateral_MSE"] = (
        mean_squared_error(lateral_gt, lateral_pred)
        if len(lateral_gt) > 0 else np.nan
    )
    metrics["Lateral_MAE"] = (
        mean_absolute_error(lateral_gt, lateral_pred)
        if len(lateral_gt) > 0 else np.nan
    )

    return metrics


# --------------------------------------------------------------------------
# Scaling baseline
# --------------------------------------------------------------------------
def evaluate_scaling(test_dataset, label) -> pd.DataFrame:
    from deterministic import get_scaled_jsw_results_df
    return get_scaled_jsw_results_df(
        test_dataset,
        medial_columns=MEDIAL_COLUMNS,
        lateral_columns=LATERAL_COLUMNS,
        label=label
    )


# --------------------------------------------------------------------------
# Deterministic baseline
# --------------------------------------------------------------------------
def evaluate_deterministic(
    test_dataset,
    label,
    coef=77.88208682895399,
    intercept=1.3779068446995613
) -> pd.DataFrame:
    from deterministic import get_deterministic_jsw_results_df
    return get_deterministic_jsw_results_df(
        test_dataset,
        medial_columns=MEDIAL_COLUMNS,
        lateral_columns=LATERAL_COLUMNS,
        label=label,
        coef=coef,
        intercept=intercept
    )



# --------------------------------------------------------------------------
# Model evaluation
# --------------------------------------------------------------------------
def evaluate_model(
    model: torch.nn.Module,
    test_dataset,
    batch_size: int = 1,
    device: str = "cuda"
) -> pd.DataFrame:

    model.to(device)
    model.eval()

    results = {"Sample_ID": []}
    for col in MEDIAL_COLUMNS + LATERAL_COLUMNS:
        results[f"{col}_Ground_Truth"] = []
        results[f"{col}_Prediction"] = []

    loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    with torch.no_grad():
        for idx, data in enumerate(loader):
            img = data["image"].to(device)
            pred = model(img).squeeze().cpu().numpy()

            results["Sample_ID"].append(idx)

            # MEDIAL-ONLY MODEL
            if pred.shape[0] == len(MEDIAL_COLUMNS):
                for i, col in enumerate(MEDIAL_COLUMNS):
                    results[f"{col}_Ground_Truth"].append(
                        data["medial_target"][0][i].item()
                    )
                    results[f"{col}_Prediction"].append(pred[i])

                for i, col in enumerate(LATERAL_COLUMNS):
                    results[f"{col}_Ground_Truth"].append(
                        data["lateral_target"][0][i].item()
                    )
                    results[f"{col}_Prediction"].append(np.nan)

            # LATERAL-ONLY MODEL
            elif pred.shape[0] == len(LATERAL_COLUMNS):
                for i, col in enumerate(MEDIAL_COLUMNS):
                    results[f"{col}_Ground_Truth"].append(
                        data["medial_target"][0][i].item()
                    )
                    results[f"{col}_Prediction"].append(np.nan)

                for i, col in enumerate(LATERAL_COLUMNS):
                    results[f"{col}_Ground_Truth"].append(
                        data["lateral_target"][0][i].item()
                    )
                    results[f"{col}_Prediction"].append(pred[i])

            else:
                raise RuntimeError(
                    f"Unexpected model output size: {pred.shape[0]}"
                )

    return pd.DataFrame(results)

def evaluate_random_forest(
    test_dataset,
    label,
    rf_medial,
    rf_lateral
):
    results = {"Sample_ID": []}

    for col in MEDIAL_COLUMNS + LATERAL_COLUMNS:
        results[f"{col}_Ground_Truth"] = []
        results[f"{col}_Prediction"] = []

    for i in range(len(test_dataset)):
        data = test_dataset[i]
        img = data["image"].unsqueeze(0)

        results["Sample_ID"].append(i)

        # --------------------
        # MEDIAL
        # --------------------
        medial_features = []
        for idx, col in enumerate(MEDIAL_COLUMNS):
            pos = int(col[-3:]) / 1000
            if label == "Right":
                pos = 1 - pos

            wp = get_white_pixel_percentage(img, pos)
            medial_features.append(wp)

        medial_preds = rf_medial.predict([medial_features])[0]

        for idx, col in enumerate(MEDIAL_COLUMNS):
            results[f"{col}_Ground_Truth"].append(
                data["medial_target"][idx]
            )
            results[f"{col}_Prediction"].append(
                medial_preds[idx]
            )

        # --------------------
        # LATERAL
        # --------------------
        lateral_features = []
        for idx, col in enumerate(LATERAL_COLUMNS):
            pos = int(col[-3:]) / 1000
            if label == "Right":
                pos = 1 - pos

            wp = get_white_pixel_percentage(img, pos)
            lateral_features.append(wp)

        lateral_preds = rf_lateral.predict([lateral_features])[0]

        for idx, col in enumerate(LATERAL_COLUMNS):
            results[f"{col}_Ground_Truth"].append(
                data["lateral_target"][idx]
            )
            results[f"{col}_Prediction"].append(
                lateral_preds[idx]
            )

    return pd.DataFrame(results)

# multitask model 
def evaluate_multitask_model(
    test_dataset,
    df_side,
    multitask_model,
    label
):
    from multitask_features import extract_multitask_features, ALL_COLUMNS

    results = {"Sample_ID": []}
    for col in ALL_COLUMNS:
        results[f"{col}_Ground_Truth"] = []
        results[f"{col}_Prediction"] = []

    for i in range(len(test_dataset)):
        data = test_dataset[i]
        img = data["image"].unsqueeze(0).cpu()

        features = extract_multitask_features(img, label)
        preds = multitask_model.predict([features])[0]

        results["Sample_ID"].append(i)

        gt = np.concatenate([
            data["medial_target"].numpy(), 
            data["lateral_target"].numpy()
            #data["medial_target"],
            #data["lateral_target"]


        ])

        for j, col in enumerate(ALL_COLUMNS):
            results[f"{col}_Ground_Truth"].append(gt[j])
            results[f"{col}_Prediction"].append(preds[j])

    return pd.DataFrame(results)


# --------------------------------------------------------------------------
# Main evaluation runner
# --------------------------------------------------------------------------
def run_evaluation(
    model_dict: Dict[str, torch.nn.Module],
    dataset_csv: str,
    mask_root: str,
    out_dir: str
    
):
    os.makedirs(out_dir, exist_ok=True)
    summary_rows = []

    df = pd.read_csv(dataset_csv)
    transforms = get_transforms()

    rf_models = {
    "L_medial": joblib.load("Models/rf/L_medial.pkl"),
    "L_lateral": joblib.load("Models/rf/L_lateral.pkl"),
    "R_medial": joblib.load("Models/rf/R_medial.pkl"),
    "R_lateral": joblib.load("Models/rf/R_lateral.pkl"),
}

    # --------------------
    # HistGradientBoosting models
    # --------------------
    hgb_models = {
        "L_medial": joblib.load("Models/rf/L_medial_hgb.pkl"),
        "L_lateral": joblib.load("Models/rf/L_lateral_hgb.pkl"),
        "R_medial": joblib.load("Models/rf/R_medial_hgb.pkl"),
        "R_lateral": joblib.load("Models/rf/R_lateral_hgb.pkl"),
}

    multitask_model = joblib.load("Models/multitask/rf_multitask.pkl") 

    for side in KNEE_SIDES:
        side_code = 1 if side == "Right" else 2
        df_side = df[df["SIDE"] == side_code].reset_index(drop=True)

        dataset = MaskDataset(df_side, mask_root, transform=transforms)

        for name, df_res in {
            "Scaling": evaluate_scaling(dataset, side),
            "Deterministic": evaluate_deterministic(dataset, side)
        }.items():
            df_res.to_csv(os.path.join(out_dir, f"{side}_{name}.csv"), index=False)
            summary_rows.append({
                "Knee": side,
                "Approach": name,
                **compute_metrics_from_df(df_res)
            })
       
        # --------------------
        # Random Forest 
        # --------------------
        df_rf = evaluate_random_forest(
            dataset,
            side,
            rf_models[f"{side[0]}_medial"],
            rf_models[f"{side[0]}_lateral"]
        )

        df_rf.to_csv(
            os.path.join(out_dir, f"{side}_RandomForest.csv"),
            index=False
        )

        summary_rows.append({
            "Knee": side,
            "Approach": "RandomForest",
            **compute_metrics_from_df(df_rf)
        })

        # --------------------
        # Multi-task Random Forest
        # --------------------
        df_mt = evaluate_multitask_model(
            dataset,
            df_side,
            multitask_model,
            side
        )

        df_mt.to_csv(
            os.path.join(out_dir, f"{side}_MultiTaskRF.csv"),
            index=False
        )

        summary_rows.append({
            "Knee": side,
            "Approach": "MultiTask-RF",
            **compute_metrics_from_df(df_mt)
        })

        # --------------------
        # HistGradientBoosting
        # --------------------
        df_hgb = evaluate_random_forest(
            dataset,
            side,
            hgb_models[f"{side[0]}_medial"],
            hgb_models[f"{side[0]}_lateral"]
        )

        df_hgb.to_csv(
            os.path.join(out_dir, f"{side}_HistGB.csv"),
            index=False
        )

        summary_rows.append({
            "Knee": side,
            "Approach": "HistGradientBoosting",
            **compute_metrics_from_df(df_hgb)
        })

        



        #for key, model in model_dict.items():
        #    if not key.startswith(side[0]):
        #        continue

        #    df_model = evaluate_model(model, dataset)
        #    df_model.to_csv(os.path.join(out_dir, f"{side}_{key}.csv"), index=False)

        #    summary_rows.append({
        #        "Knee": side,
        #       "Approach": "DL",
        #        **compute_metrics_from_df(df_model)
        #    })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, "summary_metrics.csv"), index=False)
    print(summary_df)


# --------------------------------------------------------------------------
# Example usage
# --------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #model_dict = {
    #    "L_medial": ResNet50Regressor(len(MEDIAL_COLUMNS), pretrained=False).to(device),
    #    "L_lateral": ResNet50Regressor(len(LATERAL_COLUMNS), pretrained=False).to(device),
    #    "R_medial": ResNet50Regressor(len(MEDIAL_COLUMNS), pretrained=False).to(device),
    #    "R_lateral": ResNet50Regressor(len(LATERAL_COLUMNS), pretrained=False).to(device),
    #}

    #model_dict["L_medial"].load_state_dict(torch.load(
    #    "Models/regression/model_L_medial_final.pth")["model_state_dict"])
    #model_dict["L_lateral"].load_state_dict(torch.load(
    #    "Models/regression/model_L_lateral_final.pth")["model_state_dict"])
    #model_dict["R_medial"].load_state_dict(torch.load(
    #    "Models/regression/model_R_medial_final.pth")["model_state_dict"])
    #model_dict["R_lateral"].load_state_dict(torch.load(
    #    "Models/regression/model_R_lateral_final.pth")["model_state_dict"])

    run_evaluation(
        #model_dict=model_dict,
        model_dict={},
        #dataset_csv="Dataset/jsw_dataset_final/test/jsw_measurements.csv",
        dataset_csv="/data_vault/hexai02/Explainable-JSW-Measurement-main/jsw_dataset_final/test/jsw_measurements.csv",
        #mask_root="Dataset/jsw_dataset_final/test/masks",
        mask_root="/data_vault/hexai02/Explainable-JSW-Measurement-main/jsw_dataset_final/test/masks",
        out_dir="Results"
    )