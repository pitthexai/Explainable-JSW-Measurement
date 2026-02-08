import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from dataset import MaskDataset, get_transforms
from model import ResNet50Regressor


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


def enable_dropout_only(model: torch.nn.Module) -> None:
    """
    Keep model in eval() but enable Dropout layers (MC Dropout).
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


@torch.no_grad()
def mc_forward(model: torch.nn.Module, x: torch.Tensor, T: int) -> np.ndarray:

    preds = []
    for _ in range(T):
        y = model(x).squeeze(0)  
        preds.append(y.detach().cpu().numpy())
    return np.stack(preds, axis=0)


def build_results_row(
    side: str,
    sample_id: int,
    gt_medial: np.ndarray,
    gt_lateral: np.ndarray,
    pred_medial_stats: dict | None,
    pred_lateral_stats: dict | None,
) -> dict:
    row = {
        "Side": side,
        "Sample_ID": sample_id,
    }

    for i, col in enumerate(MEDIAL_COLUMNS):
        row[f"{col}_GT"] = float(gt_medial[i]) if np.isfinite(gt_medial[i]) else np.nan
    for i, col in enumerate(LATERAL_COLUMNS):
        row[f"{col}_GT"] = float(gt_lateral[i]) if np.isfinite(gt_lateral[i]) else np.nan

    def fill(stats: dict, cols: list[str]):
        mu = stats["mu"]
        sd = stats["sd"]
        q05 = stats["q05"]
        q95 = stats["q95"]
        for i, col in enumerate(cols):
            row[f"{col}_MU"] = float(mu[i])
            row[f"{col}_SD"] = float(sd[i])
            row[f"{col}_Q05"] = float(q05[i])
            row[f"{col}_Q95"] = float(q95[i])

    if pred_medial_stats is not None:
        fill(pred_medial_stats, MEDIAL_COLUMNS)
    else:
        for col in MEDIAL_COLUMNS:
            row[f"{col}_MU"] = np.nan
            row[f"{col}_SD"] = np.nan
            row[f"{col}_Q05"] = np.nan
            row[f"{col}_Q95"] = np.nan

    if pred_lateral_stats is not None:
        fill(pred_lateral_stats, LATERAL_COLUMNS)
    else:
        for col in LATERAL_COLUMNS:
            row[f"{col}_MU"] = np.nan
            row[f"{col}_SD"] = np.nan
            row[f"{col}_Q05"] = np.nan
            row[f"{col}_Q95"] = np.nan

    return row


def compute_interval_coverage(df: pd.DataFrame, cols: list[str], interval="q05q95") -> dict:

    inside = []
    widths = []

    for col in cols:
        gt = df[f"{col}_GT"].to_numpy()

        mu = df[f"{col}_MU"].to_numpy()
        sd = df[f"{col}_SD"].to_numpy()
        q05 = df[f"{col}_Q05"].to_numpy()
        q95 = df[f"{col}_Q95"].to_numpy()

        valid = np.isfinite(gt) & np.isfinite(mu)

        if interval == "q05q95":
            lo, hi = q05, q95
        else:
            lo, hi = mu - 1.96 * sd, mu + 1.96 * sd

        v = valid & np.isfinite(lo) & np.isfinite(hi)
        if np.any(v):
            inside.append(((gt[v] >= lo[v]) & (gt[v] <= hi[v])).astype(np.float32))
            widths.append((hi[v] - lo[v]).astype(np.float32))

    if len(inside) == 0:
        return {"coverage": np.nan, "avg_width": np.nan, "n": 0}

    inside = np.concatenate(inside, axis=0)
    widths = np.concatenate(widths, axis=0)
    return {
        "coverage": float(np.mean(inside)),
        "avg_width": float(np.mean(widths)),
        "n": int(len(inside)),
    }


def compute_mae(df: pd.DataFrame, cols: list[str]) -> float:
    errs = []
    for col in cols:
        gt = df[f"{col}_GT"].to_numpy()
        mu = df[f"{col}_MU"].to_numpy()
        v = np.isfinite(gt) & np.isfinite(mu)
        if np.any(v):
            errs.append(np.abs(gt[v] - mu[v]))
    if len(errs) == 0:
        return np.nan
    return float(np.mean(np.concatenate(errs)))


def plot_uncertainty_hist(df: pd.DataFrame, cols: list[str], title: str, out_png: str) -> None:
    vals = []
    for col in cols:
        sd = df[f"{col}_SD"].to_numpy()
        sd = sd[np.isfinite(sd)]
        if sd.size > 0:
            vals.append(sd)

    if len(vals) == 0:
        return

    vals = np.concatenate(vals, axis=0)

    bar_color = "#8348f1"   
    edge_color = "black"

    bins = 25

    plt.figure()

    counts, bin_edges, _ = plt.hist(
        vals,
        bins=bins,
        color=bar_color,
        edgecolor=edge_color,
        linewidth=1.0,
        alpha=0.75
    )

    x = np.linspace(vals.min(), vals.max(), 400)
    bin_width = bin_edges[1] - bin_edges[0]

    y_curve = None
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(vals)
        y_curve = kde(x) * len(vals) * bin_width
    except Exception:
        mu = float(np.mean(vals))
        sd = float(np.std(vals) + 1e-12)
        pdf = (1.0 / (sd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sd) ** 2)
        y_curve = pdf * len(vals) * bin_width

    plt.plot(x, y_curve, color="blue", linewidth=2.0)

    plt.title(title)
    plt.xlabel("MC Dropout Std (SD)")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_error_vs_uncertainty(df: pd.DataFrame, cols: list[str], title: str, out_png: str) -> None:
    xs, ys = [], []
    for col in cols:
        gt = df[f"{col}_GT"].to_numpy()
        mu = df[f"{col}_MU"].to_numpy()
        sd = df[f"{col}_SD"].to_numpy()
        v = np.isfinite(gt) & np.isfinite(mu) & np.isfinite(sd)
        if np.any(v):
            xs.append(sd[v])
            ys.append(np.abs(gt[v] - mu[v]))
    if len(xs) == 0:
        return
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)

    plt.figure()
    plt.scatter(x, y, s=8)
    plt.title(title)
    plt.xlabel("Uncertainty (SD)")
    plt.ylabel("|Error| = |GT - MeanPred|")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_csv", required=True, help="Path to test/jsw_measurements.csv")
    ap.add_argument("--mask_root", required=True, help="Path to test/masks")
    ap.add_argument("--out_dir", default="Results_Uncertainty", help="Output directory")
    ap.add_argument("--weights_dir", required=True, help="Directory containing 4 regression weight .pth files")
    ap.add_argument("--T", type=int, default=30, help="MC samples per image (e.g., 30 or 50)")
    ap.add_argument("--batch_size", type=int, default=1, help="Keep 1 for exact match with current code")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.dataset_csv)
    transforms = get_transforms()

    device = args.device

    model_dict = {
        "L_medial": ResNet50Regressor(len(MEDIAL_COLUMNS), pretrained=False).to(device),
        "L_lateral": ResNet50Regressor(len(LATERAL_COLUMNS), pretrained=False).to(device),
        "R_medial": ResNet50Regressor(len(MEDIAL_COLUMNS), pretrained=False).to(device),
        "R_lateral": ResNet50Regressor(len(LATERAL_COLUMNS), pretrained=False).to(device),
    }

    def load_one(key: str, fname: str):
        path = os.path.join(args.weights_dir, fname)
        ckpt = torch.load(path, map_location=device)
        model_dict[key].load_state_dict(ckpt["model_state_dict"])
        enable_dropout_only(model_dict[key])

    load_one("L_medial", "model_L_medial_final.pth")
    load_one("L_lateral", "model_L_lateral_final.pth")
    load_one("R_medial", "model_R_medial_final.pth")
    load_one("R_lateral", "model_R_lateral_final.pth")

    all_rows = []
    summary_rows = []

    for side in KNEE_SIDES:
        side_code = 1 if side == "Right" else 2
        df_side = df[df["SIDE"] == side_code].reset_index(drop=True)
        dataset = MaskDataset(df_side, args.mask_root, transform=transforms)

        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        for idx, batch in enumerate(loader):
            img = batch["image"].to(device) 
            gt_medial = batch["medial_target"][0].cpu().numpy().astype(np.float32)
            gt_lateral = batch["lateral_target"][0].cpu().numpy().astype(np.float32)

            if side == "Left":
                m_med = model_dict["L_medial"]
                m_lat = model_dict["L_lateral"]
            else:
                m_med = model_dict["R_medial"]
                m_lat = model_dict["R_lateral"]

            preds_med = mc_forward(m_med, img, T=args.T)  
            preds_lat = mc_forward(m_lat, img, T=args.T)  

            med_stats = {
                "mu": preds_med.mean(axis=0),
                "sd": preds_med.std(axis=0),
                "q05": np.quantile(preds_med, 0.05, axis=0),
                "q95": np.quantile(preds_med, 0.95, axis=0),
            }
            lat_stats = {
                "mu": preds_lat.mean(axis=0),
                "sd": preds_lat.std(axis=0),
                "q05": np.quantile(preds_lat, 0.05, axis=0),
                "q95": np.quantile(preds_lat, 0.95, axis=0),
            }

            row = build_results_row(
                side=side,
                sample_id=idx,
                gt_medial=gt_medial,
                gt_lateral=gt_lateral,
                pred_medial_stats=med_stats,
                pred_lateral_stats=lat_stats,
            )
            all_rows.append(row)

        df_out = pd.DataFrame(all_rows)
        out_csv = os.path.join(args.out_dir, "mc_dropout_predictions_all.csv")
        df_out.to_csv(out_csv, index=False)

    df_out = pd.read_csv(os.path.join(args.out_dir, "mc_dropout_predictions_all.csv"))

    for side in KNEE_SIDES:
        d = df_out[df_out["Side"] == side].reset_index(drop=True)

        mae_med = compute_mae(d, MEDIAL_COLUMNS)
        mae_lat = compute_mae(d, LATERAL_COLUMNS)

        cov90_med = compute_interval_coverage(d, MEDIAL_COLUMNS, interval="q05q95")
        cov90_lat = compute_interval_coverage(d, LATERAL_COLUMNS, interval="q05q95")

        summary_rows.append({
            "Side": side,
            "MAE_Medial": mae_med,
            "MAE_Lateral": mae_lat,
            "Coverage90_Medial": cov90_med["coverage"],
            "AvgWidth90_Medial": cov90_med["avg_width"],
            "N_Medial": cov90_med["n"],
            "Coverage90_Lateral": cov90_lat["coverage"],
            "AvgWidth90_Lateral": cov90_lat["avg_width"],
            "N_Lateral": cov90_lat["n"],
            "T": args.T,
        })

        plot_uncertainty_hist(
            d, MEDIAL_COLUMNS,
            title=f"{side} - Medial MC Dropout Uncertainty (SD) Distribution",
            out_png=os.path.join(args.out_dir, f"{side}_medial_sd_hist.png"),
        )
        plot_uncertainty_hist(
            d, LATERAL_COLUMNS,
            title=f"{side} - Lateral MC Dropout Uncertainty (SD) Distribution",
            out_png=os.path.join(args.out_dir, f"{side}_lateral_sd_hist.png"),
        )
        plot_error_vs_uncertainty(
            d, MEDIAL_COLUMNS,
            title=f"{side} - Medial |Error| vs Uncertainty (SD)",
            out_png=os.path.join(args.out_dir, f"{side}_medial_error_vs_sd.png"),
        )
        plot_error_vs_uncertainty(
            d, LATERAL_COLUMNS,
            title=f"{side} - Lateral |Error| vs Uncertainty (SD)",
            out_png=os.path.join(args.out_dir, f"{side}_lateral_error_vs_sd.png"),
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(args.out_dir, "summary_uncertainty_metrics.csv"), index=False)
    print(summary_df)
    print("\nSaved:")
    print(" - mc_dropout_predictions_all.csv")
    print(" - summary_uncertainty_metrics.csv")
    print(" - *_sd_hist.png")
    print(" - *_error_vs_sd.png")


if __name__ == "__main__":
    main()