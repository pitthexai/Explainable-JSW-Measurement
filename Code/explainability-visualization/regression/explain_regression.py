import argparse
import yaml
import torch
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.dataset import MaskDataset, get_transforms
from src.model import ResNet50Regressor


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/regression_explain.yaml")
    parser.add_argument("--side", choices=["L", "R"], default=None)
    parser.add_argument("--compartment", choices=["medial", "lateral"], default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # CLI overrides YAML
    side = args.side if args.side else cfg.get("side")
    compartment = args.compartment if args.compartment else cfg.get("compartment")
    if not side or not compartment:
        raise ValueError("Specify --side and --compartment in CLI or YAML.")

    # device setup
    device_str = cfg.get("device", "cuda")
    if device_str.startswith("cuda") and (not torch.cuda.is_available()):
        device_str = "cpu"
    device = torch.device(device_str)

    # setup logic
    side_map = {"R": 1, "L": 2}
    model_path = cfg["models"][side][compartment]
    num_outputs = 7 if compartment == "medial" else 9

    medial_cols = ['V00JSW150','V00JSW175','V00JSW200','V00JSW225','V00JSW250','V00JSW275','V00JSW300']
    lateral_cols = ['V00LJSW700','V00LJSW725','V00LJSW750','V00LJSW775','V00LJSW800','V00LJSW825','V00LJSW850','V00LJSW875','V00LJSW900']
    col_list = medial_cols if compartment == "medial" else lateral_cols

    # load dataset
    df = pd.read_csv(cfg["test_csv"])
    df_filtered = df[df["SIDE"] == side_map[side]].reset_index(drop=True)
    dataset = MaskDataset(df_filtered, cfg["mask_dir"], transform=get_transforms())

    # load model checkpoint (state_dict or dict with model_state_dict)
    model = ResNet50Regressor(num_classes=num_outputs).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # EigenCAM setup (no targets needed)
    target_layers = [model.resnet.layer3[-1]]
    cam = EigenCAM(model=model, target_layers=target_layers)

    # output dir
    outdir = Path(cfg["outdir"]) / f"{side}_{compartment}"
    outdir.mkdir(parents=True, exist_ok=True)

    target_idx = int(cfg.get("explain_target_idx", 0))
    if target_idx < 0 or target_idx >= len(col_list):
        raise ValueError(f"explain_target_idx={target_idx} out of range for {compartment} (len={len(col_list)})")
    current_col = col_list[target_idx]

    max_n = int(cfg.get("max_n", 10))
    print(f"Generating explanations (EigenCAM): side={side}, compartment={compartment}, idx={target_idx} ({current_col}), max_n={max_n}")

    for i in range(min(len(dataset), max_n)):
        data = dataset[i]
        input_tensor = data["image"].unsqueeze(0).to(device)  # (1,3,224,224)

        # ground truth value (may be NaN)
        actual_val = df_filtered.iloc[i][current_col]
        actual_str = "NaN" if pd.isna(actual_val) else f"{float(actual_val):.3f}"

        # prediction
        with torch.no_grad():
            pred_all = model(input_tensor)  # (1,K)
            pred_val = float(pred_all[0, target_idx].item())

        # EigenCAM (no grads needed, no targets)
        # returns (B,H,W)
        cam_out = cam(input_tensor=input_tensor)
        grayscale_cam = cam_out[0]

        row = df_filtered.iloc[i]
        img_filename = f"{int(row['ID'])}_{side}.png"
        img_path = Path(cfg["image_dir"]) / img_filename

        if img_path.exists():
            bg_img = cv2.imread(str(img_path))
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
            bg_img = cv2.resize(bg_img, (224, 224)).astype(np.float32) / 255.0
        else:
            bg_img = data["image"].permute(1, 2, 0).numpy().astype(np.float32)
            bg_img = np.clip(bg_img, 0.0, 1.0)

        cam_viz = show_cam_on_image(bg_img, grayscale_cam, use_rgb=True)

        # plot
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(bg_img)
        axs[0].set_title("Input Image")
        axs[0].axis("off")

        axs[1].imshow(cam_viz)
        # axs[1].set_title(f"{current_col}\n GT: {actual_str} | Pred: {pred_val:.3f}")
        axs[1].set_title(f"{current_col}\n EigenCAM")
        axs[1].axis("off")

        fig.savefig(outdir / f"explain_{i:03d}.png", bbox_inches="tight", dpi=150)
        plt.close(fig)

    print(f"Done. Saved results to: {outdir}")


if __name__ == "__main__":
    main()
