import torch
import cv2
import numpy as np
from model import UNet
import yaml
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

OUTDIR = Path("Figures/segmentation/inference")  # <- folder to save annotated images
OUTDIR.mkdir(parents=True, exist_ok=True)

def load_config(path="segmentation.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_inference(model, image_path, device):
    model.eval()
    model.to(device)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image_tensor = torch.tensor(np.transpose(image_rgb, (2, 0, 1))).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        pred_mask = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0).cpu().numpy()
    return image, pred_mask

def visualize_prediction(image_path, image, pred_mask):
    plt.figure(figsize=(8,8))
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(pred_mask, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")
    filename = Path(image_path).stem + "_pred_mask.png"
    out_path = OUTDIR / filename
    plt.savefig(out_path)
    plt.close()
    print(f"✅ Saved prediction visualization to: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="Configs/segmentation.yaml", help="Path to config YAML")
    parser.add_argument("--image", type=str, required=True, help="Path to input image for inference")
    parser.add_argument("--model_path", required=True, type=str, help="Path to trained model weights (overrides config)")
    args = parser.parse_args()

    config = load_config(args.config)

    device = config.get("device", "cuda")
    num_classes = config.get("num_classes", 2)
    weights_path = args.model_path

    model = UNet(3, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    image, pred_mask = run_inference(model, args.image, device)
    visualize_prediction(args.image, image, pred_mask)

if __name__ == "__main__":
    main()