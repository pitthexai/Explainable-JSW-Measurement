import argparse
from pathlib import Path
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from src.dataset import SegmentationDataset, get_transforms
from src.model import UNet

# Grad-CAM imports
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

@torch.no_grad()
def predict_mask(model, image_tensor):
    """
    Inference helper to get the class with highest probability.
    """
    logits = model(image_tensor)  # (1,C,H,W)
    pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)[0]  # (H,W)
    return pred.detach().cpu().numpy().astype(np.uint8)

def tensor_to_numpy_image(image_tensor):
    """
    Converts a (3, H, W) float tensor in [0,1] to (H, W, 3) numpy for plotting.
    """
    img = image_tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 1)
    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="Configs/segmentation.yaml")
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="/ihome/atafti/feg48/local_folder/jsw/Figures/segmentation")
    ap.add_argument("--max_n", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Dataset setup
    dataset = SegmentationDataset(
        images_dir=cfg["test_images"],
        masks_dir=cfg["test_masks"],
        device="cpu",
        transform=get_transforms(cfg["height"], cfg["width"], mode="val"),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model setup
    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    num_classes = int(cfg.get("num_classes", 2))

    model = UNet(3, num_classes).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Grad-CAM Layer Selection
    try:
        target_layers = [model.encoder4] 
    except AttributeError:
        target_layers = [list(model.children())[-3]] 

    cam = GradCAM(model=model, target_layers=target_layers)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    vmax = num_classes - 1
    n = 0
    
    for images, true_masks in loader:
        images_gpu = images.to(device)
        
        # Prepare visual input and masks
        input_viz = tensor_to_numpy_image(images[0])
        true_mask_viz = true_masks[0].numpy().astype(np.uint8)

        # Define Grad-CAM Target 
        target_mask = np.ones((images.shape[2], images.shape[3]), dtype=np.float32)
        targets = [SemanticSegmentationTarget(category=1, mask=target_mask)]

        # Generate Grad-CAM heatmap
        grayscale_cam = cam(input_tensor=images_gpu, targets=targets)[0, :]
        cam_viz = show_cam_on_image(input_viz, grayscale_cam, use_rgb=True)

        # Plotting
        fig, axs = plt.subplots(1, 3, figsize=(15, 5)) 

        axs[0].imshow(input_viz)
        axs[0].set_title("Input Image")

        axs[1].imshow(true_mask_viz, cmap="jet", vmin=0, vmax=vmax)
        axs[1].set_title("Ground Truth")

        # Now using index 2 for the third plot
        axs[2].imshow(cam_viz)
        axs[2].set_title("Grad-CAM")

        for ax in axs:
            ax.axis("off")

        fig.tight_layout()
        fig.savefig(outdir / f"explanation_{n:04d}.png", dpi=250, bbox_inches="tight")
        plt.close(fig)

        n += 1
        if args.max_n and n >= args.max_n:
            break

    print(f"Saved {n} explanation figures to: {outdir}")

if __name__ == "__main__":
    main()