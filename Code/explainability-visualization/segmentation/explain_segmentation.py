import argparse
import os
import warnings
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Suppress technical noise in the console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=UserWarning)

from src.dataset import SegmentationDataset, get_transforms
from src.model import UNet

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def tensor_to_numpy_image(image_tensor: torch.Tensor) -> np.ndarray:
    """Converts (3,H,W) tensor to (H,W,3) numpy for visualization."""
    img = image_tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 1)
    return img

def pick_target_layers(model: torch.nn.Module):
    """
    Targets the final decoder layer to provide high-resolution explanations.
    """
    for name in ["up4", "decoder4", "dec4", "up_conv4"]:
        if hasattr(model, name):
            return [getattr(model, name)]
    
    conv_layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    return [conv_layers[-2]] if len(conv_layers) > 1 else [conv_layers[-1]]

def ensure_hw_mask(mask_2d: torch.Tensor, Hout: int, Wout: int) -> torch.Tensor:
    """Resizes masks to match model output resolution."""
    if mask_2d.ndim != 2:
        raise ValueError(f"mask_2d must be (H,W), got {tuple(mask_2d.shape)}")
    if mask_2d.shape[0] == Hout and mask_2d.shape[1] == Wout:
        return mask_2d.float()
    m = mask_2d[None, None].float()
    m = F.interpolate(m, size=(Hout, Wout), mode="nearest")[0, 0]
    return m

@torch.no_grad()
def forward_logits_pred(model: torch.nn.Module, x: torch.Tensor):
    logits = model(x)
    pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
    return logits, pred

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="Configs/segmentation.yaml")
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="./Figures/segmentation")
    ap.add_argument("--max_n", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--target_class", type=int, default=1)
    ap.add_argument("--mask_source", type=str, default="pred", choices=["all", "gt", "pred"])
    args = ap.parse_args()

    cfg = load_config(args.config)

    dataset = SegmentationDataset(
        images_dir=cfg["test_images"],
        masks_dir=cfg["test_masks"],
        device="cpu",
        transform=get_transforms(cfg["height"], cfg["width"], mode="val"),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    num_classes = int(cfg.get("num_classes", 2))

    model = UNet(3, num_classes).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    target_layers = pick_target_layers(model)
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    n = 0
    for images, true_masks in loader:
        images_gpu = images.to(device)
        logits, pred = forward_logits_pred(model, images_gpu)
        Hout, Wout = logits.shape[-2], logits.shape[-1]

        input_viz = tensor_to_numpy_image(images[0])
        
        # Ground Truth visualization
        true_mask_viz = true_masks[0].detach().cpu().numpy().astype(np.uint8)
        if true_mask_viz.ndim == 3:
            true_mask_viz = true_mask_viz[0]

        if args.mask_source == "all":
            target_mask_np = np.ones((Hout, Wout), dtype=np.float32)
        elif args.mask_source == "gt":
            gt0 = (true_masks[0] > 0).float().to(device)
            target_mask_np = ensure_hw_mask(gt0, Hout, Wout).cpu().numpy()
        else:
            target_mask_np = (pred[0] == args.target_class).float().cpu().numpy()

        if np.sum(target_mask_np) == 0:
            print(f"[{n}] Class {args.target_class} not predicted; skipping.")
            n += 1
            continue

        targets = [SemanticSegmentationTarget(category=args.target_class, mask=target_mask_np)]

        # aug_smooth=True averages out the noisy artifacts
        grayscale_cam = cam(
            input_tensor=images_gpu, 
            targets=targets,
            aug_smooth=False,
            eigen_smooth=False
        )[0, :]
        
        cam_viz = show_cam_on_image(input_viz, grayscale_cam, use_rgb=True, image_weight=0.7)

        # Final Plotting
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        axs[0].imshow(input_viz)
        axs[0].set_title("Input Image", fontsize=12, pad=8)

        axs[1].imshow(true_mask_viz, cmap="jet", vmin=0, vmax=1)
        axs[1].set_title("Ground Truth", fontsize=12, pad=8)

        axs[2].imshow(cam_viz)
        axs[2].set_title("Grad-CAM++", fontsize=12, pad=8)
        
        for ax in axs: 
            ax.axis("off")

        plt.tight_layout(pad=1.0, w_pad=0.5)

        plt.savefig(
            outdir / f"explanation_{n:04d}.png", 
            dpi=300, 
            bbox_inches='tight', 
            pad_inches=0.05  
        )
        plt.close(fig)

        n += 1
        if args.max_n and n >= args.max_n: break

    print(f"Success. {n} figures saved to {outdir}")

if __name__ == "__main__":
    main()