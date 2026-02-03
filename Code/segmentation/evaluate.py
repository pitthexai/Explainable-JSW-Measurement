import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import SegmentationDataset, get_transforms
from model import UNet
import yaml
import argparse

def load_config(path="segmentation.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def dice_score(preds, targets, num_classes, smooth=1e-6):
    preds = torch.argmax(torch.softmax(preds, dim=1), dim=1)
    preds_one_hot = torch.nn.functional.one_hot(preds, num_classes).permute(0,3,1,2).float()
    targets_one_hot = torch.nn.functional.one_hot(targets, num_classes).permute(0,3,1,2).float()
    intersection = (preds_one_hot * targets_one_hot).sum(dim=(2,3))
    union = preds_one_hot.sum(dim=(2,3)) + targets_one_hot.sum(dim=(2,3))
    return ((2 * intersection + smooth)/(union + smooth)).mean()

def iou_score(preds, targets, num_classes, smooth=1e-6):
    preds = torch.argmax(torch.softmax(preds, dim=1), dim=1)
    preds_one_hot = torch.nn.functional.one_hot(preds, num_classes).permute(0,3,1,2).float()
    targets_one_hot = torch.nn.functional.one_hot(targets, num_classes).permute(0,3,1,2).float()
    intersection = (preds_one_hot * targets_one_hot).sum(dim=(2,3))
    union = preds_one_hot.sum(dim=(2,3)) + targets_one_hot.sum(dim=(2,3)) - intersection
    return ((intersection + smooth)/(union + smooth)).mean()

def evaluate(model, test_loader, num_classes, device):
    model.eval()
    model.to(device)
    dice_scores, iou_scores = [], []

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            dice_scores.append(dice_score(outputs, masks, num_classes).item())
            iou_scores.append(iou_score(outputs, masks, num_classes).item())

    print(f"Average Dice Score: {np.mean(dice_scores):.5f}")
    print(f"Average IoU Score: {np.mean(iou_scores):.5f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="Configs/segmentation.yaml", help="Path to YAML config")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model weights (overrides config)")
    args = parser.parse_args()

    config = load_config(args.config)

    device = config.get("device", "cuda")
    num_classes = config.get("num_classes", 2)
    weights_path = args.model_path

    test_dataset = SegmentationDataset(
        config["test_images"], config["test_masks"], device,
        transform=get_transforms(config["height"], config["width"], mode="val")
    )
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    model = UNet(3, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    evaluate(model, test_loader, num_classes, device)

if __name__ == "__main__":
    main()