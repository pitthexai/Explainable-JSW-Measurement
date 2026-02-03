import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import SegmentationDataset, get_transforms
from model import UNet
import yaml
from pathlib import Path

def load_config(path="segmentation.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def train_model(config):
    device = config["device"]
    train_dataset = SegmentationDataset(
        config["train_images"], config["train_masks"], device,
        transform=get_transforms(config["height"], config["width"], mode="train")
    )
    val_dataset = SegmentationDataset(
        config["val_images"], config["val_masks"], device,
        transform=get_transforms(config["height"], config["width"], mode="val")
    )

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    model = UNet(3, config["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)

    Path(config["weights_dir"]).mkdir(exist_ok=True, parents=True)

    for epoch in tqdm(range(config["epochs"])):
        model.train()
        total_loss = 0
        for images, masks in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{config['epochs']}, Train Loss: {avg_loss:.5f}")
        torch.save(model.state_dict(), f"{config['weights_dir']}/unet_epoch_{epoch+1}.pth")


if __name__ == "__main__":
    config = load_config("Configs/segmentation.yaml")
    train_model(config)