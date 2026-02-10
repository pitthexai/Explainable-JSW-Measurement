import argparse, yaml, torch, pandas as pd
from dataset import MaskDataset, get_transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from model import ResNet50Regressor
from utils import masked_mse_loss

def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, target_key, device, num_epochs):
    train_losses,val_losses=[],[]
    for epoch in range(num_epochs):
        model.train(); t_loss=0
        for data in train_loader:
            x,y = data['image'].to(device), data[target_key].to(device)
            optimizer.zero_grad()
            out = model(x)
            mask = ~torch.isnan(y)
            y[torch.isnan(y)]=0
            loss = criterion(out,y,mask)
            loss.backward(); optimizer.step()
            t_loss+=loss.item()
        train_losses.append(t_loss/len(train_loader))

        model.eval(); v_loss=0
        with torch.no_grad():
            for data in val_loader:
                x,y = data['image'].to(device), data[target_key].to(device)
                out = model(x)
                mask = ~torch.isnan(y)
                y[torch.isnan(y)]=0
                loss = criterion(out,y,mask)
                v_loss+=loss.item()
        val_losses.append(v_loss/len(val_loader))
        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs} - Train: {train_losses[-1]:.4f}, Val: {val_losses[-1]:.4f}")
    return train_losses,val_losses,model

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="regression.yaml"); parser.add_argument("--side",choices=["L","R"],required=True)
    args=parser.parse_args()
    config=yaml.safe_load(open(args.config))

    device=config.get("device","cuda")
    num_epochs=config.get("num_epochs",150)
    batch_size=config.get("batch_size",16)

    # Load CSVs
    df_train=pd.read_csv(config["train_csv"])
    df_val=pd.read_csv(config["val_csv"])
    mask_train,mask_val=config["mask_train_dir"],config["mask_val_dir"]

    transform=get_transforms()

    if args.side=="L":
        df_train=df_train[df_train["SIDE"]==2]; df_val=df_val[df_val["SIDE"]==2]
        target_key_medial="medial_target"; target_key_lateral="lateral_target"
    else:
        df_train=df_train[df_train["SIDE"]==1]; df_val=df_val[df_val["SIDE"]==1]
        target_key_medial="medial_target"; target_key_lateral="lateral_target"

    train_dataset=MaskDataset(df_train,mask_train,transform)
    val_dataset=MaskDataset(df_val,mask_val,transform)
    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    val_loader=DataLoader(val_dataset,batch_size=batch_size,shuffle=False)

    model_medial=ResNet50Regressor(num_classes=7).to(device)
    model_lateral=ResNet50Regressor(num_classes=9).to(device)
    optimizer_medial=optim.Adam(model_medial.parameters(),lr=1e-4)
    optimizer_lateral=optim.Adam(model_lateral.parameters(),lr=1e-4)
    scheduler_medial=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_medial,T_0=5,T_mult=2)
    scheduler_lateral=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_lateral,T_0=5,T_mult=2)

    print("Training medial model...")
    train_model(model_medial,train_loader,val_loader,optimizer_medial,scheduler_medial,masked_mse_loss,target_key_medial,device,num_epochs)
    print("Training lateral model...")
    train_model(model_lateral,train_loader,val_loader,optimizer_lateral,scheduler_lateral,masked_mse_loss,target_key_lateral,device,num_epochs)

if __name__=="__main__":
    main()