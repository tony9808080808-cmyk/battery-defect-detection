import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BatteryDataset, load_file_list
from model import UNet


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("사용 디바이스:", device)

    train_files = load_file_list("splits/train.txt")
    val_files = load_file_list("splits/val.txt")

    train_dataset = BatteryDataset(
        image_dir="data/raw/images",
        mask_dir="data/processed/masks",
        file_list=train_files,
        image_size=(128, 128)
    )

    val_dataset = BatteryDataset(
        image_dir="data/raw/images",
        mask_dir="data/processed/masks",
        file_list=val_files,
        image_size=(128, 128)
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    model = UNet(in_channels=3, out_channels=3).to(device)

    class_weights = torch.tensor([0.2, 2.0, 3.0], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10
    best_val_loss = float("inf")

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}", ncols=100):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"[Val] Epoch {epoch+1}", ncols=100):
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = f"checkpoints/best_weighted_model_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"모델 저장됨: {save_path}")


if __name__ == "__main__":
    train()