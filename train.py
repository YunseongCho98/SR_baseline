import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import SR
from dataset import SRDataset


# Setup parameters
batch_size = 8
epochs = 100
lr = 1e-4
scale = 4
num_blocks = 16  # Number of residual blocks in the model
img_size = 512  # HR size (LR will be 1/4x this size)

device = torch.device("mps" if torch.backends.mps.is_available()
                        else "cuda" if torch.cuda.is_available()
                        else "cpu")

print(f"Using device: {device}")

# Dataset
dataset = SRDataset("data/LR", "data/HR", img_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Model
model = SR(num_blocks=num_blocks, scale=scale).to(device)

# Loss function & optimizer
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0

    for lr_imgs, hr_imgs in tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", leave=True):
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)

        sr_imgs = model(lr_imgs)
        loss = criterion(sr_imgs, hr_imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"[Epoch {epoch}] Loss: {epoch_loss / len(dataloader):.6f}")

    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"sr_epoch{epoch}.pth")
