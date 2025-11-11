# src/trainer.py
import os
import pickle as pk
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.utils import set_seed, prepare_data_from_train_points
from src.hconfig import  NUM_EPOCHS, PATIENCE, BATCH_SIZE,SEED

import numpy as np  # ← Needed for percentile computation

def train_model(X_train_scaled, X_val_scaled, model_fn, learning_rate, weight_decay, scheduler_factor, device=None):
    set_seed(SEED)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_size = X_train_scaled.shape[1]
    model = model_fn(input_size).to(device)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32)),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32)),
                            batch_size=2 * BATCH_SIZE, shuffle=False)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=scheduler_factor,min_lr=1e-6)
    scaler_amp = torch.cuda.amp.GradScaler()

    best_val_loss = float("inf")
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}") as pbar:
            for batch in pbar:
                inputs = batch[0].to(device)
                optimizer.zero_grad()

                with torch.amp.autocast(device_type='cuda'):
                    out = model(inputs)
                    if isinstance(out, tuple):
                        x_hat, z = out
                        recon_loss = criterion(x_hat, inputs)
                        reg_loss = model.regularization_loss(z)
                        total_loss = recon_loss + reg_loss
                    else:
                        recon_loss = criterion(out, inputs)
                        reg_loss = model.regularization_loss() if hasattr(model, "regularization_loss") else 0.0
                        total_loss = recon_loss + reg_loss

                scaler_amp.scale(total_loss).backward()
                scaler_amp.step(optimizer)
                scaler_amp.update()

                total_train_loss += total_loss.item()
                pbar.set_postfix(loss=total_loss.item(), lr=optimizer.param_groups[0]['lr'])

        # === Validation ===
        model.eval()
        total_val_loss = 0
        spe_vals = []
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0].to(device)
                with torch.amp.autocast(device_type='cuda'):
                    out = model(inputs)
                    val_out = out[0] if isinstance(out, tuple) else out
                    val_loss = criterion(val_out, inputs)
                    total_val_loss += val_loss.item()

                    spe = ((inputs - val_out) ** 2).sum(dim=1).cpu().numpy()
                    spe_vals.append(spe)

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        spe_99 = np.percentile(np.concatenate(spe_vals), 99)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}, Val 99% SPE = {spe_99:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            print(f"🔥 Model improved.")
        else:
            epochs_no_improve += 1
            print(f"🛑 No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= PATIENCE:
            print(f"⏹️ Early stopping triggered after {epoch+1} epochs.")
            break

        scheduler.step(avg_val_loss)

    

    return model, best_val_loss
