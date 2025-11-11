import os
import pickle as pk
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import joblib
from sklearn.decomposition import PCA

from src.utils import set_seed, prepare_data_from_train_points
from src.config import LEARNING_RATE, NUM_EPOCHS, PATIENCE, BATCH_SIZE, WEIGHT_DECAY, SCHEDULER_FACTOR, DROP_COLUMNS

# === Fault Cluster Definitions ===
FAULT_CLUSTERS = {
    'Controllable': [3, 9, 15],
    'BackToControl': [4, 5, 7],
    'Uncontrollable': [i for i in range(1, 22) if i not in [3, 4, 5, 7, 9, 15]],
}

# === FDR Evaluation Frequency (in epochs) ===
N_EPOCHS_FDR_EVAL = 10

def evaluate_fdr_by_cluster(model, epoch, data_dir, scaler_path, device):
    model.eval()
    spe_scores = {}

    # === Load saved scaler ===
    with open(scaler_path, 'rb') as f:
        scaler = pk.load(f)

    with torch.no_grad():
        for fault_num in range(22):  # d00 to d21
            file_path = os.path.join(data_dir, f"d{fault_num:02}_te.csv")
            df = pd.read_csv(file_path)
            df.drop(columns=DROP_COLUMNS, errors='ignore', inplace=True)
            df_values = df.values

            # Apply scaling
            X_scaled = scaler.transform(df_values)
            X = torch.tensor(X_scaled, dtype=torch.float32).to(device)

            # Forward pass
            out = model(X)
            x_hat = out[0] if isinstance(out, tuple) else out
            spe = torch.sum((X - x_hat) ** 2, dim=1).cpu().numpy()

            # Store SPE
            spe_scores[fault_num] = spe

    # === Threshold from normal data (full d00_te) ===
    q_limit = np.percentile(spe_scores[0], 95)

    # === Compute FDR per cluster ===
    cluster_fdr = {cluster: [] for cluster in FAULT_CLUSTERS}
    for cluster_name, fault_ids in FAULT_CLUSTERS.items():
        fdrs = []
        for fid in fault_ids:
            fault_spe = spe_scores[fid][160:]  # only fault portion
            fdr = np.mean(fault_spe > q_limit)
            fdrs.append(fdr)
        cluster_fdr[cluster_name] = 100 * np.mean(fdrs)

    return epoch, cluster_fdr

def train_model(df, model_fn, n_simulations, seed, model_save_dir, fdr_data_dir, device=None):
    set_seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_size = df.shape[1]-1
    print(input_size)
    model = model_fn(input_size).to(device)

    # === Data Prep ===
    X_train_scaled, X_val_scaled, scaler = prepare_data_from_train_points(df, n_simulations, seed, sim_len=500)
    os.makedirs(model_save_dir, exist_ok=True)
    with open(os.path.join(model_save_dir, "scaler.pkl"), 'wb') as f:
        pk.dump(scaler, f)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32)),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32)),
                            batch_size=BATCH_SIZE, shuffle=False)

    # === Setup ===
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=SCHEDULER_FACTOR,min_lr=1e-6)
    
    
    scaler_amp = torch.cuda.amp.GradScaler()

    best_val_loss = float("inf")
    epochs_no_improve = 0
    train_losses, val_losses = [], []
    fdr_log = []
    start_time = time.time()

    best_model_path = os.path.join(model_save_dir, "best_model.pth")
    final_model_path = os.path.join(model_save_dir, "final_model.pth")
    optimizer_path = os.path.join(model_save_dir, "optimizer.pth")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (n={n_simulations}, seed={seed})") as pbar:
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
                        if hasattr(model, "regularization_loss"):
                            reg_loss = model.regularization_loss()
                            total_loss = recon_loss + reg_loss
                        else:
                            total_loss = recon_loss

                scaler_amp.scale(total_loss).backward()
                scaler_amp.step(optimizer)
                scaler_amp.update()

                total_train_loss += total_loss.item()
                pbar.set_postfix(loss=total_loss.item(), lr=optimizer.param_groups[0]['lr'])

        # === Validation ===
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0].to(device)
                with torch.amp.autocast(device_type='cuda'):
                    out = model(inputs)
                    val_out = out[0] if isinstance(out, tuple) else out
                    val_loss = criterion(val_out, inputs)
                total_val_loss += val_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"\U0001F525 Model improved. Saved to {best_model_path}")
        else:
            epochs_no_improve += 1
            print(f"\U0001F6D1 No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= PATIENCE:
            print(f"\u23F9️ Early stopping triggered after {epoch+1} epochs.")
            break

        scheduler.step(avg_val_loss)

        # === Optional FDR Evaluation Every N Epochs ===
        if (epoch + 1) % N_EPOCHS_FDR_EVAL == 0:
            scaler_path = os.path.join(model_save_dir, "scaler.pkl")
            eval_epoch, cluster_fdrs = evaluate_fdr_by_cluster(model, epoch + 1, fdr_data_dir, scaler_path, device)
            fdr_log.append((eval_epoch, cluster_fdrs))

            print(f"[Epoch {eval_epoch}] FDR per cluster: " +
                  ", ".join(f"{k}: {v:.3f}" for k, v in cluster_fdrs.items()))

    # Final save
    torch.save(model.state_dict(), final_model_path)
    torch.save(optimizer.state_dict(), optimizer_path)

    total_time = time.time() - start_time
    print(f"\u23F1️ Total Training Time: {total_time:.2f} seconds")
    print(f"\u2705 Model and optimizer saved in '{model_save_dir}'")

    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.savefig(os.path.join(model_save_dir, "autoencoder_loss_curve.png"))
    plt.close()

    # Plot FDR profile
    if fdr_log:
        epochs = [entry[0] for entry in fdr_log]
        for cluster_name in FAULT_CLUSTERS:
            values = [entry[1][cluster_name] for entry in fdr_log]
            plt.plot(epochs, values, label=cluster_name)
        plt.xlabel("Epoch")
        plt.ylabel("Mean FDR")
        plt.title("FDR per Cluster Over Epochs")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(model_save_dir, "fdr_profiles.png"))
        plt.close()

    return best_val_loss




def train_pca_model(df, n_components,total_samples , seed, model_save_dir):
    """
    Train and evaluate a PCA model using the same data-prep logic as AEs.

    Args:
        df: pandas.DataFrame – typically the fault-free dataset
        n_components: int – number of principal components
        n_simulations: int – number of training simulations
        seed: int – random seed
        model_save_dir: str – directory where model and plots are saved
    """
    set_seed(seed)
    os.makedirs(model_save_dir, exist_ok=True)

    # === Data Preparation ===
    X_train_scaled, X_val_scaled, scaler = prepare_data_from_train_points(
        df, total_samples, seed, sim_len=500
    )
    scaler_path = os.path.join(model_save_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        joblib.dump(scaler, f)
    print(f"✅ Scaler saved to {scaler_path}")

    # === PCA Fit ===
    start_time = time.time()
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(X_train_scaled)
    total_time = time.time() - start_time

    # === Save PCA model ===
    pca_path = os.path.join(model_save_dir, "pca_model.pkl")
    joblib.dump(pca, pca_path)
    print(f"✅ PCA model saved to {pca_path}")
    print(f"🕒 Training completed in {total_time:.2f} seconds")

    # === Validation Reconstruction Error ===
    X_val_recon = pca.inverse_transform(pca.transform(X_val_scaled))
    val_loss = np.mean((X_val_scaled - X_val_recon) ** 2)
    print(f"📉 Validation Reconstruction Loss: {val_loss:.6f}")
    print(f"Explained Variance Ratio (sum): {pca.explained_variance_ratio_.sum():.4f}")

    # === Explained Variance Plot ===
    plt.figure(figsize=(8, 4))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker="o", color="tab:blue")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance Ratio")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_dir, "explained_variance.png"))
    plt.close()

    # === Reconstruction Error Distribution (Training) ===
    X_train_recon = pca.inverse_transform(pca.transform(X_train_scaled))
    train_errors = np.mean((X_train_scaled - X_train_recon) ** 2, axis=1)
    plt.hist(train_errors, bins=50, color="gray", alpha=0.7)
    plt.title("Training Reconstruction Error Distribution")
    plt.xlabel("MSE")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_dir, "reconstruction_error_hist.png"))
    plt.close()

    return val_loss
