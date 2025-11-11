import os
import pickle as pk
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from src.config import DROP_COLUMNS

def compute_spe(
    test_csv_path,
    scaler_path,
    model_path,
    output_path,
    model_fn,           # ✅ generic model constructor
    batch_size=1200
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # === Load and preprocess test data ===
    df = pd.read_csv(test_csv_path)
    df.drop(columns=DROP_COLUMNS, errors='ignore', inplace=True)

    with open(scaler_path, "rb") as f:
        scaler = pk.load(f)

    data_scaled = scaler.transform(df.values)
    dataset = TensorDataset(torch.tensor(data_scaled, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # === Build and load model ===
    model = model_fn()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # === Compute SPE and SPE per variable ===
    spe_total = []
    spe_per_variable = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)
            output = model(x)
            x_hat = output[0] if isinstance(output, tuple) else output

            squared_error = (x - x_hat) ** 2
            spe_batch = torch.sum(squared_error, dim=1).cpu().numpy()
            spe_total.extend(spe_batch)

            spe_per_variable_batch = squared_error.cpu().numpy()
            spe_per_variable.append(spe_per_variable_batch)

    # === Save SPE and per-variable SPE ===
    spe_per_variable_all = np.vstack(spe_per_variable)
    columns = [f"SPE_x{i}" for i in range(spe_per_variable_all.shape[1])]

    df_out = pd.DataFrame(spe_per_variable_all, columns=columns)
    df_out.insert(0, "SPE", spe_total)

    df_out.to_csv(output_path, index=False)
    print(f"✅ Saved SPE (total and per-variable) to: {output_path}")

import os
import pandas as pd
import numpy as np
import joblib
from src.config import DROP_COLUMNS


def compute_spe_pca(
    test_csv_path,
    scaler_path,
    pca_path,
    output_path
):
    """
    Compute Squared Prediction Error (SPE) and per-variable SPE for PCA models.

    Args:
        test_csv_path (str): path to CSV test dataset
        scaler_path (str): path to saved scaler (joblib/pkl)
        pca_path (str): path to trained PCA model (joblib/pkl)
        output_path (str): where to save computed SPE profiles
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # === Load and preprocess test data ===
    df = pd.read_csv(test_csv_path)
    df.drop(columns=DROP_COLUMNS, errors="ignore", inplace=True)

    with open(scaler_path, "rb") as f:
        scaler = joblib.load(f)
    pca = joblib.load(pca_path)

    X_scaled = scaler.transform(df.values)

    # === Reconstruction ===
    X_recon = pca.inverse_transform(pca.transform(X_scaled))

    # === SPE calculations ===
    squared_error = (X_scaled - X_recon) ** 2
    spe_total = np.sum(squared_error, axis=1)
    spe_per_variable = squared_error

    # === Save SPE (total and per-variable) ===
    columns = [f"SPE_x{i}" for i in range(spe_per_variable.shape[1])]
    df_out = pd.DataFrame(spe_per_variable, columns=columns)
    df_out.insert(0, "SPE", spe_total)

    df_out.to_csv(output_path, index=False)
    print(f"✅ Saved PCA SPE (total and per-variable) to: {output_path}")
