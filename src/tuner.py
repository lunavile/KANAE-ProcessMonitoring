import os
import csv
import optuna
import numpy as np
import pandas as pd
import torch
from src.model_variants import MODEL_VARIANTS
from src.htrainer import train_model
from src.config import DROP_COLUMNS

def evaluate_fdr_score(model, scaler, fdr_data_dir, device):
    
    FAULT_CLUSTERS = {
        'Controllable': [3, 9, 15],
        'BackToControl': [4, 5, 7],
        'Uncontrollable': [i for i in range(1, 22) if i not in [3, 4, 5, 7, 9, 15]],
    }

    model.eval()
    spe_scores = {}

    with torch.no_grad():
        for fault_num in range(22):  # d00 to d21
            df = pd.read_csv(os.path.join(fdr_data_dir, f'd{fault_num:02}.csv'))
            df.drop(columns=DROP_COLUMNS, errors='ignore', inplace=True)
            df = df.values
            X_scaled = scaler.transform(df)
            X = torch.tensor(X_scaled, dtype=torch.float32).to(device)

            out = model(X)
            x_hat = out[0] if isinstance(out, tuple) else out
            spe = torch.sum((X - x_hat) ** 2, dim=1).cpu().numpy()
            spe_scores[fault_num] = spe

    q_lim = np.percentile(spe_scores[0], 95)

    fdr = {}
    for cluster, fault_ids in FAULT_CLUSTERS.items():
        cluster_fdrs = [np.mean(spe_scores[fid] > q_lim) for fid in fault_ids]
        fdr[cluster] = np.mean(cluster_fdrs)
    
    return fdr

def suggest_hyperparameters(trial, search_space):
    params = {}
    for key, value in search_space.items():
        dist_type = str(value[0]).lower()

        if dist_type == "loguniform":
            low = float(value[1])
            high = float(value[2])
            params[key] = trial.suggest_float(key, low, high, log=True)

        elif dist_type == "uniform":
            low = float(value[1])
            high = float(value[2])
            params[key] = trial.suggest_float(key, low, high)

        elif dist_type == "int":
            low = int(value[1])
            high = int(value[2])
            params[key] = trial.suggest_int(key, low, high)

        elif dist_type == "categorical":
            categories = value[1]
            params[key] = trial.suggest_categorical(key, categories)

        else:
            raise ValueError(f"Unsupported distribution type: {value[0]}")

    return params



def objective(trial, model_name, model_config, search_space, x_train_scaled, x_val_scaled, scaler):
    trial_params = suggest_hyperparameters(trial, search_space)

    model_fn = MODEL_VARIANTS[model_name]
    full_model_config = model_config.copy()
    full_model_config.update(trial_params)

    model_wrapper = lambda input_size: model_fn(input_size, **full_model_config)

    model, val_loss = train_model(
        X_train_scaled=x_train_scaled,
        X_val_scaled=x_val_scaled,
        model_fn=model_wrapper,
        learning_rate=trial_params["learning_rate"],
        weight_decay=trial_params["weight_decay"],
        scheduler_factor=trial_params["scheduler_factor"],
        device=None
    )

    # === Compute FDRs using existing scaler ===
    fdr_scores = evaluate_fdr_score(
        model, scaler, fdr_data_dir="data/cases_val",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # === Log to CSV ===
    log_file = "hypertune_log.csv"
    row = {
        **trial_params,
        "val_loss": val_loss,
        "FDR_Controllable": fdr_scores["Controllable"],
        "FDR_BackToControl": fdr_scores["BackToControl"],
        "FDR_Uncontrollable": fdr_scores["Uncontrollable"]
    }

    file_exists = os.path.isfile(log_file)
    with open(log_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return val_loss