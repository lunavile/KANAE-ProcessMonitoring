import sys
import os
import yaml
import optuna
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import prepare_data_from_train_points
from src.loader import load_training_data
#from src.hconfig import TRAIN_PATH, VAL_PATH, DROP_COLUMNS
from src.config import TRAIN_PATH, DROP_COLUMNS
from src.tuner import objective

def main():
    # === Load model config from YAML ===
    config_path = "src/configs/orthogonal_ae.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # === Load h_config.yaml ===
    with open("src/hypertune_configs/orthogonal_ae.yaml", "r") as f:
        hcfg = yaml.safe_load(f)

    model_name = hcfg["model_name"]
    model_config = config
    search_space = hcfg["search_space"]
    n_trials = hcfg["n_trials"]

    # === Load and scale data ===
    #df_train = pd.read_csv(TRAIN_PATH)
    #df_train.drop(columns=DROP_COLUMNS, errors='ignore', inplace=True)
    #df_val = pd.read_csv(VAL_PATH)
    #df_val.drop(columns=DROP_COLUMNS, errors='ignore', inplace=True)

    #scaler = StandardScaler().fit(df_train.values)
    #x_train_scaled = scaler.transform(df_train.values)
    #x_val_scaled = scaler.transform(df_val.values)
    df = load_training_data(path=TRAIN_PATH, drop_columns=DROP_COLUMNS)
    x_train_scaled, x_val_scaled, scaler = prepare_data_from_train_points(df, 500, 43, 500) 
    # === Create Optuna study ===
    study = optuna.create_study(direction="minimize")

    # ✅ Enqueue known good config
    #study.enqueue_trial({
    #    "regularize_activation": 1.62e-4,
    #    "regularize_entropy": 8.31e-4,
    #    "learning_rate": 0.001,
    #    "weight_decay": 0.01,
    #    "scheduler_factor": 0.2,
    #    "scale_noise": 9.22999687766983e-3,
    #    "scale_base": 1.0,
    #    "scale_spline": 0.18061601820123188,
    #    "grid_eps": 0.005178538231231332
    #})

    # 🚀 Run optimization
    study.optimize(lambda trial: objective(
        trial=trial,
        model_name=model_name,
        model_config=model_config,
        search_space=search_space,
        x_train_scaled=x_train_scaled,
        x_val_scaled=x_val_scaled,
        scaler=scaler
    ), n_trials=n_trials)

    # ✅ Output best result
    print("\n✅ Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"{k}: {v}")

    print(f"\n🥇 Best Validation 99th Percentile SPE: {study.best_value:.6f}")

    # 📊 Show all trial outcomes
    print("\n📊 All trial results:")
    for trial in study.trials:
        print(f"Trial {trial.number}: value={trial.value:.6f}, params={trial.params}")

if __name__ == "__main__":
    main()
