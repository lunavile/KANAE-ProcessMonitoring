import sys
import os
import yaml
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import NUM_SIMULATIONS, SEEDS, TRAIN_PATH, MODEL_SAVE_ROOT, DROP_COLUMNS
from src.loader import load_training_data
from src.trainer import train_pca_model  

def main():
    # === Load PCA config from YAML ===
    config_path = "src/configs/pca.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_name = config.pop("model_name")  # e.g. "pca"
    n_components = config.get("n_components", None)  # number of PCA components

    # Load training data
    df = load_training_data(path=TRAIN_PATH, drop_columns=DROP_COLUMNS)

    for size in NUM_SIMULATIONS:
        for seed in SEEDS:
            print(f"\n🚀 Training PCA | Size: {size} | Seed: {seed}")

            save_dir = os.path.join(MODEL_SAVE_ROOT, model_name, f"n_{size}", f"seed_{seed}")
            os.makedirs(save_dir, exist_ok=True)

            # Train PCA
            best_val_loss = train_pca_model(
                df=df,
                n_components=n_components,
                total_samples=size,
                seed=seed,
                model_save_dir=save_dir
            )

            print(f"✅ Done: PCA | Size: {size} | Seed: {seed} | Val Loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    main()
