import sys
import os
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import NUM_SIMULATIONS, SEEDS, TRAIN_PATH, MODEL_SAVE_ROOT, DROP_COLUMNS
from src.loader import load_training_data
from src.model_variants import MODEL_VARIANTS
from src.trainer import train_model

def main():
    # === Load model config from YAML ===
    config_path = "src/configs/orthogonal_ae.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_name = config.pop("model_name")
    model_fn = MODEL_VARIANTS[model_name]
    model_config = config  # everything else is kwargs

    # Load training data
    df = load_training_data(path=TRAIN_PATH, drop_columns=DROP_COLUMNS)
    input_size = df.shape[1]

    ## Update layer sizes dynamically
    #if "layers_hidden" in model_config:
    #    model_config["layers_hidden"][0] = input_size
    #    model_config["layers_hidden"][-1] = input_size

    for size in NUM_SIMULATIONS:
        for seed in SEEDS:
            print(f"\n🚀 Training {model_name} | Size: {size} | Seed: {seed}")

            save_dir = os.path.join(MODEL_SAVE_ROOT, model_name, f"n_{size}", f"seed_{seed}")
            os.makedirs(save_dir, exist_ok=True)

            # Train
            best_val_loss = train_model(
                df=df,
                model_fn=lambda input_size: model_fn(input_size, **model_config),
                n_simulations=size,
                seed=seed,
                model_save_dir=save_dir,
                fdr_data_dir= 'data\cases'
                
            )

            print(f"✅ Done: {model_name} | Size: {size} | Seed: {seed} | Best Val Loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    main()
