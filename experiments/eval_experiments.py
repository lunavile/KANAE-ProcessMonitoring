import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np
from src.config import DROP_COLUMNS, MODEL_SAVE_ROOT, NUM_SIMULATIONS, SEEDS, CASE_DIR, TRAIN_PATH
from src.utils import load_model_config
from src.model_variants import MODEL_VARIANTS
from src.eval.compute_spe import compute_spe
from src.eval.thresholds import compute_kde_threshold
from src.eval.metrics import compute_detection_time, compute_metrics

def main():
    # === Load model config from YAML ===
    config_path = "src/configs/wavkan.yaml"
    model_name, model_config = load_model_config(config_path)
    model_fn = MODEL_VARIANTS[model_name]

    # Get input size from training data
    df = pd.read_csv(TRAIN_PATH)
    df.drop(columns=DROP_COLUMNS, errors='ignore', inplace=True)
    input_size = df.shape[1]

    ## Adjust architecture dynamically
    #if "layers_hidden" in model_config:
    #    model_config["layers_hidden"][0] = input_size
    #    model_config["layers_hidden"][-1] = input_size

    model_builder = lambda: model_fn(input_size, **model_config)

    idv_cases = [f"d{str(i).zfill(2)}" for i in range(0, 22)]
    fault_order = [3, 9, 15, 4, 5, 7, 1, 2, 6, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21]

    fault_descriptions = {
        1: ("Step", "A/C-ratio of stream 4, B composition constant"),
        2: ("Step", "B composition of stream 4, A/C-ratio constant"),
        3: ("Step", "D feed (stream 2) temperature"),
        4: ("Step", "Cooling water inlet temperature of reactor"),
        5: ("Step", "Cooling water inlet temperature of separator"),
        6: ("Step", "A feed loss (stream 1)"),
        7: ("Step", "C header pressure loss (stream 4)"),
        8: ("Random", "A/B/C composition of stream 4"),
        9: ("Random", "D feed (stream 2) temperature"),
        10: ("Random", "C feed (stream 4) temperature"),
        11: ("Random", "Cooling water inlet temperature of reactor"),
        12: ("Random", "Cooling water inlet temperature of separator"),
        13: ("Drift", "Reaction kinetics"),
        14: ("Stiction", "Cooling water outlet valve of reactor"),
        15: ("Stiction", "Cooling water outlet valve of separator"),
        16: ("Random", "(unknown); deviations of heat transfer within stripper"),
        17: ("Random", "(unknown); deviations of heat transfer within reactor"),
        18: ("Random", "(unknown); deviations of heat transfer within condenser"),
        19: ("Stiction", "(unknown); recycle valve, underflow separator/stripper"),
        20: ("Random", "(unknown)"),
        21: ("Random", "(unknown)")
    }

    for tag in NUM_SIMULATIONS:
        for seed in SEEDS:
            print(f"\n🔍 Evaluating tag={tag} | seed={seed}")
            model_dir = os.path.join(MODEL_SAVE_ROOT, model_name, f"n_{tag}", f"seed_{seed}")
            scaler_path = os.path.join(model_dir, "scaler.pkl")
            model_path = os.path.join(model_dir, "best_model.pth")

            for idv in idv_cases:
                test_csv = os.path.join(CASE_DIR, f"{idv}_te.csv")
                spe_csv = os.path.join(model_dir, f"{idv}_spe.csv")

                if not os.path.exists(test_csv):
                    print(f"⚠️ Skipping {test_csv} — not found.")
                    continue

                compute_spe(
                    test_csv_path=test_csv,
                    scaler_path=scaler_path,
                    model_path=model_path,
                    output_path=spe_csv,
                    model_fn=model_builder
                )

            # === Compute KDE threshold from IDV 0
            ref_spe_path = os.path.join(model_dir, "d00_spe.csv")
            if not os.path.exists(ref_spe_path):
                print(f"⚠️ Cannot compute KDE threshold — missing {ref_spe_path}")
                continue

            spe_values = pd.read_csv(ref_spe_path)["SPE"]
            threshold, _ = compute_kde_threshold(spe_values, confidence_level=0.95)
            print(f"📏 Threshold for tag={tag} | seed={seed}: {threshold:.4f}")

            # === Evaluate metrics for each fault
            results = []
            for fault_id in fault_order:
                profile_path = os.path.join(model_dir, f"d{fault_id:02d}_spe.csv")
                if not os.path.exists(profile_path):
                    print(f"⚠️ Missing profile for IDV {fault_id}, skipping.")
                    continue

                df_profile = pd.read_csv(profile_path)
                if "SPE" not in df_profile.columns:
                    print(f"⚠️ 'SPE' column missing in {profile_path}, skipping.")
                    continue

                spe = df_profile["SPE"].values
                detection_time = compute_detection_time(spe, threshold)
                metrics = compute_metrics(spe, threshold)
                fault_type, disturbed_value = fault_descriptions[fault_id]

                results.append({
                    "Fault": f"IDV {fault_id}",
                    "Type": fault_type,
                    "Disturbed Value": disturbed_value,
                    "Detection Time": detection_time,
                    "F1": metrics["F1"],
                    "FDR": metrics["FDR"],
                    "Recall": metrics["Recall"],
                    "Precision": metrics["Precision"],
                    "FPR": metrics["FPR"],
                    "FNR": metrics["FNR"]
                })

            # Save metrics to CSV
            results_df = pd.DataFrame(results)
            output_file = os.path.join(model_dir, f"results_n_{tag}_seed_{seed}.csv")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            results_df.to_csv(output_file, index=False)
            print(f"✅ Saved metrics to {output_file}")

if __name__ == "__main__":
    main()