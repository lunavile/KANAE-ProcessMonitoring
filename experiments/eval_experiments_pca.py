import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import joblib

from src.config import (
    DROP_COLUMNS,
    MODEL_SAVE_ROOT,
    NUM_SIMULATIONS,
    SEEDS,
    CASE_DIR,
    TRAIN_PATH,
)
from src.eval.compute_spe import compute_spe_pca
from src.eval.thresholds import compute_kde_threshold
from src.eval.metrics import compute_detection_time, compute_metrics


def main():
    # === Model and evaluation setup ===
    model_name = "pca"
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
        16: ("Random", "Heat transfer deviation in stripper"),
        17: ("Random", "Heat transfer deviation in reactor"),
        18: ("Random", "Heat transfer deviation in condenser"),
        19: ("Stiction", "Recycle valve underflow (separator/stripper)"),
        20: ("Random", "(unknown)"),
        21: ("Random", "(unknown)"),
    }

    # === Load training data for variable consistency ===
    df_train = pd.read_csv(TRAIN_PATH)
    df_train.drop(columns=DROP_COLUMNS, errors="ignore", inplace=True)
    input_size = df_train.shape[1]

    # === Evaluate each PCA model (per n_sim, per seed) ===
    for tag in NUM_SIMULATIONS:
        for seed in SEEDS:
            print(f"\n🔍 Evaluating PCA model | n_sim={tag} | seed={seed}")

            model_dir = os.path.join(MODEL_SAVE_ROOT, model_name, f"n_{tag}", f"seed_{seed}")
            scaler_path = os.path.join(model_dir, "scaler.pkl")
            pca_path = os.path.join(model_dir, "pca_model.pkl")

            if not os.path.exists(pca_path):
                print(f"⚠️ Missing PCA model: {pca_path}")
                continue

            if not os.path.exists(scaler_path):
                print(f"⚠️ Missing scaler: {scaler_path}")
                continue

            # === Compute SPE for all IDV cases ===
            for idv in idv_cases:
                test_csv = os.path.join(CASE_DIR, f"{idv}_te.csv")
                spe_csv = os.path.join(model_dir, f"{idv}_spe.csv")

                if not os.path.exists(test_csv):
                    print(f"⚠️ Skipping {test_csv} — not found.")
                    continue

                compute_spe_pca(
                    test_csv_path=test_csv,
                    scaler_path=scaler_path,
                    pca_path=pca_path,
                    output_path=spe_csv
                )

            # === KDE threshold (from IDV 0 normal data) ===
            ref_spe_path = os.path.join(model_dir, "d00_spe.csv")
            if not os.path.exists(ref_spe_path):
                print(f"⚠️ Missing reference {ref_spe_path} for thresholding.")
                continue

            spe_values = pd.read_csv(ref_spe_path)["SPE"]
            threshold, _ = compute_kde_threshold(spe_values, confidence_level=0.95)
            print(f"📏 KDE threshold (95%) for n={tag}, seed={seed}: {threshold:.4f}")

            # === Compute metrics per fault ===
            results = []
            for fault_id in fault_order:
                profile_path = os.path.join(model_dir, f"d{fault_id:02d}_spe.csv")
                if not os.path.exists(profile_path):
                    print(f"⚠️ Missing profile for IDV {fault_id}, skipping.")
                    continue

                df_profile = pd.read_csv(profile_path)
                if "SPE" not in df_profile.columns:
                    print(f"⚠️ No 'SPE' column in {profile_path}, skipping.")
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
                    "FNR": metrics["FNR"],
                })

            # === Save evaluation results ===
            results_df = pd.DataFrame(results)
            output_file = os.path.join(model_dir, f"results_n_{tag}_seed_{seed}.csv")
            results_df.to_csv(output_file, index=False)
            print(f"✅ Metrics saved to: {output_file}")


if __name__ == "__main__":
    main()
