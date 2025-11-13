import os
import pandas as pd
import numpy as np
from scipy.stats import t

# === Configuration ===
models = ["PCA","OrthogonalAE", "EfficientKAN", "FastKAN", "FourierKAN", "WavKAN"]
sample_sizes = [500,1000,1500, 2500,4000,6500,11000,18500,30500,51500,86000,144000,200000]  # Extend this list as needed
seeds = [0, 1, 3, 7, 8, 10, 11, 13, 17, 21, 23, 27, 32, 42, 43,
    55, 77, 101, 123, 256, 512, 999, 1111, 1337, 1995, 2024, 2025, 2048, 3141, 9001]
base_dir = "results"
output_dir = "tables"
os.makedirs(output_dir, exist_ok=True)

# === Fault order from the original results structure ===
ordered_faults = [
    "IDV 3", "IDV 9", "IDV 15", "IDV 4", "IDV 5", "IDV 7", "IDV 1", "IDV 2", "IDV 6",
    "IDV 8", "IDV 10", "IDV 11", "IDV 12", "IDV 13", "IDV 14", "IDV 16", "IDV 17",
    "IDV 18", "IDV 19", "IDV 20", "IDV 21"
]

# === Function to compute mean ± 95% CI ===
#def compute_ci(values):
#    n = len(values)
#    print(n)
#    mean = np.mean(values)*100
#    sem = np.std(values, ddof=1)*100 / np.sqrt(n)
#    ci = 1.96 * sem
#    return f"{mean:.2f} ± {ci:.2f}"

def compute_ci(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    
    if n < 2:
        # Not enough samples for CI
        mean = np.mean(values) * 100
        return f"{mean:.2f} ± 0.00"

    mean = np.mean(values) * 100                  # convert to %
    std = np.std(values, ddof=1)
    sem = std / np.sqrt(n)                        # standard error (fractional)
    tcrit = t.ppf(0.975, df=n - 1)                # 95% CI critical value
    ci = tcrit * sem * 100                        # convert CI to %

    return f"{mean:.2f} ± {ci:.2f}"

# === Main processing loop ===
for size in sample_sizes:
    fdr_table = pd.DataFrame()
    far_table = pd.DataFrame()
    faults_seen = set()

    for model in models:
        model_fdr = {}
        model_far = {}

        for seed in seeds:
            file_path = os.path.join(base_dir, model, f"n_{size}", f"seed_{seed}", f"results_n_{size}_seed_{seed}.csv")
            if not os.path.exists(file_path):
                continue

            df = pd.read_csv(file_path)

            for _, row in df.iterrows():
                fault = row["Fault"]
                faults_seen.add(fault)
                model_fdr.setdefault(fault, []).append(row["FDR"])
                model_far.setdefault(fault, []).append(row["FPR"])

        # Compute mean ± 95% CI per fault
        fdr_results = {fault: compute_ci(model_fdr[fault]) for fault in model_fdr}
        far_results = {fault: compute_ci(model_far[fault]) for fault in model_far}

        # Convert to Series
        fdr_series = pd.Series(fdr_results, name=model)
        far_series = pd.Series(far_results, name=model)

        # Append to tables
        fdr_table = pd.concat([fdr_table, fdr_series], axis=1)
        far_table = pd.concat([far_table, far_series], axis=1)

    # Use predefined fault order
    fdr_table = fdr_table.reindex(ordered_faults)
    far_table = far_table.reindex(ordered_faults)

    # Save the tables
    fdr_table.to_csv(os.path.join(output_dir, f"fdr_table_n_{size}.csv"))
    far_table.to_csv(os.path.join(output_dir, f"far_table_n_{size}.csv"))
