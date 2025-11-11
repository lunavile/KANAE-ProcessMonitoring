import os
import pandas as pd
from itertools import combinations
from baycomp import two_on_multiple

# === Configuration ===
models = ["OrthogonalAE", "EfficientKAN", "FastKAN", "FourierKAN", "WavKAN"]
sample_sizes = [500, 1000, 1500, 2500, 4000, 6500, 11000, 18500, 30500, 51500, 86000, 144000, 200000]
base_dir = "results"
seeds = [0, 1, 3, 7, 8, 10, 11, 13, 17, 21, 23, 27, 32, 42, 43,
         55, 77, 101, 123, 256, 512, 999, 1111, 1337, 1995, 2024, 2025, 2048, 3141, 9001]
excluded_faults = {"IDV 1","IDV 2","IDV 3","IDV 4","IDV 6","IDV 7","IDV 8","IDV 9","IDV 12","IDV 13","IDV 14","IDV 15"}
faults = [f"IDV {i}" for i in range(1, 22) if f"IDV {i}" not in excluded_faults]
rope = 0.01  # ±1% FDR treated as practically equivalent

# === Ensure output directory exists ===
output_dir = "Bayesian_Results"
os.makedirs(output_dir, exist_ok=True)

# === Loop over all sample sizes ===
for sample_size in sample_sizes:
    print(f"\n=== Processing sample size: {sample_size} ===")

    # === Step 1: Load all FDR values ===
    records = []
    for seed in seeds:
        for fault in faults:
            entry = {"Fault": fault, "Seed": seed}
            found_all = True
            for model in models:
                file_path = os.path.join(base_dir, model, f"n_{sample_size}", f"seed_{seed}",
                                         f"results_n_{sample_size}_seed_{seed}.csv")
                if not os.path.exists(file_path):
                    found_all = False
                    break
                df = pd.read_csv(file_path)
                fdr_row = df[df["Fault"] == fault]
                if fdr_row.empty:
                    found_all = False
                    break
                entry[model] = fdr_row["FDR"].values[0]
            if found_all:
                records.append(entry)

    if not records:
        print(f"⚠ No complete data found for sample size {sample_size}. Skipping.")
        continue

    df_fdr = pd.DataFrame(records)
    df_fdr.set_index(["Fault", "Seed"], inplace=True)

    # === Step 2: Average across seeds per fault ===
    mean_df = df_fdr.groupby("Fault").mean(numeric_only=True)

    # === Step 3: Run Bayesian signed-rank test for each model pair ===
    model_pairs = list(combinations(models, 2))
    bayesian_results = []

    for model_a, model_b in model_pairs:
        x = mean_df[model_a].values  # [n_faults]
        y = mean_df[model_b].values  # [n_faults]

        (p_left, p_rope, p_right), fig = two_on_multiple(
            x, y, rope=rope, runs=1, names=(model_a, model_b), plot=True
        )

        print(f"\n--- Bayesian Signed-Rank Test: {model_a} vs {model_b} ---")
        print(f"P({model_a} > {model_b}) = {p_left:.3f}")
        print(f"P(Equivalent)            = {p_rope:.3f}")
        print(f"P({model_b} > {model_a}) = {p_right:.3f}")

        bayesian_results.append({
            "Model A": model_a,
            "Model B": model_b,
            "Pleft": round(p_left, 4),
            "Prope": round(p_rope, 4),
            "Pright": round(p_right, 4)
        })

    # === Save results to CSV ===
    output_path = os.path.join(output_dir, f"bayesian_results_{sample_size}.csv")
    results_df = pd.DataFrame(bayesian_results)
    results_df.to_csv(output_path, index=False)
    print(f"✓ Results saved to {output_path}")
