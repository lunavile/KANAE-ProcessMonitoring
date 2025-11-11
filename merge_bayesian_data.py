import os
import pandas as pd

# === Configuration ===
sample_sizes = [500, 1000, 1500, 2500, 4000, 6500, 11000, 18500,
                30500, 51500, 86000, 144000, 200000]
results_dir = "Bayesian_Results"
output_path = os.path.join(results_dir, "bayesian_all_merged.csv")

# === Collect and merge all result files ===
merged = []

for size in sample_sizes:
    file_path = os.path.join(results_dir, f"bayesian_results_{size}.csv")
    if not os.path.exists(file_path):
        print(f"⚠ Skipping missing file: {file_path}")
        continue

    df = pd.read_csv(file_path)
    df["SampleSize"] = size
    merged.append(df)

# === Concatenate and save ===
if merged:
    df_all = pd.concat(merged, ignore_index=True)
    df_all.to_csv(output_path, index=False)
    print(f"✓ Merged results saved to {output_path}")
else:
    print("✖ No files found to merge.")
