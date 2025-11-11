import sys
import os
import yaml
import pandas as pd
import numpy as np

# --- Ensure project root is on the path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import TRAIN_PATH, DROP_COLUMNS
from src.loader import load_training_data
from src.model_variants import MODEL_VARIANTS
from src.trainer_benchmark import benchmark_training


def run_all_benchmarks(config_dir="src/configs"):
    yaml_files = [
        "efficient_kan.yaml",
        "fastkan.yaml",
        "fourierkan.yaml",
        "orthogonal_ae.yaml",
        "wavkan.yaml",
    ]

    BASELINE_NAME = "orthogonalae"  # lowercase comparison key

    records = []

    # === Load training data once ===
    df = load_training_data(path=TRAIN_PATH, drop_columns=DROP_COLUMNS)

    # === Run all benchmarks ===
    for cfg_file in yaml_files:
        config_path = os.path.join(config_dir, cfg_file)
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        model_name = config.pop("model_name")
        model_fn = MODEL_VARIANTS[model_name]
        model_config = config

        print(f"\n🚀 Benchmarking {model_name} on CPU & GPU...")
        res = benchmark_training(
            model_fn=lambda input_size: model_fn(input_size, **model_config),
            df=df,
            n_simulations=200000,
            seed=43,
            reps=500,              # small number for stable mean & CI
            warmup_batches=50,
            device_mode="both",
        )

        for dev_key, dev_label in [("cpu", "CPU"), ("gpu", "GPU")]:
            dev_res = res.get(dev_key)
            if not dev_res:
                continue

            fwd_mean, fwd_ci = dev_res["forward_ms"]
            bwd_mean, bwd_ci = dev_res["backward_ms"]

            records.append({
                "Model": model_name,
                "Device": dev_label,
                "Fwd_mean": fwd_mean,
                "Fwd_CI": fwd_ci,
                "Bwd_mean": bwd_mean,
                "Bwd_CI": bwd_ci,
            })

    df = pd.DataFrame(records)
    df["Model_lower"] = df["Model"].str.lower()

    # === Identify fixed baseline (OrthogonalAE) for each device ===
    base_cpu = df[(df["Model_lower"] == BASELINE_NAME) & (df["Device"] == "CPU")]
    base_gpu = df[(df["Model_lower"] == BASELINE_NAME) & (df["Device"] == "GPU")]

    if base_cpu.empty or base_gpu.empty:
        print("\n⚠️ Available models:")
        print(df[["Model", "Device"]])
        raise ValueError("❌ Baseline OrthogonalAE missing for CPU or GPU.")

    base_cpu_fwd, base_cpu_bwd = base_cpu["Fwd_mean"].iloc[0], base_cpu["Bwd_mean"].iloc[0]
    base_gpu_fwd, base_gpu_bwd = base_gpu["Fwd_mean"].iloc[0], base_gpu["Bwd_mean"].iloc[0]

    # === Compute acceleration (speed-up) relative to baseline ===
    df["Fwd_acc"] = np.where(
        df["Device"] == "CPU",
        base_cpu_fwd / df["Fwd_mean"],
        base_gpu_fwd / df["Fwd_mean"],
    )
    df["Bwd_acc"] = np.where(
        df["Device"] == "CPU",
        base_cpu_bwd / df["Bwd_mean"],
        base_gpu_bwd / df["Bwd_mean"],
    )

    # === Format display (mean ± CI + acceleration) ===
    df_display = df.copy()
    df_display["Model"] = df_display["Model"] + "-" + df_display["Device"].str.lower()

    def fmt_mean_ci(mean, ci):
        if pd.isna(mean):
            return "—"
        return f"{mean:.2f} ± {ci:.2f}"

    df_display["Fwd (ms)"] = [
        fmt_mean_ci(m, c) for m, c in zip(df_display["Fwd_mean"], df_display["Fwd_CI"])
    ]
    df_display["Bwd (ms)"] = [
        fmt_mean_ci(m, c) for m, c in zip(df_display["Bwd_mean"], df_display["Bwd_CI"])
    ]

    df_display["Fwd acc."] = df_display["Fwd_acc"].map("{:.2f}".format)
    df_display["Bwd acc."] = df_display["Bwd_acc"].map("{:.2f}".format)

    df_display = df_display[["Model", "Fwd (ms)", "Fwd acc.", "Bwd (ms)", "Bwd acc."]]

    # Sort by model, with CPU above GPU
    df_display["sort_order"] = df_display["Model"].apply(
        lambda x: (x.split("-")[0], 0 if "cpu" in x else 1)
    )
    df_display = df_display.sort_values("sort_order").drop(columns="sort_order")

    # === Output ===
    print("\n📊 Benchmark Summary (speed-up vs OrthogonalAE):")
    print(df_display.to_string(index=False))

    os.makedirs("results", exist_ok=True)
    out_path = "results/benchmark_summary_relative.csv"
    df_display.to_csv(out_path, index=False)
    print(f"\n✅ Saved summary to {out_path}")


if __name__ == "__main__":
    run_all_benchmarks()
