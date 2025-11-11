import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import rcParams

# === Configuration ===
models = ["OrthogonalAE", "EfficientKAN", "FastKAN", "FourierKAN", "WavKAN"]
model_pairs = [(a, b) for i, a in enumerate(models) for b in models[i+1:]]
sample_sizes = [500, 1000, 1500, 2500, 4000, 6500, 11000, 18500, 30500, 51500, 86000, 144000, 200000]
results_dir = "Bayesian_Results"
save_dir = os.path.join(results_dir, "profiles")
os.makedirs(save_dir, exist_ok=True)

# === Step 1: Load all results ===
# Dictionary: {(model_a, model_b): {"Pleft": [...], "Prope": [...], "Pright": [...]} }
results_by_pair = {pair: {"Pleft": [], "Prope": [], "Pright": []} for pair in model_pairs}

for size in sample_sizes:
    filepath = os.path.join(results_dir, f"bayesian_results_{size}.csv")
    if not os.path.exists(filepath):
        print(f"⚠ Missing: {filepath}")
        continue

    df = pd.read_csv(filepath)
    for _, row in df.iterrows():
        pair = (row["Model A"], row["Model B"])
        if pair not in results_by_pair:
            continue  # skip if not expected (safety)
        results_by_pair[pair]["Pleft"].append(row["Pleft"])
        results_by_pair[pair]["Prope"].append(row["Prope"])
        results_by_pair[pair]["Pright"].append(row["Pright"])

# Compact styling
rcParams.update({
    "font.size": 8,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 7,
    "ytick.labelsize": 9,
})

# === Step 2: Plot and save profiles ===
for (model_a, model_b), probs in results_by_pair.items():
    if len(probs["Pleft"]) != len(sample_sizes):
        print(f"⚠ Skipping plot for {model_a} vs {model_b}: incomplete data.")
        continue

    fig, ax = plt.subplots(figsize=(7, 4))

    # === Plot posterior probabilities ===
    line1, = ax.plot(sample_sizes, probs["Pleft"], marker='o', label=r"$P(A \gg B)$",linestyle=(0, (3, 3)))
    line2, = ax.plot(sample_sizes, probs["Prope"], marker='o', label=r"$P(A \approx B)$",linestyle=(0, (4, 2, 1, 2)))
    line3, = ax.plot(sample_sizes, probs["Pright"], marker='o', label=r"$P(A \ll B)$",linestyle=(0, (6, 2)))

    # === Shaded training regimes ===
    ax.axvspan(1, 1e4, color="#e6e6e6", alpha=0.5)
    ax.axvspan(1e4, 1e5, color="#d0e1f2", alpha=0.5)
    ax.axvspan(1e5, 2.1e5, color="#d9f2d0", alpha=0.5)

    # === Axes configuration ===
    ax.set_xscale('log')
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(5e2, 2e5)
    ax.set_xticks([5e2, 1e3, 2e3, 4e3, 8e3, 1.6e4, 3.2e4, 6.4e4, 1.28e5, 2e5])
    ax.set_xticklabels(['5.0E2', '1.0E3', '2.0E3', '4.0E3', '8.0E3',
                        '1.6E4', '3.2E4', '6.4E4', '1.28E5', '2.0E5'], rotation=45)
    ax.grid(True, linestyle="--", alpha=0.4)

    # === X-axis label (small font) ===
    ax.set_xlabel("Training Set Size (log scale)", fontsize=7)
    ax.set_ylabel("")

    # === Combined legend: probability + regime (below x-axis) ===
    regime_patches = [
        Patch(facecolor="#e6e6e6", edgecolor='k', alpha=0.5, label="Data-Scarce"),
        Patch(facecolor="#d0e1f2", edgecolor='k', alpha=0.5, label="Data-Sufficient"),
        Patch(facecolor="#d9f2d0", edgecolor='k', alpha=0.5, label="Data-Rich"),
    ]

    all_handles = [line1, line2, line3] + regime_patches

    ax.legend(
        handles=all_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.28),
        ncol=6,
        fontsize=7.5,
        frameon=False,
        handlelength=2
    )

    # === Adjust layout for extra bottom space ===
    fig.tight_layout(rect=[0, 0.1, 1, 1])

    # === Save figure ===
    plot_name = f"profile_{model_a}_vs_{model_b}.png".replace(" ", "_")
    plot_path = os.path.join(save_dir, plot_name)
    fig.savefig(plot_path, dpi=600, bbox_inches='tight')
    plt.close(fig)

    print(f"✓ Saved profile plot: {plot_path}")
