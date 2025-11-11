import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import t


# === Configuration ===
models = ["OrthogonalAE", "EfficientKAN", "FastKAN", "FourierKAN", "WavKAN","PCA"]
label_map = {
    "OrthogonalAE": "OAE",
    "EfficientKAN": "EfficientKAN-AE",
    "FastKAN": "FastKAN-AE",
    "FourierKAN": "FourierKAN-AE",
    "WavKAN": "WavKAN-AE",
    "PCA": "PCA"
}

base_dir = "results"
sample_sizes = [500,1000,1500, 2500,4000,6500,11000,18500,30500,51500,86000,144000,200000]
x_axis_values = [
    500,1000,1500, 2500,4000,6500,11000,18500,30500,51500,86000,144000,200000
]
seeds = [
    0, 1, 3, 7, 8, 10, 11, 13, 17, 21, 23, 27, 32, 42, 43,
    55, 77, 101, 123, 256, 512, 999, 1111, 1337, 1995, 2024,
    2025, 2048, 3141, 9001
]
clusters = {
    "Uncontrollable Faults": ["IDV 1", "IDV 2", "IDV 6", "IDV 8", "IDV 10", "IDV 11",
                              "IDV 12", "IDV 13", "IDV 14", "IDV 16", "IDV 17", "IDV 18",
                              "IDV 19", "IDV 20"],
    "Controllable Faults": ["IDV 3", "IDV 9", "IDV 15"],
    "Back-to-Control Faults": ["IDV 4", "IDV 5", "IDV 7"]
}
metric = "FDR"

# === Color and Style Configuration ===
colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00","#d8b365"]
line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2, 1, 2))]

output_dir = "logscale_profiles_final"
os.makedirs(output_dir, exist_ok=True)

# === Plotting ===
for cluster_name, faults in clusters.items():
    fig, ax = plt.subplots(figsize=(14, 7))
    all_means = []
    cluster_csv_data = []

    # Add regime shading
    ax.axvspan(1, 1e4, color="#e6e6e6", alpha=0.5)
    ax.axvspan(1e4, 1e5, color="#d0e1f2", alpha=0.5)
    ax.axvspan(1e5, 2.1e5, color="#d9f2d0", alpha=0.5)

    for idx, model_name in enumerate(models):
        means = []
        cis = []
        sizes = []

        for size in sample_sizes:
            seed_values = []
            for seed in seeds:
                file_path = os.path.join(
                    base_dir,
                    model_name,
                    f"n_{size}",
                    f"seed_{seed}",
                    f"results_n_{size}_seed_{seed}.csv"
                )
                if not os.path.exists(file_path):
                    continue

                df = pd.read_csv(file_path)
                df_cluster = df[df["Fault"].isin(faults)]
                if df_cluster.empty:
                    continue

                seed_avg = df_cluster[metric].mean()
                seed_values.append(seed_avg)

            #if seed_values:
            #    mean = np.mean(seed_values) * 100
            #    sem = np.std(seed_values, ddof=1) / np.sqrt(len(seed_values))
            #    ci95 = 1.96 * sem * 100
            #    means.append(round(mean, 2))
            #    cis.append(round(ci95, 2))
            #    sizes.append(size)

            if seed_values:
                n = len(seed_values)
                mean = np.mean(seed_values) * 100  # convert to %
                std = np.std(seed_values, ddof=1)
                sem = std / np.sqrt(n)
                tcrit = t.ppf(0.975, df=n - 1)     # 95% two-sided critical value
                ci95 = tcrit * sem * 100           # half-width of CI in %

                means.append(round(mean, 2))
                cis.append(round(ci95, 2))
                sizes.append(size)

        if means:
            sizes_arr = np.array(x_axis_values[:len(sizes)])
            means_arr = np.array(means)
            cis_arr = np.array(cis)
            all_means.extend(means_arr.tolist())

            ax.plot(
                sizes_arr, means_arr,
                label=label_map[model_name],
                color=colors[idx],
                linestyle=line_styles[idx],
                marker='o'
            )
            ax.fill_between(sizes_arr, means_arr - cis_arr, means_arr + cis_arr, alpha=0.2, color=colors[idx])

            df_model = pd.DataFrame({
                "Sample Size": sizes_arr,
                f"Mean {metric} (%)": means_arr,
                f"95% CI (%)": cis_arr
            })
            df_model.insert(0, "Model", label_map[model_name])
            cluster_csv_data.append(df_model)

    # X-axis config
    ax.set_xscale('log')
    ax.set_xlabel("Training Set Size (log scale)", fontsize=12)

    xticks = [5e2, 1e3, 2e3, 4e3, 8e3, 1.6e4, 3.2e4, 6.4e4, 1.28e5, 2e5]
    xtick_labels = ['5.0E2', '1.0E3', '2.0E3', '4.0E3', '8.0E3',
                    '1.6E4', '3.2E4', '6.4E4', '1.28E5', '2.0E5']
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation=45, fontsize=10)
    ax.set_xlim(min(xticks)*0.9, max(xticks)*1.05)

    # Y-axis config
    ax.set_ylabel(f"Average {metric} (%)", fontsize=12)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    #ax.set_title(f"{cluster_name}", fontsize=14)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    if all_means:
        y_min = max(0, np.nanmin(all_means) - 2)
        if cluster_name == "Back-to-Control Faults":
            y_max = 103
            yticks = np.linspace(y_min, 100, num=6)
        else:
            y_max = min(100, np.nanmax(all_means) + 2)
            yticks = np.linspace(y_min, y_max, num=6)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(yticks)

    # Add model legend (top-right)
    legend1 = ax.legend(loc='lower right', fontsize=10)

    # Add regime legend (bottom center, no title)
    regime_patches = [
        Patch(facecolor="#e6e6e6", edgecolor='k', alpha=0.5, label="Data-Scarce"),
        Patch(facecolor="#d0e1f2", edgecolor='k', alpha=0.5, label="Data-Sufficient"),
        Patch(facecolor="#d9f2d0", edgecolor='k', alpha=0.5, label="Data-Rich"),
    ]
    legend2 = ax.legend(handles=regime_patches, loc='lower center',
                        bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=10, frameon=False)
    ax.add_artist(legend1)

    plt.tight_layout()

    # Save figure
    filename = f"{cluster_name.replace(' ', '_')}_{metric}_logscale_profile_final.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=600)
    plt.close()
    print(f"📈 Saved: {filepath}")

    # Save CSV
    if cluster_csv_data:
        df_cluster_all = pd.concat(cluster_csv_data, ignore_index=True)
        csv_path = os.path.join(output_dir, f"{cluster_name.replace(' ', '_')}_{metric}_plot_data.csv")
        df_cluster_all.to_csv(csv_path, index=False)
        print(f"🧾 Saved CSV: {csv_path}")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter

# === Configuration ===
models = ["OrthogonalAE", "EfficientKAN", "FastKAN", "FourierKAN", "WavKAN", "PCA"]
label_map = {
    "OrthogonalAE": "OAE",
    "EfficientKAN": "EfficientKAN-AE",
    "FastKAN": "FastKAN-AE",
    "FourierKAN": "FourierKAN-AE",
    "WavKAN": "WavKAN-AE",
    "PCA": "PCA"
}

base_dir = "results"
sample_sizes = [500, 1000, 1500, 2500, 4000, 6500, 11000, 18500, 30500, 51500, 86000, 144000, 200000]
x_axis_values = sample_sizes
seeds = [
    0, 1, 3, 7, 8, 10, 11, 13, 17, 21, 23, 27, 32, 42, 43,
    55, 77, 101, 123, 256, 512, 999, 1111, 1337, 1995, 2024,
    2025, 2048, 3141, 9001
]
clusters = {
    "Uncontrollable Faults": ["IDV 1", "IDV 2", "IDV 6", "IDV 8", "IDV 10", "IDV 11",
                              "IDV 12", "IDV 13", "IDV 14", "IDV 16", "IDV 17", "IDV 18",
                              "IDV 19", "IDV 20"],
    "Controllable Faults": ["IDV 3", "IDV 9", "IDV 15"],
    "Back-to-Control Faults": ["IDV 4", "IDV 5", "IDV 7"]
}

# ⬇️ switch metric here
metric = "FAR"   # now using False Alarm Rate (FPR in CSV)

# === Color and Style Configuration ===
colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#d8b365"]
line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2, 1, 2))]

output_dir = "logscale_profiles_final"
os.makedirs(output_dir, exist_ok=True)

# === Plotting ===
for cluster_name, faults in clusters.items():
    fig, ax = plt.subplots(figsize=(14, 7))
    all_means = []
    cluster_csv_data = []

    # Add regime shading
    ax.axvspan(1, 1e4, color="#e6e6e6", alpha=0.5)
    ax.axvspan(1e4, 1e5, color="#d0e1f2", alpha=0.5)
    ax.axvspan(1e5, 2.1e5, color="#d9f2d0", alpha=0.5)

    for idx, model_name in enumerate(models):
        means, cis, sizes = [], [], []

        for size in sample_sizes:
            seed_values = []
            for seed in seeds:
                file_path = os.path.join(
                    base_dir,
                    model_name,
                    f"n_{size}",
                    f"seed_{seed}",
                    f"results_n_{size}_seed_{seed}.csv"
                )
                if not os.path.exists(file_path):
                    continue

                df = pd.read_csv(file_path)
                df_cluster = df[df["Fault"].isin(faults)]
                if df_cluster.empty:
                    continue

                # use FPR column as FAR
                if "FPR" not in df_cluster.columns:
                    continue

                seed_avg = df_cluster["FPR"].mean() # convert to %
                seed_values.append(seed_avg)

            #if seed_values:
            #    mean = np.mean(seed_values)
            #    sem = np.std(seed_values, ddof=1) / np.sqrt(len(seed_values))
            #    ci95 = 1.96 * sem
            #    means.append(round(mean, 2))
            #    cis.append(round(ci95, 2))
            #    sizes.append(size)
            if seed_values:
                n = len(seed_values)
                mean = np.mean(seed_values) * 100  # convert to %
                std = np.std(seed_values, ddof=1)
                sem = std / np.sqrt(n)
                tcrit = t.ppf(0.975, df=n - 1)     # 95% two-sided critical value
                ci95 = tcrit * sem * 100           # half-width of CI in %

                means.append(round(mean, 2))
                cis.append(round(ci95, 2))
                sizes.append(size)    

        # plot
        if means:
            sizes_arr = np.array(x_axis_values[:len(sizes)])
            means_arr = np.array(means)
            cis_arr = np.array(cis)
            all_means.extend(means_arr.tolist())

            ax.plot(
                sizes_arr, means_arr,
                label=label_map[model_name],
                color=colors[idx],
                linestyle=line_styles[idx],
                marker='o'
            )
            ax.fill_between(sizes_arr, means_arr - cis_arr, means_arr + cis_arr, alpha=0.2, color=colors[idx])

            df_model = pd.DataFrame({
                "Sample Size": sizes_arr,
                f"Mean {metric} (%)": means_arr,
                f"95% CI (%)": cis_arr
            })
            df_model.insert(0, "Model", label_map[model_name])
            cluster_csv_data.append(df_model)

    # X-axis config
    ax.set_xscale('log')
    ax.set_xlabel("Training Set Size (log scale)", fontsize=12)
    xticks = [5e2, 1e3, 2e3, 4e3, 8e3, 1.6e4, 3.2e4, 6.4e4, 1.28e5, 2e5]
    xtick_labels = ['5.0E2', '1.0E3', '2.0E3', '4.0E3', '8.0E3',
                    '1.6E4', '3.2E4', '6.4E4', '1.28E5', '2.0E5']
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation=45, fontsize=10)
    ax.set_xlim(min(xticks)*0.9, max(xticks)*1.05)

    # Y-axis config
    ax.set_ylabel(f"Average {metric} (%)", fontsize=12)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    if all_means:
        y_min = max(0, np.nanmin(all_means) - 1)
        y_max = min(100, np.nanmax(all_means) + 1)
        yticks = np.linspace(y_min, y_max, num=6)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(yticks)

    # === Add Nominal FAR reference line ===
    nominal_far = 5.0
    ref_color = "#4d4d4d"  # charcoal gray
    ax.axhline(
        nominal_far,
        color=ref_color,
        linestyle="--",
        linewidth=2.2,
        alpha=0.9,
    )

    # Add reference line to legend
    ref_line = plt.Line2D(
        [0], [0],
        color=ref_color,
        linestyle="--",
        linewidth=2.2,
        label="Nominal FAR (5%)"
    )

    # Add model legend (top-right)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(ref_line)
    labels.append("Nominal FAR (5%)")
    legend1 = ax.legend(handles, labels, loc="lower right", fontsize=10)

    # Add regime legend (bottom center)
    regime_patches = [
        Patch(facecolor="#e6e6e6", edgecolor='k', alpha=0.5, label="Data-Scarce"),
        Patch(facecolor="#d0e1f2", edgecolor='k', alpha=0.5, label="Data-Sufficient"),
        Patch(facecolor="#d9f2d0", edgecolor='k', alpha=0.5, label="Data-Rich"),
    ]
    legend2 = ax.legend(handles=regime_patches, loc='lower center',
                        bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=10, frameon=False)
    ax.add_artist(legend1)

    plt.tight_layout()

    # Save figure
    filename = f"{cluster_name.replace(' ', '_')}_{metric}_logscale_profile_final.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=600)
    plt.close()
    print(f"📈 Saved: {filepath}")

    # Save CSV
    if cluster_csv_data:
        df_cluster_all = pd.concat(cluster_csv_data, ignore_index=True)
        csv_path = os.path.join(output_dir, f"{cluster_name.replace(' ', '_')}_{metric}_plot_data.csv")
        df_cluster_all.to_csv(csv_path, index=False)
        print(f"🧾 Saved CSV: {csv_path}")
