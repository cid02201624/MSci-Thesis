import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

# =========================
# Load data
# =========================
predict_df = pd.read_csv("Model_9_Better_fusion/training_figures9/test_predictions_binary.csv")
meta_df = pd.read_csv("Training_Data_Generation/pt_dataset/test_sample_metadata.csv")

df = predict_df.merge(
    meta_df[["sample_index", "class_type", "rho_net"]],
    on="sample_index",
    how="inner"
)

# Binary target: GW vs non-GW
df["is_gw_true"] = (df["class_type"] == "GW").astype(int)

score_col = "p_gw"

save_dir = "M9"
os.makedirs(save_dir, exist_ok=True)

# =========================
# Split data
# =========================
gw_df = df[df["class_type"] == "GW"].copy()
non_gw_df = df[df["class_type"] != "GW"].copy()   # includes glitch + noise
noise_df = df[df["class_type"] == "noise"].copy()

# =========================
# Build 5 GW rho bins
# =========================
rho_min = gw_df["rho_net"].min()
rho_max = gw_df["rho_net"].max()
bin_edges = np.linspace(rho_min, rho_max, 6)
bin_labels = [f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})" for i in range(5)]

roc_results = {}
pr_results = {}
summary_rows = []

for i in range(5):
    left = bin_edges[i]
    right = bin_edges[i + 1]
    bin_label = bin_labels[i]

    if i == 4:
        gw_bin = gw_df[(gw_df["rho_net"] >= left) & (gw_df["rho_net"] <= right)]
    else:
        gw_bin = gw_df[(gw_df["rho_net"] >= left) & (gw_df["rho_net"] < right)]

    # Include all non-GW in every bin, including noise
    eval_df = pd.concat([gw_bin, non_gw_df], ignore_index=True)

    y_true = eval_df["is_gw_true"].to_numpy()
    y_score = eval_df[score_col].to_numpy()

    n_gw = len(gw_bin)
    n_non_gw = len(non_gw_df)
    n_total = len(eval_df)

    summary_rows.append({
        "rho_bin": bin_label,
        "rho_left": left,
        "rho_right": right,
        "n_gw_in_bin": n_gw,
        "n_non_gw_all": n_non_gw,
        "n_total_eval": n_total
    })

    if n_gw == 0 or np.unique(y_true).size < 2:
        continue

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    roc_results[bin_label] = (fpr, tpr, roc_auc)
    pr_results[bin_label] = (recall, precision, ap)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(save_dir, "rho_bin_summary.csv"), index=False)

# =========================
# Save ROC plot
# =========================

import re

def round_bin_label(bin_label):
    nums = re.findall(r"[-+]?\d*\.?\d+", bin_label)
    low, high = [round(float(x)) for x in nums]
    return f"[{low}, {high})"

plt.figure(figsize=(10, 8))

colors = [
    "darkblue",
    "cornflowerblue",
    "powderblue",
    "turquoise",
    "darkcyan"
]

for color, (bin_label, (fpr, tpr, roc_auc)) in zip(colors, roc_results.items()):
    plt.plot(
        fpr,
        tpr,
        color=color,
        label=f"SNR={round_bin_label(bin_label)} (AUC={roc_auc:.2f})"
    )

# for bin_label, (fpr, tpr, roc_auc) in roc_results.items():
#     plt.plot(fpr, tpr, label=f"SNR={bin_label:.0f} (AUC={roc_auc:.2f})")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate", fontsize=18)
plt.ylabel("True Positive Rate", fontsize=18)
plt.title("ROC Curves by GW SNR Bin", fontsize=19)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=14)
plt.grid(True)

plt.savefig(
    os.path.join(save_dir, "roc_curves_by_rho_bin_5bins.pdf"),
    format="pdf",
    bbox_inches="tight"
)
plt.close()

# =========================
# Save PR plot
# =========================
plt.figure(figsize=(10, 8))


for color, (bin_label, (recall, precision, ap)) in zip(colors, pr_results.items()):
    plt.plot(
        recall,
        precision,
        color=color,
        label=f"SNR={round_bin_label(bin_label)} (AP={ap:.2f})"
    )

plt.xlabel("Recall", fontsize=18)
plt.ylabel("Precision", fontsize=18)
plt.title("PR Curves by GW SNR Bin", fontsize=19)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=14)
plt.grid(True)

plt.savefig(
    os.path.join(save_dir, "pr_curves_by_rho_bin_5bins.pdf"),
    format="pdf",
    bbox_inches="tight"
)
plt.close()

# =========================
# Noise threshold file
# =========================
if len(noise_df) == 0:
    raise ValueError("No rows found with class_type == 'noise'.")

noise_scores = noise_df[score_col].dropna().to_numpy()

# Gaussian coverage probabilities within ±N sigma
sigma_targets = {
    0: 0.5,
    1: 0.6826894921370859,
    2: 0.9544997361036416,
    3: 0.9973002039367398,
    4: 1,
}

threshold_rows = []

for n_sigma, target_acc in sigma_targets.items():
    threshold = np.quantile(noise_scores, target_acc)

    achieved_acc = np.mean(noise_scores < threshold)
    false_positive_rate_on_noise = np.mean(noise_scores >= threshold)

    threshold_rows.append({
        "sigma_level": n_sigma,
        "target_noise_accuracy": target_acc,
        "target_noise_accuracy_percent": 100 * target_acc,
        "threshold_p_gw": threshold,
        "achieved_noise_accuracy": achieved_acc,
        "achieved_noise_accuracy_percent": 100 * achieved_acc,
        "noise_false_positive_rate": false_positive_rate_on_noise,
        "noise_false_positive_rate_percent": 100 * false_positive_rate_on_noise,
        "n_noise_samples": len(noise_scores)
    })

threshold_df = pd.DataFrame(threshold_rows)
threshold_df.to_csv(
    os.path.join(save_dir, "noise_thresholds_by_std.csv"),
    index=False
)

print(f"Saved files in: {save_dir}")
print("Created:")
print("- roc_curves_by_rho_bin_5bins.pdf")
print("- pr_curves_by_rho_bin_5bins.pdf")
print("- rho_bin_summary.csv")
print("- noise_thresholds_by_std.csv")