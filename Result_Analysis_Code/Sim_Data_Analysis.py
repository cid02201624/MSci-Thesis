import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from gwpy.timeseries import TimeSeries

# from Training_Data_Generation.Processing2 import QTransformDataset
# from Reproduce_Plot import plot_spec_on_ax

# Load CSVs into DataFrame
predict_df = pd.read_csv("training_figures5/test_predictions.csv") #change according to which model you want to test
meta_df = pd.read_csv("Training_Data_Generation/pt_dataset/test_sample_metadata.csv")

merged = meta_df.merge(predict_df, on="sample_index", how="inner")

plot_df = merged[[
    "GPS",
    "correct",
    "true_class",
    "pred_class",
    "chirp_mass", 
    "merger_family", 
    "rho_H1",
    "rho_L1", 
    "rho_net",
    "glitch_type",
    "glitch_detector",
]].copy()

# Convert numeric columns
numeric_cols = [
    "GPS",
    "correct",
    "true_class",
    "pred_class",
    "chirp_mass", 
    "merger_family", 
    "rho_H1",
    "rho_L1", 
    "rho_net",
    "glitch_type",
    "glitch_detector",
]

for col in numeric_cols:
    plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

# Detector-attention asymmetry
plot_df["attention_asymmetry"] = (plot_df["rho_H1"] - plot_df["rho_L1"]).abs()

# Map class labels to readable names
class_labels = {
    0: "Noise",
    1: "Glitch",
    2: "Gravitational Wave"
}

class_colours = {
    0: "powderblue",     # noise
    1: "cornflowerblue",   # glitch
    2: "darkblue"     # gw
}



def plot_stacked_with_accuracy(df, group_col, title, bins=None, is_categorical=False):
    import matplotlib.pyplot as plt

    temp = df.copy()

    # Bin if needed
    if not is_categorical:
        temp["group"] = pd.cut(temp[group_col], bins=bins)
    else:
        temp["group"] = temp[group_col]

    # Drop NaNs
    temp = temp.dropna(subset=["group"])

    # Prediction distribution
    dist = (
        temp.groupby(["group", "pred_class"])
        .size()
        .unstack(fill_value=0)
    )

    # Convert to %
    dist_pct = dist.div(dist.sum(axis=1), axis=0)

    # Accuracy
    acc = temp.groupby("group", observed=False)["correct"].mean()

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 5))

    bottom = np.zeros(len(dist_pct))

    for cls in [0, 1, 2]:
        if cls in dist_pct.columns:
            ax1.bar(
                dist_pct.index.astype(str),
                dist_pct[cls],
                bottom=bottom,
                label=class_labels[cls],
                color=class_colours[cls],
            )
            bottom += dist_pct[cls].values

    ax1.set_ylabel("Prediction Composition (%)")
    ax1.set_title(title)
    ax1.set_xticklabels(dist_pct.index.astype(str), rotation=45)

    # Accuracy line
    ax2 = ax1.twinx()
    ax2.plot(dist_pct.index.astype(str), acc, color="black", marker="o", label="Accuracy")
    ax2.set_ylabel("Accuracy")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.savefig(f"{title}.pdf", bbox_inches='tight')


# GW SNR Stuff
snr_bins = np.logspace(np.log10(5), np.log10(35), 8)

gw_df = plot_df[plot_df["true_class"] == 2]

plot_stacked_with_accuracy(
    gw_df,
    "rho_net",
    "GW Detection vs SNR",
    bins=snr_bins
)

# Glitch SNR Stuff
glitch_df = plot_df[plot_df["true_class"] == 1]

plot_stacked_with_accuracy(
    glitch_df,
    "rho_net",
    "Glitch Detection vs SNR",
    bins=snr_bins
)

# GW detection vs attention asymmetry
bins = np.linspace(0, plot_df["attention_asymmetry"].max(), 8)

plot_stacked_with_accuracy(
    gw_df,
    "attention_asymmetry",
    "GW Detection vs Detector Asymmetry",
    bins=bins
)


# GW detection vs chirp mass
bins = np.linspace(gw_df["chirp_mass"].min(), gw_df["chirp_mass"].max(), 8)

plot_stacked_with_accuracy(
    gw_df,
    "chirp_mass",
    "GW Detection vs Chirp Mass",
    bins=bins
)

# GW detection vs merger family
plot_stacked_with_accuracy(
    gw_df,
    "merger_family",
    "GW Detection vs Merger Type",
    is_categorical=True
)

# Glitch detection vs glitch type
plot_stacked_with_accuracy(
    glitch_df,
    "glitch_type",
    "Glitch Detection vs Glitch Type",
    is_categorical=True
)


# Total Accuracy vs SNR
snr_groups = pd.cut(plot_df["rho_net"], bins=snr_bins, include_lowest=True)
acc_snr = plot_df.groupby(snr_groups, observed=False)["correct"].mean()

bin_centres = np.sqrt(snr_bins[:-1] * snr_bins[1:])

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(bin_centres, acc_snr.values, marker="o")
ax.set_xscale("log")
ax.set_title("Accuracy vs SNR")
ax.set_xlabel("rho_net")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig("Accuracy vs SNR.pdf", bbox_inches="tight")
plt.close(fig)


# Heat map: SNR vs chirp mass
pivot = gw_df.copy()
pivot["snr_bin"] = pd.cut(pivot["rho_net"], bins=snr_bins, include_lowest=True)
pivot["mass_bin"] = pd.cut(pivot["chirp_mass"], bins=6, include_lowest=True)

heat = pivot.groupby(["snr_bin", "mass_bin"], observed=False)["correct"].mean().unstack()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(heat, cmap="viridis", vmin=0, vmax=1, ax=ax)
ax.set_title("GW Accuracy: SNR vs Chirp Mass")
fig.tight_layout()
fig.savefig("GW Accuracy SNR vs Chirp Mass.pdf", bbox_inches="tight")
plt.close(fig)








