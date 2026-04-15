import pandas as pd
import matplotlib.pyplot as plt
import requests

# path to your file
# csv_path = "real_event_predictions/real_events_predictions.csv"
# df = pd.read_csv(csv_path)
# df = df[df["status"] == "ok"].copy()

# total = len(df_ok)  # number of events
# correct = (df_ok["pred_3class"] == 2).sum() # correctly classified GW events (3-class head)
# accuracy = correct / total * 100

# print(f"Total events evaluated: {total}")
# print(f"Correctly classified as GW: {correct}")
# print(f"Accuracy: {accuracy:.2f}%")


# Load the two files
ev = pd.read_csv("event-versions.csv") #downloaded from GWOSC
pred = pd.read_csv("real_event_predictions/real_events_predictions9.csv")

# Build a join key from the first table:
# name -> name-v1 so it matches event_name in the second table
ev["event_name"] = ev["shortName"]

# Join
merged = ev.merge(pred, on="event_name", how="inner")

# Keep only rows that have finite values for the plot columns
plot_df = merged[["event_name",
                  "gps_x",
                  "catalog",
                  "network_matched_filter_snr",
                  "p_astro",
                  "far",
                  "luminosity_distance",
                  "chirp_mass_source",
                  "mass_1_source",
                  "mass_2_source",
                  "chi_eff",
                  "p_gw",
                #   "pred_3class",
                #   "attn_H1",
                #   "attn_L1"
]].copy()

# Convert numeric columns
numeric_cols = [
    "gps_x",
    "network_matched_filter_snr",
    "p_astro",
    "far",
    "luminosity_distance",
    "chirp_mass_source",
    "mass_1_source",
    "mass_2_source",
    "chi_eff",
    "p_gw",
    # "pred_3class",
    # "attn_H1",
    # "attn_L1"
]

for col in numeric_cols:
    plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

plot_df = plot_df.dropna(subset=["network_matched_filter_snr", "p_gw", "pred_3class"])

# Detector-attention asymmetry
# plot_df["attention_asymmetry"] = (plot_df["attn_H1"] - plot_df["attn_L1"]).abs()

# Optional: map class labels to readable names
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

# IMPORTANT, SNR against probability, color-coded by pred_3class
plt.figure(figsize=(8, 6))

for cls, group in plot_df.groupby("pred_3class"):
    plt.scatter(
        group["network_matched_filter_snr"],
        group["p_gw"],
        label=class_labels.get(int(cls), f"class {int(cls)}"),
        color=class_colours[cls],
        alpha=0.8
    )


plt.xlabel("Network Matched Filter SNR")
plt.ylabel("Probability of Gravitational Wave")
plt.title("Gravitational Wave Predictions with Network Matched Filter SNR")
plt.legend(title="Predicted Class")
plt.grid(True, alpha=0.3)
plt.savefig("Network_SNR_New_Data.pdf", bbox_inches="tight", transparent=True)


noise_cluster = plot_df[
    (plot_df["pred_3class"] == 0) &
    (plot_df["network_matched_filter_snr"] <= 15) &
    (plot_df["p_gw"] <= 0.55)
].copy()

low_snr_gw = plot_df[
    (plot_df["pred_3class"] == 2) &
    (plot_df["network_matched_filter_snr"] <= 11.5)
].copy()

# print("\nCounts")
# print("Noise-cluster events:", len(noise_cluster))
# print("Low-SNR GW events:", len(low_snr_gw))


# ------------------------------------------------------------------
# Summary statistics for testing hypotheses
# ------------------------------------------------------------------
summary_cols = [
    "network_matched_filter_snr",
    "p_gw",
    "p_astro",
    "far",
    "luminosity_distance",
    "chirp_mass_source",
    "mass_1_source",
    "mass_2_source",
    "chi_eff"
]

print("\nNoise-cluster summary:")
print(noise_cluster[summary_cols].describe())

print("\nLow-SNR GW summary:")
print(low_snr_gw[summary_cols].describe())

print("\nCatalog counts: noise-cluster")
print(noise_cluster["catalog"].value_counts(dropna=False))

print("\nCatalog counts: low-SNR GW")
print(low_snr_gw["catalog"].value_counts(dropna=False))

print("\nExample noise-cluster events:")
print(
    noise_cluster[
        ["event_name", "catalog", "network_matched_filter_snr", "p_gw", "p_astro", "far",
         "luminosity_distance", "chirp_mass_source", "mass_1_source", "mass_2_source", "chi_eff"]
    ].sort_values(["network_matched_filter_snr", "p_gw"])
)

# print("\nAttention summary: noise cluster")
# print(noise_cluster[["attn_H1", "attn_L1", "attention_asymmetry"]].describe())

# print("\nAttention summary: low-SNR GW")
# print(low_snr_gw[["attn_H1", "attn_L1", "attention_asymmetry"]].describe())

# ------------------------------------------------------------------
# Diagnostic plots to test theories
# ------------------------------------------------------------------

# attn_df = plot_df.dropna(subset=["attention_asymmetry"]).copy()

# plt.figure(figsize=(8, 6))
# for cls, group in attn_df.groupby("pred_3class"):
#     plt.scatter(
#         group["attention_asymmetry"],
#         group["p_gw"],
#         label=class_labels.get(int(cls), f"class {int(cls)}"),
#         color=class_colours[int(cls)],
#         alpha=0.8
#     )
# plt.xlabel("Absolute Attention Difference |attn_H1 - attn_L1|")
# plt.ylabel("Probability of Gravitational Wave")
# plt.title("Model Probability vs Detector Attention Asymmetry")
# plt.legend(title="Predicted Class")
# plt.grid(True, alpha=0.3)
# plt.savefig("attention_asymmetry_vs_pgw.pdf", bbox_inches="tight", transparent=True)
# plt.show()


# 1) Compare p_gw to p_astro
plt.figure(figsize=(8, 6))
plt.scatter(
    plot_df["p_astro"],
    plot_df["p_gw"],
    c=plot_df["pred_3class"].map(class_colours),
    alpha=0.7
)
plt.xlabel("LIGO p_astro")
plt.ylabel("Model p_gw")
plt.title("Model Probability vs LIGO Astrophysical Probability")
plt.grid(True, alpha=0.3)
plt.savefig("pastro_vs_pgw.pdf", bbox_inches="tight", transparent=True)

# 2) Compare FAR to p_gw
far_df = plot_df.dropna(subset=["far"]).copy()
far_df = far_df[far_df["far"] > 0]

plt.figure(figsize=(8, 6))
for cls, group in far_df.groupby("pred_3class"):
    plt.scatter(
        group["far"],
        group["p_gw"],
        label=class_labels.get(int(cls), f"class {int(cls)}"),
        color=class_colours[int(cls)],
        alpha=0.8
    )
plt.xscale("log")
plt.xlabel("False Alarm Rate (log scale)")
plt.ylabel("Probability of Gravitational Wave")
plt.title("Model Probability vs FAR")
plt.legend(title="Predicted Class")
plt.grid(True, alpha=0.3)
plt.savefig("far_vs_pgw.pdf", bbox_inches="tight", transparent=True)

# 3) GPS time test: is the cluster concentrated in one era?
gps_df = plot_df.dropna(subset=["gps_x"]).copy()

plt.figure(figsize=(8, 6))
for cls, group in gps_df.groupby("pred_3class"):
    plt.scatter(
        group["gps_x"],
        group["p_gw"],
        label=class_labels.get(int(cls), f"class {int(cls)}"),
        color=class_colours[int(cls)],
        alpha=0.8
    )
plt.xlabel("GPS Time")
plt.ylabel("Probability of Gravitational Wave")
plt.title("Model Probability vs GPS Time")
plt.legend(title="Predicted Class")
plt.grid(True, alpha=0.3)
plt.savefig("gps_vs_pgw.pdf", bbox_inches="tight", transparent=True)

# 4) Chirp mass test: morphology / source-type differences
cm_df = plot_df.dropna(subset=["chirp_mass_source"]).copy()

plt.figure(figsize=(8, 6))
for cls, group in cm_df.groupby("pred_3class"):
    plt.scatter(
        group["chirp_mass_source"],
        group["p_gw"],
        label=class_labels.get(int(cls), f"class {int(cls)}"),
        color=class_colours[int(cls)],
        alpha=0.8
    )
plt.xlabel("Chirp Mass Source")
plt.ylabel("Probability of Gravitational Wave")
plt.title("Model Probability vs Chirp Mass")
plt.legend(title="Predicted Class")
plt.grid(True, alpha=0.3)
plt.savefig("chirp_mass_vs_pgw.pdf", bbox_inches="tight", transparent=True)

# 5) Distance test: are cluster events just farther / weaker in morphology?
dist_df = plot_df.dropna(subset=["luminosity_distance"]).copy()

plt.figure(figsize=(8, 6))
for cls, group in dist_df.groupby("pred_3class"):
    plt.scatter(
        group["luminosity_distance"],
        group["p_gw"],
        label=class_labels.get(int(cls), f"class {int(cls)}"),
        color=class_colours[int(cls)],
        alpha=0.8
    )
plt.xlabel("Luminosity Distance")
plt.ylabel("Probability of Gravitational Wave")
plt.title("Model Probability vs Luminosity Distance")
plt.legend(title="Predicted Class")
plt.grid(True, alpha=0.3)
plt.savefig("distance_vs_pgw.pdf", bbox_inches="tight", transparent=True)
