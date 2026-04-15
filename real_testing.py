import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Colours
# -----------------------------
class_colours = {
    0: "powderblue",      # noise
    1: "cornflowerblue",  # glitch
    2: "darkblue"         # gw
}

# -----------------------------
# Load files
# -----------------------------
thresholds_path = "M9/noise_thresholds_by_std.csv"
events_path = "real_event_predictions/real_events_predictions9_dunno.csv"

thresholds_df = pd.read_csv(thresholds_path)
events_df = pd.read_csv(events_path)

# -----------------------------
# True GW events
# -----------------------------
gw_df = events_df[events_df["y_true"] == 2].copy()
total_gw = len(gw_df)

print(f"Total GW events: {total_gw}")

# -----------------------------
# Calculate GW accuracy
# -----------------------------
results = []

for _, row in thresholds_df.iterrows():
    sigma_level = row["sigma_level"]
    threshold = row["threshold_p_gw"]

    # use achieved noise accuracy from file
    noise_accuracy_percent = row["achieved_noise_accuracy_percent"]

    # predicted as GW above threshold
    gw_predicted_count = (gw_df["p_gw"] >= threshold).sum()

    gw_accuracy_percent = (
        gw_predicted_count / total_gw * 100
        if total_gw > 0 else 0
    )

    results.append({
        "sigma_level": sigma_level,
        "noise_accuracy_percent": noise_accuracy_percent,
        "gw_accuracy_percent": gw_accuracy_percent,
        "threshold_p_gw": threshold
    })

results_df = pd.DataFrame(results)

print("\nResults:")
print(results_df)

# Save table
results_df.to_csv(
    "M9/gw_noise_accuracy_by_sigma_level.csv",
    index=False
)

# -----------------------------
# Plot grouped bars
# -----------------------------
x = np.arange(len(results_df))
width = 0.35

plt.figure(figsize=(9, 5))

plt.bar(
    x - width/2,
    results_df["noise_accuracy_percent"],
    width,
    label="Noise accuracy",
    color=class_colours[0]
)

plt.bar(
    x + width/2,
    results_df["gw_accuracy_percent"],
    width,
    label="GW accuracy",
    color=class_colours[2]
)

plt.xlabel("Sigma Level")
plt.ylabel("Accuracy (%)")
plt.title("Noise vs GW Accuracy by Sigma Level")
plt.xticks(x, results_df["sigma_level"].astype(int))
plt.ylim(0, 105)
plt.legend()

# value labels
for i, val in enumerate(results_df["noise_accuracy_percent"]):
    plt.text(
        x[i] - width/2,
        val + 1,
        f"{val:.1f}%",
        ha="center",
        fontsize=9
    )

for i, val in enumerate(results_df["gw_accuracy_percent"]):
    plt.text(
        x[i] + width/2,
        val + 1,
        f"{val:.1f}%",
        ha="center",
        fontsize=9
    )

plt.tight_layout()

# -----------------------------
# Save PDF
# -----------------------------
pdf_path = "M9/gw_noise_accuracy_vs_sigma_level.pdf"
plt.savefig(pdf_path, format="pdf", bbox_inches="tight")

print(f"\nSaved PDF to: {pdf_path}")

plt.show()