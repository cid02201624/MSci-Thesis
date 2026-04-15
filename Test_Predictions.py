import os
import torch
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Model_5.Training import MultiViewConvNeXtGWWithTime, load_checkpoint


def make_prediction_dataframe(
    checkpoint_path,
    shards_dir,
    metadata_csv,
    device=None,
    partial_save_path="partial_predictions.csv",
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Rebuild model exactly as in training
    model = MultiViewConvNeXtGWWithTime(
        pretrained=True,
        use_aux_3class=True,
        fuse_dim=384,
        head_hidden=256,
        dropout=0.30,
    ).to(device)

    # Load trained weights
    load_checkpoint(checkpoint_path, model, optimiser=None, map_location=device)
    model.eval()

    # Load metadata
    df_meta = pd.read_csv(metadata_csv)

    # Get shard list in order
    manifest_path = os.path.join(shards_dir, "manifest.pt")
    if os.path.exists(manifest_path):
        manifest = torch.load(manifest_path, map_location="cpu")
        shard_paths = [os.path.join(shards_dir, f) for f in manifest["files"]]
    else:
        shard_paths = sorted(Path(shards_dir).glob("shard_*.pt"))
        shard_paths = [str(p) for p in shard_paths]

    predictions = []

    if os.path.exists(partial_save_path):
        df_partial = pd.read_csv(partial_save_path)
        predictions = df_partial["model_prediction"].tolist()
        print(f"Resuming from {len(predictions)} existing predictions")

    with torch.no_grad():
        for shard_path in shard_paths:
            shard = torch.load(shard_path, map_location="cpu")

            X = shard["X"].to(device=device, dtype=torch.float32)
            t_feat = shard["t_feat"].to(device=device, dtype=torch.float32)

            outputs = model(X, t_feat)

            # 3-class prediction: 0=noise, 1=glitch, 2=GW
            logits_3class = outputs["logits_3class"]
            pred = logits_3class.argmax(dim=1)

            predictions.extend(pred.cpu().tolist())

            df_partial = df_meta.iloc[:len(predictions)].copy()
            df_partial.insert(0, "model_prediction", predictions)

            df_partial.to_csv(partial_save_path, index=False)

            # print(f"Saved partial predictions after shard {shard_i+1}/{len(shard_paths)}")


    if len(predictions) != len(df_meta):
        raise ValueError(
            f"Number of predictions ({len(predictions)}) does not match "
            f"number of metadata rows ({len(df_meta)})."
        )

    df_out = df_meta.copy()
    df_out.insert(0, "model_prediction", predictions)

    df_out = df_out[
        [
            "model_prediction",
            "GPS",
            "example_seed",
            "y",
            "class_type",
            "final_snr",
            "chirp_mass",
            "merger_family",
            "approximant",
            "rho_H1",
            "rho_L1",
            "rho_net",
            "glitch_type",
            "glitch_detector",
            "sample_index",
        ]
    ]

    return df_out




# def _accuracy_by_binned_numeric(df, true_class, x_col, pred_col="model_prediction", y_col="y", bin_width=1.0):
#     """
#     Compute class-specific accuracy within bins of a numeric column.
#     Only uses rows with y == true_class.
#     """
#     sub = df[df[y_col] == true_class].copy()
#     sub = sub[sub[x_col].notna()].copy()

#     if len(sub) == 0:
#         raise ValueError(f"No rows found for class={true_class} with non-null {x_col}")

#     x_min = np.floor(sub[x_col].min())
#     x_max = np.ceil(sub[x_col].max())

#     # include final edge
#     bins = np.arange(x_min, x_max + bin_width, bin_width)
#     if len(bins) < 2:
#         bins = np.array([x_min, x_min + bin_width])

#     sub["bin"] = pd.cut(sub[x_col], bins=bins, right=False, include_lowest=True)

#     grouped = sub.groupby("bin", observed=False)
#     out = grouped.apply(
#         lambda g: pd.Series({
#             "n": len(g),
#             "accuracy": (g[pred_col] == true_class).mean()
#         })
#     ).reset_index()

#     # cleaner labels like 0-1, 1-2, ...
#     out["bin_label"] = out["bin"].apply(lambda b: f"{int(b.left)}-{int(b.right)}" if pd.notna(b) else "NA")
#     return out


# def _accuracy_by_category(df, true_class, cat_col, pred_col="model_prediction", y_col="y"):
#     """
#     Compute class-specific accuracy by category.
#     Only uses rows with y == true_class.
#     """
#     sub = df[df[y_col] == true_class].copy()
#     sub = sub[sub[cat_col].notna()].copy()

#     if len(sub) == 0:
#         raise ValueError(f"No rows found for class={true_class} with non-null {cat_col}")

#     out = (
#         sub.groupby(cat_col, observed=False)
#         .apply(lambda g: pd.Series({
#             "n": len(g),
#             "accuracy": (g[pred_col] == true_class).mean()
#         }))
#         .reset_index()
#         .sort_values("n", ascending=False)
#     )
#     return out


# def plot_accuracy_vs_final_snr(df, save_dir=None):
#     # GW = 2
#     gw_stats = _accuracy_by_binned_numeric(df, true_class=2, x_col="final_snr", bin_width=0.1)

#     plt.figure(figsize=(8, 5))
#     plt.plot(gw_stats["bin_label"], gw_stats["accuracy"], marker="o")
#     plt.xticks(rotation=45)
#     plt.ylim(0, 1.05)
#     plt.xlabel("Final SNR")
#     plt.ylabel("GW Accuracy")
#     plt.title("GW Accuracy vs Final SNR")
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     if save_dir:
#         plt.savefig(f"{save_dir}/gw_accuracy_vs_final_snr.pdf", bbox_inches="tight", transparent=True)
#     plt.show()

#     # Glitch = 1
#     glitch_stats = _accuracy_by_binned_numeric(df, true_class=1, x_col="final_snr", bin_width=1.0)

#     plt.figure(figsize=(8, 5))
#     plt.plot(glitch_stats["bin_label"], glitch_stats["accuracy"], marker="o")
#     plt.xticks(rotation=45)
#     plt.ylim(0, 1.05)
#     plt.xlabel("Final SNR")
#     plt.ylabel("Glitch Accuracy")
#     plt.title("Glitch Accuracy vs Final SNR")
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     if save_dir:
#         plt.savefig(f"{save_dir}/glitch_accuracy_vs_final_snr.pdf", bbox_inches="tight", transparent=True)
#     plt.show()

#     return gw_stats, glitch_stats


# def plot_accuracy_vs_chirp_mass(df, save_dir=None):
#     # Only GW accuracy vs chirp mass makes sense
#     gw_stats = _accuracy_by_binned_numeric(df, true_class=2, x_col="chirp_mass", bin_width=1.0)

#     plt.figure(figsize=(8, 5))
#     plt.plot(gw_stats["bin_label"], gw_stats["accuracy"], marker="o")
#     plt.xticks(rotation=45)
#     plt.ylim(0, 1.05)
#     plt.xlabel("Chirp Mass bin")
#     plt.ylabel("GW Accuracy")
#     plt.title("GW Accuracy against Chirp Mass")
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     if save_dir:
#         plt.savefig(f"{save_dir}/gw_accuracy_vs_chirp_mass.pdf", bbox_inches="tight", transparent=True)
#     plt.show()

#     return gw_stats


# def plot_accuracy_vs_glitch_type(df, save_dir=None):
#     # Only glitch accuracy vs glitch_type makes sense
#     glitch_stats = _accuracy_by_category(df, true_class=1, cat_col="glitch_type")

#     plt.figure(figsize=(9, 5))
#     plt.bar(glitch_stats["glitch_type"], glitch_stats["accuracy"])
#     plt.xticks(rotation=45, ha="right")
#     plt.ylim(0, 1.05)
#     plt.xlabel("Glitch Type")
#     plt.ylabel("Glitch Accuracy")
#     plt.title("Glitch Accuracy vs Glitch Type")
#     plt.grid(True, axis="y", alpha=0.3)
#     plt.tight_layout()
#     if save_dir:
#         plt.savefig(f"{save_dir}/glitch_accuracy_vs_glitch_type.pdf", bbox_inches="tight", transparent=True)
#     plt.show()

#     return glitch_stats


# Example usage
if __name__ == "__main__":
    checkpoint_path = "checkpoints5/best.pt"
    shards_dir = "Training_Data_Generation/pt_dataset/test"
    metadata_csv = "Training_Data_Generation/pt_dataset/test_sample_metadata.csv"

    df_predictions = make_prediction_dataframe(
        checkpoint_path=checkpoint_path,
        shards_dir=shards_dir,
        metadata_csv=metadata_csv,
    )

#     print(df_predictions.head())
    df_predictions.to_csv("test_predictions.csv", index=False)

    # gw_snr_stats, glitch_snr_stats = plot_accuracy_vs_final_snr(df_predictions, save_dir="plots")
    # gw_chirp_stats = plot_accuracy_vs_chirp_mass(df_predictions, save_dir="plots")
    # glitch_type_stats = plot_accuracy_vs_glitch_type(df_predictions, save_dir="plots")