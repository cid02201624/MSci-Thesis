import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from gwpy.timeseries import TimeSeries

from Training_Data_Generation.Processing2 import QTransformDataset



def plot_spec_on_ax(qspec, ax, title, vmin=0, vmax=30):
    # Build edges for pcolormesh (length N+1)
    x = np.concatenate((qspec.xindex.value, qspec.xspan[-1:]))
    y = np.concatenate((qspec.yindex.value, qspec.yspan[-1:]))
    X, Y = np.meshgrid(x, y, copy=False, sparse=True)

    # Note transpose: qspec.value is (time, freq)
    m = ax.pcolormesh(X, Y, qspec.value.T, shading="auto", vmin=vmin, vmax=vmax)

    ax.set_title(title)
    ax.set_ylabel("Frequency [Hz]")
    ax.set_xlabel("Time [s]")

    # Match typical q-transform look
    ax.set_yscale("log")
    ax.set_ylim(20, 300)

    return m


def plot_qtransform_grid_by_y(
    y: int,
    *,
    split_csv: str,
    save_path: str | None = None,
    **dataset_kwargs,
):
    """
    Find the first 6 GPSs for the input y from the split CSV, recompute the
    q-transform for each, and plot them in a 3 column x 2 row grid.

    dataset_kwargs are passed into QTransformDataset, e.g.
    seglen=8, sample_rate=4096, padding=30, qrange=(3,100), frange=(20,300), ...
    """
    df = pd.read_csv(split_csv)

    hits = df.loc[df["y"] == int(y)].head(6).reset_index(drop=True)

    if len(hits) == 0:
        raise ValueError(f"No rows found for y={y}")

    if len(hits) < 6:
        raise ValueError(f"Found only {len(hits)} rows for y={y}, need at least 6")

    fig, axes = plt.subplots(
        2, 3, figsize=(18, 8), sharex=True, sharey=True, constrained_layout=True
    )
    axes = axes.ravel()

    mappable = None

    for i, (_, row) in enumerate(hits.iterrows()):
        row_df = pd.DataFrame([row])

        ds = QTransformDataset(
            segment_df=row_df,
            cache=True,
            return_metadata=True,
            visualise=True,
            **dataset_kwargs,
        )

        q_H, q_L, meta = ds[0]

        # Keep structure simple: plot H1 only, one q-transform per subplot
        q_H.name = "H1_qtransform"

        gps = int(row["GPS"])
        seed = int(row["example_seed"])

        mappable = plot_spec_on_ax(
            q_H,
            axes[i],
            title=f"GPS={gps}\nseed={seed}",
            vmin=0,
            vmax=30,
        )

    cbar = fig.colorbar(mappable, ax=axes.tolist(), pad=0.02, shrink=0.95)
    cbar.set_label("Normalised energy")

    fig.suptitle(f"First 6 q-transforms for y={y}")

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    plot_qtransform_grid_by_y(
        y=2,
        split_csv="Training_Data_Generation/splits/precompute_300k/test_30k.csv",
        seglen=8,
        sample_rate=4096,
        padding=30,
        qrange=(3, 100),
        frange=(20, 300),
        save_path="TEST_qtransform_grid_y1.png",
    )










# def plot_spec_on_ax(qspec, ax, title, vmin=0, vmax=30):
#     # Build edges for pcolormesh (length N+1)
#     x = np.concatenate((qspec.xindex.value, qspec.xspan[-1:]))
#     y = np.concatenate((qspec.yindex.value, qspec.yspan[-1:]))
#     X, Y = np.meshgrid(x, y, copy=False, sparse=True)

#     # Note transpose: qspec.value is (time, freq)
#     m = ax.pcolormesh(X, Y, qspec.value.T, shading="auto", vmin=vmin, vmax=vmax)

#     ax.set_title(title)
#     ax.set_ylabel("Frequency [Hz]")
#     ax.set_xlabel("Time [s]")

#     # Match typical q-transform look (optional)
#     ax.set_yscale("log")
#     ax.set_ylim(20, 300)

#     return m

# def plot_qtransform_by_recomputing(
#     gps: int,
#     *,
#     split_csv: str,
#     example_seed: int | None = None,
#     y: int | None = None,
#     save_path: str | None = None,
#     **dataset_kwargs,
# ):
#     """
#     Recompute a deterministic sample from the split CSV and plot the resulting
#     stored-style q-transform tensors.

#     dataset_kwargs are passed into QTransformDataset, e.g.
#     seglen=8, sample_rate=4096, padding=30, qrange=(3,100), frange=(20,300), ...
#     """
#     df = pd.read_csv(split_csv)

#     m = df["GPS"] == int(gps)
#     if example_seed is not None:
#         m &= df["example_seed"] == int(example_seed)
#     if y is not None:
#         m &= df["y"] == int(y)

#     hits = df.loc[m].copy()

#     if len(hits) == 0:
#         raise ValueError(f"No row found for GPS={gps}, example_seed={example_seed}, y={y}")

#     row_df = hits.iloc[[0]].reset_index(drop=True)

#     ds = QTransformDataset(
#         segment_df=row_df,
#         cache=True,
#         return_metadata=True,
#         visualise=True,
#         **dataset_kwargs,
#     )

#     q_H, q_L, meta = ds[0]

#     q_H.name = "H1_qtransform"
#     q_L.name = "L1_qtransform"

#     fig, (axH, axL) = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True, constrained_layout=True)

#     mH = plot_spec_on_ax(q_H, axH, "H1", vmin=0, vmax=30)
#     mL = plot_spec_on_ax(q_L, axL, "L1", vmin=0, vmax=30)

#     # One shared colorbar for both panels
#     cbar = fig.colorbar(mH, ax=[axH, axL], pad=0.02, shrink=0.95)
#     cbar.set_label("Normalised energy")

#     fig.suptitle(
#     f"GPS={gps} | seed={int(row_df.iloc[0]['example_seed'])} | "#y={int(y)}
# )

#     if save_path is not None:
#         fig.savefig(save_path, bbox_inches="tight")#, transparent=True)










# if __name__ == "__main__":
#     plot_qtransform_by_recomputing(
#         gps=1388665251, #1389314548, 1389323161, 1389067517, 1388030308, 1389300097, 1389310116
#         # example_seed=2318690297,
#         split_csv="Training_Data_Generation/splits/precompute_300k/test_30k.csv",
#         seglen=8,
#         sample_rate=4096,
#         padding=30,
#         qrange=(3, 100),
#         frange=(20, 300),
#         save_path="TEST_qtransform_gw6.png",
#         )