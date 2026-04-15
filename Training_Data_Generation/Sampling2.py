"""
Author: Saskia Knight
Date: January 2026
Description: 
1. Use network_segments_list.json to create deterministic separation of GPS times into non-overlapping 68s segments, then assigns training/ validation/ testing 
(time separated), finally assigns class according to (deterministic/ seeded) probability distribution. Output = "all_available_starts_blocksplit.csv".
2. Uses "all_available_starts_blocksplit.csv"
"""

# System stuff
from __future__ import annotations
import re, os
import requests, json
from typing import Optional, Sequence, Tuple, List

# Basics
import numpy as np
import pandas as pd
import random, math
import heapq
from bisect import bisect_left

# GW Related
from gwosc.timeline import get_segments
from gwosc.datasets import query_events


Segment = Tuple[int, int]

def _valid_start_bounds(seg: Segment, padding: int, seglen: int) -> Optional[Tuple[int, int]]:
    """Return inclusive integer start bounds [lo, hi] for this segment, after padding rules."""
    s, e = int(seg[0]), int(seg[1])
    lo = s + padding
    hi = e - (seglen + padding)
    if lo <= hi:
        return lo, hi
    return None

def _mix32(x: np.ndarray) -> np.ndarray:
    """
    Vectorized 32-bit mixing hash. Input/Output dtype: uint32.
    Deterministic across platforms (pure integer ops).
    """
    x = x.astype(np.uint32, copy=False)
    x ^= (x >> np.uint32(16))
    x *= np.uint32(0x7feb352d)
    x ^= (x >> np.uint32(15))
    x *= np.uint32(0x846ca68b)
    x ^= (x >> np.uint32(16))
    return x

def write_splits_csv(
    out_dir: str,
    *,
    # segments source (must be coincident network segments)
    file_name: str = "network_segments_list.json",
    # windowing
    seglen: int = 8,
    padding: int = 30,
    # block sizes (absolute counts; remainder is train)
    n_val: int = 500_000,
    n_test: int = 500_000,
    # deterministic assignment
    seed: int = 2026,
    class_probs: Sequence[float] = (1/3, 1/3, 1/3),
    # output
    out_csv_name: str = "all_available_starts_blocksplit.csv",
    chunksize: int = 1_000_000,
    compression: Optional[str] = None,  # e.g. "gzip" to write compressed
) -> str:
    """
    Build a CSV containing ALL valid integer GPS start times (after padding rules),
    assigned by time-block split: train (earliest) -> val -> test (latest).

    Output columns:
      GPS, split_id, example_seed, y

    split_id mapping:
      0=train, 1=val, 2=test

    Determinism:
      - example_seed is a deterministic hash of (GPS, seed)
      - y is a deterministic mapping from a hashed uniform to class_probs
    """

    with open(os.path.join(os.path.dirname(__file__), file_name), "r") as f:
        segs = [tuple(seg) for seg in json.load(f)]
    segs = sorted(segs)

    # First pass: compute total number of valid starts and store valid ranges
    total = 0
    valid_ranges: List[Tuple[int, int]] = []
    for seg in segs:
        bounds = _valid_start_bounds(seg, padding=padding, seglen=seglen)
        if bounds is None:
            continue
        lo, hi = bounds
        valid_ranges.append((lo, hi))
        total += (hi - lo + 1)

    if total == 0:
        raise ValueError("No valid integer start times after applying padding/seglen constraints.")

    n_val = int(max(0, min(n_val, total)))
    n_test = int(max(0, min(n_test, total - n_val)))
    n_train = total - n_val - n_test

    train_end = n_train            # [0, train_end)
    val_end = n_train + n_val      # [train_end, val_end)
    # test is [val_end, total)

    # normalise class probs
    p = np.asarray(class_probs, dtype=np.float64)
    if (p <= 0).any():
        raise ValueError("class_probs must be all > 0.")
    p = p / p.sum()
    c0 = float(p[0])
    c1 = float(p[0] + p[1])

    os.makedirs(out_dir, exist_ok=True)

    # choose output path
    out_path = os.path.join(out_dir, out_csv_name)
    if compression == "gzip" and not out_path.endswith(".gz"):
        out_path = out_path + ".gz"

    # overwrite any existing file
    if os.path.exists(out_path):
        os.remove(out_path)

    seed_u32 = np.uint32(seed)
    wrote_header = False
    global_idx = 0

    # Second pass: stream-write all GPS start times in chronological order
    for lo, hi in valid_ranges:
        start = lo
        while start <= hi:
            end = min(hi, start + chunksize - 1)
            gps = np.arange(start, end + 1, dtype=np.int64)
            L = gps.size

            # split_id by block boundaries
            if global_idx + L <= train_end:
                split_id = np.zeros(L, dtype=np.uint8)
            elif global_idx >= val_end:
                split_id = np.full(L, 2, dtype=np.uint8)
            elif global_idx >= train_end and (global_idx + L) <= val_end:
                split_id = np.ones(L, dtype=np.uint8)
            else:
                split_id = np.full(L, 2, dtype=np.uint8)
                cursor = 0
                if global_idx < train_end:
                    n0 = min(L, train_end - global_idx)
                    split_id[:n0] = 0
                    cursor = n0
                if (global_idx + cursor) < val_end and cursor < L:
                    n1 = min(L - cursor, val_end - (global_idx + cursor))
                    split_id[cursor:cursor + n1] = 1
                    cursor += n1

            gps_u32 = gps.astype(np.uint32, copy=False)

            # deterministic per-example seed (uint32) derived from (GPS, seed)
            example_seed = _mix32(gps_u32 ^ seed_u32)

            # deterministic y from hashed uniform in [0,1)
            u = _mix32(gps_u32 + (seed_u32 * np.uint32(0x9e3779b9)))
            u01 = (u.astype(np.float64) / float(2**32))
            y = np.zeros(L, dtype=np.uint8)
            y[u01 >= c0] = 1
            y[u01 >= c1] = 2

            df = pd.DataFrame({
                "GPS": gps,
                "split_id": split_id,
                "example_seed": example_seed.astype(np.uint32),
                "y": y,
            })

            df.to_csv(
                out_path,
                mode="a",
                header=(not wrote_header),
                index=False,
                compression=compression,
            )
            wrote_header = True

            global_idx += L
            start = end + 1

    return out_path





_SPLIT_MAP = {"train": 0, "val": 1, "test": 2}

# A fast, deterministic 64-bit mixer (SplitMix64 finaliser style)
def _mix64(x: np.uint64) -> np.uint64:
    with np.errstate(over="ignore"):
        x ^= x >> np.uint64(30)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(27)
        x *= np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
    return x

def load_split_from_allocated_csv(
    path: str,
    split: str,
    n: Optional[int] = None,
    seed: int = 1234,
    usecols=("GPS", "split_id", "example_seed", "y"),
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    """
    Load rows for one split from the big allocated CSV.
    If n is provided, returns an ORDER-INVARIANT deterministic sample of size n,
    and a deterministic random ordering (by sampled hash key).
    """
    split_id = _SPLIT_MAP[split]

    if n is None:
        parts = []
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
            c = chunk.loc[chunk["split_id"] == split_id].copy()
            if not c.empty:
                parts.append(c)
        if not parts:
            return pd.DataFrame(columns=["GPS", "example_seed", "y"])
        out = pd.concat(parts, ignore_index=True)
        return out.drop(columns=["split_id"])

    # --- ORDER-INVARIANT sampling ---
    # Keep the n *smallest* keys. We use a max-heap of size n via (-key, row) so we can quickly discard worse (larger-key) candidates.
    heap = []
    seed64 = np.uint64(seed)

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        c = chunk.loc[chunk["split_id"] == split_id].copy()
        if c.empty:
            continue

        # Convert to numpy for speed; preserve column order in `usecols`
        arr = c.to_numpy()

        # Column indices (based on usecols)
        # usecols = ("GPS", "split_id", "example_seed", "y")
        GPS_i, split_i, ex_i, y_i = 0, 1, 2, 3

        for row in arr:
            # Build a stable 64-bit key from seed + row identity
            # If GPS is unique, (seed, GPS) is enough; adding example_seed and y
            # makes it robust if GPS can repeat.
            gps = int(row[GPS_i])
            exs = int(row[ex_i])
            yy  = int(row[y_i])

            key = int(_mix64(seed64 ^ np.uint64(gps) ^ (np.uint64(exs) << 1) ^ (np.uint64(yy) << 33)))

            # IMPORTANT: include gps as secondary key to avoid array comparison
            entry = (-key, -gps, row.copy())

            if len(heap) < n:
                heapq.heappush(heap, entry)
            else:
                if entry > heap[0]:
                    heapq.heapreplace(heap, entry)

    if not heap:
        return pd.DataFrame(columns=["GPS", "example_seed", "y"])

    # Extract rows and sort by key to get deterministic random ordering
    # (smallest key first = “random” order derived from key)
    rows_with_key = [(-neg_key, -neg_gps, row) for (neg_key, neg_gps, row) in heap]

    # sort by (key, gps) for deterministic “random” order with tie-break
    rows_with_key.sort(key=lambda t: (t[0], t[1]))

    selected = np.stack([row for _, __, row in rows_with_key], axis=0)
    out = pd.DataFrame(selected, columns=list(usecols))
    return out.drop(columns=["split_id"])


def write_fixed_size_split_csvs_from_allocated(
    allocated_csv: str,
    out_dir: str,
    n_train: int = 240_000,
    n_val: int = 30_000,
    n_test: int = 30_000,
    seed: int = 2026,
):
    """
    Create small fixed-size split CSVs (train/val/test) from the big allocated CSV.
    Preserves your deterministic labels/seeds from the allocator.
    """
    os.makedirs(out_dir, exist_ok=True)

    train_df = load_split_from_allocated_csv(allocated_csv, "train", n=n_train, seed=seed)
    val_df   = load_split_from_allocated_csv(allocated_csv, "val",   n=n_val,   seed=seed + 1)
    test_df  = load_split_from_allocated_csv(allocated_csv, "test",  n=n_test,  seed=seed + 2)

    # Keep types clean/stable
    for df in (train_df, val_df, test_df):
        if len(df) == 0:
            raise ValueError("One split came back empty. Check allocated CSV and split sizes.")
        df["GPS"] = df["GPS"].astype(np.int64)
        df["example_seed"] = df["example_seed"].astype(np.uint32)
        df["y"] = df["y"].astype(np.uint8)

    # Sort by GPS for faster GWOSC file reuse during precompute (important)
    # train_df = train_df.sort_values("GPS").reset_index(drop=True)
    # val_df   = val_df.sort_values("GPS").reset_index(drop=True)
    # test_df  = test_df.sort_values("GPS").reset_index(drop=True)

    train_path = os.path.join(out_dir, f"train_{n_train//1_000_000}M.csv")
    val_path   = os.path.join(out_dir, f"val_{n_val//1000}k.csv")
    test_path  = os.path.join(out_dir, f"test_{n_test//1000}k.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    return {"train": train_path, "val": val_path, "test": test_path}


# from Data_Generation.Sampling_Module import write_splits_csv
# path = write_splits_csv(
#     out_dir="splits",
#     file_name="network_segments_list.json",
#     seglen=8,
#     padding=30,
#     n_val=30_000,
#     n_test=30_000,
#     seed=2026,
#     class_probs=(1/3, 1/3, 1/3),
#     # compression="gzip",  # recommended for size
# )
# print("Wrote:", path)



