"""
Author: Saskia Knight
Date: January 2026
Description: 
1. Find the GPS times when data is good quality, has no events, and is available for both H1 and L1. 
2. Randomly select segments within approved times, outputting batches for each class.
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
from bisect import bisect_left

# GW Related
from gwosc.timeline import get_segments
from gwosc.datasets import query_events



Segment = Tuple[int, int]

def intersect_two(a: List[Segment], b: List[Segment], min_len: int = 0) -> List[Segment]:
    """
    Intersection of two segment lists. Keeps only overlapping parts.
    Assumes each list is a list of (start, end) with start < end.
    Works even if boundaries don't line up.
    """
    if not a or not b:
        return []

    a = sorted(a)
    b = sorted(b)

    i = j = 0
    out: List[Segment] = []

    while i < len(a) and j < len(b):
        a0, a1 = a[i]
        b0, b1 = b[j]

        start = max(a0, b0)
        end = min(a1, b1)

        if end > start:
            if end - start >= min_len:
                out.append((start, end))

        # Advance the one that ends first
        if a1 <= b1:
            i += 1
        else:
            j += 1

    return out

def intersect_many(lists: List[List[Segment]], min_len: int = 0) -> List[Segment]:
    """Intersection across many segment lists."""
    # If any list is empty => empty intersection
    if any(not lst for lst in lists):
        return []

    lists = [sorted(lst) for lst in lists]
    result = lists[0]
    for lst in lists[1:]:
        result = intersect_two(result, lst, min_len=0)  # filter at end
        if not result:
            break

    if min_len > 0:
        result = [(s, e) for s, e in result if (e - s) >= min_len]

    return result


def subtract_segments(good: List[Segment], bad: List[Segment]) -> List[Segment]:
    """
    Subtract bad segments from good segments.
    Returns pieces of good that do not overlap bad.
    """
    if not good:
        return []
    if not bad:
        return sorted(good)

    good = sorted(good)
    bad = sorted(bad)

    out: List[Segment] = []
    j = 0

    for gs, ge in good:
        cur = gs

        # advance bad until it might overlap
        while j < len(bad) and bad[j][1] <= cur:
            j += 1

        k = j
        while k < len(bad) and bad[k][0] < ge:
            bs, be = bad[k]

            # if there's a gap before this bad segment, keep it
            if bs > cur:
                out.append((cur, min(bs, ge)))

            # move cur past the bad segment
            if be > cur:
                cur = max(cur, be)

            if cur >= ge:
                break
            k += 1

        if cur < ge:
            out.append((cur, ge))

    return out


def filter_min_len(segs: List[Segment], min_len: int) -> List[Segment]:
    return [(s, e) for (s, e) in segs if (e - s) >= min_len]


def event_pad_seconds(m1, m2, ns_max=2.9):  # 2.9 = max NS mass
    # double check the justification behind these times - maybe plot spectrograms?
    if m1 is None or m2 is None: # Unclassified
        return 2
    elif m1 < ns_max and m2 < ns_max: # BNS
        return 8
    elif (m1 < ns_max) ^ (m2 < ns_max): # NSBH
        return 4
    else:  # BBH
        return 2


GWOSC_EVENT_DETAIL_URL = "https://gwosc.org/eventapi/json/event/{event_name}/v{version}/"
_EVENT_ID_RE = re.compile(r"^(?P<name>.+)-v(?P<ver>\d+)$") # magic DONT TOUCH

def fetch_event_detail(session: requests.Session, start=1368195220, end=1389456018):
    """
    Call GWOSC Event Portal API and return the GPS times and required padding for each event.
    """
    # Find event names within timeframe
    event_list = query_events(select=[f"{end} >= gps-time >= {start}"]) 

    GPS_list = []
    padding_list = []
    for event_id in event_list:

        # Split name into sections for URL building
        m = _EVENT_ID_RE.match(event_id.strip())
        event_name = m.group("name")
        version = int(m.group("ver"))

        url = GWOSC_EVENT_DETAIL_URL.format(event_name=event_name, version=version)

        r = session.get(url)
        r.raise_for_status()
        params = r.json()['events'][event_id]

        GPS = params['GPS']
        mass1 = params['mass_1_source']
        mass2 = params['mass_2_source']
        padding = event_pad_seconds(mass1, mass2)

        GPS_list.append(GPS)
        padding_list.append(padding)

    return GPS_list, padding_list


def get_event_veto_windows(start=1368195220, end=1389456018) -> List[Segment]:
    """
    Build veto windows around known GW event GPS times.
    Query events in [start, end] and convert to GPS.

    """
    session = requests.Session()
    GPS_list, padding_list = fetch_event_detail(session, start, end)

    veto = []
    for t, pad in zip(GPS_list, padding_list):
        veto.append((max(start, t - (0.9*pad)), min(end, t + (0.1*pad))))  # Event is not in the middle
    return veto


def get_science_segments(ifo, start=1368195220, end=1389456018, min_len: int = 68):
    # Data quality problems
    burst2 = get_segments(f"{ifo}_BURST_CAT2", start, end)  # includes burst1 & burst2
    cbc2   = get_segments(f"{ifo}_CBC_CAT2", start, end)    # includes cbc1 & cbc2
    # cw1    = get_segments(f"{ifo}_CW_CAT1", start, end)
    # stoch1 = get_segments(f"{ifo}_STOCH_CAT1", start, end)

    # Hardware injections (these are usually "NO_*" == good times)
    burst_hw_inj   = get_segments(f"{ifo}_NO_BURST_HW_INJ", start, end)
    cbc_hw_inj     = get_segments(f"{ifo}_NO_CBC_HW_INJ", start, end)
    # cw_hw_inj      = get_segments(f"{ifo}_NO_CW_HW_INJ", start, end)
    # detchar_hw_inj = get_segments(f"{ifo}_NO_DETCHAR_HW_INJ", start, end)
    # stoch_hw_inj   = get_segments(f"{ifo}_NO_STOCH_HW_INJ", start, end)

    # Intersection of all GOOD criteria
    segs = intersect_many([burst2, cbc2, 
                        #    cw1, stoch1, 
                           burst_hw_inj, cbc_hw_inj, 
                        #    cw_hw_inj, detchar_hw_inj, stoch_hw_inj
                           ])

    # 2) Remove windows around known GW events
    event_veto = get_event_veto_windows(start, end)
    segs = subtract_segments(segs, event_veto)

    # 3) Drop short leftovers
    segs = filter_min_len(segs, min_len)

    return segs

# Both detectors good
def get_network_science_segments(ifos=("H1", "L1"), start=1368195220, end=1389456018, min_len: int = 68, save=False):
    """
    Return segments where ALL ifos have good-quality data simultaneously.
    """
    per_ifo = [get_science_segments(ifo, start=start, end=end, min_len=min_len) for ifo in ifos]
    # intersect_many expects a list of segment lists
    net = intersect_many(per_ifo)
    net = filter_min_len(net, min_len)

    if save:
        filename = "network_segments_list.json"
        # convert tuples -> lists for JSON compatibility
        net_serialisable = [tuple(seg) for seg in net]

        with open(filename, "w") as f:
            json.dump(net_serialisable, f, indent=2)

    return net



# def random_start_time_with_padding(
#     segs: List[Segment],
#     padding: int = 30,
#     seglen: int = 8,
#     rng: Optional[np.random.Generator] = None,
# ) -> int:
#     """
#     Pick an integer start time t uniformly from the union of segments, with padding constraints:
#       - t is at least padding seconds after segment start
#       - t + seglen + padding is within the segment end
#     """
#     if rng is None:
#         rng = np.random.default_rng()

#     end_margin = seglen + padding

#     cleaned = []
#     prefix = []
#     total = 0

#     for a, b in segs:
#         start = min(a, b)
#         end = max(a, b)

#         lo_f = start + padding
#         hi_f = end - end_margin

#         lo = math.ceil(lo_f)
#         hi = math.floor(hi_f)

#         if lo > hi:
#             continue

#         length = hi - lo + 1
#         total += length
#         prefix.append(total)
#         cleaned.append((lo, hi))

#     if total == 0:
#         raise ValueError("No valid start times after applying padding constraints.")

#     # uniform integer in [1, total]
#     r = int(rng.integers(1, total + 1))
#     i = bisect_left(prefix, r)

#     prev = prefix[i - 1] if i > 0 else 0
#     lo, _hi = cleaned[i]
#     offset = r - prev - 1
#     return lo + offset


# def segment_lists(
#     samples: int,
#     require_coincident: bool = True,
#     start: int = 1368195220,
#     end: int = 1389456018,
#     min_len: int = 67,
#     file_name: str = "network_segments_list.json",
#     seed: Optional[int] = None,
#     unique: bool = True,
# ):
#     """
#     Deterministic segment sampler if seed is provided.
#     Returns:
#       - if require_coincident=True: (df, segs)
#       - else: df
#     """
#     rng = np.random.default_rng(seed)

#     ifos = ["H1", "L1"]

#     if require_coincident:
#         if not file_name:
#             segs = get_network_science_segments(ifos=ifos, start=start, end=end, min_len=min_len)
#         else:
#             with open(file_name) as f:
#                 segs = [tuple(seg) for seg in json.load(f)]

#         gps_times = []
#         seen = set()

#         while len(gps_times) < samples:
#             gps_time = random_start_time_with_padding(segs, padding=30, seglen=8, rng=rng)
#             if unique:
#                 if gps_time in seen:
#                     continue
#                 seen.add(gps_time)
#             gps_times.append(gps_time)

#         df = pd.DataFrame({"H1": gps_times, "L1": gps_times})
#         return df, segs

#     # non-coincident mode
#     gps_times = {ifo: [] for ifo in ifos}
#     for ifo in ifos:
#         segs = get_science_segments(ifo, start=start, end=end, min_len=min_len)
#         seen = set()
#         while len(gps_times[ifo]) < samples:
#             gps_time = random_start_time_with_padding(segs, padding=30, seglen=8, rng=rng)
#             if unique:
#                 if gps_time in seen:
#                     continue
#                 seen.add(gps_time)
#             gps_times[ifo].append(gps_time)

#     df = pd.DataFrame(gps_times)
#     return df



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

# def load_split_from_allocated_csv(
#     path: str,
#     split: str,
#     n: Optional[int] = None,
#     seed: int = 1234,
#     usecols=("GPS", "split_id", "example_seed", "y"),
#     chunksize: int = 1_000_000,
# ) -> pd.DataFrame:
#     """
#     Load rows for one split from the big allocated CSV.
#     If n is provided, returns a deterministic reservoir sample of size n.
#     """
#     split_id = _SPLIT_MAP[split]

#     if n is None:
#         parts = []
#         for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
#             c = chunk[chunk["split_id"] == split_id]
#             if not c.empty:
#                 parts.append(c)
#         if not parts:
#             return pd.DataFrame(columns=["GPS", "example_seed", "y"])
#         out = pd.concat(parts, ignore_index=True)
#         return out.drop(columns=["split_id"])

#     rng = np.random.default_rng(seed)
#     reservoir = None
#     seen = 0

#     for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
#         c = chunk[chunk["split_id"] == split_id]
#         if c.empty:
#             continue
#         arr = c.to_numpy()
#         for row in arr:
#             if seen < n:
#                 if reservoir is None:
#                     reservoir = np.empty((n, arr.shape[1]), dtype=arr.dtype)
#                 reservoir[seen] = row
#             else:
#                 j = int(rng.integers(0, seen + 1))
#                 if j < n:
#                     reservoir[j] = row
#             seen += 1

#     if reservoir is None:
#         return pd.DataFrame(columns=["GPS", "example_seed", "y"])

#     out = pd.DataFrame(reservoir[:min(n, seen)], columns=list(usecols))
#     return out.drop(columns=["split_id"])

from typing import Optional
import numpy as np
import pandas as pd
import heapq

# A fast, deterministic 64-bit mixer (SplitMix64 finalizer style)
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
    # Keep the n *smallest* keys. We use a max-heap of size n via (-key, row)
    # so we can quickly discard worse (larger-key) candidates.
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
    n_train: int = 40_000,
    n_val: int = 5_000,
    n_test: int = 5_000,
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
#     n_val=20_000,
#     n_test=20_000,
#     seed=2026,
#     class_probs=(1/3, 1/3, 1/3),
#     # compression="gzip",  # recommended for size
# )
# print("Wrote:", path)



