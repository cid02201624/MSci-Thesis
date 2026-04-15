"""
Author: Saskia Knight
Date: January 2026
Description: 
1. Find the GPS times when data is good quality, has no events, and is available for both H1 and L1. 
2. Randomly select segments within approved times, download strain data, process and add injections.
"""

# Basics
from __future__ import annotations
from multiprocessing.util import debug
import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, tf2sos, sosfiltfilt
from functools import lru_cache
import os
from pathlib import Path
import json

# GW Related
from gwpy.timeseries import TimeSeries

# ML Related
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import time
import random
import requests
from requests.exceptions import RequestException, HTTPError, Timeout, ConnectionError
import glob

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import matplotlib as mpl

# mpl.rcParams["text.usetex"] = False
# mpl.rcParams["font.family"] = "serif"
# mpl.rcParams["font.serif"] = ["DejaVu Serif"]


# Go up 2 levels: New_version → Data_Generation 
# PROJECT_ROOT = Path.cwd().parents[1]
# sys.path.insert(0, str(PROJECT_ROOT))

# Personal Modules
from Training_Data_Generation.Simulation import SignalGenerator, GlitchGenerator
# from Training_Data_Generation.Sampling import segment_lists
# from Notching_Module import notch_bandpass_lines

# AUXILIARY FUNCTIONS
def zero_pad_timeseries(ts: TimeSeries, pad_seconds: float) -> TimeSeries:
    """
    Zero-pad a GwPy TimeSeries on both sides.
    """

    # Fast sample rate extraction
    fs = 1.0 / float(ts.dt.value)
    pad_samples = int(pad_seconds * fs)

    if pad_samples <= 0:
        return ts  # nothing to do

    n = len(ts)
    new_len = n + 2 * pad_samples

    # Preallocate once (fastest possible method)
    new_data = np.zeros(new_len, dtype=ts.value.dtype)

    # Insert original data into center
    new_data[pad_samples:pad_samples + n] = ts.value

    # Shift start time backward (use float GPS seconds — fastest + safest)
    new_t0 = ts.t0.value - pad_seconds

    return TimeSeries(new_data, t0=new_t0, dt=ts.dt, unit=ts.unit)


def zero_like(ts: TimeSeries) -> TimeSeries:
    return TimeSeries(np.zeros(len(ts), dtype=ts.value.dtype), t0=ts.t0, dt=ts.dt, unit=ts.unit)


# def scipy_bandpass_notch(ts, fs=4096, low=20, high=1700,
#                          notches=(60, 120, 180, 300, 505, 600, 1000), q=30, order=4): 

#     x = ts.value.astype(float)
#     x = x - x.mean()  # important

#     # bandpass as SOS
#     sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")

#     # append notch SOS sections
#     if notches is not None:
#         for f0 in notches:
#             b, a = iirnotch(w0=f0, Q=q, fs=fs)
#             sos_notch = tf2sos(b, a)
#             sos = np.vstack([sos, sos_notch])
#     y = sosfiltfilt(sos, x)

#     # preserve start time + sample spacing
#     return TimeSeries(y, t0=ts.t0, dt=ts.dt, unit=ts.unit)

@lru_cache(maxsize=32)
def _cached_sos(fs, low, high, notches, q, order):
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    if notches is not None:
        for f0 in notches:
            b, a = iirnotch(w0=f0, Q=q, fs=fs)
            sos_notch = tf2sos(b, a)
            sos = np.vstack([sos, sos_notch])
    return sos


def scipy_bandpass_notch(ts, fs=4096, low=20, high=1700,
                         notches=(60, 120, 180, 300, 505, 600, 1000), q=30, order=4): 

    x = ts.value.astype(float)
    x = x - x.mean()  # important

    notches_key = None if notches is None else tuple(float(f) for f in notches)
    sos = _cached_sos(float(fs), float(low), float(high), notches_key, float(q), int(order))

    y = sosfiltfilt(sos, x)
    return TimeSeries(y, t0=ts.t0, dt=ts.dt, unit=ts.unit)


SECONDS_PER_DAY = 86400.0
SECONDS_PER_WEEK = 7 * SECONDS_PER_DAY

def gps_to_cyclical_time_features(gps_start: float, include_dow: bool = False) -> np.ndarray:
    """
    Convert GPS seconds to cyclic features.

    Returns:
      if include_dow=False:  [sin(tod), cos(tod)]                shape (2,)
      if include_dow=True:   [sin(tod), cos(tod), sin(dow), cos(dow)] shape (4,)

    Notes:
    - Uses only modulo arithmetic -> no encoding of long-term calendar drift.
    - This is NOT civil local time; it's a stable periodic proxy.
    """
    # time-of-day phase
    tod_phase = 2.0 * np.pi * ((gps_start % SECONDS_PER_DAY) / SECONDS_PER_DAY)
    feats = [np.sin(tod_phase), np.cos(tod_phase)]

    if include_dow:
        dow_phase = 2.0 * np.pi * ((gps_start % SECONDS_PER_WEEK) / SECONDS_PER_WEEK)
        feats += [np.sin(dow_phase), np.cos(dow_phase)]

    return np.asarray(feats, dtype=np.float32)


def _to_fixed_spectrogram(qgram, out_f=256, out_t=256, log_eps=1e-12):
    """
    Convert gwpy QGram / Spectrogram-like object to a fixed-size torch.Tensor [F, T].
    """
    # gwpy spectrogram-like objects typically expose .value as a 2D numpy array
    arr = np.asarray(qgram.value, dtype=np.float32)#.T #transpose!

    arr = np.abs(arr)
    # log scaling (common for TF images)
    arr = np.log10(arr + log_eps)

    # normalise per image (optional but often helps)
    m = arr.mean()
    s = arr.std() + 1e-6
    arr = (arr - m) / s

    # resize to fixed shape using torch interpolate
    x = torch.from_numpy(arr)[None, None, ...]  # [1,1,F,T]
    x = F.interpolate(x, size=(out_f, out_t), mode="bilinear", align_corners=False)
    return x[0, 0]  # [F,T]


def _scan_existing_shards(out_dir: str):
    shard_paths = sorted(glob.glob(os.path.join(out_dir, "shard_*.pt")))
    shard_files, shard_lengths = [], []
    total = 0

    for p in shard_paths:
        try:
            payload = torch.load(p, map_location="cpu")
            n = int(payload["y"].numel())
            shard_files.append(os.path.basename(p))
            shard_lengths.append(n)
            total += n
        except Exception as e:
            # Corrupt/incomplete shard -> ignore it (or move it aside)
            print(f"[resume] ignoring unreadable shard: {p} ({e})", flush=True)
            # optional: os.rename(p, p + ".corrupt")
            break

    return shard_files, shard_lengths, total



_CLASS_NAME_MAP = {0: "noise", 1: "glitch", 2: "GW"}

def _safe_float(x):
    return None if x is None else float(x)


def _build_sample_metadata_row(gps, example_seed, y, inj_meta):
    """
    Normalised row for the sidecar CSV.
    final_snr:
    - noise  -> blank
    - glitch -> single-detector SNR actually injected
    - GW     -> network SNR (rho_net) after scaling
    """
    row = {
        "GPS": int(gps),
        "example_seed": None if example_seed is None else int(example_seed),
        "y": int(y),
        "class_type": _CLASS_NAME_MAP[int(y)],

        "final_snr": None,

        "chirp_mass": None,
        "merger_family": "",
        "approximant": "",

        "rho_H1": None,
        "rho_L1": None,
        "rho_net": None,

        "glitch_type": "",
        "glitch_detector": "",
    }

    if not inj_meta:
        return row

    if int(y) == 1:
        row.update({
            "final_snr": _safe_float(inj_meta.get("final_snr")),
            "glitch_type": inj_meta.get("glitch_type", ""),
            "glitch_detector": inj_meta.get("detector", ""),
        })

    elif int(y) == 2:
        row.update({
            "final_snr": _safe_float(inj_meta.get("rho_net")),
            "chirp_mass": _safe_float(inj_meta.get("chirp_mass")),
            "merger_family": inj_meta.get("source_class", ""),
            "approximant": inj_meta.get("approximant", ""),
            "rho_H1": _safe_float(inj_meta.get("rhoH")),
            "rho_L1": _safe_float(inj_meta.get("rhoL")),
            "rho_net": _safe_float(inj_meta.get("rho_net")),
        })

    return row





# MAIN FUNCTION
class QTransformDataset(Dataset):
    """
    Returns (X, t_feat, y) where:
      X: torch.FloatTensor [2, F, T]
      t_feat: torch.FloatTensor [2] (sin/cos tod)
      y: torch.LongTensor scalar in {0=noise, 1=glitch, 2=signal}
    """

    def __init__(
        self,
        segment_df: pd.DataFrame,
        seglen=8,
        sample_rate=4096,
        padding=30,
        qrange=(3, 100),
        frange=(20, 300),
        out_f=256,
        out_t=256,
        class_probs=(1/3, 1/3, 1/3),  # only used if no "y" column
        cache=False,                  # IMPORTANT for storage limits
        return_metadata=False,
        metadata_only=False,
        visualise=False,
    ):
        self.segment_df = segment_df.reset_index(drop=True)
        self.seglen = seglen
        self.sample_rate = sample_rate
        self.padding = padding

        self.qrange = qrange
        self.frange = frange
        self.out_f = out_f
        self.out_t = out_t

        self.cache = bool(cache)

        p = np.array(class_probs, dtype=np.float64)
        self.class_probs = p / p.sum()

        # Column conventions:
        # - if "example_seed" exists -> deterministic injection params
        # - if "y" exists -> deterministic label
        self.has_seed = "example_seed" in self.segment_df.columns
        self.has_y = "y" in self.segment_df.columns

        self.return_metadata = bool(return_metadata)
        self.metadata_only = bool(metadata_only)

        self.visualise = bool(visualise)

    def __len__(self):
        return len(self.segment_df)

    def _row_rng(self, row, idx: int) -> np.random.Generator:
        if self.has_seed:
            seed = int(row["example_seed"])
        else:
            # fallback deterministic seed (still reproducible)
            seed = (1234 + idx) & 0xFFFFFFFF
        # SeedSequence avoids platform-dependent hashing
        ss = np.random.SeedSequence([seed])
        return np.random.default_rng(ss)

    def _fetch_and_preprocess(self, ifo: str, gps: float) -> TimeSeries:
        max_retries = 6
        base_sleep = 1.0  # seconds

        last_err = None
        for attempt in range(max_retries):
            try:
                raw = TimeSeries.fetch_open_data(
                    ifo,
                    gps - self.padding,
                    gps + self.seglen + self.padding,
                    sample_rate=self.sample_rate,
                    cache=self.cache,
                )
                x = raw.copy()
                x = scipy_bandpass_notch(x, fs=self.sample_rate)
                return x

            except HTTPError as e:
                last_err = e
                status = getattr(getattr(e, "response", None), "status_code", None)

                # Retry only transient HTTPs
                if status not in (429, 500, 502, 503, 504):
                    raise

            except (Timeout, ConnectionError, RequestException) as e:
                last_err = e

            # exponential backoff + jitter
            sleep_s = base_sleep * (2 ** attempt) + random.uniform(0, 0.5)
            print(
                f"[fetch retry] ifo={ifo} gps={gps} attempt={attempt+1}/{max_retries} "
                f"sleep={sleep_s:.2f}s err={type(last_err).__name__}: {last_err}",
                flush=True,
            )
            time.sleep(sleep_s)

        raise RuntimeError(f"Failed to fetch {ifo} at gps={gps} after {max_retries} retries") from last_err

    def _estimate_asd(self, ts: TimeSeries):
        return ts.asd(
            fftlength=self.seglen,
            overlap=self.seglen / 2,
            window="hann",
            method="median",
        )

    # def _make_injection(self, y: int, H1_noise, L1_noise, gps_H1: float, gps_L1: float, H1_asd, L1_asd, H1_dt, L1_dt, rng: np.random.Generator):
    #     """
    #     Deterministic injection using rng-derived seeds.
    #     """
    #     if y == 0:
    #         # return zero_like(TimeSeries(np.zeros(len(H1_asd)*2 - 2), t0=gps_H1, dt=H1_dt)), \
    #         #        zero_like(TimeSeries(np.zeros(len(L1_asd)*2 - 2), t0=gps_L1, dt=L1_dt))
    #         H1_inj_pro = zero_like(H1_noise)
    #         L1_inj_pro = zero_like(L1_noise)
    #         return H1_inj_pro, L1_inj_pro

    #     inj_seed = int(rng.integers(1, 2**31 - 1))

    #     if y == 1:
    #         g = GlitchGenerator(
    #             sample_rate=self.sample_rate,
    #             seglen=self.seglen,
    #             snr_min=4,
    #             snr_max=20,
    #             snr_dist="loguniform",
    #             seed=inj_seed,
    #         )
    #         H1_inj, L1_inj = g.generate(asd_H1=H1_asd, asd_L1=L1_asd, epoch=gps_H1)

    #         # Force time alignment
    #         H1_inj = TimeSeries(H1_inj.value*10, t0=gps_H1, dt=H1_dt)
    #         L1_inj = TimeSeries(L1_inj.value*10, t0=gps_L1, dt=L1_dt)

    #     elif y == 2:
    #         s = SignalGenerator(
    #             sample_rate=self.sample_rate,
    #             seglen=self.seglen,
    #             f_lower=20,
    #             source_class=None,
    #             snr_enabled=True,
    #             veto_enabled=False,
    #             seed=inj_seed,
    #         )
    #         H1_inj, L1_inj = s.generate(
    #             epoch=gps_H1,
    #             asd_H1=H1_asd,
    #             asd_L1=L1_asd,
    #             target_snr=None,      # let generator draw from snr_min..snr_max
    #             coa_time=None,
    #             return_metadata=False,
    #         )

    #         H1_inj = TimeSeries(H1_inj.value, t0=gps_H1, dt=H1_dt)
    #         L1_inj = TimeSeries(L1_inj.value, t0=gps_L1, dt=L1_dt)

    #     else:
    #         raise ValueError(f"Unknown class y={y}")

    #     H1_inj = zero_pad_timeseries(H1_inj, self.padding)
    #     L1_inj = zero_pad_timeseries(L1_inj, self.padding)

    #     H1_inj_pro = scipy_bandpass_notch(H1_inj.copy(), fs=self.sample_rate)
    #     L1_inj_pro = scipy_bandpass_notch(L1_inj.copy(), fs=self.sample_rate)

    #     return H1_inj_pro, L1_inj_pro

    def _make_injection(
        self,
        y: int,
        H1_noise,
        L1_noise,
        gps_H1: float,
        gps_L1: float,
        H1_asd,
        L1_asd,
        H1_dt,
        L1_dt,
        rng: np.random.Generator,
        return_metadata: bool = False,
    ):
        """
        Deterministic injection using rng-derived seeds.
        """
        inj_meta = None

        if y == 0:
            H1_inj_pro = zero_like(H1_noise)
            L1_inj_pro = zero_like(L1_noise)

            if return_metadata:
                return H1_inj_pro, L1_inj_pro, {"class_type": "noise"}

            return H1_inj_pro, L1_inj_pro

        inj_seed = int(rng.integers(1, 2**31 - 1))

        if y == 1:
            glitch_amp_boost = 10.0 

            g = GlitchGenerator(
                sample_rate=self.sample_rate,
                seglen=self.seglen,
                snr_min=4,
                snr_max=20,
                snr_dist="loguniform",
                seed=inj_seed,
            )

            if return_metadata:
                H1_inj, L1_inj, inj_meta = g.generate(
                    asd_H1=H1_asd,
                    asd_L1=L1_asd,
                    epoch=gps_H1,
                    return_metadata=True,
                )
            else:
                H1_inj, L1_inj = g.generate(
                    asd_H1=H1_asd,
                    asd_L1=L1_asd,
                    epoch=gps_H1,
                )

            # Force time alignment + preserve your extra x10 glitch scaling
            H1_inj = TimeSeries(H1_inj.value * glitch_amp_boost, t0=gps_H1, dt=H1_dt)
            L1_inj = TimeSeries(L1_inj.value * glitch_amp_boost, t0=gps_L1, dt=L1_dt)

            if return_metadata and inj_meta is not None:
                if inj_meta.get("target_snr") is not None:
                    inj_meta["target_snr"] = float(inj_meta["target_snr"]) * glitch_amp_boost
                if inj_meta.get("final_snr") is not None:
                    inj_meta["final_snr"] = float(inj_meta["final_snr"]) * glitch_amp_boost

        elif y == 2:
            s = SignalGenerator(
                sample_rate=self.sample_rate,
                seglen=self.seglen,
                f_lower=20,
                source_class=None,
                snr_enabled=True,
                veto_enabled=False,
                seed=inj_seed,
            )

            if return_metadata:
                H1_inj, L1_inj, inj_meta = s.generate(
                    epoch=gps_H1,
                    asd_H1=H1_asd,
                    asd_L1=L1_asd,
                    target_snr=None,
                    coa_time=None,
                    return_metadata=True,
                )
            else:
                H1_inj, L1_inj = s.generate(
                    epoch=gps_H1,
                    asd_H1=H1_asd,
                    asd_L1=L1_asd,
                    target_snr=None,
                    coa_time=None,
                    return_metadata=False,  # we can ignore this if not needed
                )

            H1_inj = TimeSeries(H1_inj.value, t0=gps_H1, dt=H1_dt)
            L1_inj = TimeSeries(L1_inj.value, t0=gps_L1, dt=L1_dt)

        else:
            raise ValueError(f"Unknown class y={y}")

        H1_inj = zero_pad_timeseries(H1_inj, self.padding)
        L1_inj = zero_pad_timeseries(L1_inj, self.padding)

        H1_inj_pro = scipy_bandpass_notch(H1_inj.copy(), fs=self.sample_rate)
        L1_inj_pro = scipy_bandpass_notch(L1_inj.copy(), fs=self.sample_rate)

        if return_metadata:
            return H1_inj_pro, L1_inj_pro, inj_meta

        return H1_inj_pro, L1_inj_pro

    def __getitem__(self, idx: int):
        row = self.segment_df.iloc[idx]
        gps = float(row.GPS)
        gps_H1 = gps
        gps_L1 = gps

        rng = self._row_rng(row, idx)

        # label: deterministic if column exists
        if self.has_y:
            y = int(row["y"])
        else:
            y = int(rng.choice([0, 1, 2], p=self.class_probs))

        example_seed = int(row["example_seed"]) if self.has_seed else None

        H1_processed = self._fetch_and_preprocess("H1", gps_H1)
        L1_processed = self._fetch_and_preprocess("L1", gps_L1)

        H1_asd = self._estimate_asd(H1_processed)
        L1_asd = self._estimate_asd(L1_processed)

        need_meta = self.return_metadata or self.metadata_only

        if need_meta:
            H1_inj_pro, L1_inj_pro, inj_meta = self._make_injection(
                y=y,
                H1_noise=H1_processed,
                L1_noise=L1_processed,
                gps_H1=gps_H1,
                gps_L1=gps_L1,
                H1_asd=H1_asd,
                L1_asd=L1_asd,
                H1_dt=H1_processed.dt,
                L1_dt=L1_processed.dt,
                rng=rng,
                return_metadata=True,
            )
        else:
            H1_inj_pro, L1_inj_pro = self._make_injection(
                y=y,
                H1_noise=H1_processed,
                L1_noise=L1_processed,
                gps_H1=gps_H1,
                gps_L1=gps_L1,
                H1_asd=H1_asd,
                L1_asd=L1_asd,
                H1_dt=H1_processed.dt,
                L1_dt=L1_processed.dt,
                rng=rng,
                return_metadata=False,
            )
            inj_meta = None

        if self.metadata_only:
            meta_row = _build_sample_metadata_row(
                gps=gps,
                example_seed=example_seed,
                y=y,
                inj_meta=inj_meta,
            )
            return json.dumps(meta_row, separators=(",", ":"))

        H1_strain = H1_processed + H1_inj_pro
        L1_strain = L1_processed + L1_inj_pro

        # crop edges (reduce filter artifacts)
        H1_strain = H1_strain.crop(gps_H1 - self.padding + 1, gps_H1 + self.seglen + self.padding - 1)
        L1_strain = L1_strain.crop(gps_L1 - self.padding + 1, gps_L1 + self.seglen + self.padding - 1)

        H1_whitened = H1_strain.whiten(asd=H1_asd, pad=self.padding, remove_corrupted=True)
        L1_whitened = L1_strain.whiten(asd=L1_asd, pad=self.padding, remove_corrupted=True)

        H1_whitened = H1_whitened.crop(gps_H1 - 0.1, gps_H1 + self.seglen + 0.1)
        L1_whitened = L1_whitened.crop(gps_L1 - 0.1, gps_L1 + self.seglen + 0.1)

        H1_q = H1_whitened.q_transform(
            qrange=self.qrange,
            frange=self.frange,
            whiten=False,
        ).crop(gps_H1, gps_H1 + self.seglen)

        L1_q = L1_whitened.q_transform(
            qrange=self.qrange,
            frange=self.frange,
            whiten=False,
        ).crop(gps_L1, gps_L1 + self.seglen)

        if self.visualise and need_meta:
            return H1_q, L1_q, inj_meta

        H1_img = _to_fixed_spectrogram(H1_q, out_f=self.out_f, out_t=self.out_t)
        L1_img = _to_fixed_spectrogram(L1_q, out_f=self.out_f, out_t=self.out_t)

        t_feat = gps_to_cyclical_time_features(gps_H1, include_dow=False).astype(np.float32)
        t_feat = torch.from_numpy(t_feat)

        X = torch.stack([H1_img, L1_img], dim=0)
        y_tensor = torch.tensor(y, dtype=torch.long)

        if self.return_metadata:
            meta_row = _build_sample_metadata_row(
                gps=gps,
                example_seed=example_seed,
                y=y,
                inj_meta=inj_meta,
            )
            return X, t_feat, y_tensor, json.dumps(meta_row, separators=(",", ":"))

        return X, t_feat, y_tensor




def make_dataloader(
    segment_df: pd.DataFrame,
    batch_size=64,
    num_workers=4,
    pin_memory=True,
    shuffle=True,
    cache=False,
    loader_seed: int = 1234,
    drop_last: bool = False,
):
    ds = QTransformDataset(segment_df=segment_df, cache=cache)
    g = torch.Generator()
    g.manual_seed(loader_seed)  # deterministic shuffling if you want it

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        generator=g,
    )


def precompute_split_to_pt_shards(
    segment_df: pd.DataFrame,
    out_dir: str,
    *,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = False,   # CPU-side precompute; pinning usually not needed
    cache: bool = True,         # enable GWPy/GWOSC cache during precompute
    loader_seed: int = 2026,
    x_dtype: torch.dtype = torch.float16,  # saves lots of space and I/O
    seglen: int = 8,
    sample_rate: int = 4096,
    padding: int = 30,
    qrange=(3, 100),
    frange=(20, 300),
    out_f: int = 256,
    out_t: int = 256,
):
    """
    Materialise a split as shard_XXXXX.pt files.
    Each shard contains:
      {"X": [B,2,F,T], "t_feat": [B,2], "y": [B]}
    plus manifest.pt with shard metadata.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Resume from existing shards if present
    existing_files, existing_lengths, already_done = _scan_existing_shards(out_dir)
    if already_done > 0:
        print(f"[resume] found {len(existing_files)} shards, already_done={already_done}", flush=True)

    # If everything is already done, just rebuild manifest and return
    if already_done >= len(segment_df):
        manifest = {
            "files": existing_files,
            "lengths": existing_lengths,
            "n_total": already_done,
            "x_dtype": str(x_dtype),
            "shape_per_sample": [2, out_f, out_t],
            "seglen": seglen,
            "sample_rate": sample_rate,
            "padding": padding,
            "qrange": tuple(qrange),
            "frange": tuple(frange),
        }
        torch.save(manifest, os.path.join(out_dir, "manifest.pt"))
        print(f"[resume] dataset already complete: {already_done} samples", flush=True)
        return os.path.join(out_dir, "manifest.pt")

    # Resume on remaining rows only
    segment_df = segment_df.iloc[already_done:].reset_index(drop=True)
    shard_idx_offset = len(existing_files)

    ds = QTransformDataset(
        segment_df=segment_df,
        seglen=seglen,
        sample_rate=sample_rate,
        padding=padding,
        qrange=qrange,
        frange=frange,
        out_f=out_f,
        out_t=out_t,
        cache=cache,
    )

    g = torch.Generator()
    g.manual_seed(loader_seed)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,  # IMPORTANT: keep GPS-sorted order for fetch locality
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        generator=g,
    )

    shard_files = list(existing_files)
    shard_lengths = list(existing_lengths)
    total = int(already_done)

    for local_shard_idx, (X, t_feat, y) in enumerate(loader):
        shard_idx = shard_idx_offset + local_shard_idx

        if x_dtype is not None:
            X = X.to(dtype=x_dtype)
        t_feat = t_feat.to(dtype=torch.float32)
        y = y.to(dtype=torch.long)

        payload = {
            "X": X.contiguous(),
            "t_feat": t_feat.contiguous(),
            "y": y.contiguous(),
        }

        shard_name = f"shard_{shard_idx:05d}.pt"
        shard_path = os.path.join(out_dir, shard_name)
        tmp_path = shard_path + ".tmp"

        torch.save(payload, tmp_path)
        os.replace(tmp_path, shard_path)  # atomic rename

        n_this = int(y.numel())
        shard_files.append(shard_name)
        shard_lengths.append(n_this)
        total += n_this

        # write partial manifest every shard (cheap, useful for crash recovery)
        partial_manifest = {
            "files": shard_files,
            "lengths": shard_lengths,
            "n_total": total,
            "x_dtype": str(x_dtype),
            "shape_per_sample": [2, out_f, out_t],
            "seglen": seglen,
            "sample_rate": sample_rate,
            "padding": padding,
            "qrange": tuple(qrange),
            "frange": tuple(frange),
        }
        torch.save(partial_manifest, os.path.join(out_dir, "manifest.partial.pt"))

        print(f"[{out_dir}] wrote {shard_name} ({n_this} samples) | total={total}", flush=True)

    final_manifest = {
        "files": shard_files,
        "lengths": shard_lengths,
        "n_total": total,
        "x_dtype": str(x_dtype),
        "shape_per_sample": [2, out_f, out_t],
        "seglen": seglen,
        "sample_rate": sample_rate,
        "padding": padding,
        "qrange": tuple(qrange),
        "frange": tuple(frange),
    }
    tmp_manifest = os.path.join(out_dir, "manifest.pt.tmp")
    final_manifest_path = os.path.join(out_dir, "manifest.pt")
    torch.save(final_manifest, tmp_manifest)
    os.replace(tmp_manifest, final_manifest_path)

    return os.path.join(out_dir, "manifest.pt")


class PrecomputedPTShardDataset(Dataset):
    """
    Lazy loader for shard_XXXXX.pt files written by precompute_split_to_pt_shards.
    Keeps one shard in memory at a time.
    """
    def __init__(self, shard_dir: str, cast_x_to_float32: bool = True, max_samples: int | None = None):
        self.shard_dir = Path(shard_dir)
        # manifest_path = self.shard_dir / "manifest.partial.pt"  #I CHANGED HERE
        manifest_path = self.shard_dir / "manifest.pt"
        if not manifest_path.exists():
            manifest_path = self.shard_dir / "manifest.partial.pt"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")

        manifest = torch.load(manifest_path, map_location="cpu")

        files = [self.shard_dir / f for f in manifest["files"]]
        lengths = np.asarray(manifest["lengths"], dtype=np.int64)

        if max_samples is not None:
            max_samples = int(max_samples)
            cum = np.cumsum(lengths)

            if max_samples < cum[-1]:
                cut_shard = int(np.searchsorted(cum, max_samples, side="right"))
                prev_cum = 0 if cut_shard == 0 else int(cum[cut_shard - 1])
                keep_in_last = max_samples - prev_cum

                files = files[: cut_shard + 1]
                lengths = lengths[: cut_shard + 1].copy()
                lengths[-1] = keep_in_last

        self.files = files
        self.lengths = lengths
        self.cum = np.cumsum(self.lengths)
        self.n_total = int(self.lengths.sum())
        self.cast_x_to_float32 = bool(cast_x_to_float32)

        self._cache_shard_idx = None
        self._cache = None

    def __len__(self):
        return self.n_total

    def _load_shard(self, shard_idx: int):
        if self._cache_shard_idx != shard_idx:
            self._cache = torch.load(self.files[shard_idx], map_location="cpu")
            self._cache_shard_idx = shard_idx

    def __getitem__(self, idx: int):
        if idx < 0:
            idx += self.n_total
        if idx < 0 or idx >= self.n_total:
            raise IndexError(idx)

        shard_idx = int(np.searchsorted(self.cum, idx, side="right"))
        prev_cum = 0 if shard_idx == 0 else int(self.cum[shard_idx - 1])
        local_idx = idx - prev_cum

        self._load_shard(shard_idx)
        X = self._cache["X"][local_idx]
        t_feat = self._cache["t_feat"][local_idx]
        y = self._cache["y"][local_idx]

        if self.cast_x_to_float32 and X.dtype != torch.float32:
            X = X.float()

        return X, t_feat, y


# def write_split_metadata_csv(
#     segment_df: pd.DataFrame,
#     out_csv: str,
#     *,
#     batch_size: int = 64,
#     num_workers: int = 4,
#     cache: bool = True,
#     loader_seed: int = 2026,
#     seglen: int = 8,
#     sample_rate: int = 4096,
#     padding: int = 30,
#     qrange=(3, 100),
#     frange=(20, 300),
#     out_f: int = 256,
#     out_t: int = 256,
# ):
#     """
#     Build a sidecar CSV for one split (typically TEST only), using the same
#     deterministic GPS/example_seed/y rows as the main dataset.

#     Output columns:
#       GPS, example_seed, y, class_type, final_snr, chirp_mass,
#       merger_family, approximant, rho_H1, rho_L1, rho_net,
#       glitch_type, glitch_detector, sample_index
#     """
#     os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
#     if os.path.exists(out_csv):
#         os.remove(out_csv)

#     ds = QTransformDataset(
#         segment_df=segment_df,
#         seglen=seglen,
#         sample_rate=sample_rate,
#         padding=padding,
#         qrange=qrange,
#         frange=frange,
#         out_f=out_f,
#         out_t=out_t,
#         cache=cache,
#         return_metadata=False,
#         metadata_only=True,
#     )

#     g = torch.Generator()
#     g.manual_seed(loader_seed)

#     loader = DataLoader(
#         ds,
#         batch_size=batch_size,
#         shuffle=False,   # keep same order as segment_df / test CSV
#         drop_last=False,
#         num_workers=num_workers,
#         pin_memory=False,
#         persistent_workers=(num_workers > 0),
#         prefetch_factor=2 if num_workers > 0 else None,
#         generator=g,
#     )

#     wrote_header = False
#     sample_index = 0

#     for meta_json_batch in loader:
#         rows = [json.loads(s) for s in meta_json_batch]

#         for r in rows:
#             r["sample_index"] = sample_index
#             sample_index += 1

#         pd.DataFrame(rows).to_csv(
#             out_csv,
#             mode="a",
#             header=(not wrote_header),
#             index=False,
#         )
#         wrote_header = True

#     return out_csv

def write_split_metadata_csv(
    segment_df: pd.DataFrame,
    out_csv: str,
    *,
    batch_size: int = 64,
    num_workers: int = 4,
    cache: bool = True,
    loader_seed: int = 2026,
    seglen: int = 8,
    sample_rate: int = 4096,
    padding: int = 30,
    qrange=(3, 100),
    frange=(20, 300),
    out_f: int = 256,
    out_t: int = 256,
    resume: bool = True,
):
    """
    Build a sidecar CSV for one split (typically TEST only), using the same
    deterministic GPS/example_seed/y rows as the main dataset.

    If resume=True and out_csv already exists, continue from the first
    unwritten sample instead of starting over.
    """
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    already_done = 0
    wrote_header = False

    if resume and os.path.exists(out_csv):
        # Count completed rows already on disk
        existing = pd.read_csv(out_csv, usecols=["sample_index"])
        already_done = len(existing)
        wrote_header = already_done > 0

        if already_done > len(segment_df):
            raise ValueError(
                f"Existing metadata file has {already_done} rows, "
                f"but segment_df has only {len(segment_df)} rows."
            )

        if already_done == len(segment_df):
            print(f"[resume] metadata already complete: {already_done} samples", flush=True)
            return out_csv

        print(f"[resume] metadata file has {already_done} completed samples", flush=True)

        # Skip completed rows
        segment_df = segment_df.iloc[already_done:].reset_index(drop=True)

    else:
        # Fresh run: remove any old file only if not resuming
        if os.path.exists(out_csv):
            os.remove(out_csv)

    ds = QTransformDataset(
        segment_df=segment_df,
        seglen=seglen,
        sample_rate=sample_rate,
        padding=padding,
        qrange=qrange,
        frange=frange,
        out_f=out_f,
        out_t=out_t,
        cache=cache,
        return_metadata=False,
        metadata_only=True,
    )

    g = torch.Generator()
    g.manual_seed(loader_seed)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        generator=g,
    )

    sample_index = already_done

    for meta_json_batch in loader:
        rows = [json.loads(s) for s in meta_json_batch]

        for r in rows:
            r["sample_index"] = sample_index
            sample_index += 1

        pd.DataFrame(rows).to_csv(
            out_csv,
            mode="a",
            header=(not wrote_header),
            index=False,
        )
        wrote_header = True

        print(f"[resume] wrote through sample_index={sample_index - 1}", flush=True)

    return out_csv

def write_test_metadata_csv_from_csv(
    test_csv_path: str,
    out_csv: str,
    **kwargs,
):
    test_df = pd.read_csv(test_csv_path)
    return write_split_metadata_csv(test_df, out_csv, **kwargs)