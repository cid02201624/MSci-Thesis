import os
import json
from pathlib import Path
import time
import random
from requests.exceptions import HTTPError, Timeout, ConnectionError, RequestException

import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, tf2sos, sosfiltfilt

import torch
import torch.nn.functional as F

from gwpy.timeseries import TimeSeries
from gwosc.timeline import get_segments
from gwosc.datasets import query_events, event_gps

from Model_9_Better_fusion.Training import JointConvNeXtGWWithTime, load_checkpoint


O4A_START = 1368195220
O4A_END   = 1389456018
SECONDS_PER_DAY = 86400.0
SECONDS_PER_WEEK = 7 * SECONDS_PER_DAY

def fetch_open_data_retry(ifo, start, end, sample_rate, max_retries=4):
    """
    Robust wrapper around gwpy TimeSeries.fetch_open_data.
    Retries transient network failures.
    """
    base_sleep = 1.0
    last_err = None

    for attempt in range(max_retries):
        try:
            return TimeSeries.fetch_open_data(
                ifo,
                start,
                end,
                sample_rate=sample_rate,
            )

        except HTTPError as e:
            last_err = e
            status = getattr(getattr(e, "response", None), "status_code", None)

            # only retry transient HTTPs
            if status not in (429, 500, 502, 503, 504):
                raise

        except (Timeout, ConnectionError, RequestException) as e:
            last_err = e

        sleep_s = base_sleep * (2 ** attempt) + random.uniform(0, 0.5)

        print(
            f"[retry] {ifo} attempt {attempt+1}/{max_retries} "
            f"sleep {sleep_s:.1f}s err={type(last_err).__name__}",
            flush=True,
        )

        time.sleep(sleep_s)

    raise RuntimeError(f"Failed to fetch {ifo} after {max_retries} retries") from last_err

def intersect_two(a, b):
    if not a or not b:
        return []

    a = sorted(a)
    b = sorted(b)

    i = j = 0
    out = []

    while i < len(a) and j < len(b):
        a0, a1 = a[i]
        b0, b1 = b[j]

        start = max(a0, b0)
        end = min(a1, b1)

        if end > start:
            out.append((start, end))

        if a1 <= b1:
            i += 1
        else:
            j += 1

    return out


def gps_in_segments(gps, segments):
    return any(start <= gps < end for start, end in segments)


def get_o4_events_with_h1_l1_data(save=False, filename="o4_events_h1_l1.json"):
    # Times when each detector has data
    h1 = get_segments("H1_DATA", O4A_START, O4A_END)
    l1 = get_segments("L1_DATA", O4A_START, O4A_END)

    # Times when both have data simultaneously
    both = intersect_two(h1, l1)

    # Events in the O4a time window
    event_list = query_events(select=[f"{O4A_END} >= gps-time >= {O4A_START}"])

    out = []
    for event in event_list:
        gps = event_gps(event)
        if gps_in_segments(gps, both):
            out.append((event, gps))

    out.sort(key=lambda x: x[1])

    if save:
        with open(filename, "w") as f:
            json.dump(out, f, indent=2)

    return out


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


def to_fixed_spectrogram(qgram, out_f=256, out_t=256, log_eps=1e-12):
    """
    Convert gwpy QGram / Spectrogram-like object to a fixed-size torch.Tensor [F, T].
    """
    # gwpy spectrogram-like objects typically expose .value as a 2D numpy array
    arr = np.asarray(qgram.value, dtype=np.float32).T #transpose!

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


def finite_timeseries(ts):
    y = np.asarray(ts.value, dtype=float)
    return np.isfinite(y).all()


def clean_timeseries(ts):
    y = np.asarray(ts.value, dtype=float).copy()
    bad = ~np.isfinite(y)
    if bad.any():
        y[bad] = 0.0
    return TimeSeries(y, t0=ts.t0, dt=ts.dt, unit=ts.unit)


def scipy_bandpass_notch(ts, fs=4096, low=20, high=1700,
                         notches=(60, 120, 180, 300, 505, 600, 1000), q=30, order=4): 

    x = ts.value.astype(float)
    x = x - x.mean()  # important

    # bandpass as SOS
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")

    # append notch SOS sections
    if notches is not None:
        for f0 in notches:
            b, a = iirnotch(w0=f0, Q=q, fs=fs)
            sos_notch = tf2sos(b, a)
            sos = np.vstack([sos, sos_notch])
    y = sosfiltfilt(sos, x)

    # preserve start time + sample spacing
    return TimeSeries(y, t0=ts.t0, dt=ts.dt, unit=ts.unit)


def build_event_sample(gps, seglen=4.0, padding=30.0, sample_rate=4096):
    """
    Build one model input sample from real detector data around a GPS time.

    Returns
    -------
    X : torch.Tensor
        Shape [2, F, T]
    t_feat : torch.Tensor
        Shape [2]
    """
    # H1_raw = TimeSeries.fetch_open_data("H1", gps - seglen - padding, gps + seglen + padding, sample_rate=sample_rate)
    # L1_raw = TimeSeries.fetch_open_data("L1", gps - seglen - padding, gps + seglen + padding, sample_rate=sample_rate)

    start = gps - seglen - padding
    end   = gps + seglen + padding

    H1_raw = fetch_open_data_retry("H1", start, end, sample_rate)
    L1_raw = fetch_open_data_retry("L1", start, end, sample_rate)

    H1_processed = scipy_bandpass_notch(H1_raw.copy(), fs=sample_rate)
    L1_processed = scipy_bandpass_notch(L1_raw.copy(), fs=sample_rate)

    H1_asd = H1_processed.asd(
        fftlength=seglen * 2,
        overlap=seglen,
        window="hann",
        method="median",
    )
    L1_asd = L1_processed.asd(
        fftlength=seglen * 2,
        overlap=seglen,
        window="hann",
        method="median",
    )

    H1_whitened = H1_processed.whiten(asd=H1_asd, pad=padding, remove_corrupted=True)
    L1_whitened = L1_processed.whiten(asd=L1_asd, pad=padding, remove_corrupted=True)

    H1_whitened = H1_whitened.crop(gps - seglen - 0.1, gps + seglen + 0.1)
    L1_whitened = L1_whitened.crop(gps - seglen - 0.1, gps + seglen + 0.1)

    if not finite_timeseries(H1_whitened) or not finite_timeseries(L1_whitened):
        raise ValueError("Non-finite values after whitening")

    H1_whitened = clean_timeseries(H1_whitened)
    L1_whitened = clean_timeseries(L1_whitened)

    q_H = H1_whitened.q_transform(
        qrange=(3, 100),
        frange=(20, 300),
        whiten=False,
        outseg=(gps - seglen, gps + seglen),
    )
    q_L = L1_whitened.q_transform(
        qrange=(3, 100),
        frange=(20, 300),
        whiten=False,
        outseg=(gps - seglen, gps + seglen),
    )

    H1_img = to_fixed_spectrogram(q_H, out_f=256, out_t=256)
    L1_img = to_fixed_spectrogram(q_L, out_f=256, out_t=256)

    t_feat = gps_to_cyclical_time_features(gps, include_dow=False).astype(np.float32)
    t_feat = torch.from_numpy(t_feat)

    X = torch.stack([H1_img, L1_img], dim=0).to(torch.float32)  # [2, F, T]

    return X, t_feat


def load_model(checkpoint_path, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = JointConvNeXtGWWithTime(
        pretrained=True,
        use_aux_3class=False,
        head_hidden=256,
        dropout=0.30,
    ).to(device)

    load_checkpoint(checkpoint_path, model, optimiser=None, map_location=device)
    model.eval()
    return model, device



def predict_real_events(
    checkpoint_path,
    events,
    seglen=4.0,
    padding=30.0,
    sample_rate=4096,
    gw_threshold=0.5,
    batch_size=16,
    device=None,
):
    """
    Parameters
    ----------
    checkpoint_path : str
        Path to trained checkpoint.
    events : list of tuples
        [(name, gps), ...]
    """

    model, device = load_model(checkpoint_path, device=device)

    rows = []
    batch_items = []

    def run_batch(batch_items):
        if not batch_items:
            return

        names = [item["name"] for item in batch_items]
        gpss = [item["gps"] for item in batch_items]

        X = torch.stack([item["X"] for item in batch_items], dim=0).to(device=device, dtype=torch.float32)
        t_feat = torch.stack([item["t_feat"] for item in batch_items], dim=0).to(device=device, dtype=torch.float32)

        with torch.no_grad():
            outputs = model(X, t_feat)

            p_gw = torch.sigmoid(outputs["logit_gw"]).cpu().numpy()
            pred_gw_bin = (p_gw > gw_threshold).astype(int)

            logits_3class = outputs.get("logits_3class", None)
            if logits_3class is not None:
                probs_3class = torch.softmax(logits_3class, dim=1).cpu().numpy()
                pred_3class = logits_3class.argmax(dim=1).cpu().numpy()
            else:
                probs_3class = None
                pred_3class = None

            attn = outputs.get("attn_weights", None)
            if attn is not None:
                attn = attn.cpu().numpy()

        for i in range(len(batch_items)):
            row = {
                "event_name": names[i],
                "gps": float(gpss[i]),
                "y_true": 2,
                "pred_gw_bin": int(pred_gw_bin[i]),
                "p_gw": float(p_gw[i]),
                "status": "ok",
            }

            if pred_3class is not None:
                row.update({
                    "pred_3class": int(pred_3class[i]),
                    "p_noise": float(probs_3class[i, 0]),
                    "p_glitch": float(probs_3class[i, 1]),
                    "p_gw_3class": float(probs_3class[i, 2]),
                    "correct_3class": int(pred_3class[i] == 2),
                })
            else:
                row.update({
                    "pred_3class": np.nan,
                    "p_noise": np.nan,
                    "p_glitch": np.nan,
                    "p_gw_3class": np.nan,
                    "correct_3class": np.nan,
                })

            if attn is not None:
                row.update({
                    "attn_H1": float(attn[i, 0]),
                    "attn_L1": float(attn[i, 1]),
                })
            else:
                row.update({
                    "attn_H1": np.nan,
                    "attn_L1": np.nan,
                })

            rows.append(row)

    for i, (name, gps) in enumerate(events):
        try:
            X, t_feat = build_event_sample(
                gps=gps,
                seglen=seglen,
                padding=padding,
                sample_rate=sample_rate,
            )

            batch_items.append({
                "name": name,
                "gps": gps,
                "X": X,
                "t_feat": t_feat,
            })

        except Exception as e:
            rows.append({
                "event_name": name,
                "gps": float(gps),
                "y_true": 2,
                "pred_gw_bin": np.nan,
                "p_gw": np.nan,
                "pred_3class": np.nan,
                "p_noise": np.nan,
                "p_glitch": np.nan,
                "p_gw_3class": np.nan,
                "correct_3class": np.nan,
                "attn_H1": np.nan,
                "attn_L1": np.nan,
                "status": f"failed: {e}",
            })

        if len(batch_items) >= batch_size:
            run_batch(batch_items)
            batch_items = []

        print(f"Prepared {i + 1}/{len(events)}: {name}", flush=True)

    # final partial batch
    run_batch(batch_items)

    df = pd.DataFrame(rows)

    # optional: restore original event order
    df["__order"] = pd.Categorical(df["event_name"], categories=[name for name, _ in events], ordered=True)
    df = df.sort_values("__order").drop(columns="__order").reset_index(drop=True)

    return df


def load_events_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["shortName", "gps"]).copy()
    df["gps"] = pd.to_numeric(df["gps"], errors="coerce")
    df = df.dropna(subset=["gps"]).sort_values("gps")
    return [(row["shortName"], float(row["gps"])) for _, row in df.iterrows()]

if __name__ == "__main__":
    checkpoint_path = "Model_9_Better_fusion/training_figures9_460k/best.pt"

    events = load_events_from_csv("event-versions.csv")

    df_predictions = predict_real_events(
        checkpoint_path=checkpoint_path,
        events=events,
        seglen=4.0,
        padding=30.0,
        sample_rate=4096,
        gw_threshold=0.5,
        batch_size=16,
    )


    out_dir = "real_event_predictions"
    os.makedirs(out_dir, exist_ok=True)

    df_predictions.to_csv(os.path.join(out_dir, "real_events_predictions_9_latest.csv"), index=False)
