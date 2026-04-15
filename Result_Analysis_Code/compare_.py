from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.calibration import calibration_curve

try:
    from scipy.stats import chi2
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


def load_and_align(file1: str, file2: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load two CSV files and align them on sample_index.
    """
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    required = {"sample_index", "true_is_gw", "p_gw"}
    for name, df in [("file1", df1), ("file2", df2)]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")

    merged = df1.merge(
        df2,
        on="sample_index",
        suffixes=("_m1", "_m2"),
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:
        raise ValueError("No overlapping sample_index values found.")

    # sanity check: true labels should match
    if not np.array_equal(merged["true_is_gw_m1"].values, merged["true_is_gw_m2"].values):
        mismatch = (merged["true_is_gw_m1"] != merged["true_is_gw_m2"]).sum()
        raise ValueError(
            f"true_is_gw does not match across files for {mismatch} aligned rows."
        )

    return df1, df2, merged


def bootstrap_ap_difference(
    y_true: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    n_boot: int = 5000,
    seed: int = 42,
) -> dict:
    """
    Paired bootstrap for AP difference: AP(model2) - AP(model1).
    Resamples examples with replacement, preserving pairing between models.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)

    ap1 = average_precision_score(y_true, p1)
    ap2 = average_precision_score(y_true, p2)
    observed_diff = ap2 - ap1

    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]

        # AP is undefined if bootstrap sample has only one class
        if len(np.unique(y_b)) < 2:
            continue

        p1_b = p1[idx]
        p2_b = p2[idx]

        ap1_b = average_precision_score(y_b, p1_b)
        ap2_b = average_precision_score(y_b, p2_b)
        diffs.append(ap2_b - ap1_b)

    diffs = np.array(diffs)
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])

    # Approximate two-sided bootstrap p-value against diff == 0
    p_value = 2 * min(np.mean(diffs >= 0), np.mean(diffs <= 0))

    return {
        "ap1": ap1,
        "ap2": ap2,
        "diff_m2_minus_m1": observed_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "bootstrap_p_value": p_value,
        "n_valid_bootstraps": len(diffs),
    }


def summarize_threshold_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    thresholds: Iterable[float],
    model_name: str,
) -> pd.DataFrame:
    rows = []
    for thr in thresholds:
        pred = (scores >= thr).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)

        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan

        rows.append(
            {
                "model": model_name,
                "threshold": thr,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "specificity": specificity,
                "fpr": fpr,
                "fnr": fnr,
                "positive_rate": pred.mean(),
            }
        )
    return pd.DataFrame(rows)


def threshold_for_target_precision(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_precision: float = 0.95,
) -> tuple[float | None, dict]:
    """
    Find the threshold achieving at least target precision with maximum recall.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)

    # sklearn returns precision/recall of length len(thresholds)+1
    candidate_rows = []
    for i, thr in enumerate(thresholds):
        p = precision[i + 1]
        r = recall[i + 1]
        if p >= target_precision:
            candidate_rows.append((thr, p, r))

    if not candidate_rows:
        return None, {}

    best_thr, best_p, best_r = max(candidate_rows, key=lambda x: x[2])
    return best_thr, {"precision": best_p, "recall": best_r}


def mcnemar_test(y_true: np.ndarray, pred1: np.ndarray, pred2: np.ndarray) -> dict:
    """
    McNemar's test on paired binary correctness.
    Uses continuity-corrected chi-square approximation.
    """
    correct1 = (pred1 == y_true)
    correct2 = (pred2 == y_true)

    b = np.sum((correct1 == True) & (correct2 == False))
    c = np.sum((correct1 == False) & (correct2 == True))

    if b + c == 0:
        return {"b": int(b), "c": int(c), "statistic": 0.0, "p_value": 1.0}

    stat = (abs(b - c) - 1) ** 2 / (b + c)

    if SCIPY_AVAILABLE:
        p_value = 1 - chi2.cdf(stat, df=1)
    else:
        p_value = np.nan

    return {"b": int(b), "c": int(c), "statistic": float(stat), "p_value": float(p_value)}


def plot_pr_curve(y_true: np.ndarray, p1: np.ndarray, p2: np.ndarray, outpath: Path) -> None:
    ap1 = average_precision_score(y_true, p1)
    ap2 = average_precision_score(y_true, p2)

    prec1, rec1, _ = precision_recall_curve(y_true, p1)
    prec2, rec2, _ = precision_recall_curve(y_true, p2)

    plt.figure(figsize=(7, 6))
    plt.plot(rec1, prec1, label=f"Model 1 (AP={ap1:.4f})")
    plt.plot(rec2, prec2, label=f"Model 2 (AP={ap2:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_score_histograms(y_true: np.ndarray, p1: np.ndarray, p2: np.ndarray, outpath: Path) -> None:
    plt.figure(figsize=(8, 6))
    bins = 40

    plt.hist(p1[y_true == 0], bins=bins, alpha=0.5, density=True, label="Model 1 negatives")
    plt.hist(p1[y_true == 1], bins=bins, alpha=0.5, density=True, label="Model 1 positives")
    plt.hist(p2[y_true == 0], bins=bins, histtype="step", linewidth=2, density=True, label="Model 2 negatives")
    plt.hist(p2[y_true == 1], bins=bins, histtype="step", linewidth=2, density=True, label="Model 2 positives")

    plt.xlabel("Predicted p_gw")
    plt.ylabel("Density")
    plt.title("Score distributions by class")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_calibration(y_true: np.ndarray, p1: np.ndarray, p2: np.ndarray, outpath: Path) -> None:
    frac_pos1, mean_pred1 = calibration_curve(y_true, p1, n_bins=10, strategy="quantile")
    frac_pos2, mean_pred2 = calibration_curve(y_true, p2, n_bins=10, strategy="quantile")

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.plot(mean_pred1, frac_pos1, marker="o", label="Model 1")
    plt.plot(mean_pred2, frac_pos2, marker="o", label="Model 2")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive fraction")
    plt.title("Calibration curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare two model output CSVs.")
    parser.add_argument("file1", type=str, help="CSV for model 1")
    parser.add_argument("file2", type=str, help="CSV for model 2")
    parser.add_argument("--outdir", type=str, default="comparison_output", help="Output directory")
    parser.add_argument("--n_boot", type=int, default=5000, help="Number of bootstrap samples")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=[0.3, 0.5, 0.7],
        help="Thresholds for operating-point comparison",
    )
    parser.add_argument(
        "--target_precision",
        type=float,
        default=0.95,
        help="Target precision for threshold search",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    _, _, merged = load_and_align(args.file1, args.file2)

    y_true = merged["true_is_gw_m1"].to_numpy().astype(int)
    p1 = merged["p_gw_m1"].to_numpy().astype(float)
    p2 = merged["p_gw_m2"].to_numpy().astype(float)

    # Global metrics
    results = {}
    results["n_samples"] = len(y_true)
    results["positive_rate"] = float(y_true.mean())

    results["model1_ap"] = float(average_precision_score(y_true, p1))
    results["model2_ap"] = float(average_precision_score(y_true, p2))
    results["model1_roc_auc"] = float(roc_auc_score(y_true, p1))
    results["model2_roc_auc"] = float(roc_auc_score(y_true, p2))
    results["model1_brier"] = float(brier_score_loss(y_true, p1))
    results["model2_brier"] = float(brier_score_loss(y_true, p2))

    # Paired bootstrap AP difference
    boot = bootstrap_ap_difference(y_true, p1, p2, n_boot=args.n_boot)
    results.update({
        "ap_diff_m2_minus_m1": float(boot["diff_m2_minus_m1"]),
        "ap_diff_ci_low": float(boot["ci_low"]),
        "ap_diff_ci_high": float(boot["ci_high"]),
        "ap_diff_bootstrap_p_value": float(boot["bootstrap_p_value"]),
        "n_valid_bootstraps": int(boot["n_valid_bootstraps"]),
    })

    # Threshold summaries
    th_df1 = summarize_threshold_metrics(y_true, p1, args.thresholds, "model1")
    th_df2 = summarize_threshold_metrics(y_true, p2, args.thresholds, "model2")
    threshold_df = pd.concat([th_df1, th_df2], ignore_index=True)

    # Best threshold for target precision
    thr1, info1 = threshold_for_target_precision(y_true, p1, args.target_precision)
    thr2, info2 = threshold_for_target_precision(y_true, p2, args.target_precision)

    # McNemar at threshold 0.5
    pred1_05 = (p1 >= 0.5).astype(int)
    pred2_05 = (p2 >= 0.5).astype(int)
    mcnemar_05 = mcnemar_test(y_true, pred1_05, pred2_05)

    # Save numeric outputs
    summary_df = pd.DataFrame([results])
    summary_df.to_csv(outdir / "summary_metrics.csv", index=False)
    threshold_df.to_csv(outdir / "threshold_metrics.csv", index=False)

    pd.DataFrame(
        [
            {
                "model": "model1",
                "target_precision": args.target_precision,
                "best_threshold": thr1,
                "achieved_precision": info1.get("precision", np.nan),
                "achieved_recall": info1.get("recall", np.nan),
            },
            {
                "model": "model2",
                "target_precision": args.target_precision,
                "best_threshold": thr2,
                "achieved_precision": info2.get("precision", np.nan),
                "achieved_recall": info2.get("recall", np.nan),
            },
        ]
    ).to_csv(outdir / "target_precision_comparison.csv", index=False)

    pd.DataFrame([mcnemar_05]).to_csv(outdir / "mcnemar_threshold_0.5.csv", index=False)

    # Plots
    plot_pr_curve(y_true, p1, p2, outdir / "pr_curve.png")
    plot_score_histograms(y_true, p1, p2, outdir / "score_histograms.png")
    plot_calibration(y_true, p1, p2, outdir / "calibration_curve.png")

    # Console report
    print("\n=== OVERALL METRICS ===")
    print(f"N samples:                {results['n_samples']}")
    print(f"Positive rate:            {results['positive_rate']:.4f}")
    print(f"Model 1 AP:               {results['model1_ap']:.6f}")
    print(f"Model 2 AP:               {results['model2_ap']:.6f}")
    print(f"AP diff (m2 - m1):        {results['ap_diff_m2_minus_m1']:.6f}")
    print(
        f"95% bootstrap CI:         [{results['ap_diff_ci_low']:.6f}, "
        f"{results['ap_diff_ci_high']:.6f}]"
    )
    print(f"Bootstrap p-value:        {results['ap_diff_bootstrap_p_value']:.6f}")
    print(f"Model 1 ROC AUC:          {results['model1_roc_auc']:.6f}")
    print(f"Model 2 ROC AUC:          {results['model2_roc_auc']:.6f}")
    print(f"Model 1 Brier score:      {results['model1_brier']:.6f}")
    print(f"Model 2 Brier score:      {results['model2_brier']:.6f}")

    print("\n=== TARGET PRECISION COMPARISON ===")
    if thr1 is None:
        print(f"Model 1: no threshold reaches precision >= {args.target_precision:.2f}")
    else:
        print(
            f"Model 1: threshold={thr1:.6f}, "
            f"precision={info1['precision']:.6f}, recall={info1['recall']:.6f}"
        )

    if thr2 is None:
        print(f"Model 2: no threshold reaches precision >= {args.target_precision:.2f}")
    else:
        print(
            f"Model 2: threshold={thr2:.6f}, "
            f"precision={info2['precision']:.6f}, recall={info2['recall']:.6f}"
        )

    print("\n=== MCNEMAR TEST AT THRESHOLD 0.5 ===")
    print(f"b (m1 correct, m2 wrong): {mcnemar_05['b']}")
    print(f"c (m1 wrong, m2 correct): {mcnemar_05['c']}")
    print(f"Statistic:                {mcnemar_05['statistic']:.6f}")
    print(f"P-value:                  {mcnemar_05['p_value']:.6f}")

    print(f"\nSaved outputs to: {outdir.resolve()}")
    print("Files:")
    print("  - summary_metrics.csv")
    print("  - threshold_metrics.csv")
    print("  - target_precision_comparison.csv")
    print("  - mcnemar_threshold_0.5.csv")
    print("  - pr_curve.png")
    print("  - score_histograms.png")
    print("  - calibration_curve.png")


if __name__ == "__main__":
    main()