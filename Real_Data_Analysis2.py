import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


NUMERIC_CANDIDATES = [
    "mass_1_source",
    "mass_2_source",
    "network_matched_filter_snr",
    "luminosity_distance",
    "chi_eff",
    "total_mass_source",
    "chirp_mass_source",
    "chirp_mass",
    "redshift",
    "far",
    "p_astro",
    "final_mass_source",
]


BIN_CANDIDATES = {
    "network_matched_filter_snr": [0, 8, 10, 12, 15, 20, 30, np.inf],
    "luminosity_distance": [0, 500, 1000, 2000, 4000, 8000, np.inf],
    "redshift": [0, 0.05, 0.1, 0.2, 0.4, 0.8, np.inf],
    "chirp_mass_source": [0, 10, 20, 30, 50, 80, 120, np.inf],
    "total_mass_source": [0, 20, 40, 60, 100, 150, 250, np.inf],
    "p_astro": [0, 0.5, 0.8, 0.95, 0.99, 0.999, 1.000001],
    "far": [0, 1e-8, 1e-6, 1e-4, 1e-2, 1, np.inf],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join prediction results to GW event metadata and summarize which event "
            "properties are associated with correct pred_gw predictions."
        )
    )
    parser.add_argument(
        "predictions_csv",
        type=Path,
        help="Path to real_event_predictions9.csv",
    )
    parser.add_argument(
        "events_csv",
        type=Path,
        help="Path to event_versions.csv",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("gw_analysis_output"),
        help="Directory where output CSV and text summary files will be written.",
    )
    parser.add_argument(
        "--positive-class-column",
        default="pred_gw_bin",
        help="Prediction column where 1 means the model predicted GW correctly/positively.",
    )
    parser.add_argument(
        "--event-column",
        default="event_name",
        help="Event-name column in the predictions CSV.",
    )
    parser.add_argument(
        "--match-column",
        default="shortName",
        help="Matching name column in the events CSV.",
    )
    return parser.parse_args()



def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")



def text_header(title: str) -> str:
    return f"\n{title}\n" + "=" * len(title) + "\n"



def load_and_prepare(predictions_path: Path, events_path: Path, event_col: str, match_col: str, pred_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred = pd.read_csv(predictions_path)
    events = pd.read_csv(events_path)

    missing_pred = [c for c in [event_col, pred_col] if c not in pred.columns]
    missing_events = [c for c in [match_col] if c not in events.columns]
    if missing_pred:
        raise ValueError(f"Missing columns in predictions CSV: {missing_pred}")
    if missing_events:
        raise ValueError(f"Missing columns in events CSV: {missing_events}")

    pred = pred.copy()
    events = events.copy()

    pred[event_col] = pred[event_col].astype(str).str.strip()
    events[match_col] = events[match_col].astype(str).str.strip()

    pred[pred_col] = safe_numeric(pred[pred_col])
    pred["predicted_gw_positive"] = (pred[pred_col] == 1).astype(int)

    merged = pred.merge(
        events,
        how="left",
        left_on=event_col,
        right_on=match_col,
        suffixes=("_pred", "_event"),
        indicator=True,
    )

    numeric_cols = [c for c in NUMERIC_CANDIDATES if c in merged.columns]
    for col in numeric_cols:
        merged[col] = safe_numeric(merged[col])

    return pred, events, merged



def event_level_summary(merged: pd.DataFrame, event_col: str) -> pd.DataFrame:
    grouped = (
        merged.groupby(event_col, dropna=False)
        .agg(
            n_predictions=("predicted_gw_positive", "size"),
            n_positive_pred_gw=("predicted_gw_positive", "sum"),
        )
        .reset_index()
    )
    grouped["positive_pred_rate"] = grouped["n_positive_pred_gw"] / grouped["n_predictions"]

    metadata_cols = [
        c for c in [
            "catalog",
            "version",
            "mass_1_source",
            "mass_2_source",
            "network_matched_filter_snr",
            "luminosity_distance",
            "chi_eff",
            "total_mass_source",
            "chirp_mass_source",
            "chirp_mass",
            "redshift",
            "far",
            "p_astro",
            "final_mass_source",
        ] if c in merged.columns
    ]

    meta = (
        merged[[event_col] + metadata_cols]
        .drop_duplicates(subset=[event_col])
    )

    return grouped.merge(meta, on=event_col, how="left")



def numeric_association_table(event_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    target = safe_numeric(event_df["positive_pred_rate"])
    target_count = safe_numeric(event_df["n_positive_pred_gw"])
    total_count = safe_numeric(event_df["n_predictions"])

    for col in NUMERIC_CANDIDATES:
        if col not in event_df.columns:
            continue
        x = safe_numeric(event_df[col])
        valid = x.notna() & target.notna()
        if valid.sum() < 3:
            continue

        rows.append(
            {
                "feature": col,
                "n_events": int(valid.sum()),
                "pearson_with_positive_rate": x[valid].corr(target[valid], method="pearson"),
                "spearman_with_positive_rate": x[valid].corr(target[valid], method="spearman"),
                "pearson_with_positive_count": x[valid].corr(target_count[valid], method="pearson"),
                "spearman_with_positive_count": x[valid].corr(target_count[valid], method="spearman"),
                "pearson_with_total_predictions": x[valid].corr(total_count[valid], method="pearson"),
                "median_feature": float(x[valid].median()),
                "mean_feature": float(x[valid].mean()),
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["abs_spearman_rate"] = out["spearman_with_positive_rate"].abs()
    return out.sort_values(["abs_spearman_rate", "feature"], ascending=[False, True])



def binned_tables(event_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    outputs = {}
    for feature, bins in BIN_CANDIDATES.items():
        if feature not in event_df.columns:
            continue
        x = safe_numeric(event_df[feature])
        valid = x.notna() & event_df["positive_pred_rate"].notna()
        if valid.sum() < 3:
            continue

        labels = [f"[{bins[i]}, {bins[i+1]})" for i in range(len(bins) - 1)]
        cat = pd.cut(x[valid], bins=bins, labels=labels, include_lowest=True, right=False)
        table = (
            pd.DataFrame(
                {
                    feature + "_bin": cat,
                    "positive_pred_rate": event_df.loc[valid, "positive_pred_rate"],
                    "n_positive_pred_gw": event_df.loc[valid, "n_positive_pred_gw"],
                    "n_predictions": event_df.loc[valid, "n_predictions"],
                }
            )
            .groupby(feature + "_bin", observed=False)
            .agg(
                n_events=("positive_pred_rate", "size"),
                mean_positive_pred_rate=("positive_pred_rate", "mean"),
                median_positive_pred_rate=("positive_pred_rate", "median"),
                mean_positive_count=("n_positive_pred_gw", "mean"),
                mean_total_predictions=("n_predictions", "mean"),
            )
            .reset_index()
        )
        outputs[feature] = table
    return outputs



def matched_unmatched_summary(merged: pd.DataFrame, event_col: str, match_col: str) -> str:
    n_rows = len(merged)
    n_matched = int((merged["_merge"] == "both").sum())
    unmatched_names = (
        merged.loc[merged["_merge"] != "both", event_col]
        .dropna()
        .astype(str)
        .value_counts()
    )

    lines = []
    lines.append(text_header("Join Summary"))
    lines.append(f"Prediction rows: {n_rows}\n")
    lines.append(f"Matched rows on {event_col} -> {match_col}: {n_matched} ({n_matched / max(n_rows, 1):.2%})\n")
    lines.append(f"Unmatched rows: {n_rows - n_matched} ({(n_rows - n_matched) / max(n_rows, 1):.2%})\n")
    if len(unmatched_names) > 0:
        lines.append("Top unmatched event names:\n")
        lines.append(unmatched_names.head(20).to_string())
        lines.append("\n")
    return "".join(lines)



def strongest_findings(corr_df: pd.DataFrame) -> str:
    lines = []
    lines.append(text_header("Strongest Numeric Associations"))
    if corr_df.empty:
        lines.append("No usable numeric metadata columns were found for association analysis.\n")
        return "".join(lines)

    top_pos = corr_df.sort_values("spearman_with_positive_rate", ascending=False).head(5)
    top_neg = corr_df.sort_values("spearman_with_positive_rate", ascending=True).head(5)

    lines.append("Top positive associations with positive prediction rate (Spearman):\n")
    lines.append(top_pos[["feature", "n_events", "spearman_with_positive_rate", "pearson_with_positive_rate"]].to_string(index=False))
    lines.append("\n\nTop negative associations with positive prediction rate (Spearman):\n")
    lines.append(top_neg[["feature", "n_events", "spearman_with_positive_rate", "pearson_with_positive_rate"]].to_string(index=False))
    lines.append("\n")
    return "".join(lines)



def overall_summary(pred: pd.DataFrame, merged: pd.DataFrame) -> str:
    n_rows = len(pred)
    n_positive = int(pred["predicted_gw_positive"].sum())
    lines = []
    lines.append(text_header("Overall Prediction Summary"))
    lines.append(f"Total prediction rows: {n_rows}\n")
    lines.append(f"Rows with pred_gw_bin == 1: {n_positive} ({n_positive / max(n_rows, 1):.2%})\n")
    if "y_true" in pred.columns:
        yt = safe_numeric(pred["y_true"])
        valid = yt.notna() & pred["predicted_gw_positive"].notna()
        if valid.sum() > 0:
            true_positive_rows = int(((yt == 1) & (pred["predicted_gw_positive"] == 1)).sum())
            lines.append(f"Rows where y_true == 1 and pred_gw_bin == 1: {true_positive_rows}\n")
    if "status" in merged.columns:
        lines.append("\nStatus counts:\n")
        lines.append(merged["status"].fillna("<NA>").astype(str).value_counts().to_string())
        lines.append("\n")
    return "".join(lines)



def save_outputs(outdir: Path, merged: pd.DataFrame, event_df: pd.DataFrame, corr_df: pd.DataFrame, binned: dict[str, pd.DataFrame], report_text: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    merged.to_csv(outdir / "merged_predictions_events.csv", index=False)
    event_df.to_csv(outdir / "event_level_summary.csv", index=False)
    corr_df.to_csv(outdir / "numeric_associations.csv", index=False)

    for feature, table in binned.items():
        table.to_csv(outdir / f"binned_summary_{feature}.csv", index=False)

    (outdir / "summary_report.txt").write_text(report_text, encoding="utf-8")



def main() -> int:
    args = parse_args()

    try:
        pred, events, merged = load_and_prepare(
            args.predictions_csv,
            args.events_csv,
            args.event_column,
            args.match_column,
            args.positive_class_column,
        )
    except Exception as exc:
        print(f"ERROR loading data: {exc}", file=sys.stderr)
        return 1

    event_df = event_level_summary(merged, args.event_column)
    corr_df = numeric_association_table(event_df)
    binned = binned_tables(event_df)

    report_parts = [
        overall_summary(pred, merged),
        matched_unmatched_summary(merged, args.event_column, args.match_column),
        strongest_findings(corr_df),
    ]

    report_parts.append(text_header("Generated Files"))
    report_parts.append("merged_predictions_events.csv\n")
    report_parts.append("event_level_summary.csv\n")
    report_parts.append("numeric_associations.csv\n")
    for feature in sorted(binned):
        report_parts.append(f"binned_summary_{feature}.csv\n")
    report_parts.append("summary_report.txt\n")

    report_text = "".join(report_parts)
    save_outputs(args.outdir, merged, event_df, corr_df, binned, report_text)

    print(report_text)
    print(f"\nOutputs written to: {args.outdir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
