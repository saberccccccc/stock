"""Diagnose which M0 Top-N names a reranker promotes and demotes."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="reranker_data_20260614/m0_validation_2024/reranker_dataset.parquet",
    )
    parser.add_argument(
        "--validation-dir",
        default="reranker_validation_20260615",
    )
    parser.add_argument("--baseline", default="m0w100")
    parser.add_argument("--variants", default="m0w075,m0w050,m0w025,m0w000")
    parser.add_argument("--transform", default="maxret095")
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument(
        "--output-dir",
        default="reranker_validation_20260615/replacement_diagnostics",
    )
    return parser.parse_args()


def load_alpha(path):
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[pd.Timestamp(row["date"])] = [str(code) for code in row["codes"]]
    return rows


def safe_mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else np.nan


def main():
    args = parse_args()
    validation_dir = Path(args.validation_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    columns = [
        "date",
        "code",
        "candidate_position",
        "market_regime",
        "future_target",
        "future_h5_raw",
        "future_rank_pct",
        "relevance",
        "m0_rank_pct",
    ]
    labels = pd.read_parquet(args.dataset, columns=columns)
    labels["date"] = pd.to_datetime(labels["date"])
    labels["code"] = labels["code"].astype(str)
    labels = labels.set_index(["date", "code"]).sort_index()

    baseline_path = (
        validation_dir / args.baseline / f"alpha_{args.transform}.jsonl"
    )
    baseline = load_alpha(baseline_path)
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]

    detail_rows = []
    daily_rows = []
    for variant in variants:
        variant_path = validation_dir / variant / f"alpha_{args.transform}.jsonl"
        ranked = load_alpha(variant_path)
        common_dates = sorted(set(baseline).intersection(ranked))
        for date in common_dates:
            base_list = baseline[date][: args.top_n]
            variant_list = ranked[date][: args.top_n]
            base_set = set(base_list)
            variant_set = set(variant_list)
            promoted = [code for code in variant_list if code not in base_set]
            demoted = [code for code in base_list if code not in variant_set]
            unchanged = [code for code in variant_list if code in base_set]

            daily = {
                "variant": variant,
                "date": date,
                "month": date.strftime("%Y-%m"),
                "overlap_count": len(unchanged),
                "replacement_count": len(promoted),
                "overlap_ratio": len(unchanged) / args.top_n,
            }
            for role, codes in (
                ("promoted", promoted),
                ("demoted", demoted),
                ("unchanged", unchanged),
            ):
                values = []
                for position, code in enumerate(codes):
                    key = (date, code)
                    if key not in labels.index:
                        continue
                    row = labels.loc[key]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    record = {
                        "variant": variant,
                        "date": date,
                        "month": date.strftime("%Y-%m"),
                        "role": role,
                        "code": code,
                        "role_position": position,
                    }
                    record.update(row.to_dict())
                    detail_rows.append(record)
                    values.append(record)

                daily[f"{role}_future_target"] = safe_mean(
                    [value["future_target"] for value in values]
                )
                daily[f"{role}_future_h5_raw"] = safe_mean(
                    [value["future_h5_raw"] for value in values]
                )
                daily[f"{role}_future_rank_pct"] = safe_mean(
                    [value["future_rank_pct"] for value in values]
                )
                daily[f"{role}_relevance"] = safe_mean(
                    [value["relevance"] for value in values]
                )

            daily["replacement_target_delta"] = (
                daily["promoted_future_target"] - daily["demoted_future_target"]
            )
            daily["replacement_h5_delta"] = (
                daily["promoted_future_h5_raw"] - daily["demoted_future_h5_raw"]
            )
            daily_rows.append(daily)

    detail = pd.DataFrame(detail_rows)
    daily = pd.DataFrame(daily_rows)
    detail.to_csv(output_dir / "replacement_details.csv", index=False)
    daily.to_csv(output_dir / "replacement_daily.csv", index=False)

    summary_rows = []
    for variant, group in daily.groupby("variant", sort=False):
        valid_delta = group["replacement_target_delta"].dropna()
        valid_h5 = group["replacement_h5_delta"].dropna()
        summary_rows.append(
            {
                "variant": variant,
                "dates": len(group),
                "mean_replacements": group["replacement_count"].mean(),
                "mean_overlap_ratio": group["overlap_ratio"].mean(),
                "promoted_future_target": group["promoted_future_target"].mean(),
                "demoted_future_target": group["demoted_future_target"].mean(),
                "replacement_target_delta": valid_delta.mean(),
                "target_delta_positive_days": (valid_delta > 0).mean(),
                "promoted_future_h5_raw": group["promoted_future_h5_raw"].mean(),
                "demoted_future_h5_raw": group["demoted_future_h5_raw"].mean(),
                "replacement_h5_delta": valid_h5.mean(),
                "h5_delta_positive_days": (valid_h5 > 0).mean(),
                "promoted_relevance": group["promoted_relevance"].mean(),
                "demoted_relevance": group["demoted_relevance"].mean(),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "replacement_summary.csv", index=False)

    monthly = (
        daily.groupby(["variant", "month"], sort=False)
        .agg(
            dates=("date", "count"),
            mean_replacements=("replacement_count", "mean"),
            overlap_ratio=("overlap_ratio", "mean"),
            target_delta=("replacement_target_delta", "mean"),
            h5_delta=("replacement_h5_delta", "mean"),
        )
        .reset_index()
    )
    monthly.to_csv(output_dir / "replacement_monthly.csv", index=False)

    print(summary.to_string(index=False))
    print(f"Saved diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
