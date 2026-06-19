"""Build 2024 reranker blends and run strict constrained portfolio validation."""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from core.research_protocol import assert_alpha_rows_within_research
from run.backtest_retention_execution_constraints import (
    load_alpha_rows as load_backtest_rows,
    load_close_money,
    recompute_adv,
    run_constrained,
)
from run.backtest_temporal_retention import load_index_returns
from run.transform_alpha_for_execution import (
    load_alpha_rows,
    load_signal_returns,
    transform_rows,
)


BLEND_WEIGHTS = (1.00, 0.75, 0.50, 0.25, 0.00)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="reranker_data_20260614/m0_validation_2024/reranker_dataset.parquet",
    )
    parser.add_argument(
        "--model",
        default="reranker_models_20260615/lambdarank_v1/reranker_model.pkl",
    )
    parser.add_argument(
        "--m0-alpha",
        default="multi_loss_validation_20260614/portfolio/m0_nomulti_e6/alpha_raw.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="reranker_validation_20260615",
    )
    parser.add_argument("--data-dir", default="data/raw")
    return parser.parse_args()


def write_alpha(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def descending_percentile(values):
    order = np.argsort(-np.asarray(values, dtype=np.float64), kind="mergesort")
    result = np.empty(len(order), dtype=np.float64)
    result[order] = 1.0 - np.arange(len(order), dtype=np.float64) / max(len(order) - 1, 1)
    return result


def build_blended_rows(dataset, model_payload, m0_rows):
    model = model_payload["model"]
    features = model_payload["feature_columns"]
    missing = [column for column in features if column not in dataset.columns]
    if missing:
        raise ValueError(f"2024 dataset is missing model features: {missing[:10]}")

    dataset = dataset.copy()
    dataset["date"] = pd.to_datetime(dataset["date"])
    dataset["reranker_score"] = model.predict(dataset[features])
    dataset["reranker_pct"] = dataset.groupby("date")["reranker_score"].transform(
        descending_percentile
    )
    dataset["m0_pct"] = 1.0 - dataset["m0_rank_pct"].astype(float)

    by_date = {
        pd.Timestamp(date): group.copy()
        for date, group in dataset.groupby("date", sort=False)
    }
    outputs = {weight: [] for weight in BLEND_WEIGHTS}
    for row in m0_rows:
        date = pd.Timestamp(row["date"])
        if date not in by_date:
            raise ValueError(f"No reranker candidates for {date.date()}")
        group = by_date[date]
        original_codes = [str(code) for code in row["codes"]]
        original_position = {code: index for index, code in enumerate(original_codes)}
        candidate_codes = set(group["code"].astype(str))
        if not candidate_codes.issubset(original_position):
            missing_codes = sorted(candidate_codes.difference(original_position))
            raise ValueError(f"{date.date()} candidate codes absent from M0 Alpha: {missing_codes[:5]}")

        for weight in BLEND_WEIGHTS:
            score = (
                weight * group["m0_pct"].to_numpy(dtype=np.float64)
                + (1.0 - weight) * group["reranker_pct"].to_numpy(dtype=np.float64)
            )
            ranked_candidates = [
                code
                for code, _, _ in sorted(
                    zip(
                        group["code"].astype(str),
                        score,
                        group["m0_pct"].to_numpy(dtype=np.float64),
                    ),
                    key=lambda item: (-item[1], -item[2], original_position[item[0]]),
                )
            ]
            remaining = [code for code in original_codes if code not in candidate_codes]
            final_codes = ranked_candidates + remaining
            n = len(final_codes)
            final_alpha = (
                1.0 - np.arange(n, dtype=np.float64) / max(n - 1, 1)
            ).tolist()
            outputs[weight].append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "codes": final_codes,
                    "alpha": final_alpha,
                    "n_stocks": n,
                    "reranker_blend": {
                        "m0_weight": weight,
                        "reranker_weight": 1.0 - weight,
                        "candidate_count": len(ranked_candidates),
                    },
                }
            )
    return outputs


def make_run_args(portfolio_value, scenario):
    cost_mult = scenario["cost_mult"]
    return SimpleNamespace(
        max_weight=0.05,
        market_timing_mode="legacy",
        market_min_mult=0.20,
        market_max_mult=1.00,
        legacy_bear_mult=0.70,
        legacy_crash_mult=0.30,
        commission_rate=0.0001 * cost_mult,
        stamp_tax_rate=0.0005 * cost_mult,
        slippage_rate=0.0005 * cost_mult,
        portfolio_value=float(portfolio_value),
        adv_participation_cap=scenario["adv_cap"],
        min_adv_cny=3_000_000.0,
        limit_threshold=0.095,
        execution_lag=scenario["lag"],
        lot_size=100,
        min_commission_cny=5.0,
        rebalance_band=0.20,
        allow_forward=False,
    )


def main():
    args = parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_parquet(ROOT / args.dataset)
    with (ROOT / args.model).open("rb") as handle:
        model_payload = pickle.load(handle)
    m0_rows = load_alpha_rows(ROOT / args.m0_alpha)
    assert_alpha_rows_within_research(m0_rows, context="M0 reranker validation")

    blend_rows = build_blended_rows(dataset, model_payload, m0_rows)
    alpha_paths = {}
    for weight, rows in blend_rows.items():
        tag = f"m0w{int(round(weight * 100)):03d}"
        raw_path = output_dir / tag / "alpha_raw.jsonl"
        filtered_path = output_dir / tag / "alpha_maxret095.jsonl"
        write_alpha(raw_path, rows)
        codes = {code for row in rows for code in row["codes"]}
        signal_returns = load_signal_returns(
            args.data_dir,
            codes,
            pd.Timestamp(rows[0]["date"]),
            pd.Timestamp(rows[-1]["date"]),
        )
        filtered = transform_rows(rows, signal_returns, max_signal_return=0.095)
        write_alpha(filtered_path, filtered)
        alpha_paths[(tag, "raw")] = raw_path
        alpha_paths[(tag, "maxret095")] = filtered_path
        demoted = sum(row["execution_transform"]["demoted_count"] for row in filtered)
        print(f"Built {tag}: rows={len(rows)} demoted={demoted}", flush=True)

    all_rows = {key: load_backtest_rows(path) for key, path in alpha_paths.items()}
    all_codes = sorted(
        {
            code
            for rows in all_rows.values()
            for row in rows
            for code in row["codes"]
        }
    )
    print(f"Loading shared market data: codes={len(all_codes)}", flush=True)
    close, money = load_close_money(args.data_dir, all_codes, 1000.0, 1000)
    adv = recompute_adv(money, 20)
    idx_close, idx_daily = load_index_returns(args.data_dir, "hs300_index.csv", close.index)

    scenarios = {
        "base": {"adv_cap": 0.05, "cost_mult": 1.0, "lag": 0},
        "cap3": {"adv_cap": 0.03, "cost_mult": 1.0, "lag": 0},
        "cost2x": {"adv_cap": 0.05, "cost_mult": 2.0, "lag": 0},
        "lag1": {"adv_cap": 0.05, "cost_mult": 1.0, "lag": 1},
    }
    summary = []
    for (model_name, transform), rows in all_rows.items():
        for scenario_name, scenario in scenarios.items():
            for portfolio_value in (500_000.0, 1_000_000.0):
                result, returns_df, diag_df = run_constrained(
                    rows,
                    close,
                    adv,
                    0.006,
                    0.10,
                    make_run_args(portfolio_value, scenario),
                    idx_close,
                    idx_daily,
                )
                result.update(
                    {
                        "model": model_name,
                        "transform": transform,
                        "scenario": scenario_name,
                        "portfolio_value": portfolio_value,
                    }
                )
                summary.append(result)
                tag = (
                    f"{model_name}_{transform}_{scenario_name}_"
                    f"{int(portfolio_value / 10000)}w"
                )
                returns_df.to_csv(output_dir / f"returns_{tag}.csv", index=False)
                diag_df.to_csv(output_dir / f"diagnostics_{tag}.csv", index=False)
                print(
                    f"{tag}: ann={result['ann']:.2f}% "
                    f"sharpe={result['sharpe']:.3f} "
                    f"mdd={result['mdd'] * 100:.2f}%",
                    flush=True,
                )

    frame = pd.DataFrame(summary)
    frame.to_csv(output_dir / "reranker_validation_summary.csv", index=False)
    print(f"Saved validation to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
