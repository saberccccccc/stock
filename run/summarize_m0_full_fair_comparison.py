"""Summarize the complete CC/OO/OO-lag1 validation and test comparison."""

from pathlib import Path

import pandas as pd


ROOT = Path("v14_m0_full_fair_comparison_20260622")
SPLITS = ("val_2024", "test_2025_20260518")
MODELS = ("cc_e14", "oo_e15", "oo_lag1_e15")
CELL_KEYS = (
    "stress",
    "rebalance_band",
    "portfolio_value",
    "target_frac",
    "hold_frac",
)


def load_backtests(root=ROOT):
    frames = []
    for split in SPLITS:
        for model in MODELS:
            path = (
                root
                / split
                / "parameter_surface_by_model"
                / model
                / "open_price_ledger_param_sweep_summary.csv"
            )
            frame = pd.read_csv(path)
            if len(frame) != 288:
                raise ValueError(f"{path} has {len(frame)} rows, expected 288")
            frame["split"] = split
            frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if len(result) != 1728:
        raise ValueError(f"combined backtests have {len(result)} rows, expected 1728")
    return result


def aggregate_metrics(frame, group_keys):
    return (
        frame.groupby(group_keys, as_index=False)
        .agg(
            rows=("sharpe", "size"),
            ann_median=("ann", "median"),
            ann_min=("ann", "min"),
            sharpe_median=("sharpe", "median"),
            sharpe_min=("sharpe", "min"),
            mdd_median=("mdd", "median"),
            mdd_max=("mdd", "max"),
            turnover_median=("avg_turnover", "median"),
            executed_turnover_median=("avg_executed_turnover", "median"),
        )
        .sort_values(group_keys)
    )


def matched_cell_wins(frame):
    rows = []
    for split, split_frame in frame.groupby("split"):
        pivot = split_frame.pivot(
            index=list(CELL_KEYS), columns="alpha_name", values="sharpe"
        )
        winners = pivot.idxmax(axis=1)
        for model in MODELS:
            rows.append(
                {
                    "split": split,
                    "stress": "all",
                    "model": model,
                    "wins": int((winners == model).sum()),
                    "cells": int(len(winners)),
                    "win_rate": float((winners == model).mean()),
                }
            )
        for stress, stress_pivot in pivot.groupby(level="stress"):
            stress_winners = stress_pivot.idxmax(axis=1)
            for model in MODELS:
                rows.append(
                    {
                        "split": split,
                        "stress": stress,
                        "model": model,
                        "wins": int((stress_winners == model).sum()),
                        "cells": int(len(stress_winners)),
                        "win_rate": float((stress_winners == model).mean()),
                    }
                )
    return pd.DataFrame(rows)


def val_selected_test_confirmation(frame):
    val = frame[frame["split"] == "val_2024"]
    test = frame[frame["split"] == "test_2025_20260518"]
    rows = []
    for model in MODELS:
        candidates = (
            val[val["alpha_name"] == model]
            .groupby(["target_frac", "hold_frac", "rebalance_band"], as_index=False)
            .agg(
                val_ann_median=("ann", "median"),
                val_sharpe_median=("sharpe", "median"),
                val_sharpe_min=("sharpe", "min"),
                val_mdd_median=("mdd", "median"),
                val_turnover_median=("avg_turnover", "median"),
            )
            .sort_values(
                ["val_sharpe_median", "val_sharpe_min", "val_turnover_median"],
                ascending=[False, False, True],
            )
        )
        selected = candidates.iloc[0]
        confirmed = test[
            (test["alpha_name"] == model)
            & (test["target_frac"] == selected["target_frac"])
            & (test["hold_frac"] == selected["hold_frac"])
            & (test["rebalance_band"] == selected["rebalance_band"])
        ]
        rows.append(
            {
                "model": model,
                **selected.to_dict(),
                "test_ann_median": confirmed["ann"].median(),
                "test_sharpe_median": confirmed["sharpe"].median(),
                "test_sharpe_min": confirmed["sharpe"].min(),
                "test_mdd_median": confirmed["mdd"].median(),
                "test_mdd_max": confirmed["mdd"].max(),
                "test_turnover_median": confirmed["avg_turnover"].median(),
            }
        )
    return pd.DataFrame(rows)


def load_prediction_quality(root=ROOT):
    frames = []
    for split in SPLITS:
        frame = pd.read_csv(root / split / "signal_quality" / "summary.csv")
        frame["split"] = split
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main():
    output_dir = ROOT / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    backtests = load_backtests()
    backtests.to_csv(output_dir / "all_backtest_results_1728.csv", index=False)
    aggregate_metrics(backtests, ["split", "alpha_name"]).to_csv(
        output_dir / "overall_summary.csv", index=False
    )
    aggregate_metrics(backtests, ["split", "stress", "alpha_name"]).to_csv(
        output_dir / "stress_summary.csv", index=False
    )
    matched_cell_wins(backtests).to_csv(
        output_dir / "matched_cell_sharpe_wins.csv", index=False
    )
    val_selected_test_confirmation(backtests).to_csv(
        output_dir / "val_selected_test_confirmation.csv", index=False
    )
    load_prediction_quality().to_csv(
        output_dir / "prediction_quality_val_test.csv", index=False
    )
    print(f"Saved full comparison summary to {output_dir}")


if __name__ == "__main__":
    main()
