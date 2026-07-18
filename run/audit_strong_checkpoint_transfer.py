"""Audit validation checkpoint metrics against existing strict-OOS pilot months.

This command performs inference only. It never fits a model and never promotes a
per-window checkpoint from its OOS result.
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.runtime import build_v9_backtest_config, load_dl_predictor
from data.cache_metadata import load_explicit_cross_section_meta
from data.pipeline import samples_from_precomputed_metadata
from experiments.checkpoint_selection import load_epoch_metrics
from experiments.strong_rolling import load_json
from experiments.workflow_records import _resolve_label_raw_path
from run.generate_v9_inference_alpha import compute_rows
from run.v9_long_only_optimization import V9RankPredictor


DEFAULT_CACHE = (
    "cache/cross_section_v14_multilabel_open_tech_market_funda_shareh_restr_macro_"
    "all_s40_t5_h10_min30_mad_rawlab_end20260518_meta.pkl"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-root",
        default="reports/experiments/strong_e19_staged_pilot_20260717",
    )
    parser.add_argument(
        "--schedule",
        default="configs/monthly_rolling_compact_4y6m1m_2024_2025.json",
    )
    parser.add_argument("--cache-meta", default=DEFAULT_CACHE)
    parser.add_argument(
        "--output-dir",
        default="reports/strong_checkpoint_transfer_audit_20260718",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--windows", default="oos_2024_01,oos_2025_01,oos_2025_12")
    return parser.parse_args(argv)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def collect_epoch_records(window_root: Path) -> list[dict]:
    records: dict[int, dict] = {}
    for metrics_path in sorted(window_root.glob("training/*/epochs/epoch_metrics.jsonl")):
        frame = load_epoch_metrics(metrics_path)
        for row in frame.to_dict(orient="records"):
            epoch = int(row["epoch"])
            checkpoint = metrics_path.parent / f"epoch_{epoch:03d}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            if epoch in records:
                raise ValueError(f"duplicate epoch metrics in {window_root}: {epoch}")
            records[epoch] = {
                **row,
                "checkpoint": str(checkpoint.resolve()),
                "legacy_stage_id": metrics_path.parents[1].name,
            }
    if sorted(records) != list(range(1, 20)):
        raise ValueError(f"expected epochs 1..19 in {window_root}, got {sorted(records)}")
    return [records[epoch] for epoch in sorted(records)]


def evaluate_alpha_rows(
    rows: list[dict],
    *,
    labels: np.memmap,
    lag1_shift: int,
    date_to_index: dict[str, int],
    code_to_index: dict[str, int],
    horizon_indices: tuple[int, ...] = (0, 2, 4, 6),
    horizon_weights: tuple[float, ...] = (0.15, 0.25, 0.35, 0.25),
    top_fraction: float = 0.006,
) -> dict[str, float]:
    daily = []
    previous_top: set[str] | None = None
    turnover = []
    weights = np.asarray(horizon_weights, dtype=np.float64)
    weights = weights / weights.sum()
    for row in rows:
        date = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
        time_index = date_to_index[date]
        codes = [str(code) for code in row["codes"]]
        scores = np.asarray(row["alpha"], dtype=np.float64)
        stock_indices = np.asarray([code_to_index.get(code, -1) for code in codes], dtype=np.int64)
        valid_codes = stock_indices >= 0
        target = np.full(len(codes), np.nan, dtype=np.float64)
        lag1_target = np.full(len(codes), np.nan, dtype=np.float64)
        if valid_codes.any():
            selected = stock_indices[valid_codes]
            target[valid_codes] = labels[selected, time_index][:, horizon_indices] @ weights
            lag1_target[valid_codes] = labels[selected, time_index + lag1_shift][:, horizon_indices] @ weights
        valid = np.isfinite(scores) & np.isfinite(target) & np.isfinite(lag1_target)
        if int(valid.sum()) < 3:
            continue
        score_valid = scores[valid]
        target_valid = target[valid]
        lag1_valid = lag1_target[valid]
        valid_codes_list = np.asarray(codes, dtype=object)[valid]
        top_n = max(1, int(np.ceil(valid.sum() * top_fraction)))
        top_indices = np.argsort(score_valid)[-top_n:]
        top_codes = set(valid_codes_list[top_indices].tolist())
        if previous_top is not None:
            turnover.append(1.0 - len(previous_top & top_codes) / max(len(previous_top), 1))
        previous_top = top_codes
        daily.append(
            {
                "rank_ic_oo": pd.Series(score_valid).rank().corr(pd.Series(target_valid).rank()),
                "rank_ic_lag1": pd.Series(score_valid).rank().corr(pd.Series(lag1_valid).rank()),
                "top_return_oo": float(target_valid[top_indices].mean()),
                "top_return_lag1": float(lag1_valid[top_indices].mean()),
            }
        )
    if not daily:
        raise ValueError("checkpoint produced no evaluable OOS rows")
    frame = pd.DataFrame(daily)
    return {
        "oos_days": int(len(frame)),
        "oos_rank_ic_oo": float(frame["rank_ic_oo"].mean()),
        "oos_rank_ic_lag1": float(frame["rank_ic_lag1"].mean()),
        "oos_top0p6_return_oo": float(frame["top_return_oo"].mean()),
        "oos_top0p6_return_lag1": float(frame["top_return_lag1"].mean()),
        "oos_top0p6_turnover_proxy": float(np.mean(turnover)) if turnover else 0.0,
    }


def metric_transfer_summary(frame: pd.DataFrame) -> pd.DataFrame:
    validation_metrics = (
        "rawtopstable_h5_top0p6",
        "rawtopret_h5_top0p6",
        "alpha",
    )
    oos_metrics = (
        "oos_rank_ic_oo",
        "oos_rank_ic_lag1",
        "oos_top0p6_return_oo",
        "oos_top0p6_return_lag1",
        "oos_top0p6_turnover_proxy",
    )
    rows = []
    for window, group in frame.groupby("window", sort=True):
        for validation_metric in validation_metrics:
            for oos_metric in oos_metrics:
                rows.append(
                    {
                        "window": window,
                        "validation_metric": validation_metric,
                        "oos_metric": oos_metric,
                        "spearman_across_epochs": float(
                            group[validation_metric].corr(group[oos_metric], method="spearman")
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _write_report(results: pd.DataFrame, transfer: pd.DataFrame, output: Path) -> None:
    primary = transfer[
        (transfer["validation_metric"] == "rawtopstable_h5_top0p6")
        & transfer["oos_metric"].isin(
            ["oos_rank_ic_oo", "oos_rank_ic_lag1", "oos_top0p6_return_oo", "oos_top0p6_return_lag1"]
        )
    ]
    selected_rows = []
    fixed_rows = []
    for window, group in results.groupby("window", sort=True):
        selected = group.loc[group["rawtopstable_h5_top0p6"].idxmax()]
        selected_rows.append(selected)
        for epoch in (6, 15, 19):
            row = group[group["epoch"] == epoch].iloc[0].copy()
            row["rule"] = f"exact_e{epoch}"
            fixed_rows.append(row)
        selected_copy = selected.copy()
        selected_copy["rule"] = "rawtopstable_best"
        fixed_rows.append(selected_copy)
    selected = pd.DataFrame(selected_rows)
    fixed = pd.DataFrame(fixed_rows)
    aggregate = (
        transfer.groupby(["validation_metric", "oos_metric"], as_index=False)
        .agg(
            mean_spearman=("spearman_across_epochs", "mean"),
            positive_windows=("spearman_across_epochs", lambda values: int((values > 0).sum())),
            negative_windows=("spearman_across_epochs", lambda values: int((values < 0).sum())),
        )
    )
    rawtop_return = aggregate[
        (aggregate["validation_metric"] == "rawtopstable_h5_top0p6")
        & aggregate["oos_metric"].isin(["oos_top0p6_return_oo", "oos_top0p6_return_lag1"])
    ]
    rawtop_consistent = bool(
        len(rawtop_return)
        and ((rawtop_return["positive_windows"] == 3) | (rawtop_return["negative_windows"] == 3)).all()
    )
    lines = [
        "# 强模型 checkpoint 跨月一致性诊断",
        "",
        "本报告只使用已有 checkpoint 做推理，没有训练，也没有按单月 OOS 事后选择 checkpoint。",
        "2026 Forward 未参与。OOS 指标用于审计验证指标能否迁移到下一月，不构成逐窗选优规则。",
        "",
        "## rawtopstable 与下一月 OOS 的 epoch 横截面相关",
        "",
        primary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 三个窗口的平均相关与方向计数",
        "",
        aggregate.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 每窗由 rawtopstable 选出的 epoch",
        "",
        selected[
            [
                "window",
                "epoch",
                "rawtopstable_h5_top0p6",
                "oos_rank_ic_oo",
                "oos_rank_ic_lag1",
                "oos_top0p6_return_oo",
                "oos_top0p6_return_lag1",
                "oos_top0p6_turnover_proxy",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 固定边界 epoch 与内部 selected 对照",
        "",
        fixed[
            [
                "window",
                "rule",
                "epoch",
                "oos_rank_ic_oo",
                "oos_top0p6_return_oo",
                "oos_top0p6_return_lag1",
                "oos_top0p6_turnover_proxy",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 审计结论",
        "",
        (
            "`rawtopstable_h5_top0p6` 在三个窗口中没有形成方向一致的下一月收益关系，"
            "不能作为可靠的跨月 checkpoint 选择规则。"
            if not rawtop_consistent
            else "`rawtopstable_h5_top0p6` 在本次三个窗口中方向一致，但样本仍不足以晋级为正式规则。"
        ),
        "现有 `alpha` 与 `rawtopret_h5_top0p6` 同样没有在所有窗口形成一致方向，"
        "因此本轮不应根据这三个窗口临时拼出新的综合分。",
        "固定 e6/e15/e19 的相对表现随窗口变化，说明增加 epoch 并不单调改善下一月结果；"
        "完整 Rolling 若继续，应同时保留 exact 与 selected 信号，但正式判断仍需完整 Val/Test ledger。",
        "",
        "## 解释规则",
        "",
        "- 正相关表示该验证指标在同一窗口的 19 个 epoch 中倾向于选出下一月更好的 checkpoint。",
        "- 负相关表示验证指标与下一月目标方向相反；接近零表示选优信息弱。",
        "- 换手代理越低越好，因此它与验证收益指标出现正相关不一定是好事。",
        "- 只有跨多个窗口方向一致的关系，才值得进入新的预注册 checkpoint 规则。",
        "",
    ]
    (output / "CHECKPOINT_TRANSFER_AUDIT_ZH.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    pilot_root = _resolve(args.pilot_root)
    schedule = load_json(_resolve(args.schedule))
    cache_meta = _resolve(args.cache_meta)
    output = _resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    requested = tuple(value.strip() for value in args.windows.split(",") if value.strip())
    window_map = {item["name"]: item for item in schedule["windows"]}

    meta = load_explicit_cross_section_meta(
        cache_meta,
        project_root=ROOT,
        expected_input_dim=250,
        required_label_families=("oo", "oo_lag1"),
        logical_end=max(window_map[name]["predict_end"] for name in requested),
    )
    raw_path, _ = _resolve_label_raw_path(meta, cache_meta, "oo")
    _, lag1_shift = _resolve_label_raw_path(meta, cache_meta, "oo_lag1")
    labels = np.memmap(
        raw_path,
        dtype=np.float32,
        mode="r",
        shape=(len(meta["all_codes"]), len(meta["all_dates"]), int(meta["max_horizon"])),
    )
    date_index = pd.DatetimeIndex(pd.to_datetime(meta["all_dates"])).normalize()
    date_to_index = {value.strftime("%Y-%m-%d"): index for index, value in enumerate(date_index)}
    code_to_index = {str(code): index for index, code in enumerate(meta["all_codes"])}
    schema_meta = dict(meta)
    schema_meta["train_indices"] = [meta["train_indices"][0]]
    train_samples = samples_from_precomputed_metadata(schema_meta, "train")
    cfg = build_v9_backtest_config()
    results = []

    for window_name in requested:
        window = window_map[window_name]
        start, end = pd.Timestamp(window["predict_start"]), pd.Timestamp(window["predict_end"])
        indices = [index for index, date in enumerate(date_index) if start <= date <= end]
        samples = samples_from_precomputed_metadata(meta, time_indices=indices, require_labels=False)
        samples.sort(key=lambda sample: pd.Timestamp(sample["date"]))
        epoch_records = collect_epoch_records(pilot_root / "windows" / window_name)
        for position, record in enumerate(epoch_records, start=1):
            print(f"{window_name}: checkpoint {position}/19 epoch={record['epoch']}", flush=True)
            base = load_dl_predictor(record["checkpoint"], train_samples, cfg, args.device)
            predictor = V9RankPredictor(base, "v9_raw", cache={})
            alpha_rows = compute_rows(samples, predictor, progress_every=0)
            metrics = evaluate_alpha_rows(
                alpha_rows,
                labels=labels,
                lag1_shift=lag1_shift,
                date_to_index=date_to_index,
                code_to_index=code_to_index,
            )
            results.append(
                {
                    "window": window_name,
                    "epoch": int(record["epoch"]),
                    "legacy_stage_id": record["legacy_stage_id"],
                    "checkpoint": record["checkpoint"],
                    "alpha": float(record["alpha"]),
                    "rawtopret_h5_top0p6": float(record["rawtopret_h5_top0p6"]),
                    "rawtopstable_h5_top0p6": float(record["rawtopstable_h5_top0p6"]),
                    **metrics,
                }
            )
            del predictor, base, alpha_rows
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    frame = pd.DataFrame(results).sort_values(["window", "epoch"])
    transfer = metric_transfer_summary(frame)
    frame.to_csv(output / "checkpoint_oos_metrics.csv", index=False)
    transfer.to_csv(output / "metric_transfer_correlations.csv", index=False)
    _write_report(frame, transfer, output)
    manifest = {
        "schema": "strong_checkpoint_transfer_audit_v1",
        "training_launched": False,
        "forward_used": False,
        "pilot_root": str(pilot_root),
        "cache_meta": str(cache_meta),
        "windows": list(requested),
        "checkpoints": int(len(frame)),
        "outputs": [
            "checkpoint_oos_metrics.csv",
            "metric_transfer_correlations.csv",
            "CHECKPOINT_TRANSFER_AUDIT_ZH.md",
        ],
    }
    (output / "audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
