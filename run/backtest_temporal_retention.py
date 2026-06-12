"""Retention-first daily long-only backtest for temporal alpha checkpoints."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from backtest.reports import calc_extended_metrics, calc_metrics
from core.temporal_train_utils import TemporalMemmapDataset
from run.backtest_temporal_daily_top import (
    build_target_weights,
    build_universe_matrix_light,
    compute_daily_alphas,
    explicit_cost,
    load_price_volume_light,
)
from run.eval_temporal_full import apply_checkpoint_horizons, build_cfg, build_model, load_meta


def parse_args():
    parser = argparse.ArgumentParser(description="Temporal retention-first daily backtest")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path, or comma-separated paths for an ensemble")
    parser.add_argument("--ensemble-mode", default="rank_mean", choices=["rank_mean", "alpha_mean"])
    parser.add_argument("--meta", default=None)
    parser.add_argument("--split", default="val", choices=["val", "test", "trainval", "all"])
    parser.add_argument("--output-dir", default="backtest_results_temporal_retention_20260531")
    parser.add_argument("--target-fracs", default="0.05,0.10")
    parser.add_argument("--hold-fracs", default="0.10,0.15,0.20")
    parser.add_argument("--temporal-chunk", type=int, default=512)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--commission-rate", type=float, default=0.0001)
    parser.add_argument("--stamp-tax-rate", type=float, default=0.0005)
    parser.add_argument("--slippage-rate", type=float, default=0.0005)
    parser.add_argument("--max-weight", type=float, default=0.05)
    parser.add_argument("--weight-mode", default="equal", choices=["equal", "rank_linear", "alpha_positive"])
    parser.add_argument("--market-timing-mode", default="none", choices=["none", "legacy", "dynamic"])
    parser.add_argument("--market-min-mult", type=float, default=0.20)
    parser.add_argument("--market-max-mult", type=float, default=1.00)
    parser.add_argument("--index-file", default="hs300_index.csv")
    parser.add_argument("--progress-every", type=int, default=40)
    parser.add_argument("--limit-dates", type=int, default=None)
    return parser.parse_args()


def parse_checkpoint_paths(raw):
    return [p.strip() for p in str(raw).split(",") if p.strip()]


def combine_alpha_rows(alpha_rows_list, mode="rank_mean"):
    if len(alpha_rows_list) == 1:
        return alpha_rows_list[0]

    by_date = []
    for rows in alpha_rows_list:
        by_date.append({pd.Timestamp(row["date"]): row for row in rows})
    common_dates = sorted(set.intersection(*[set(d.keys()) for d in by_date]))
    combined = []
    for date in common_dates:
        scores = {}
        counts = {}
        for date_map in by_date:
            row = date_map[date]
            codes = row["codes"]
            alphas = row["alpha"]
            n = len(codes)
            if n == 0:
                continue
            if mode == "rank_mean":
                denom = max(n - 1, 1)
                for rank, code in enumerate(codes):
                    scores[code] = scores.get(code, 0.0) + (1.0 - rank / denom)
                    counts[code] = counts.get(code, 0) + 1
            elif mode == "alpha_mean":
                vals = np.asarray(alphas, dtype=np.float64)
                finite = np.isfinite(vals)
                if np.count_nonzero(finite) >= 2:
                    vals = (vals - np.nanmean(vals[finite])) / (np.nanstd(vals[finite]) + 1e-8)
                else:
                    vals = np.zeros_like(vals)
                for code, value in zip(codes, vals):
                    scores[code] = scores.get(code, 0.0) + float(value)
                    counts[code] = counts.get(code, 0) + 1
            else:
                raise ValueError(f"Unknown ensemble mode: {mode}")

        items = [(code, score / max(counts.get(code, 1), 1)) for code, score in scores.items()]
        items.sort(key=lambda x: x[1], reverse=True)
        combined.append({
            "date": date,
            "codes": [code for code, _ in items],
            "alpha": [float(score) for _, score in items],
            "n_stocks": len(items),
        })
    return combined


def _empty_row(target_frac, hold_frac):
    return {
        "target_frac": target_frac,
        "hold_frac": hold_frac,
        "n_return_days": 0,
        "ann": 0.0,
        "sharpe": 0.0,
        "mdd": 0.0,
        "calmar": 0.0,
        "sortino": 0.0,
        "win_rate": 0.0,
        "avg_daily_return": 0.0,
        "vol": 0.0,
        "avg_turnover": 0.0,
        "avg_holding_days": 0.0,
        "avg_names": 0.0,
        "weight_mode": "",
        "total_cost": 0.0,
        "total_commission": 0.0,
        "total_stamp_tax": 0.0,
        "total_slippage": 0.0,
    }


def build_weight_target(selected, code2idx, n_codes, args, rank_map=None, alpha_map=None):
    if args.weight_mode == "equal":
        return build_target_weights(selected, code2idx, n_codes, args.max_weight)

    w = np.zeros(n_codes, dtype=np.float64)
    if not selected:
        return w
    selected_sorted = sorted(selected, key=lambda c: rank_map.get(c, len(selected)) if rank_map else 0)
    if args.weight_mode == "rank_linear":
        raw = np.arange(len(selected_sorted), 0, -1, dtype=np.float64)
    elif args.weight_mode == "alpha_positive":
        vals = np.asarray([alpha_map.get(c, np.nan) for c in selected_sorted], dtype=np.float64)
        finite = np.isfinite(vals)
        if np.count_nonzero(finite) >= 2:
            vals = np.where(finite, vals, np.nanmean(vals[finite]))
            z = (vals - np.mean(vals)) / (np.std(vals) + 1e-8)
            raw = np.maximum(z, 0.0) + 0.05
        else:
            raw = np.ones(len(selected_sorted), dtype=np.float64)
    else:
        raw = np.ones(len(selected_sorted), dtype=np.float64)

    raw = raw / max(float(np.sum(raw)), 1e-12)
    if args.max_weight > 0:
        raw = np.minimum(raw, float(args.max_weight))
        raw = raw / max(float(np.sum(raw)), 1e-12)
    for code, weight in zip(selected_sorted, raw):
        idx = code2idx.get(code)
        if idx is not None:
            w[idx] = float(weight)
    gross = np.sum(np.abs(w))
    if gross > 1e-12:
        w = w / gross
    return w


def build_retention_targets(alpha_rows, all_dates, code2idx, n_codes, target_frac, hold_frac, args):
    target_by_entry = {}
    diagnostics = []
    holding_ages = {}
    closed_ages = []
    current_selected = []

    for row in alpha_rows:
        pos = all_dates.searchsorted(row["date"], side="right")
        if pos <= 0 or pos >= len(all_dates):
            continue

        codes = row["codes"]
        n = len(codes)
        if n == 0:
            continue
        target_n = max(1, int(n * target_frac))
        hold_n = max(target_n, int(n * hold_frac))
        rank_map = {code: i for i, code in enumerate(codes)}
        alpha_map = {code: float(alpha) for code, alpha in zip(codes, row["alpha"])}

        kept = [code for code in current_selected if rank_map.get(code, n + 1) < hold_n]
        if len(kept) > target_n:
            kept = sorted(kept, key=lambda c: rank_map.get(c, n + 1))[:target_n]

        selected_set = set(kept)
        selected = list(kept)
        for code in codes:
            if len(selected) >= target_n:
                break
            if code not in selected_set:
                selected.append(code)
                selected_set.add(code)

        prev_set = set(current_selected)
        for code in prev_set - selected_set:
            closed_ages.append(holding_ages.get(code, 1))
            holding_ages.pop(code, None)
        for code in selected:
            holding_ages[code] = holding_ages.get(code, 0) + 1

        current_selected = selected
        target_by_entry[int(pos)] = build_weight_target(
            selected,
            code2idx,
            n_codes,
            args,
            rank_map=rank_map,
            alpha_map=alpha_map,
        )
        diagnostics.append({
            "day": int(pos),
            "date": str(all_dates[int(pos)]),
            "target_n": int(target_n),
            "hold_n": int(hold_n),
            "kept_n": int(len(kept)),
            "selected_n": int(len(selected)),
            "avg_live_age": float(np.mean(list(holding_ages.values()))) if holding_ages else 0.0,
            "weight_mode": args.weight_mode,
        })

    closed_ages.extend(holding_ages.values())
    avg_holding_days = float(np.mean(closed_ages)) if closed_ages else 0.0
    return target_by_entry, pd.DataFrame(diagnostics), avg_holding_days


def load_index_returns(data_dir, index_file, all_dates):
    path = Path(data_dir) / index_file
    if not path.exists():
        return pd.Series(np.nan, index=all_dates), pd.Series(0.0, index=all_dates)
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    date_col = "trade_date" if "trade_date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    close = df["close"].astype(float).reindex(all_dates)
    daily = close.pct_change().fillna(0.0)
    return close, daily


def compute_market_multiplier(
    idx_close,
    idx_daily,
    ret_daily,
    col_cur,
    mode,
    min_mult,
    max_mult,
    legacy_bear_mult=0.7,
    legacy_crash_mult=0.3,
):
    if mode in (None, "", "none"):
        return 1.0
    if col_cur < 60 or not np.isfinite(idx_close.iloc[col_cur]):
        return float(max_mult)

    idx_cur = float(idx_close.iloc[col_cur])
    idx_ma60 = float(idx_close.iloc[col_cur - 60:col_cur].mean())
    if mode == "legacy":
        market_mult = float(legacy_bear_mult) if idx_cur < idx_ma60 else 1.0
        if col_cur >= 120 and np.isfinite(idx_close.iloc[col_cur - 120]) and idx_close.iloc[col_cur - 120] > 0:
            idx_ret_6m = idx_cur / float(idx_close.iloc[col_cur - 120]) - 1.0
            if idx_ret_6m < -0.10:
                market_mult = min(market_mult, float(legacy_crash_mult))
        return float(np.clip(market_mult, 0.0, max_mult))

    if mode != "dynamic":
        raise ValueError(f"Unknown market_timing_mode: {mode}")

    ma_score = 1.0 if idx_cur >= idx_ma60 else 0.0
    mom_score = 0.5
    if col_cur >= 20 and np.isfinite(idx_close.iloc[col_cur - 20]) and idx_close.iloc[col_cur - 20] > 0:
        mom20 = idx_cur / float(idx_close.iloc[col_cur - 20]) - 1.0
        mom_score = float(np.clip((mom20 + 0.08) / 0.16, 0.0, 1.0))

    breadth_score = 0.5
    if col_cur >= 20 and ret_daily.shape[1] >= col_cur:
        recent_rets = ret_daily[:, col_cur - 20:col_cur]
        finite = np.isfinite(recent_rets)
        if np.any(finite):
            breadth_score = float(np.nanmean(recent_rets[finite] > 0))

    vol_score = 0.5
    if col_cur >= 20:
        recent_idx_ret = np.asarray(idx_daily.iloc[col_cur - 20:col_cur], dtype=float)
        recent_idx_ret = recent_idx_ret[np.isfinite(recent_idx_ret)]
        if len(recent_idx_ret) > 5:
            ann_vol = float(np.std(recent_idx_ret) * np.sqrt(252))
            vol_score = float(1.0 - np.clip((ann_vol - 0.15) / 0.25, 0.0, 1.0))

    score = 0.4 * ma_score + 0.3 * mom_score + 0.2 * breadth_score + 0.1 * vol_score
    market_mult = min_mult + (max_mult - min_mult) * score
    if col_cur >= 120 and np.isfinite(idx_close.iloc[col_cur - 120]) and idx_close.iloc[col_cur - 120] > 0:
        idx_ret_6m = idx_cur / float(idx_close.iloc[col_cur - 120]) - 1.0
        if idx_ret_6m < -0.10:
            market_mult = min(market_mult, max(min_mult, 0.35))
    return float(np.clip(market_mult, min_mult, max_mult))


def run_one_retention(
    alpha_rows,
    price_mat,
    all_dates,
    code2idx,
    target_frac,
    hold_frac,
    args,
    idx_close=None,
    idx_daily=None,
):
    n_codes, t_total = price_mat.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        ret_daily = price_mat[:, 1:] / price_mat[:, :-1] - 1.0
    ret_daily[~np.isfinite(ret_daily)] = 0.0

    target_by_entry, state_df, avg_holding_days = build_retention_targets(
        alpha_rows, all_dates, code2idx, n_codes, target_frac, hold_frac, args
    )
    entry_days = sorted(target_by_entry)
    if not entry_days:
        return _empty_row(target_frac, hold_frac), np.asarray([], dtype=np.float64), state_df

    first_entry = entry_days[0]
    last_return_day = min(entry_days[-1] + 1, t_total - 1)

    current = np.zeros(n_codes, dtype=np.float64)
    weights = np.zeros((t_total, n_codes), dtype=np.float32)
    daily_costs = np.zeros(t_total, dtype=np.float64)
    trade_rows = []
    for day in range(t_total):
        target = target_by_entry.get(day)
        if target is not None:
            market_mult = 1.0
            if args.market_timing_mode != "none" and idx_close is not None and idx_daily is not None:
                market_mult = compute_market_multiplier(
                    idx_close,
                    idx_daily,
                    ret_daily,
                    max(day - 1, 0),
                    args.market_timing_mode,
                    args.market_min_mult,
                    args.market_max_mult,
                    getattr(args, "legacy_bear_mult", 0.7),
                    getattr(args, "legacy_crash_mult", 0.3),
                )
                target = target * float(market_mult)
            trade = target - current
            c = explicit_cost(trade, args.commission_rate, args.stamp_tax_rate, args.slippage_rate)
            daily_costs[day] = c["cost"]
            c["market_mult"] = float(market_mult)
            trade_rows.append({"day": day, "date": str(all_dates[day]), **c})
            current = target
        weights[day] = current.astype(np.float32)

    returns = []
    for day in range(1, t_total):
        w = weights[day].astype(np.float64)
        lev = np.sum(np.abs(w))
        if lev > 1.0:
            w = w / lev
        returns.append(float(np.dot(w, ret_daily[:, day - 1]) - daily_costs[day]))
    returns = np.asarray(returns, dtype=np.float64)

    start_idx = max(first_entry - 1, 0)
    end_idx = max(last_return_day, start_idx)
    returns_active = returns[start_idx:end_idx]

    ann, sharpe, mdd = calc_metrics(returns_active)
    ext = calc_extended_metrics(returns_active)
    trade_df = pd.DataFrame(trade_rows)
    diag_df = state_df.merge(trade_df, on=["day", "date"], how="left")
    for col in ["cost", "buy_turnover", "sell_turnover", "turnover", "commission", "stamp_tax", "slippage"]:
        if col in diag_df:
            diag_df[col] = diag_df[col].fillna(0.0)

    row = {
        "target_frac": target_frac,
        "hold_frac": hold_frac,
        "n_return_days": int(len(returns_active)),
        "ann": float(ann),
        "sharpe": float(sharpe),
        "mdd": float(mdd),
        "calmar": float(ext.get("calmar", 0.0)),
        "sortino": float(ext.get("sortino", 0.0)),
        "win_rate": float(ext.get("win_rate", 0.0)),
        "avg_daily_return": float(np.mean(returns_active)) if len(returns_active) else 0.0,
        "vol": float(np.std(returns_active) * np.sqrt(252)) if len(returns_active) else 0.0,
        "avg_turnover": float(diag_df["turnover"].mean()) if "turnover" in diag_df else 0.0,
        "avg_holding_days": avg_holding_days,
        "avg_names": float(diag_df["selected_n"].mean()) if "selected_n" in diag_df else 0.0,
        "weight_mode": args.weight_mode,
        "market_timing_mode": args.market_timing_mode,
        "avg_market_mult": float(diag_df["market_mult"].mean()) if "market_mult" in diag_df else 1.0,
        "total_cost": float(daily_costs.sum()),
        "total_commission": float(diag_df["commission"].sum()) if "commission" in diag_df else 0.0,
        "total_stamp_tax": float(diag_df["stamp_tax"].sum()) if "stamp_tax" in diag_df else 0.0,
        "total_slippage": float(diag_df["slippage"].sum()) if "slippage" in diag_df else 0.0,
    }
    return row, returns_active, diag_df


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_paths = parse_checkpoint_paths(args.checkpoint)
    checkpoint = torch.load(checkpoint_paths[0], map_location=device, weights_only=False)
    meta_path, meta = load_meta(args, checkpoint)
    cfg = apply_checkpoint_horizons(build_cfg(meta), checkpoint)
    ds = TemporalMemmapDataset(meta, split=args.split)

    alpha_rows_list = []
    for ckpt_idx, ckpt_path in enumerate(checkpoint_paths, start=1):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        ckpt_cfg = apply_checkpoint_horizons(build_cfg(meta), ckpt)
        model = build_model(meta, ckpt_cfg, ckpt, device)
        print(
            f"Computing temporal alphas: checkpoint={ckpt_idx}/{len(checkpoint_paths)} "
            f"dates={len(ds)}, split={args.split}, device={device}",
            flush=True,
        )
        alpha_rows_list.append(compute_daily_alphas(args, model, ds, ckpt_cfg, device))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    alpha_rows = combine_alpha_rows(alpha_rows_list, args.ensemble_mode)
    alpha_path = out_dir / "temporal_daily_alpha_top_order.jsonl"
    with alpha_path.open("w", encoding="utf-8") as f:
        for row in alpha_rows:
            f.write(json.dumps({**row, "date": str(row["date"])}, ensure_ascii=False) + "\n")

    all_codes = sorted(set(code for row in alpha_rows for code in row["codes"]))
    print(f"Loading prices for {len(all_codes)} temporal codes...", flush=True)
    price_dict, vol_dict = load_price_volume_light("data/raw", all_codes)
    price_mat, _, all_dates, code2idx = build_universe_matrix_light(price_dict, vol_dict, all_codes)
    idx_close, idx_daily = load_index_returns("data/raw", args.index_file, all_dates)

    summary_rows = []
    target_fracs = [float(x.strip()) for x in args.target_fracs.split(",") if x.strip()]
    hold_fracs = [float(x.strip()) for x in args.hold_fracs.split(",") if x.strip()]
    for target_frac in target_fracs:
        for hold_frac in hold_fracs:
            if hold_frac < target_frac:
                continue
            row, returns, diag_df = run_one_retention(
                alpha_rows, price_mat, all_dates, code2idx, target_frac, hold_frac, args, idx_close, idx_daily
            )
            summary_rows.append(row)
            tag = f"target{int(round(target_frac * 1000)):03d}_hold{int(round(hold_frac * 1000)):03d}"
            pd.DataFrame({"return": returns}).to_csv(out_dir / f"returns_{tag}.csv", index=False)
            diag_df.to_csv(out_dir / f"diagnostics_{tag}.csv", index=False)
            print(
                f"target={target_frac:.3f} hold={hold_frac:.3f} ann={row['ann']:.2f}% "
                f"sharpe={row['sharpe']:.3f} mdd={row['mdd'] * 100:.2f}% "
                f"turnover={row['avg_turnover']:.3f} hold_days={row['avg_holding_days']:.2f} "
                f"weight={row['weight_mode']} market={row['market_timing_mode']} "
                f"mult={row['avg_market_mult']:.2f}",
                flush=True,
            )

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "temporal_retention_summary.csv"
    summary.to_csv(summary_path, index=False)

    md_lines = [
        "# Temporal retention-first backtest",
        "",
        f"- checkpoint: `{args.checkpoint}`",
        f"- checkpoint_count: `{len(checkpoint_paths)}`",
        f"- ensemble_mode: `{args.ensemble_mode}`",
        f"- checkpoint epoch: `{checkpoint.get('epoch')}`",
        f"- meta: `{meta_path}`",
        f"- split: `{args.split}`",
        f"- costs: commission={args.commission_rate}, stamp_tax={args.stamp_tax_rate}, slippage={args.slippage_rate}",
        f"- weight_mode: `{args.weight_mode}`",
        f"- market_timing_mode: `{args.market_timing_mode}`",
        "",
        "| target_frac | hold_frac | weight_mode | market | ann | sharpe | mdd | avg_turnover | avg_holding_days | avg_market_mult | total_cost |",
        "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        md_lines.append(
            f"| {row['target_frac']:.3f} | {row['hold_frac']:.3f} | {row['weight_mode']} | "
            f"{row['market_timing_mode']} | {row['ann']:.2f}% | "
            f"{row['sharpe']:.3f} | {row['mdd'] * 100:.2f}% | {row['avg_turnover']:.3f} | "
            f"{row['avg_holding_days']:.2f} | {row['avg_market_mult']:.2f} | {row['total_cost']:.4f} |"
        )
    (out_dir / "temporal_retention_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Saved summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
