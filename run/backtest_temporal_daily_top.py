"""Daily long-only backtest for a TemporalCrossAlphaModel checkpoint."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from backtest.reports import calc_extended_metrics, calc_metrics
from core.temporal_train_utils import SENTINEL, TemporalMemmapDataset
from run.eval_temporal_full import (
    apply_checkpoint_horizons,
    build_cfg,
    build_model,
    forward_temporal_chunked,
    load_meta,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Backtest temporal alpha daily top portfolios")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--meta", default=None)
    parser.add_argument("--split", default="val", choices=["val", "test", "trainval", "all"])
    parser.add_argument("--output-dir", default="backtest_results_temporal_daily_top_20260531")
    parser.add_argument("--top-fracs", default="0.05,0.10")
    parser.add_argument("--temporal-chunk", type=int, default=512)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--commission-rate", type=float, default=0.0001)
    parser.add_argument("--stamp-tax-rate", type=float, default=0.0005)
    parser.add_argument("--slippage-rate", type=float, default=0.0005)
    parser.add_argument("--max-weight", type=float, default=0.05)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--limit-dates", type=int, default=None)
    return parser.parse_args()


def load_price_volume_light(data_dir, needed_codes):
    price_dict = {}
    vol_dict = {}
    needed = set(needed_codes)
    for code in sorted(needed):
        path = Path(data_dir) / f"{code}.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, usecols=["trade_date", "close", "volume"])
            df.columns = df.columns.str.strip().str.lower()
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df.set_index("trade_date", inplace=True)
            price_dict[code] = df["close"].astype(float)
            vol_dict[code] = df["volume"].astype(float)
        except Exception:
            continue
    return price_dict, vol_dict


def build_universe_matrix_light(price_dict, vol_dict, all_codes):
    all_dates = sorted(set().union(*[price_dict[c].index for c in all_codes if c in price_dict]))
    all_dates = pd.DatetimeIndex(all_dates)
    n_codes, n_dates = len(all_codes), len(all_dates)
    price_mat = np.full((n_codes, n_dates), np.nan, dtype=np.float64)
    vol_mat = np.full((n_codes, n_dates), np.nan, dtype=np.float64)
    code2idx = {c: i for i, c in enumerate(all_codes)}
    for code in all_codes:
        i = code2idx[code]
        if code in price_dict:
            price_mat[i] = price_dict[code].reindex(all_dates).values
        if code in vol_dict:
            vol_mat[i] = vol_dict[code].reindex(all_dates).values
    return price_mat, vol_mat, all_dates, code2idx


def valid_indices_for_item(ds, item):
    t = int(item["time_index"].item())
    start = t - ds.lookback + 1
    valid_today = (ds.X_mm[:, t, 0] != SENTINEL) & (ds.Y_mm[:, t] != SENTINEL)
    valid_seq = (ds.SEQ_mm[:, start:t + 1, 0] != SENTINEL).all(axis=1)
    return np.where(valid_today & valid_seq)[0]


@torch.no_grad()
def compute_daily_alphas(args, model, ds, cfg, device):
    regime_dim = cfg.risk_full_dim if hasattr(cfg, "risk_full_dim") else None
    from core.train_utils import get_regime_dim
    regime_dim = get_regime_dim(cfg)
    rows = []
    n_dates = len(ds) if args.limit_dates is None else min(len(ds), args.limit_dates)
    t0 = time.time()
    for i in range(n_dates):
        item = ds[i]
        n = int(item["X"].shape[0])
        if n < 10:
            continue
        valid_idx = valid_indices_for_item(ds, item)
        codes = ds.all_codes[valid_idx]
        if len(codes) != n:
            raise RuntimeError(f"code count mismatch at {i}: codes={len(codes)} tensor={n}")

        X = item["X"].unsqueeze(0).to(device)
        X_seq = item["X_seq"].unsqueeze(0).to(device)
        risk = item["risk"].unsqueeze(0).to(device)
        industry_ids = item["industry_ids"].unsqueeze(0).to(device)
        mask = torch.ones(1, n, dtype=torch.bool, device=device)

        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
            alpha_raw, _ = forward_temporal_chunked(
                model,
                X,
                X_seq,
                risk[..., :regime_dim],
                mask,
                industry_ids,
                args.temporal_chunk,
            )
        alpha = alpha_raw[0].float().cpu().numpy()
        date = pd.Timestamp(ds.all_dates[int(item["time_index"].item())])
        order = np.argsort(alpha)[::-1]
        rows.append({
            "date": date,
            "codes": codes[order].tolist(),
            "alpha": alpha[order].astype(float).tolist(),
            "n_stocks": n,
        })
        if args.progress_every > 0 and (i + 1) % args.progress_every == 0:
            print(f"alpha {i + 1}/{n_dates} | n={n} | time={(time.time() - t0) / 60:.1f}m", flush=True)
    return rows


def explicit_cost(trade, commission_rate, stamp_tax_rate, slippage_rate):
    buy = float(np.sum(np.maximum(trade, 0.0)))
    sell = float(np.sum(np.maximum(-trade, 0.0)))
    gross = buy + sell
    return {
        "cost": commission_rate * gross + stamp_tax_rate * sell + slippage_rate * gross,
        "buy_turnover": buy,
        "sell_turnover": sell,
        "turnover": gross,
        "commission": commission_rate * gross,
        "stamp_tax": stamp_tax_rate * sell,
        "slippage": slippage_rate * gross,
    }


def build_target_weights(selected, code2idx, n_codes, max_weight):
    w = np.zeros(n_codes, dtype=np.float64)
    if not selected:
        return w
    ew = min(max_weight, 1.0 / len(selected))
    for code in selected:
        idx = code2idx.get(code)
        if idx is not None:
            w[idx] = ew
    gross = np.sum(w)
    if gross > 1e-12:
        w = w / gross
    return w


def run_one_backtest(alpha_rows, price_mat, all_dates, code2idx, top_frac, args):
    n_codes, t_total = price_mat.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        ret_daily = price_mat[:, 1:] / price_mat[:, :-1] - 1.0
    ret_daily[~np.isfinite(ret_daily)] = 0.0

    target_by_entry = {}
    diagnostics = []
    for row in alpha_rows:
        pos = all_dates.searchsorted(row["date"], side="right")
        if pos <= 0 or pos >= t_total:
            continue
        k = max(1, int(len(row["codes"]) * top_frac))
        selected = row["codes"][:k]
        target_by_entry[int(pos)] = build_target_weights(selected, code2idx, n_codes, args.max_weight)

    entry_days = sorted(target_by_entry)
    if not entry_days:
        empty = np.asarray([], dtype=np.float64)
        return {
            "top_frac": top_frac,
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
            "total_cost": 0.0,
            "total_commission": 0.0,
            "total_stamp_tax": 0.0,
            "total_slippage": 0.0,
        }, empty, pd.DataFrame()
    first_entry = entry_days[0]
    last_return_day = min(entry_days[-1] + 1, t_total - 1)

    current = np.zeros(n_codes, dtype=np.float64)
    weights = np.zeros((t_total, n_codes), dtype=np.float32)
    daily_costs = np.zeros(t_total, dtype=np.float64)
    for day in range(t_total):
        target = target_by_entry.get(day)
        if target is not None:
            trade = target - current
            c = explicit_cost(trade, args.commission_rate, args.stamp_tax_rate, args.slippage_rate)
            daily_costs[day] = c["cost"]
            diagnostics.append({"day": day, "date": str(all_dates[day]), **c})
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
    diag_df = pd.DataFrame(diagnostics)
    row = {
        "top_frac": top_frac,
        "n_return_days": int(len(returns_active)),
        "ann": float(ann),
        "sharpe": float(sharpe),
        "mdd": float(mdd),
        "calmar": float(ext.get("calmar", 0.0)),
        "sortino": float(ext.get("sortino", 0.0)),
        "win_rate": float(ext.get("win_rate", 0.0)),
        "avg_daily_return": float(np.mean(returns_active)) if len(returns_active) else 0.0,
        "vol": float(np.std(returns_active) * np.sqrt(252)) if len(returns_active) else 0.0,
        "avg_turnover": float(diag_df["turnover"].mean()) if not diag_df.empty else 0.0,
        "total_cost": float(daily_costs.sum()),
        "total_commission": float(diag_df["commission"].sum()) if not diag_df.empty else 0.0,
        "total_stamp_tax": float(diag_df["stamp_tax"].sum()) if not diag_df.empty else 0.0,
        "total_slippage": float(diag_df["slippage"].sum()) if not diag_df.empty else 0.0,
    }
    return row, returns_active, diag_df


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    meta_path, meta = load_meta(args, checkpoint)
    cfg = build_cfg(meta)
    cfg = apply_checkpoint_horizons(cfg, checkpoint)
    ds = TemporalMemmapDataset(meta, split=args.split)
    model = build_model(meta, cfg, checkpoint, device)

    print(f"Computing temporal alphas: dates={len(ds)}, split={args.split}, device={device}", flush=True)
    alpha_rows = compute_daily_alphas(args, model, ds, cfg, device)
    alpha_path = out_dir / "temporal_daily_alpha_top_order.jsonl"
    with alpha_path.open("w", encoding="utf-8") as f:
        for row in alpha_rows:
            serial = {**row, "date": str(row["date"])}
            f.write(json.dumps(serial, ensure_ascii=False) + "\n")

    all_codes = sorted(set(code for row in alpha_rows for code in row["codes"]))
    print(f"Loading prices for {len(all_codes)} temporal codes...", flush=True)
    price_dict, vol_dict = load_price_volume_light("data/raw", all_codes)
    price_mat, _, all_dates, code2idx = build_universe_matrix_light(price_dict, vol_dict, all_codes)

    summary_rows = []
    for top_frac in [float(x.strip()) for x in args.top_fracs.split(",") if x.strip()]:
        row, returns, diag_df = run_one_backtest(alpha_rows, price_mat, all_dates, code2idx, top_frac, args)
        summary_rows.append(row)
        tag = int(round(top_frac * 1000))
        pd.DataFrame({"return": returns}).to_csv(out_dir / f"returns_top{tag:03d}.csv", index=False)
        diag_df.to_csv(out_dir / f"diagnostics_top{tag:03d}.csv", index=False)
        print(f"top_frac={top_frac:.3f} ann={row['ann']:.2f}% sharpe={row['sharpe']:.3f} "
              f"mdd={row['mdd'] * 100:.2f}% turnover={row['avg_turnover']:.3f}", flush=True)

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "temporal_daily_top_summary.csv"
    summary.to_csv(summary_path, index=False)

    md_lines = [
        "# Temporal daily top backtest",
        "",
        f"- checkpoint: `{args.checkpoint}`",
        f"- checkpoint epoch: `{checkpoint.get('epoch')}`",
        f"- meta: `{meta_path}`",
        f"- split: `{args.split}`",
        f"- costs: commission={args.commission_rate}, stamp_tax={args.stamp_tax_rate}, slippage={args.slippage_rate}",
        "",
        "| top_frac | ann | sharpe | mdd | avg_turnover | total_cost |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        md_lines.append(
            f"| {row['top_frac']:.3f} | {row['ann']:.2f}% | {row['sharpe']:.3f} | "
            f"{row['mdd'] * 100:.2f}% | {row['avg_turnover']:.3f} | {row['total_cost']:.4f} |"
        )
    (out_dir / "temporal_daily_top_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Saved summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
