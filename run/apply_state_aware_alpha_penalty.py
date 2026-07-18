"""Apply a state-aware soft rerank to saved Alpha JSONL files.

The transform is deliberately small and interpretable: keep the base Alpha
ranking as the main signal, then softly penalize candidates that become fragile
when global US/HK pressure is elevated. The output is still a normal Alpha JSONL
file, so it can be tested by the official realistic open-ledger backtest.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha.io import load_alpha_rows, write_alpha_rows
from alpha.transforms import percentile_map
from backtest.open_ledger import load_industry_map, normalize_ts_code


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--global-features", required=True)
    parser.add_argument("--global-pressure-col", default="global_us_hk_pressure")
    parser.add_argument("--global-threshold", type=float, default=0.035)
    parser.add_argument("--global-width", type=float, default=0.055)
    parser.add_argument("--industry-csv", default="data/stock_industry.csv")
    parser.add_argument("--top-frac", type=float, default=0.006)
    parser.add_argument("--min-top-n", type=int, default=30)
    parser.add_argument("--penalty", type=float, default=0.05)
    parser.add_argument("--momentum-weight", type=float, default=0.40)
    parser.add_argument("--beta-weight", type=float, default=0.20)
    parser.add_argument("--vol-weight", type=float, default=0.20)
    parser.add_argument("--industry-weight", type=float, default=0.20)
    parser.add_argument("--drawdown-weight", type=float, default=0.0)
    parser.add_argument("--momentum-mode", choices=["high", "extreme"], default="high")
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args(argv)


def _read_table(path):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_global_pressure(path, pressure_col):
    frame = _read_table(path)
    if "date" not in frame.columns:
        raise ValueError(f"{path} is missing date column")
    if pressure_col not in frame.columns:
        raise ValueError(f"{path} is missing {pressure_col}")
    frame = frame[["date", pressure_col]].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame[pressure_col] = pd.to_numeric(frame[pressure_col], errors="coerce").fillna(0.0)
    return dict(zip(frame["date"], frame[pressure_col].astype(float)))


def load_index_returns(data_dir, start_date, end_date):
    path = Path(data_dir) / "hs300_index.csv"
    if not path.exists():
        return pd.Series(dtype=np.float64)
    header = pd.read_csv(path, nrows=0)
    date_col = "trade_date" if "trade_date" in header.columns else "date"
    frame = pd.read_csv(path, usecols=[date_col, "close"], parse_dates=[date_col])
    frame = frame.rename(columns={date_col: "trade_date"})
    frame = frame.sort_values("trade_date")
    frame["trade_date"] = frame["trade_date"].dt.normalize()
    frame = frame[
        (frame["trade_date"] >= start_date - pd.Timedelta(days=160))
        & (frame["trade_date"] <= end_date)
    ]
    close = pd.to_numeric(frame["close"], errors="coerce")
    ret = close.pct_change()
    return pd.Series(ret.to_numpy(dtype=np.float64), index=frame["trade_date"])


def _rank01(values):
    arr = np.asarray(values, dtype=np.float64)
    out = np.full(arr.shape, 0.5, dtype=np.float64)
    mask = np.isfinite(arr)
    if mask.sum() <= 1:
        return out
    order = np.argsort(arr[mask], kind="mergesort")
    ranks = np.empty(order.shape[0], dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, order.shape[0])
    out[mask] = ranks
    return out


def _clip01(values):
    return np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)


def _load_code_features(data_dir, code, start_date, end_date, index_returns):
    path = Path(data_dir) / f"{code}.csv"
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path, usecols=["trade_date", "close"], parse_dates=["trade_date"])
    except Exception:
        return {}
    if frame.empty:
        return {}
    frame = frame.sort_values("trade_date")
    frame["trade_date"] = frame["trade_date"].dt.normalize()
    frame = frame[
        (frame["trade_date"] >= start_date - pd.Timedelta(days=170))
        & (frame["trade_date"] <= end_date)
    ].copy()
    if frame.empty:
        return {}
    close = pd.to_numeric(frame["close"], errors="coerce")
    ret = close.pct_change()
    frame["momentum20"] = close / close.shift(20) - 1.0
    frame["volatility60"] = ret.rolling(60, min_periods=20).std() * np.sqrt(252.0)
    frame["drawdown20"] = close / close.rolling(20, min_periods=5).max() - 1.0
    if not index_returns.empty:
        stock_ret = pd.Series(ret.to_numpy(dtype=np.float64), index=frame["trade_date"])
        aligned = pd.concat(
            [stock_ret.rename("stock"), index_returns.rename("index")],
            axis=1,
            sort=False,
        )
        cov = aligned["stock"].rolling(60, min_periods=20).cov(aligned["index"])
        var = aligned["index"].rolling(60, min_periods=20).var()
        beta = (cov / var.replace(0.0, np.nan)).reindex(frame["trade_date"])
        frame["beta60"] = beta.to_numpy(dtype=np.float64)
    else:
        frame["beta60"] = np.nan
    feature_cols = ["momentum20", "volatility60", "beta60", "drawdown20"]
    frame = frame[["trade_date", *feature_cols]]
    frame = frame[(frame["trade_date"] >= start_date) & (frame["trade_date"] <= end_date)]
    return {
        pd.Timestamp(row.trade_date): (
            float(row.momentum20) if np.isfinite(row.momentum20) else np.nan,
            float(row.volatility60) if np.isfinite(row.volatility60) else np.nan,
            float(row.beta60) if np.isfinite(row.beta60) else np.nan,
            float(row.drawdown20) if np.isfinite(row.drawdown20) else np.nan,
        )
        for row in frame.itertuples(index=False)
    }


def load_stock_features(data_dir, codes, start_date, end_date, progress_every=1000):
    index_returns = load_index_returns(data_dir, start_date, end_date)
    features = {}
    for i, code in enumerate(sorted(codes), start=1):
        code_norm = normalize_ts_code(code)
        code_features = _load_code_features(data_dir, code_norm, start_date, end_date, index_returns)
        if code_features:
            features[code_norm] = code_features
        if progress_every > 0 and i % progress_every == 0:
            print(f"loaded state-aware stock features {i}/{len(codes)}", flush=True)
    return features


def candidate_industry_share(codes, industry_map, top_n):
    top_codes = [normalize_ts_code(code) for code in codes[:top_n]]
    industries = [industry_map.get(code, "UNKNOWN") for code in top_codes]
    counts = {}
    for industry in industries:
        counts[industry] = counts.get(industry, 0) + 1
    denom = max(len(top_codes), 1)
    return {industry: count / denom for industry, count in counts.items()}


def compute_adjusted_row(
    row,
    global_pressure,
    stock_features,
    industry_map,
    args,
):
    date = pd.Timestamp(row["date"]).normalize()
    codes = [normalize_ts_code(code) for code in row.get("codes", [])]
    base_scores = percentile_map(codes)
    n = len(codes)
    top_n = min(n, max(int(args.min_top_n), int(np.ceil(n * float(args.top_frac)))))
    industry_share = candidate_industry_share(codes, industry_map, top_n)

    pressure = float(global_pressure.get(date, 0.0))
    width = max(float(args.global_width), 1e-12)
    stress = float(np.clip((pressure - float(args.global_threshold)) / width, 0.0, 1.0))

    raw_features = []
    for code in codes:
        raw_features.append(stock_features.get(code, {}).get(date, (np.nan, np.nan, np.nan, np.nan)))
    raw_features = np.asarray(raw_features, dtype=np.float64) if raw_features else np.empty((0, 4))

    if raw_features.size:
        momentum_rank = _rank01(raw_features[:, 0])
        if args.momentum_mode == "extreme":
            momentum_risk = _clip01(np.abs(momentum_rank - 0.5) * 2.0)
        else:
            momentum_risk = momentum_rank
        vol_risk = _rank01(raw_features[:, 1])
        beta_risk = _clip01((np.nan_to_num(raw_features[:, 2], nan=1.0) - 0.8) / 1.0)
        drawdown_risk = _clip01(np.abs(np.nan_to_num(raw_features[:, 3], nan=0.0)) / 0.20)
    else:
        momentum_risk = vol_risk = beta_risk = drawdown_risk = np.zeros(n, dtype=np.float64)

    ind_risk = np.asarray(
        [industry_share.get(industry_map.get(code, "UNKNOWN"), 0.0) for code in codes],
        dtype=np.float64,
    )
    ind_risk = _clip01(ind_risk / max(float(args.top_frac) * 10.0, 0.10))

    total_weight = (
        float(args.momentum_weight)
        + float(args.beta_weight)
        + float(args.vol_weight)
        + float(args.industry_weight)
        + float(args.drawdown_weight)
    )
    if total_weight <= 0:
        fragility = np.zeros(n, dtype=np.float64)
    else:
        fragility = (
            float(args.momentum_weight) * momentum_risk
            + float(args.beta_weight) * beta_risk
            + float(args.vol_weight) * vol_risk
            + float(args.industry_weight) * ind_risk
            + float(args.drawdown_weight) * drawdown_risk
        ) / total_weight

    adjusted = {}
    for i, code in enumerate(codes):
        adjusted[code] = float(base_scores[code] - float(args.penalty) * stress * fragility[i])

    items = sorted(adjusted.items(), key=lambda item: (-item[1], item[0]))
    before_top = set(codes[:top_n])
    after_top = {code for code, _ in items[:top_n]}
    moved_out = len(before_top - after_top)
    avg_penalty = float((float(args.penalty) * stress * fragility).mean()) if n else 0.0

    return {
        "date": date.strftime("%Y-%m-%d"),
        "codes": [code for code, _ in items],
        "alpha": [score for _, score in items],
        "n_stocks": len(items),
        "state_aware_alpha_penalty": {
            "global_pressure_col": args.global_pressure_col,
            "global_pressure": pressure,
            "global_threshold": float(args.global_threshold),
            "global_width": float(args.global_width),
            "stress": stress,
            "penalty": float(args.penalty),
            "momentum_weight": float(args.momentum_weight),
            "beta_weight": float(args.beta_weight),
            "vol_weight": float(args.vol_weight),
            "industry_weight": float(args.industry_weight),
            "drawdown_weight": float(args.drawdown_weight),
            "momentum_mode": args.momentum_mode,
            "top_n": int(top_n),
            "top_pool_moved_out": int(moved_out),
            "avg_effective_penalty": avg_penalty,
        },
    }


def transform_rows(rows, global_pressure, stock_features, industry_map, args):
    return [
        compute_adjusted_row(row, global_pressure, stock_features, industry_map, args)
        for row in rows
    ]


def main(argv=None):
    args = parse_args(argv)
    rows = load_alpha_rows(args.input, timestamp_dates=True)
    if not rows:
        raise ValueError("Input Alpha file is empty")
    start_date = min(pd.Timestamp(row["date"]).normalize() for row in rows)
    end_date = max(pd.Timestamp(row["date"]).normalize() for row in rows)
    codes = {normalize_ts_code(code) for row in rows for code in row.get("codes", [])}

    global_pressure = load_global_pressure(args.global_features, args.global_pressure_col)
    industry_map = load_industry_map(args.industry_csv)
    stock_features = load_stock_features(
        args.data_dir,
        codes,
        start_date,
        end_date,
        progress_every=args.progress_every,
    )
    transformed = transform_rows(rows, global_pressure, stock_features, industry_map, args)
    output = write_alpha_rows(args.output, transformed)
    moved = sum(row["state_aware_alpha_penalty"]["top_pool_moved_out"] for row in transformed)
    avg_stress = float(np.mean([row["state_aware_alpha_penalty"]["stress"] for row in transformed]))
    print(
        json.dumps(
            {
                "input": args.input,
                "output": str(output),
                "dates": len(transformed),
                "codes": len(codes),
                "avg_stress": avg_stress,
                "top_pool_moved_out_total": int(moved),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
