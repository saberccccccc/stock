"""Build state-aware portfolio-policy rows from saved Alpha JSONL files.

This dataset is for a lightweight reranker / portfolio policy.  It uses only
signal-day and earlier information as features, and labels each candidate with
future next-open executable returns.  The label is not Alpha IC; it is a proxy
for whether a candidate improves the marginal fill decisions of the retention
portfolio under realistic open-ledger timing.
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

from alpha.io import load_alpha_rows
from backtest.open_ledger import load_industry_map, normalize_ts_code
from run.backtest_retention_open_ledger import filter_alpha_rows_by_date


HORIZONS = (1, 3, 5, 10)
HORIZON_WEIGHTS = np.asarray((0.15, 0.25, 0.35, 0.25), dtype=np.float64)
ROUND_TRIP_COST = 0.0017
DEFAULT_GLOBAL_COLS = (
    "global_us_hk_pressure",
    "global_defensive_pressure",
    "global_hk_risk_pressure",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha-jsonl", required=True)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--global-features", default=None)
    parser.add_argument("--industry-csv", default="data/stock_industry.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--candidate-end", type=int, default=120)
    parser.add_argument("--target-frac", type=float, default=0.006)
    parser.add_argument("--hold-frac", type=float, default=0.10)
    parser.add_argument("--max-reranked-fills", type=int, default=3)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--max-label-date", default=None)
    parser.add_argument("--money-scale", type=float, default=1000.0)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--global-cols", default=",".join(DEFAULT_GLOBAL_COLS))
    return parser.parse_args(argv)


def rank_pct(position, n):
    return float(position) / max(int(n) - 1, 1)


def daily_normalize(values):
    values = pd.to_numeric(values, errors="coerce")
    median = values.median()
    scale = (values - median).abs().median()
    if not np.isfinite(scale) or scale < 1e-8:
        scale = values.std()
    scale = max(float(scale), 1e-8)
    return ((values - median) / scale).clip(-5.0, 5.0)


def read_price_frame(path, start_date=None, end_date=None):
    columns = ["trade_date", "open", "high", "low", "close", "money", "volume"]
    header = pd.read_csv(path, nrows=0)
    usecols = [column for column in columns if column in header.columns]
    if "trade_date" not in usecols:
        return pd.DataFrame()
    frame = pd.read_csv(path, usecols=usecols, parse_dates=["trade_date"])
    frame = frame.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
    if start_date is not None:
        frame = frame[frame["trade_date"] >= start_date]
    if end_date is not None:
        frame = frame[frame["trade_date"] <= end_date]
    if frame.empty:
        return frame
    frame = frame.set_index("trade_date")
    for column in ("open", "high", "low", "close", "money", "volume"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_price_cache(data_dir, codes, min_date, max_date, progress_every=500):
    cache = {}
    start = pd.Timestamp(min_date) - pd.Timedelta(days=180)
    end = pd.Timestamp(max_date) + pd.Timedelta(days=20)
    for i, code in enumerate(sorted(codes), start=1):
        path = Path(data_dir) / f"{code}.csv"
        if path.exists():
            frame = read_price_frame(path, start, end)
            if not frame.empty:
                cache[code] = frame
        if progress_every > 0 and i % progress_every == 0:
            print(f"loaded policy price cache {i}/{len(codes)}", flush=True)
    return cache


def load_index_returns(data_dir, min_date, max_date):
    path = Path(data_dir) / "hs300_index.csv"
    if not path.exists():
        return pd.Series(dtype=np.float64)
    header = pd.read_csv(path, nrows=0)
    date_col = "trade_date" if "trade_date" in header.columns else "date"
    frame = pd.read_csv(path, usecols=[date_col, "close"], parse_dates=[date_col])
    frame = frame.rename(columns={date_col: "trade_date"})
    frame = frame.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
    frame = frame[
        (frame["trade_date"] >= pd.Timestamp(min_date) - pd.Timedelta(days=180))
        & (frame["trade_date"] <= pd.Timestamp(max_date))
    ]
    close = pd.to_numeric(frame["close"], errors="coerce")
    return pd.Series(close.pct_change().to_numpy(dtype=np.float64), index=frame["trade_date"])


def stock_signal_features(frame, date, index_returns):
    out = {
        "ret_1d": np.nan,
        "ret_5d": np.nan,
        "ret_20d": np.nan,
        "vol_20d": np.nan,
        "vol_60d": np.nan,
        "drawdown_20d": np.nan,
        "beta_60d": np.nan,
        "specific_vol_60d": np.nan,
        "money_ma20": np.nan,
    }
    if frame is None or frame.empty or date not in frame.index:
        return out
    pos = int(frame.index.get_loc(date))
    close = frame["close"].to_numpy(dtype=np.float64)
    current = close[pos]
    if not np.isfinite(current) or current <= 0:
        return out
    for window in (1, 5, 20):
        if pos - window >= 0 and np.isfinite(close[pos - window]) and close[pos - window] > 0:
            out[f"ret_{window}d"] = float(current / close[pos - window] - 1.0)
    start20 = max(0, pos - 20)
    high20 = np.nanmax(close[start20 : pos + 1])
    if np.isfinite(high20) and high20 > 0:
        out["drawdown_20d"] = float(current / high20 - 1.0)
    returns = pd.Series(close, index=frame.index).pct_change()
    for window in (20, 60):
        hist = returns.iloc[max(0, pos - window + 1) : pos + 1].dropna()
        if len(hist) >= max(5, window // 3):
            out[f"vol_{window}d"] = float(hist.std() * np.sqrt(252.0))
    if "money" in frame.columns:
        money = frame["money"].iloc[max(0, pos - 19) : pos + 1]
        out["money_ma20"] = float(money.mean()) if len(money.dropna()) else np.nan
    if not index_returns.empty:
        stock_ret = returns.iloc[max(0, pos - 59) : pos + 1].dropna()
        idx_ret = index_returns.reindex(stock_ret.index).dropna()
        aligned = stock_ret.reindex(idx_ret.index).dropna()
        idx_ret = idx_ret.reindex(aligned.index)
        if len(aligned) >= 20:
            idx_values = idx_ret.to_numpy(dtype=np.float64)
            stock_values = aligned.to_numpy(dtype=np.float64)
            idx_centered = idx_values - np.nanmean(idx_values)
            stock_centered = stock_values - np.nanmean(stock_values)
            idx_var = float(np.nanmean(idx_centered ** 2))
            if idx_var > 1e-12:
                beta = float(np.nanmean(stock_centered * idx_centered) / idx_var)
                residual = stock_centered - beta * idx_centered
                out["beta_60d"] = beta
                out["specific_vol_60d"] = float(np.nanstd(residual) * np.sqrt(252.0))
    return out


def executable_label(frame, date, max_label_date):
    if frame is None or frame.empty or date not in frame.index:
        return {}
    pos = int(frame.index.get_loc(date))
    entry_pos = pos + 1
    required_pos = entry_pos + 1 + max(HORIZONS)
    if required_pos >= len(frame.index):
        return {}
    if max_label_date is not None and frame.index[required_pos] > pd.Timestamp(max_label_date):
        return {}
    opens = frame["open"].to_numpy(dtype=np.float64)
    lows = frame["low"].to_numpy(dtype=np.float64) if "low" in frame else opens
    closes = frame["close"].to_numpy(dtype=np.float64)
    entry = opens[entry_pos]
    delayed_entry = opens[entry_pos + 1]
    signal_close = closes[pos]
    if not all(np.isfinite(x) and x > 0 for x in (entry, delayed_entry, signal_close)):
        return {}
    forward = np.asarray(
        [opens[entry_pos + horizon] / entry - 1.0 for horizon in HORIZONS],
        dtype=np.float64,
    )
    delayed = np.asarray(
        [opens[entry_pos + 1 + horizon] / delayed_entry - 1.0 for horizon in HORIZONS],
        dtype=np.float64,
    )
    path_lows = lows[entry_pos : entry_pos + max(HORIZONS) + 1]
    signal_to_entry = entry / signal_close - 1.0
    if not (
        np.isfinite(forward).all()
        and np.isfinite(delayed).all()
        and np.isfinite(path_lows).all()
        and np.isfinite(signal_to_entry)
    ):
        return {}
    base_return = float(np.dot(forward, HORIZON_WEIGHTS))
    delayed_return = float(np.dot(delayed, HORIZON_WEIGHTS))
    max_downside = float(max(0.0, -np.min(path_lows / entry - 1.0)))
    blocked_buy = int(signal_to_entry >= 0.095)
    target_raw = (
        0.60 * base_return
        + 0.25 * delayed_return
        - 0.50 * max_downside
        - ROUND_TRIP_COST
        - 0.10 * blocked_buy
    )
    output = {
        "exec_target_raw": float(target_raw),
        "exec_base_return": base_return,
        "exec_delayed_return": delayed_return,
        "exec_max_downside": max_downside,
        "exec_signal_to_entry": float(signal_to_entry),
        "exec_blocked_buy": blocked_buy,
    }
    output.update({f"exec_return_{h}d": float(v) for h, v in zip(HORIZONS, forward)})
    return output


def load_global_features(path, cols):
    if path is None:
        return pd.DataFrame()
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    if "date" not in frame.columns:
        return pd.DataFrame()
    keep = ["date"] + [col for col in cols if col in frame.columns]
    frame = frame[keep].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame.drop_duplicates("date", keep="last").set_index("date").sort_index()


def build_rows(alpha_rows, price_cache, index_returns, global_frame, industry_map, args):
    current_selected = []
    holding_ages = {}
    rows = []
    daily_audit = []
    for row in alpha_rows:
        date = pd.Timestamp(row["date"]).normalize()
        original = [normalize_ts_code(code) for code in row.get("codes", [])]
        n = len(original)
        if n == 0:
            continue
        target_n = max(1, int(n * float(args.target_frac)))
        hold_n = max(target_n, int(n * float(args.hold_frac)))
        rank_map = {code: rank for rank, code in enumerate(original)}
        kept = [code for code in current_selected if rank_map.get(code, n + 1) < hold_n]
        if len(kept) > target_n:
            kept = sorted(kept, key=lambda code: rank_map.get(code, n + 1))[:target_n]
        vacancies = max(target_n - len(kept), 0)
        fill_candidates = [code for code in original if code not in set(kept)]
        baseline_fills = fill_candidates[:vacancies]
        protected_count = max(vacancies - int(args.max_reranked_fills), 0)
        protected_fills = set(baseline_fills[:protected_count])
        slots = min(vacancies, int(args.max_reranked_fills))
        candidate_codes = original[: min(int(args.candidate_end), n)]
        current_set = set(current_selected)
        kept_set = set(kept)
        baseline_fill_set = set(baseline_fills)
        global_values = {}
        if not global_frame.empty and date in global_frame.index:
            global_values = {
                col: float(global_frame.loc[date, col])
                for col in global_frame.columns
                if pd.notna(global_frame.loc[date, col])
            }
        top_industries = [industry_map.get(code, "UNKNOWN") for code in original[:target_n]]
        counts = pd.Series(top_industries).value_counts() if top_industries else pd.Series(dtype=int)
        top_industry_share = float(counts.iloc[0] / max(len(top_industries), 1)) if len(counts) else 0.0
        industry_hhi = float(((counts / max(len(top_industries), 1)) ** 2).sum()) if len(counts) else 0.0
        baseline_targets = []
        row_start = len(rows)
        for code in candidate_codes:
            pos = rank_map[code]
            is_eligible = int(
                code not in protected_fills
                and code not in kept_set
                and pos < int(args.candidate_end)
                and slots > 0
            )
            features = stock_signal_features(price_cache.get(code), date, index_returns)
            label = executable_label(price_cache.get(code), date, args.max_label_date)
            label_available = int(bool(label))
            industry = industry_map.get(code, "UNKNOWN")
            record = {
                "split": args.split_name,
                "date": date.strftime("%Y-%m-%d"),
                "code": code,
                "universe_size": int(n),
                "candidate_position": int(pos),
                "candidate_rank_pct": rank_pct(pos, n),
                "target_n": int(target_n),
                "hold_n": int(hold_n),
                "vacancies": int(vacancies),
                "rerank_slots": int(slots),
                "was_held": int(code in current_set),
                "is_kept": int(code in kept_set),
                "holding_age": int(holding_ages.get(code, 0)),
                "protected_fill": int(code in protected_fills),
                "baseline_fill": int(code in baseline_fill_set),
                "eligible": is_eligible,
                "top_industry_share": top_industry_share,
                "top_industry_hhi": industry_hhi,
                "candidate_industry_top_share": float(
                    top_industries.count(industry) / max(len(top_industries), 1)
                ) if top_industries else 0.0,
                "label_available": label_available,
            }
            record.update(global_values)
            record.update(features)
            record.update(label)
            if code in baseline_fill_set and label_available:
                baseline_targets.append(label["exec_target_raw"])
            rows.append(record)
        baseline_target = float(np.mean(baseline_targets)) if baseline_targets else np.nan
        for record in rows[row_start:]:
            record["baseline_target_raw"] = baseline_target
            target = record.get("exec_target_raw", np.nan)
            record["edge_vs_baseline"] = (
                float(target - baseline_target)
                if np.isfinite(target) and np.isfinite(baseline_target)
                else np.nan
            )
            record["beats_baseline"] = (
                int(target > baseline_target)
                if np.isfinite(target) and np.isfinite(baseline_target)
                else np.nan
            )

        selected = kept + baseline_fills
        selected_set = set(selected)
        for code in list(holding_ages):
            if code not in selected_set:
                holding_ages.pop(code, None)
        for code in selected:
            holding_ages[code] = holding_ages.get(code, 0) + 1
        current_selected = selected
        daily_audit.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "universe_size": int(n),
                "target_n": int(target_n),
                "kept_n": int(len(kept)),
                "vacancies": int(vacancies),
                "rerank_slots": int(slots),
                "candidate_rows": int(len(candidate_codes)),
                "labelled_rows": int(sum(1 for r in rows[row_start:] if r["label_available"])),
                "eligible_rows": int(sum(1 for r in rows[row_start:] if r["eligible"])),
                "baseline_target_raw": baseline_target,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(daily_audit)


def add_daily_labels(frame):
    frame = frame.copy()
    frame["exec_target"] = np.nan
    labelled = frame["exec_target_raw"].notna() if "exec_target_raw" in frame else pd.Series(False, index=frame.index)
    if labelled.any():
        frame.loc[labelled, "exec_target"] = (
            frame.loc[labelled]
            .groupby("date")["exec_target_raw"]
            .transform(daily_normalize)
            .to_numpy()
        )
    return frame


def validate_dataset(frame):
    if frame.empty:
        raise ValueError("policy dataset is empty")
    dup = int(frame.duplicated(["date", "code"]).sum())
    if dup:
        raise ValueError(f"duplicate date/code rows: {dup}")
    if "exec_target" in frame and frame["exec_target"].notna().any():
        leaked = [
            c
            for c in frame.columns
            if c.startswith("exec_") and c not in {
                "exec_target",
                "exec_target_raw",
                "exec_base_return",
                "exec_delayed_return",
                "exec_max_downside",
                "exec_signal_to_entry",
                "exec_blocked_buy",
                *{f"exec_return_{h}d" for h in HORIZONS},
            }
        ]
        if leaked:
            raise ValueError(f"unexpected exec columns: {leaked}")


def main(argv=None):
    args = parse_args(argv)
    rows = load_alpha_rows(args.alpha_jsonl, timestamp_dates=True)
    rows = filter_alpha_rows_by_date(rows, args.start_date, args.end_date)
    if not rows:
        raise ValueError("no alpha rows after date filter")
    min_date = min(pd.Timestamp(row["date"]).normalize() for row in rows)
    max_date = max(pd.Timestamp(row["date"]).normalize() for row in rows)
    codes = {
        normalize_ts_code(code)
        for row in rows
        for code in row.get("codes", [])[: int(args.candidate_end)]
    }
    price_cache = load_price_cache(
        args.data_dir,
        codes,
        min_date,
        pd.Timestamp(args.max_label_date) if args.max_label_date else max_date,
        progress_every=args.progress_every,
    )
    index_returns = load_index_returns(args.data_dir, min_date, max_date)
    global_cols = [col.strip() for col in str(args.global_cols).split(",") if col.strip()]
    global_frame = load_global_features(args.global_features, global_cols)
    industry_map = load_industry_map(args.industry_csv)
    frame, audit = build_rows(
        rows,
        price_cache,
        index_returns,
        global_frame,
        industry_map,
        args,
    )
    frame = add_daily_labels(frame)
    validate_dataset(frame)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset_path = output / "policy_dataset.parquet"
    frame.to_parquet(dataset_path, index=False)
    audit.to_csv(output / "daily_audit.csv", index=False)
    feature_cols = [
        col
        for col in frame.columns
        if pd.api.types.is_numeric_dtype(frame[col])
        and not col.startswith("exec_")
        and col
        not in {
            "beats_baseline",
            "label_available",
            "baseline_target_raw",
            "edge_vs_baseline",
        }
    ]
    summary = {
        "alpha_jsonl": str(args.alpha_jsonl),
        "split_name": args.split_name,
        "rows": int(len(frame)),
        "dates": int(frame["date"].nunique()),
        "date_start": str(frame["date"].min()),
        "date_end": str(frame["date"].max()),
        "candidate_end": int(args.candidate_end),
        "labelled_rows": int(frame["label_available"].sum()),
        "eligible_labelled_rows": int((frame["eligible"].eq(1) & frame["label_available"].eq(1)).sum()),
        "eligible_rows": int(frame["eligible"].sum()),
        "mean_vacancies": float(audit["vacancies"].mean()),
        "mean_rerank_slots": float(audit["rerank_slots"].mean()),
        "mean_baseline_target_raw": float(audit["baseline_target_raw"].mean()),
        "feature_count": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "target_formula": (
            "0.60*weighted_next_open_return + 0.25*delayed_weighted_next_open_return "
            "- 0.50*max_intrahorizon_downside - 0.0017 - 0.10*signal_to_entry>=9.5%"
        ),
        "selection_protocol": "Use 2024 val + 2025 test only; forward rows are observational.",
    }
    (output / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
