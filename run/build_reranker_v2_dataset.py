"""Add executable continuous labels and trailing market features to reranker data."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HORIZONS = (1, 3, 5, 10)
HORIZON_WEIGHTS = np.asarray((0.15, 0.25, 0.35, 0.25), dtype=np.float64)
ROUND_TRIP_COST = 0.0017
BOUNDARY_START = 27
BOUNDARY_END = 80


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="reranker_data_20260614")
    parser.add_argument("--output-root", default="reranker_v2_data_20260615")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--index-file", default="hs300_index.csv")
    return parser.parse_args()


def market_features(path):
    index = pd.read_csv(path)
    date_col = "date" if "date" in index.columns else "trade_date"
    index[date_col] = pd.to_datetime(index[date_col])
    index = index.sort_values(date_col).set_index(date_col)
    close = pd.to_numeric(index["close"], errors="coerce")
    daily = close.pct_change()
    output = pd.DataFrame(index=close.index)
    for window in (5, 20, 60):
        output[f"market_return_{window}d"] = close.pct_change(window)
    output["market_vol_20d"] = daily.rolling(20, min_periods=10).std()
    output["market_vol_60d"] = daily.rolling(60, min_periods=30).std()
    output["market_drawdown_60d"] = close / close.rolling(60, min_periods=20).max() - 1.0
    output["market_ma20_gap"] = close / close.rolling(20, min_periods=10).mean() - 1.0
    output["market_ma60_gap"] = close / close.rolling(60, min_periods=30).mean() - 1.0
    return output


def stock_label_rows(path, wanted_dates, max_label_date):
    frame = pd.read_csv(path, usecols=["trade_date", "close"])
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame = frame.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
    frame = frame.set_index("trade_date")
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=np.float64)
    dates = frame.index
    positions = pd.Series(np.arange(len(dates)), index=dates)
    records = []
    for date in wanted_dates:
        if date not in positions.index:
            continue
        signal_pos = int(positions.loc[date])
        entry_pos = signal_pos + 1
        if entry_pos + 1 + max(HORIZONS) >= len(close):
            continue
        required_date = dates[entry_pos + 1 + max(HORIZONS)]
        if required_date > max_label_date:
            continue
        entry = close[entry_pos]
        if not np.isfinite(entry) or entry <= 0:
            continue
        forward = np.asarray(
            [close[entry_pos + horizon] / entry - 1.0 for horizon in HORIZONS],
            dtype=np.float64,
        )
        delayed_entry = close[entry_pos + 1]
        delayed = np.asarray(
            [
                close[entry_pos + 1 + horizon] / delayed_entry - 1.0
                for horizon in HORIZONS
            ],
            dtype=np.float64,
        )
        path_returns = close[entry_pos + 1 : entry_pos + max(HORIZONS) + 1] / entry - 1.0
        signal_to_entry = entry / close[signal_pos] - 1.0
        if not (
            np.isfinite(forward).all()
            and np.isfinite(delayed).all()
            and np.isfinite(path_returns).all()
            and np.isfinite(signal_to_entry)
        ):
            continue
        base_return = float(np.dot(forward, HORIZON_WEIGHTS))
        delayed_return = float(np.dot(delayed, HORIZON_WEIGHTS))
        max_downside = float(max(0.0, -np.min(path_returns)))
        blocked_penalty = 0.10 if signal_to_entry >= 0.095 else 0.0
        executable_target = (
            0.60 * base_return
            + 0.25 * delayed_return
            - 0.50 * max_downside
            - ROUND_TRIP_COST
            - blocked_penalty
        )
        record = {
            "date": date,
            "exec_target_raw": executable_target,
            "exec_base_return": base_return,
            "exec_delayed_return": delayed_return,
            "exec_max_downside": max_downside,
            "exec_signal_to_entry": signal_to_entry,
            "exec_blocked_buy": int(signal_to_entry >= 0.095),
        }
        record.update(
            {f"exec_return_{horizon}d": value for horizon, value in zip(HORIZONS, forward)}
        )
        records.append(record)
    return records


def stock_open_label_rows(path, wanted_dates, max_label_date):
    """Build labels that match next-open execution in the share ledger."""
    frame = pd.read_csv(path, usecols=["trade_date", "open", "low", "close"])
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame = frame.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
    frame = frame.set_index("trade_date")
    opens = pd.to_numeric(frame["open"], errors="coerce").to_numpy(dtype=np.float64)
    lows = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=np.float64)
    closes = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=np.float64)
    dates = frame.index
    positions = pd.Series(np.arange(len(dates)), index=dates)
    records = []
    for date in wanted_dates:
        if date not in positions.index:
            continue
        signal_pos = int(positions.loc[date])
        entry_pos = signal_pos + 1
        required_pos = entry_pos + 1 + max(HORIZONS)
        if required_pos >= len(opens) or dates[required_pos] > max_label_date:
            continue
        entry = opens[entry_pos]
        delayed_entry = opens[entry_pos + 1]
        signal_close = closes[signal_pos]
        if not all(
            np.isfinite(value) and value > 0
            for value in (entry, delayed_entry, signal_close)
        ):
            continue
        forward = np.asarray(
            [opens[entry_pos + horizon] / entry - 1.0 for horizon in HORIZONS],
            dtype=np.float64,
        )
        delayed = np.asarray(
            [
                opens[entry_pos + 1 + horizon] / delayed_entry - 1.0
                for horizon in HORIZONS
            ],
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
            continue
        base_return = float(np.dot(forward, HORIZON_WEIGHTS))
        delayed_return = float(np.dot(delayed, HORIZON_WEIGHTS))
        max_downside = float(max(0.0, -np.min(path_lows / entry - 1.0)))
        blocked_penalty = 0.10 if signal_to_entry >= 0.095 else 0.0
        executable_target = (
            0.60 * base_return
            + 0.25 * delayed_return
            - 0.50 * max_downside
            - ROUND_TRIP_COST
            - blocked_penalty
        )
        record = {
            "date": date,
            "exec_target_raw": executable_target,
            "exec_base_return": base_return,
            "exec_delayed_return": delayed_return,
            "exec_max_downside": max_downside,
            "exec_signal_to_entry": signal_to_entry,
            "exec_blocked_buy": int(signal_to_entry >= 0.095),
        }
        record.update(
            {f"exec_return_{horizon}d": value for horizon, value in zip(HORIZONS, forward)}
        )
        records.append(record)
    return records


def daily_normalize(frame):
    def normalize(values):
        values = values.astype(float)
        median = values.median()
        scale = (values - median).abs().median()
        if not np.isfinite(scale) or scale < 1e-8:
            scale = values.std()
        scale = max(float(scale), 1e-8)
        return ((values - median) / scale).clip(-5.0, 5.0)

    frame["exec_target"] = frame.groupby("date")["exec_target_raw"].transform(normalize)
    return frame


def process_dataset(source, destination, data_dir, market):
    frame = pd.read_parquet(source)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["code"] = frame["code"].astype(str)
    frame = frame[
        (frame["candidate_position"] >= BOUNDARY_START)
        & (frame["candidate_position"] < BOUNDARY_END)
    ].copy()
    max_label_date = pd.Timestamp(f"{frame['date'].dt.year.max()}-12-31")
    wanted = {
        code: set(group["date"])
        for code, group in frame.groupby("code", sort=False)
    }
    label_records = []
    for index, (code, dates) in enumerate(wanted.items(), start=1):
        path = data_dir / f"{code}.csv"
        if path.exists():
            records = stock_label_rows(path, dates, max_label_date)
            for record in records:
                record["code"] = code
            label_records.extend(records)
        if index % 500 == 0:
            print(f"{source.parent.name}: labels {index}/{len(wanted)}", flush=True)

    labels = pd.DataFrame(label_records)
    frame = frame.merge(labels, on=["date", "code"], how="left", validate="one_to_one")
    frame = frame.merge(
        market.reset_index().rename(columns={market.index.name or "index": "date"}),
        on="date",
        how="left",
        validate="many_to_one",
    )
    labelled = frame["exec_target_raw"].notna()
    frame["exec_target"] = np.nan
    normalized = daily_normalize(frame.loc[labelled].copy())
    frame.loc[labelled, "exec_target"] = normalized["exec_target"].to_numpy()
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
    summary = {
        "source": str(source),
        "rows": int(len(frame)),
        "dates": int(frame["date"].nunique()),
        "labelled_rows": int(frame["exec_target"].notna().sum()),
        "labelled_dates": int(frame.loc[frame["exec_target"].notna(), "date"].nunique()),
        "date_start": str(frame["date"].min().date()),
        "date_end": str(frame["date"].max().date()),
        "boundary_start_zero_based": BOUNDARY_START,
        "boundary_end_exclusive": BOUNDARY_END,
        "target_formula": (
            "0.60*weighted_return + 0.25*delayed_weighted_return "
            "- 0.50*max_downside - 0.0017 - 0.10*blocked_buy"
        ),
    }
    destination.with_name("v2_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


def main():
    args = parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    data_dir = Path(args.data_dir)
    market = market_features(data_dir / args.index_file)
    sources = sorted(source_root.glob("oof_F[1-6]_*/reranker_dataset.parquet"))
    sources.append(source_root / "m0_validation_2024/reranker_dataset.parquet")
    for source in sources:
        destination = output_root / source.parent.name / "reranker_v2_dataset.parquet"
        process_dataset(source, destination, data_dir, market)


if __name__ == "__main__":
    main()
