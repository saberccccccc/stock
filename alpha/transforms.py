"""Reusable transforms for daily alpha ranking JSONL rows."""

from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from alpha.io import assert_same_date, iter_aligned_alpha_rows, write_alpha_rows


def make_rank_alpha(n):
    return (1.0 - np.arange(n, dtype=np.float64) / max(n - 1, 1)).tolist()


def percentile_map(codes):
    n = len(codes)
    if n <= 1:
        return {code: 1.0 for code in codes}
    return {code: 1.0 - rank / (n - 1) for rank, code in enumerate(codes)}


def nested_get(row, path, default=None):
    value = row
    for part in str(path).split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def load_signal_returns(data_dir, codes, start_date, end_date):
    by_date = {}
    for code in sorted(codes):
        path = Path(data_dir) / f"{code}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, usecols=["trade_date", "close"])
        if frame.empty:
            continue
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame = frame.sort_values("trade_date")
        frame["signal_return"] = pd.to_numeric(frame["close"], errors="coerce").pct_change()
        frame = frame[
            (frame["trade_date"] >= start_date) & (frame["trade_date"] <= end_date)
        ]
        for date, value in zip(frame["trade_date"], frame["signal_return"]):
            if np.isfinite(value):
                by_date.setdefault(pd.Timestamp(date), {})[code] = float(value)
    return by_date


def load_stall_signals(
    data_dir,
    codes,
    start_date,
    end_date,
    surge_return,
    recent_abs_return=0.03,
    max_range=0.08,
    surge_lookback=20,
    recent_window=5,
):
    """Find stocks that surged earlier, then stalled near the signal date."""
    if surge_return is None:
        return {}
    if not 0 < recent_window < surge_lookback:
        raise ValueError("recent_window must be between 0 and surge_lookback")

    by_date = {}
    for code in sorted(codes):
        path = Path(data_dir) / f"{code}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, usecols=["trade_date", "high", "low", "close"])
        if frame.empty:
            continue
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame = frame.sort_values("trade_date")
        close = pd.to_numeric(frame["close"], errors="coerce")
        high = pd.to_numeric(frame["high"], errors="coerce")
        low = pd.to_numeric(frame["low"], errors="coerce")
        frame["stall_surge_return"] = (
            close.shift(recent_window) / close.shift(surge_lookback) - 1.0
        )
        frame["stall_recent_return"] = close / close.shift(recent_window) - 1.0
        frame["stall_recent_range"] = (
            high.rolling(recent_window).max()
            / low.rolling(recent_window).min()
            - 1.0
        )
        frame = frame[
            (frame["trade_date"] >= start_date) & (frame["trade_date"] <= end_date)
        ]
        stalled = frame[
            (frame["stall_surge_return"] >= surge_return)
            & (frame["stall_recent_return"].abs() <= recent_abs_return)
            & (frame["stall_recent_range"] <= max_range)
        ]
        for date in stalled["trade_date"]:
            by_date.setdefault(pd.Timestamp(date), set()).add(code)
    return by_date


def transform_execution_rows(
    rows,
    signal_returns,
    stability_window=0,
    current_weight=1.0,
    max_signal_return=None,
    stall_signals=None,
    stall_config=None,
):
    if stability_window < 0:
        raise ValueError("stability_window must be non-negative")
    if not 0.0 <= current_weight <= 1.0:
        raise ValueError("current_weight must be between 0 and 1")

    history = deque(maxlen=stability_window)
    stall_signals = stall_signals or {}
    output = []
    for row in rows:
        current = percentile_map(row.get("codes", []))
        scores = {}
        demoted = 0
        chase_demoted = 0
        stall_demoted = 0
        date_returns = signal_returns.get(pd.Timestamp(row["date"]), {})
        date_stalls = stall_signals.get(pd.Timestamp(row["date"]), set())
        for code, current_score in current.items():
            prior = [rank[code] for rank in history if code in rank]
            prior_score = float(np.mean(prior)) if prior else current_score
            score = current_weight * current_score + (1.0 - current_weight) * prior_score
            signal_return = date_returns.get(code)
            chase_hit = (
                max_signal_return is not None
                and signal_return is not None
                and signal_return >= max_signal_return
            )
            stall_hit = code in date_stalls
            if chase_hit:
                score = -1.0 + current_score * 1e-6
                chase_demoted += 1
            if stall_hit:
                score = -1.0 + current_score * 1e-6
                stall_demoted += 1
            if chase_hit or stall_hit:
                demoted += 1
            scores[code] = float(score)

        items = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        output.append(
            {
                "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                "codes": [code for code, _ in items],
                "alpha": [score for _, score in items],
                "n_stocks": len(items),
                "execution_transform": {
                    "stability_window": stability_window,
                    "current_weight": current_weight,
                    "max_signal_return": max_signal_return,
                    "demoted_count": demoted,
                    "chase_demoted_count": chase_demoted,
                    "stall_demoted_count": stall_demoted,
                    "stall_config": stall_config,
                },
            }
        )
        if stability_window:
            history.append(current)
    return output


def edge_rerank_row(base, rerank, start_rank, end_rank, source_full_rerank):
    assert_same_date(base, rerank)
    codes = [str(code) for code in base["codes"]]
    rerank_codes = [str(code) for code in rerank["codes"]]
    n = len(codes)
    lo = min(max(0, int(start_rank)), n)
    hi = min(int(end_rank), n)
    if hi <= lo:
        return base, False

    edge = codes[lo:hi]
    edge_set = set(edge)
    base_pos = {code: i for i, code in enumerate(codes)}
    rerank_pos = {code: i for i, code in enumerate(rerank_codes) if code in edge_set}
    edge_sorted = sorted(edge, key=lambda code: rerank_pos.get(code, n + base_pos[code]))
    changed = edge_sorted != edge
    final_codes = codes[:lo] + edge_sorted + codes[hi:]
    out_row = dict(base)
    out_row["codes"] = final_codes
    out_row["alpha"] = make_rank_alpha(len(final_codes))
    out_row["n_stocks"] = len(final_codes)
    out_row["edge_reranker"] = {
        "source_full_rerank": str(source_full_rerank),
        "start_rank": lo,
        "end_rank": hi,
    }
    return out_row, changed


def negative_filter_row(base, rerank, start_rank, end_rank, drop_n, source_full_rerank):
    assert_same_date(base, rerank)
    codes = [str(code) for code in base["codes"]]
    rerank_codes = [str(code) for code in rerank["codes"]]
    n = len(codes)
    lo = min(max(0, int(start_rank)), n)
    hi = min(int(end_rank), n)
    drop_n = max(0, int(drop_n))
    if drop_n <= 0 or hi <= lo:
        return base, 0

    window = codes[lo:hi]
    base_pos = {code: i for i, code in enumerate(codes)}
    rerank_pos = {code: i for i, code in enumerate(rerank_codes)}
    worst = sorted(
        window,
        key=lambda code: rerank_pos.get(code, n + base_pos[code]),
        reverse=True,
    )[: min(drop_n, len(window))]
    if not worst:
        return base, 0

    drop_set = set(worst)
    final_codes = [code for code in codes if code not in drop_set] + worst
    out_row = dict(base)
    out_row["codes"] = final_codes
    out_row["alpha"] = make_rank_alpha(len(final_codes))
    out_row["n_stocks"] = len(final_codes)
    out_row["negative_filter"] = {
        "source_full_rerank": str(source_full_rerank),
        "start_rank": lo,
        "end_rank": hi,
        "drop_n": int(len(worst)),
    }
    return out_row, len(worst)


def conditional_negative_filter_row(
    base,
    rerank,
    trigger_path,
    start_rank,
    end_rank,
    drop_n,
    source_full_rerank,
):
    assert_same_date(base, rerank)
    triggered = bool(nested_get(base, trigger_path, False))
    if not triggered:
        return dict(base), False, 0

    out_row, dropped = negative_filter_row(
        base,
        rerank,
        start_rank=start_rank,
        end_rank=end_rank,
        drop_n=drop_n,
        source_full_rerank=source_full_rerank,
    )
    if dropped:
        meta = out_row.pop("negative_filter")
        out_row["conditional_negative_filter"] = {
            "trigger_path": trigger_path,
            **meta,
        }
    return out_row, True, dropped


def write_edge_rerank_alpha(base_alpha, full_rerank_alpha, output_alpha, start_rank, end_rank):
    rows = 0
    changed = 0
    out_rows = []
    for base, rerank in iter_aligned_alpha_rows(base_alpha, full_rerank_alpha):
        out_row, did_change = edge_rerank_row(
            base,
            rerank,
            start_rank=start_rank,
            end_rank=end_rank,
            source_full_rerank=full_rerank_alpha,
        )
        out_rows.append(out_row)
        rows += 1
        changed += int(did_change)
    output = write_alpha_rows(output_alpha, out_rows)
    return {"output": str(output), "rows": rows, "changed_dates": changed}


def write_negative_filter_alpha(
    base_alpha,
    full_rerank_alpha,
    output_alpha,
    start_rank=30,
    end_rank=100,
    drop_n=5,
):
    rows = 0
    changed = 0
    dropped_total = 0
    out_rows = []
    for base, rerank in iter_aligned_alpha_rows(base_alpha, full_rerank_alpha):
        out_row, dropped = negative_filter_row(
            base,
            rerank,
            start_rank=start_rank,
            end_rank=end_rank,
            drop_n=drop_n,
            source_full_rerank=full_rerank_alpha,
        )
        out_rows.append(out_row)
        rows += 1
        changed += int(dropped > 0)
        dropped_total += dropped
    output = write_alpha_rows(output_alpha, out_rows)
    return {
        "output": str(output),
        "rows": rows,
        "changed_dates": changed,
        "dropped_total": dropped_total,
    }


def write_conditional_negative_filter_alpha(
    base_alpha,
    full_rerank_alpha,
    output_alpha,
    trigger_path="breadth_market_transform.triggered",
    start_rank=30,
    end_rank=100,
    drop_n=3,
):
    rows = 0
    triggered_dates = 0
    changed_dates = 0
    dropped_total = 0
    out_rows = []
    for base, rerank in iter_aligned_alpha_rows(base_alpha, full_rerank_alpha):
        out_row, triggered, dropped = conditional_negative_filter_row(
            base,
            rerank,
            trigger_path=trigger_path,
            start_rank=start_rank,
            end_rank=end_rank,
            drop_n=drop_n,
            source_full_rerank=full_rerank_alpha,
        )
        out_rows.append(out_row)
        rows += 1
        triggered_dates += int(triggered)
        changed_dates += int(dropped > 0)
        dropped_total += dropped
    output = write_alpha_rows(output_alpha, out_rows)
    return {
        "output": str(output),
        "rows": rows,
        "triggered_dates": triggered_dates,
        "changed_dates": changed_dates,
        "dropped_total": dropped_total,
    }
