"""Parameter sweep for open-price share-ledger Alpha execution."""

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
if ROOT_STR in sys.path:
    sys.path.remove(ROOT_STR)
sys.path.insert(0, ROOT_STR)

from backtest.open_ledger import (
    infer_ohlc_load_window,
    load_alpha_rows,
    load_global_risk_features,
    load_index_returns,
    load_execution_market_frames,
    prepare_open_ledger_context,
    parse_float_list,
    prepare_execution_constraint_masks,
    recompute_adv,
    run_open_ledger,
)
from backtest.market_data_contract import add_execution_market_data_args
from backtest.stress import STRESSES, get_stress


def parse_alpha_specs(raw):
    specs = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        name, path = item.split("=", 1)
        specs.append((name.strip(), Path(path.strip())))
    if not specs:
        raise ValueError("No alpha specs provided")
    return specs


def parse_stress_names(raw):
    names = [name.strip() for name in raw.split(",") if name.strip()]
    unknown = [name for name in names if name not in STRESSES]
    if unknown:
        raise ValueError(f"Unknown stress names: {unknown}")
    if not names:
        raise ValueError("At least one stress name is required")
    return names


def apply_stress_overrides(args, stress_name):
    stressed = SimpleNamespace(**vars(args))
    for key, value in get_stress(stress_name).overrides.items():
        setattr(stressed, key, value)
    return stressed


def filter_alpha_rows(rows, start_date=None, end_date=None):
    start = pd.Timestamp(start_date) if start_date else None
    end = pd.Timestamp(end_date) if end_date else None
    if start is not None and end is not None and start > end:
        raise ValueError("start_date must be on or before end_date")
    return [
        row for row in rows
        if (start is None or pd.Timestamp(row["date"]) >= start)
        and (end is None or pd.Timestamp(row["date"]) <= end)
    ]


def parse_int_list(raw):
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def retention_param_grid(max_new_names, exit_hold_fracs, switch_gap_fracs):
    grid = []
    for max_new in max_new_names:
        if max_new <= 0:
            candidate = (0, None, 0.0)
            if candidate not in grid:
                grid.append(candidate)
            continue
        for exit_hold in exit_hold_fracs:
            normalized_exit = None if exit_hold <= 0 else float(exit_hold)
            for switch_gap in switch_gap_fracs:
                grid.append((int(max_new), normalized_exit, float(switch_gap)))
    return grid


def _key_float(value):
    if pd.isna(value):
        return 0.0
    return round(float(value), 10)


def _key_text(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def sweep_key(
    alpha_name,
    stress_name,
    band,
    industry_cap,
    risk_target_frac,
    risk_target_market_mult_below,
    defensive_tilt,
    defensive_tilt_market_mult_below,
    capital,
    max_new,
    exit_hold,
    switch_gap,
    target,
    hold,
    global_risk_overlay_mode="none",
    global_risk_pressure_threshold=0.0,
    global_risk_market_scale=1.0,
    global_risk_target_frac=None,
    active_drawdown_throttle_lookback=0,
    active_drawdown_throttle_trigger=0.0,
    active_drawdown_throttle_scale=1.0,
    active_drawdown_throttle_cooldown=0,
    selection_policy="retention",
    top_k=0,
    n_drop=0,
    state_aware_selection_mode="none",
    state_aware_selection_pressure_col=None,
    state_aware_selection_pressure_threshold=0.0,
    state_aware_selection_pressure_width=0.0,
    state_aware_selection_min_stress=0.0,
    state_aware_selection_rank_penalty=0.0,
    state_aware_selection_top_frac=0.006,
    state_aware_selection_crowd_scale=0.10,
    state_aware_selection_momentum_weight=0.40,
    state_aware_selection_beta_weight=0.20,
    state_aware_selection_vol_weight=0.20,
    state_aware_selection_industry_weight=0.20,
    state_aware_selection_risk_delta_threshold=0.15,
):
    return (
        str(alpha_name),
        str(stress_name),
        _key_float(band),
        _key_float(industry_cap),
        _key_float(0.0 if risk_target_frac is None else risk_target_frac),
        _key_float(risk_target_market_mult_below),
        _key_float(defensive_tilt),
        _key_float(defensive_tilt_market_mult_below),
        _key_float(capital),
        int(max_new),
        _key_float(0.0 if exit_hold is None else exit_hold),
        _key_float(switch_gap),
        _key_float(target),
        _key_float(hold),
        str(global_risk_overlay_mode),
        _key_float(global_risk_pressure_threshold),
        _key_float(global_risk_market_scale),
        _key_float(0.0 if global_risk_target_frac is None else global_risk_target_frac),
        int(active_drawdown_throttle_lookback or 0),
        _key_float(active_drawdown_throttle_trigger),
        _key_float(active_drawdown_throttle_scale),
        int(active_drawdown_throttle_cooldown or 0),
        str(selection_policy),
        int(top_k or 0),
        int(n_drop or 0),
        str(state_aware_selection_mode),
        _key_text(state_aware_selection_pressure_col),
        _key_float(state_aware_selection_pressure_threshold),
        _key_float(state_aware_selection_pressure_width),
        _key_float(state_aware_selection_min_stress),
        _key_float(state_aware_selection_rank_penalty),
        _key_float(state_aware_selection_top_frac),
        _key_float(state_aware_selection_crowd_scale),
        _key_float(state_aware_selection_momentum_weight),
        _key_float(state_aware_selection_beta_weight),
        _key_float(state_aware_selection_vol_weight),
        _key_float(state_aware_selection_industry_weight),
        _key_float(state_aware_selection_risk_delta_threshold),
    )


def completed_keys_from_summary(frame):
    if frame.empty:
        return set()
    required = {
        "alpha_name",
        "stress",
        "rebalance_band",
        "portfolio_value",
        "max_new_names",
        "target_frac",
        "hold_frac",
    }
    missing = required - set(frame.columns)
    if missing:
        print(f"resume ignored: summary missing columns {sorted(missing)}", flush=True)
        return set()
    keys = set()
    for _, row in frame.iterrows():
        keys.add(sweep_key(
            row["alpha_name"],
            row["stress"],
            row["rebalance_band"],
            row.get("max_industry_weight", 0.0),
            row.get("risk_target_frac", 0.0),
            row.get("risk_target_market_mult_below", 1.0),
            row.get("defensive_tilt", 0.0),
            row.get("defensive_tilt_market_mult_below", 1.0),
            row["portfolio_value"],
            row["max_new_names"],
            row.get("exit_hold_frac", 0.0),
            row.get("switch_gap_frac", 0.0),
            row["target_frac"],
            row["hold_frac"],
            row.get("global_risk_overlay_mode", "none"),
            row.get("global_risk_pressure_threshold", 0.0),
            row.get("global_risk_market_scale", 1.0),
            row.get("global_risk_target_frac", 0.0),
            row.get("active_drawdown_throttle_lookback", 0),
            row.get("active_drawdown_throttle_trigger", 0.0),
            row.get("active_drawdown_throttle_scale", 1.0),
            row.get("active_drawdown_throttle_cooldown", 0),
            row.get("selection_policy", "retention"),
            row.get("top_k", 0),
            row.get("n_drop", 0),
            row.get("state_aware_selection_mode", "none"),
            row.get("state_aware_selection_pressure_col", None),
            row.get("state_aware_selection_pressure_threshold", 0.0),
            row.get("state_aware_selection_pressure_width", 0.0),
            row.get("state_aware_selection_min_stress", 0.0),
            row.get("state_aware_selection_rank_penalty", 0.0),
            row.get("state_aware_selection_top_frac", 0.006),
            row.get("state_aware_selection_crowd_scale", 0.10),
            row.get("state_aware_selection_momentum_weight", 0.40),
            row.get("state_aware_selection_beta_weight", 0.20),
            row.get("state_aware_selection_vol_weight", 0.20),
            row.get("state_aware_selection_industry_weight", 0.20),
            row.get("state_aware_selection_risk_delta_threshold", 0.15),
        ))
    return keys


def append_summary_rows(summary_path, rows):
    """Append a completed sweep chunk without rewriting prior grid results."""
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(
        summary_path,
        mode="a",
        header=not summary_path.exists() or summary_path.stat().st_size == 0,
        index=False,
    )


def write_path_artifacts(path_dir, path_tag, returns_df, diag_df, execution_trace, position_trace):
    """Persist genuine ledger evidence without interpreting or changing execution."""
    path_dir = Path(path_dir)
    path_dir.mkdir(parents=True, exist_ok=True)
    path_tag = str(path_tag)
    longest_prefix = "diagnostics_"
    max_windows_path = 240
    tag_budget = max_windows_path - len(str(path_dir.resolve())) - len(longest_prefix) - len(".csv") - 1
    if len(path_tag) > tag_budget:
        digest = hashlib.sha256(path_tag.encode("utf-8")).hexdigest()[:16]
        readable = path_tag[: max(tag_budget - len(digest) - 1, 0)]
        path_tag = f"{readable}_{digest}" if readable else digest
    artifact_paths = {
        "equity_curve": path_dir / f"returns_{path_tag}.csv",
        "diagnostics": path_dir / f"diagnostics_{path_tag}.csv",
        "positions": path_dir / f"positions_{path_tag}.csv",
        "orders": path_dir / f"orders_{path_tag}.csv",
        "rejections": path_dir / f"rejections_{path_tag}.csv",
        "costs": path_dir / f"costs_{path_tag}.csv",
    }
    order_columns = [
        "date", "code", "side", "current_shares", "target_shares",
        "requested_shares", "executed_shares", "price", "executed_value_cny",
        "commission_cny", "stamp_tax_cny", "slippage_cny", "total_cost_cny",
        "status", "reason",
    ]
    position_columns = [
        "date", "code", "shares", "mark_price", "market_value_cny", "weight",
    ]
    orders_df = pd.DataFrame(execution_trace or [], columns=order_columns)
    positions_df = pd.DataFrame(position_trace or [], columns=position_columns)
    rejections_df = (
        orders_df.loc[orders_df["status"] == "rejected"].copy()
        if not orders_df.empty
        else orders_df.copy()
    )
    cost_columns = [
        "date", "code", "side", "executed_shares", "executed_value_cny",
        "commission_cny", "stamp_tax_cny", "slippage_cny", "total_cost_cny",
    ]
    costs_df = (
        orders_df.loc[
            orders_df["status"] == "filled",
            [column for column in cost_columns if column in orders_df],
        ].copy()
        if not orders_df.empty
        else pd.DataFrame(columns=cost_columns)
    )
    returns_df.to_csv(artifact_paths["equity_curve"], index=False)
    diag_df.to_csv(artifact_paths["diagnostics"], index=False)
    positions_df.to_csv(artifact_paths["positions"], index=False)
    orders_df.to_csv(artifact_paths["orders"], index=False)
    rejections_df.to_csv(artifact_paths["rejections"], index=False)
    costs_df.to_csv(artifact_paths["costs"], index=False)
    return artifact_paths


def process_rss_mb():
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / (1024 ** 2), 1)
    except Exception:
        if sys.platform != "win32":
            return None
        try:
            import ctypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("page_fault_count", ctypes.c_ulong),
                    ("peak_working_set_size", ctypes.c_size_t),
                    ("working_set_size", ctypes.c_size_t),
                    ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                    ("quota_paged_pool_usage", ctypes.c_size_t),
                    ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                    ("quota_non_paged_pool_usage", ctypes.c_size_t),
                    ("pagefile_usage", ctypes.c_size_t),
                    ("peak_pagefile_usage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            psapi.GetProcessMemoryInfo.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ProcessMemoryCounters),
                ctypes.c_ulong,
            ]
            psapi.GetProcessMemoryInfo.restype = ctypes.c_int
            ok = psapi.GetProcessMemoryInfo(
                kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
            )
            return round(counters.working_set_size / (1024 ** 2), 1) if ok else None
        except (AttributeError, OSError):
            return None


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep open-price share-ledger params")
    parser.add_argument("--alpha-specs", required=True, help="Comma list like max090=path,max095=path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--performance-report",
        default=None,
        help="Optional JSON path for alpha/OHLC/context/grid timing and process RSS.",
    )
    parser.add_argument(
        "--save-path-details",
        action="store_true",
        help="Save per-grid returns and ledger diagnostics under output_dir/paths.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip rows already present in the sweep summary CSV.")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--target-fracs", default="0.004,0.005,0.006,0.008")
    parser.add_argument("--hold-fracs", default="0.08,0.10,0.12,0.15")
    parser.add_argument("--rebalance-bands", default="0.15,0.20,0.25,0.30")
    parser.add_argument("--stresses", default="normal")
    parser.add_argument("--portfolio-values", default="500000,1000000")
    parser.add_argument("--max-weight", type=float, default=0.05)
    parser.add_argument("--industry-csv", default="data/stock_industry.csv")
    parser.add_argument(
        "--max-industry-weights",
        default="0",
        help="Comma list of selection-level industry caps; 0 disables.",
    )
    parser.add_argument("--market-timing-mode", default="legacy", choices=["none", "legacy", "dynamic"])
    parser.add_argument("--market-min-mult", type=float, default=0.20)
    parser.add_argument("--market-max-mult", type=float, default=1.00)
    parser.add_argument("--legacy-bear-mult", type=float, default=0.70)
    parser.add_argument("--legacy-crash-mult", type=float, default=0.30)
    parser.add_argument(
        "--risk-target-fracs",
        default="",
        help="Comma list for weak-market target caps; empty disables risk target.",
    )
    parser.add_argument(
        "--risk-target-market-mult-belows",
        default="1.0",
        help="Comma list of market multiplier thresholds for risk target caps.",
    )
    parser.add_argument(
        "--defensive-tilts",
        default="0",
        help="Comma list of defensive tilt strengths; 0 disables.",
    )
    parser.add_argument(
        "--defensive-tilt-market-mult-belows",
        default="1.0",
        help="Comma list of market_mult thresholds for applying defensive tilt; 1.0 keeps legacy/global behavior.",
    )
    parser.add_argument(
        "--global-risk-features",
        default=None,
        help="Optional A-share-date-aligned global overnight feature file (.parquet/.csv).",
    )
    parser.add_argument(
        "--global-risk-overlay-mode",
        default="none",
        choices=["none", "defensive_pressure"],
    )
    parser.add_argument(
        "--global-risk-pressure-col",
        default="global_defensive_pressure",
    )
    parser.add_argument(
        "--global-risk-pressure-thresholds",
        default="0.04",
        help="Comma list; used only when global risk overlay is enabled.",
    )
    parser.add_argument(
        "--global-risk-market-scales",
        default="0.8",
        help="Comma list; gross multiplier applied on triggered global-risk days.",
    )
    parser.add_argument(
        "--global-risk-target-fracs",
        default="",
        help="Comma list of target caps on triggered days; empty means no extra target cap.",
    )
    parser.add_argument(
        "--active-drawdown-throttle-lookback",
        type=int,
        default=0,
        help="Trailing realized active-return lookback in trading days; 0 disables.",
    )
    parser.add_argument(
        "--active-drawdown-throttle-trigger",
        type=float,
        default=0.0,
        help="Trigger threshold for compounded trailing active return, e.g. -0.03.",
    )
    parser.add_argument(
        "--active-drawdown-throttle-scale",
        type=float,
        default=1.0,
        help="Gross multiplier while throttle is active; values below 1 reduce risk.",
    )
    parser.add_argument(
        "--active-drawdown-throttle-cooldown",
        type=int,
        default=0,
        help="Number of trading days to keep the throttle active after a trigger.",
    )
    parser.add_argument("--commission-rate", type=float, default=0.0001)
    parser.add_argument("--stamp-tax-rate", type=float, default=0.0005)
    parser.add_argument("--slippage-rate", type=float, default=0.0005)
    parser.add_argument("--index-file", default="hs300_index.csv")
    parser.add_argument("--adv-window", type=int, default=20)
    parser.add_argument("--adv-participation-cap", type=float, default=0.05)
    parser.add_argument("--min-adv-cny", type=float, default=3000000.0)
    parser.add_argument("--money-scale", type=float, default=1000.0)
    parser.add_argument("--limit-threshold", type=float, default=0.095)
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--min-commission-cny", type=float, default=5.0)
    parser.add_argument("--max-new-names", type=int, default=5)
    parser.add_argument(
        "--selection-policy",
        choices=("retention", "topk_dropout"),
        default="retention",
        help="Score-to-target policy. Both policies use the same realistic open ledger.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="TopK target size for topk_dropout; 0 derives it from target_frac.",
    )
    parser.add_argument(
        "--n-drop",
        type=int,
        default=0,
        help="Maximum replacements per signal day for topk_dropout.",
    )
    parser.add_argument(
        "--state-aware-selection-mode",
        default="none",
        choices=["none", "risk_rank", "risk_suppress"],
        help="Optional selection-layer risk rerank or incumbent-preserving suppression for new candidates.",
    )
    parser.add_argument(
        "--state-aware-selection-pressure-col",
        default=None,
        help="Global feature column for selection stress; defaults to --global-risk-pressure-col.",
    )
    parser.add_argument("--state-aware-selection-pressure-threshold", type=float, default=0.035)
    parser.add_argument("--state-aware-selection-pressure-width", type=float, default=0.055)
    parser.add_argument("--state-aware-selection-min-stress", type=float, default=0.0)
    parser.add_argument(
        "--state-aware-selection-rank-penalty",
        type=float,
        default=0.0,
        help="Rank penalty as a fraction of the candidate universe at full stress.",
    )
    parser.add_argument("--state-aware-selection-top-frac", type=float, default=0.006)
    parser.add_argument("--state-aware-selection-crowd-scale", type=float, default=0.10)
    parser.add_argument("--state-aware-selection-momentum-weight", type=float, default=0.40)
    parser.add_argument("--state-aware-selection-beta-weight", type=float, default=0.20)
    parser.add_argument("--state-aware-selection-vol-weight", type=float, default=0.20)
    parser.add_argument("--state-aware-selection-industry-weight", type=float, default=0.20)
    parser.add_argument(
        "--state-aware-selection-risk-delta-threshold",
        type=float,
        default=0.15,
        help="For risk_suppress, keep the incumbent when candidate risk exceeds it by this amount.",
    )
    parser.add_argument(
        "--max-new-names-list",
        default=None,
        help="Optional comma list; overrides --max-new-names for retention sweeps.",
    )
    parser.add_argument(
        "--max-new-names-mode",
        choices=("legacy", "at_most"),
        default="legacy",
    )
    parser.add_argument("--exit-hold-fracs", default="0")
    parser.add_argument("--switch-gap-fracs", default="0")
    parser.add_argument("--execution-lag", type=int, default=0)
    parser.add_argument(
        "--max-data-date",
        default=None,
        help=(
            "Optional upper bound for loaded market data. "
            "Defaults to no truncation so arbitrary date ranges are controlled by "
            "--start-date/--end-date."
        ),
    )
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--allow-forward", action="store_true", help="Accepted for old commands; open-ledger sweeps now allow arbitrary date ranges.")
    parser.add_argument("--progress-every", type=int, default=2500)
    parser.add_argument("--ohlc-cache-dir", default="cache/open_ledger_ohlc")
    parser.add_argument("--no-ohlc-cache", action="store_true")
    parser.add_argument("--ohlc-matrix-cache-dir", default="cache/open_ledger_ohlc_matrix")
    parser.add_argument("--no-ohlc-matrix-cache", action="store_true")
    add_execution_market_data_args(parser)
    parser.add_argument("--execution-mask-cache-dir", default="cache/open_ledger_execution_masks")
    parser.add_argument("--no-execution-mask-cache", action="store_true")
    parser.add_argument("--rebuild-ohlc-matrix-cache", action="store_true")
    parser.add_argument("--load-lookback-days", type=int, default=160)
    parser.add_argument(
        "--execution-constraint-mode",
        choices=("proxy", "realistic"),
        default="proxy",
        help="proxy keeps legacy 9.5%% open-gap blocking; realistic uses OHLC volume/money and limit masks.",
    )
    parser.add_argument(
        "--block-intraday-limit-touch",
        action="store_true",
        help="In realistic mode, also block orders when high/low touched the limit intraday.",
    )
    parser.add_argument("--limit-price-tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--min-buy-listing-days",
        type=int,
        default=60,
        help="In realistic mode, block new buys until this many trading days after listing; 0 disables.",
    )
    parser.add_argument(
        "--no-limit-first-trading-days",
        type=int,
        default=5,
        help="In realistic mode, disable limit-up/down checks during the first N trading days after listing.",
    )
    return parser.parse_args()


def main():
    started = time.perf_counter()
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "open_price_ledger_param_sweep_summary.csv"
    path_index_path = output_dir / "path_artifact_index.csv"
    if summary_path.exists() and not args.resume:
        summary_path.unlink()
    if path_index_path.exists() and not args.resume:
        path_index_path.unlink()

    alpha_specs = parse_alpha_specs(args.alpha_specs)
    all_codes = set()
    loaded_alpha_specs = []
    all_alpha_rows = []
    for name, path in alpha_specs:
        rows = filter_alpha_rows(load_alpha_rows(path), args.start_date, args.end_date)
        if not rows:
            raise ValueError(f"No alpha rows remain for {name} after date filtering")
        all_codes.update(code for row in rows for code in row["codes"])
        print(f"loaded alpha {name}: days={len(rows)} path={path}", flush=True)
        loaded_alpha_specs.append((name, path, rows))
        all_alpha_rows.extend(rows)

    alpha_loaded_at = time.perf_counter()

    all_codes = sorted(all_codes)
    max_data_date = pd.Timestamp(args.max_data_date).normalize() if args.max_data_date else None
    load_start, load_end = infer_ohlc_load_window(
        all_alpha_rows,
        max_data_date=max_data_date,
        execution_lag=args.execution_lag,
        lookback_days=args.load_lookback_days,
    )
    print(
        f"loading shared OHLC: codes={len(all_codes)} window="
        f"{load_start.date() if load_start is not None else 'all'}.."
        f"{load_end.date() if load_end is not None else 'all'} backend={args.ohlc_backend} "
        f"matrix_cache="
        f"{'off' if args.no_ohlc_matrix_cache else args.ohlc_matrix_cache_dir} "
        f"window_cache={'off' if args.no_ohlc_cache else args.ohlc_cache_dir}",
        flush=True,
    )
    market_frames = load_execution_market_frames(
        args,
        all_codes,
        start_date=load_start,
        end_date=load_end,
    )
    open_df = market_frames["open"]
    close_df = market_frames["close"]
    money_df = market_frames["money"]
    ohlc_loaded_at = time.perf_counter()
    adv_df = recompute_adv(money_df, args.adv_window)
    execution_masks = prepare_execution_constraint_masks(
        args,
        all_codes,
        open_df,
        close_df,
        money_df,
        load_start=load_start,
        load_end=load_end,
        ohlcv_frames=market_frames if args.ohlc_backend != "legacy" else None,
    )
    constraints_ready_at = time.perf_counter()
    if execution_masks is not None:
        print(
            "using realistic execution constraints "
            f"block_intraday_touch={args.block_intraday_limit_touch}",
            flush=True,
        )
    idx_close, idx_daily = load_index_returns(args.data_dir, args.index_file, close_df.index)
    prepared_context = prepare_open_ledger_context(open_df, close_df)
    context_ready_at = time.perf_counter()

    targets = parse_float_list(args.target_fracs)
    holds = parse_float_list(args.hold_fracs)
    bands = parse_float_list(args.rebalance_bands)
    capitals = parse_float_list(args.portfolio_values)
    industry_caps = parse_float_list(args.max_industry_weights)
    risk_targets = [None]
    if str(args.risk_target_fracs).strip():
        risk_targets.extend(parse_float_list(args.risk_target_fracs))
    risk_thresholds = parse_float_list(args.risk_target_market_mult_belows)
    defensive_tilts = parse_float_list(args.defensive_tilts)
    defensive_tilt_thresholds = parse_float_list(args.defensive_tilt_market_mult_belows)
    if args.global_risk_overlay_mode == "none":
        global_risk_thresholds = [0.0]
        global_risk_scales = [1.0]
        global_risk_targets = [None]
    else:
        global_risk_thresholds = parse_float_list(args.global_risk_pressure_thresholds)
        global_risk_scales = parse_float_list(args.global_risk_market_scales)
        global_risk_targets = [None]
        if str(args.global_risk_target_fracs).strip():
            global_risk_targets.extend(parse_float_list(args.global_risk_target_fracs))
    global_risk_frame = None
    needs_global_risk_frame = (
        args.global_risk_overlay_mode != "none"
        or args.state_aware_selection_mode != "none"
    )
    if needs_global_risk_frame:
        if not args.global_risk_features:
            raise ValueError(
                "--global-risk-features is required when global risk overlay "
                "or state-aware selection is enabled"
            )
        global_risk_frame = load_global_risk_features(args.global_risk_features, open_df.index)
    stress_names = parse_stress_names(args.stresses)
    max_new_values = (
        parse_int_list(args.max_new_names_list)
        if args.max_new_names_list is not None
        else [int(args.max_new_names)]
    )
    retention_grid = retention_param_grid(
        max_new_values,
        parse_float_list(args.exit_hold_fracs),
        parse_float_list(args.switch_gap_fracs),
    )

    pending_rows = []
    completed = set()
    if args.resume and summary_path.exists():
        existing = pd.read_csv(summary_path)
        completed = completed_keys_from_summary(existing)
        print(
            f"resume enabled: loaded {len(existing)} rows, "
            f"{len(completed)} completed keys from {summary_path}",
            flush=True,
        )

    total = (
        len(alpha_specs)
        * len(stress_names)
        * len(bands)
        * len(industry_caps)
        * len(risk_targets)
        * len(risk_thresholds)
        * len(defensive_tilts)
        * len(defensive_tilt_thresholds)
        * len(global_risk_thresholds)
        * len(global_risk_scales)
        * len(global_risk_targets)
        * len(capitals)
        * len(retention_grid)
        * sum(1 for t in targets for h in holds if h >= t)
    )
    done = len(completed)
    grid_started_at = time.perf_counter()
    for alpha_name, alpha_path, alpha_rows in loaded_alpha_specs:
        for stress_name in stress_names:
            stress_args = apply_stress_overrides(args, stress_name)
            for band in bands:
                for industry_cap in industry_caps:
                    for risk_target in risk_targets:
                        thresholds = risk_thresholds if risk_target is not None else [1.0]
                        for risk_threshold in thresholds:
                            for defensive_tilt in defensive_tilts:
                                for defensive_tilt_threshold in defensive_tilt_thresholds:
                                    for global_risk_threshold in global_risk_thresholds:
                                        for global_risk_scale in global_risk_scales:
                                            for global_risk_target in global_risk_targets:
                                                for capital in capitals:
                                                    for max_new, exit_hold, switch_gap in retention_grid:
                                                        run_args = SimpleNamespace(**vars(stress_args))
                                                        run_args.rebalance_band = float(band)
                                                        run_args.max_industry_weight = float(industry_cap)
                                                        run_args.risk_target_frac = risk_target
                                                        run_args.risk_target_market_mult_below = float(risk_threshold)
                                                        run_args.defensive_tilt = float(defensive_tilt)
                                                        run_args.defensive_tilt_market_mult_below = float(defensive_tilt_threshold)
                                                        run_args.global_risk_pressure_threshold = float(global_risk_threshold)
                                                        run_args.global_risk_market_scale = float(global_risk_scale)
                                                        run_args.global_risk_target_frac = global_risk_target
                                                        run_args._global_risk_frame = global_risk_frame
                                                        run_args.portfolio_value = float(capital)
                                                        run_args.max_new_names = int(max_new)
                                                        run_args.exit_hold_frac = exit_hold
                                                        run_args.switch_gap_frac = float(switch_gap)
                                                        for target in targets:
                                                            for hold in holds:
                                                                if hold < target:
                                                                    continue
                                                                key = sweep_key(
                                                                    alpha_name,
                                                                    stress_name,
                                                                    band,
                                                                    industry_cap,
                                                                    risk_target,
                                                                    risk_threshold,
                                                                    defensive_tilt,
                                                                    defensive_tilt_threshold,
                                                                    capital,
                                                                    max_new,
                                                                    exit_hold,
                                                                    switch_gap,
                                                                    target,
                                                                    hold,
                                                                    args.global_risk_overlay_mode,
                                                                    global_risk_threshold,
                                                                    global_risk_scale,
                                                                    global_risk_target,
                                                                    run_args.active_drawdown_throttle_lookback,
                                                                    run_args.active_drawdown_throttle_trigger,
                                                                    run_args.active_drawdown_throttle_scale,
                                                                    run_args.active_drawdown_throttle_cooldown,
                                                                    run_args.selection_policy,
                                                                    run_args.top_k,
                                                                    run_args.n_drop,
                                                                    run_args.state_aware_selection_mode,
                                                                    run_args.state_aware_selection_pressure_col,
                                                                    run_args.state_aware_selection_pressure_threshold,
                                                                    run_args.state_aware_selection_pressure_width,
                                                                    run_args.state_aware_selection_min_stress,
                                                                    run_args.state_aware_selection_rank_penalty,
                                                                    run_args.state_aware_selection_top_frac,
                                                                    run_args.state_aware_selection_crowd_scale,
                                                                    run_args.state_aware_selection_momentum_weight,
                                                                    run_args.state_aware_selection_beta_weight,
                                                                    run_args.state_aware_selection_vol_weight,
                                                                    run_args.state_aware_selection_industry_weight,
                                                                    run_args.state_aware_selection_risk_delta_threshold,
                                                                )
                                                                if key in completed:
                                                                    continue
                                                                execution_trace = [] if args.save_path_details else None
                                                                position_trace = [] if args.save_path_details else None
                                                                row, returns_df, diag_df = run_open_ledger(
                                                                    alpha_rows,
                                                                    open_df,
                                                                    close_df,
                                                                    adv_df,
                                                                    target,
                                                                    hold,
                                                                    run_args,
                                                                    idx_close,
                                                                    idx_daily,
                                                                    execution_masks=execution_masks,
                                                                    prepared_context=prepared_context,
                                                                    execution_trace_sink=execution_trace,
                                                                    position_trace_sink=position_trace,
                                                                )
                                                                if not row:
                                                                    continue
                                                                signal_dates = [
                                                                    pd.Timestamp(item["date"])
                                                                    for item in alpha_rows
                                                                ]
                                                                backtest_dates = (
                                                                    pd.to_datetime(returns_df["date"])
                                                                    if not returns_df.empty and "date" in returns_df
                                                                    else pd.DatetimeIndex([])
                                                                )
                                                                row.update({
                                                                    "alpha_name": alpha_name,
                                                                    "stress": stress_name,
                                                                    "portfolio_value": float(capital),
                                                                    "max_industry_weight": float(industry_cap),
                                                                    "defensive_tilt": float(defensive_tilt),
                                                                    "defensive_tilt_market_mult_below": float(defensive_tilt_threshold),
                                                                    "adv_participation_cap": float(run_args.adv_participation_cap),
                                                                    "min_adv_cny": float(run_args.min_adv_cny),
                                                                    "limit_threshold": float(run_args.limit_threshold),
                                                                    "max_new_names_mode": run_args.max_new_names_mode,
                                                                    "execution_constraint_mode": args.execution_constraint_mode,
                                                                    "ohlc_backend": args.ohlc_backend,
                                                                    "market_daily_store_root": (
                                                                        str(Path(args.market_daily_store_root).resolve())
                                                                        if args.ohlc_backend == "monthly"
                                                                        else ""
                                                                    ),
                                                                    "ohlc_monthly_cache_dir": (
                                                                        str(Path(args.ohlc_monthly_cache_dir).resolve())
                                                                        if args.ohlc_backend == "monthly"
                                                                        else ""
                                                                    ),
                                                                    "block_intraday_limit_touch": bool(args.block_intraday_limit_touch),
                                                                    "min_buy_listing_days": int(args.min_buy_listing_days),
                                                                    "no_limit_first_trading_days": int(args.no_limit_first_trading_days),
                                                                    "signal_start": str(min(signal_dates).date()),
                                                                    "signal_end": str(max(signal_dates).date()),
                                                                    "backtest_start": (
                                                                        str(backtest_dates.min().date())
                                                                        if len(backtest_dates)
                                                                        else ""
                                                                    ),
                                                                    "backtest_end": (
                                                                        str(backtest_dates.max().date())
                                                                        if len(backtest_dates)
                                                                        else ""
                                                                    ),
                                                                    "state_aware_selection_min_stress": float(run_args.state_aware_selection_min_stress),
                                                                    "state_aware_selection_top_frac": float(run_args.state_aware_selection_top_frac),
                                                                    "state_aware_selection_crowd_scale": float(run_args.state_aware_selection_crowd_scale),
                                                                    "state_aware_selection_momentum_weight": float(run_args.state_aware_selection_momentum_weight),
                                                                    "state_aware_selection_beta_weight": float(run_args.state_aware_selection_beta_weight),
                                                                    "state_aware_selection_vol_weight": float(run_args.state_aware_selection_vol_weight),
                                                                    "state_aware_selection_industry_weight": float(run_args.state_aware_selection_industry_weight),
                                                                    "state_aware_selection_risk_delta_threshold": float(run_args.state_aware_selection_risk_delta_threshold),
                                                                })
                                                                if args.save_path_details:
                                                                    path_dir = output_dir / "paths"
                                                                    path_dir.mkdir(parents=True, exist_ok=True)
                                                                    key_hash = hashlib.sha256(
                                                                        repr(key).encode("utf-8")
                                                                    ).hexdigest()[:12]
                                                                    path_tag = (
                                                                        f"{alpha_name}_{stress_name}_"
                                                                        f"pv{int(float(capital) / 10000):04d}w_"
                                                                        f"target{int(round(target * 1000)):03d}_"
                                                                        f"hold{int(round(hold * 1000)):03d}_"
                                                                        f"{key_hash}"
                                                                    )
                                                                    artifact_paths = write_path_artifacts(
                                                                        path_dir,
                                                                        path_tag,
                                                                        returns_df,
                                                                        diag_df,
                                                                        execution_trace,
                                                                        position_trace,
                                                                    )
                                                                    append_summary_rows(
                                                                        path_index_path,
                                                                        [{
                                                                            "sweep_key_sha256": key_hash,
                                                                            "alpha_name": alpha_name,
                                                                            "stress": stress_name,
                                                                            "portfolio_value": float(capital),
                                                                            "signal_start": row["signal_start"],
                                                                            "signal_end": row["signal_end"],
                                                                            "backtest_start": row["backtest_start"],
                                                                            "backtest_end": row["backtest_end"],
                                                                            **{
                                                                                name: str(path.resolve())
                                                                                for name, path in artifact_paths.items()
                                                                            },
                                                                        }],
                                                                    )
                                                                pending_rows.append(row)
                                                                completed.add(key)
                                                                done += 1
                                                                if done % 10 == 0 or done == total:
                                                                    append_summary_rows(summary_path, pending_rows)
                                                                    pending_rows.clear()
                                                                    print(
                                                                        f"{done}/{total} {alpha_name} stress={stress_name} "
                                                                        f"band={band:.2f} indcap={industry_cap:.2f} "
                                                                        f"risk={risk_target}<{risk_threshold:.2f} "
                                                                        f"tilt={defensive_tilt:.2f}<{defensive_tilt_threshold:.2f} "
                                                                        f"global={args.global_risk_overlay_mode}:{global_risk_threshold:.3f}x{global_risk_scale:.2f} "
                                                                        f"pv={capital/1e4:.0f}w target={target:.3f} hold={hold:.2f} "
                                                                        f"maxnew={max_new} exit={exit_hold} gap={switch_gap:.3f} "
                                                                        f"ann={row['ann']:.2f}% sharpe={row['sharpe']:.3f}",
                                                                        flush=True,
                                                                    )
        gc.collect()

    grid_completed_at = time.perf_counter()
    append_summary_rows(summary_path, pending_rows)
    summary = pd.read_csv(summary_path).copy() if summary_path.exists() else pd.DataFrame()
    if not summary.empty:
        sort_cols = ["sharpe", "ann"]
        top = summary.sort_values(sort_cols, ascending=[False, False]).head(50)
        top.to_csv(output_dir / "top50_by_sharpe.csv", index=False)
        balanced = summary.copy()
        balanced["score"] = (
            balanced["sharpe"] * 10.0
            + balanced["ann"] / 10.0
            - balanced["mdd"] * 10.0
            - balanced["total_cost"]
        )
        balanced = balanced.sort_values("score", ascending=False).head(50)
        balanced.to_csv(output_dir / "top50_balanced_score.csv", index=False)
    print(f"Saved sweep summary: {summary_path}", flush=True)
    if args.performance_report:
        report_path = Path(args.performance_report)
        if not report_path.is_absolute():
            report_path = output_dir / report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "alpha_load_seconds": round(alpha_loaded_at - started, 3),
            "ohlc_load_seconds": round(ohlc_loaded_at - alpha_loaded_at, 3),
            "constraint_prepare_seconds": round(constraints_ready_at - ohlc_loaded_at, 3),
            "context_prepare_seconds": round(context_ready_at - constraints_ready_at, 3),
            "grid_execute_seconds": round(grid_completed_at - grid_started_at, 3),
            "total_seconds": round(time.perf_counter() - started, 3),
            "rss_mb": process_rss_mb(),
            "alpha_count": len(alpha_specs),
            "signal_rows": sum(len(rows) for _, _, rows in loaded_alpha_specs),
            "ohlc_days": len(open_df.index),
            "ohlc_codes": len(open_df.columns),
            "completed_grid_rows": len(summary),
            "execution_mode": args.execution_constraint_mode,
            "stresses": stress_names,
        }
        report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"Saved performance report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
