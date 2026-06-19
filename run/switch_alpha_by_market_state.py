"""Switch between two alpha JSONL files by a simple index market state.

This is intentionally low-memory: it streams two JSONL files line by line and
uses only the HS300 index close series to choose a row source for each date.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha.io import iter_aligned_alpha_rows, load_alpha_dates, write_alpha_rows
from backtest.market_state import load_index_states
from core.research_protocol import (
    assert_alpha_dates_within_forward,
    assert_alpha_dates_within_research,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Switch alpha rows by market state")
    parser.add_argument("--base-alpha", required=True, help="Alpha used in bear/crash states")
    parser.add_argument("--risk-on-alpha", required=True, help="Alpha used outside bear/crash states")
    parser.add_argument("--output-alpha", required=True)
    parser.add_argument("--index-file", default="data/raw/hs300_index.csv")
    parser.add_argument("--ma-window", type=int, default=60)
    parser.add_argument("--crash-ret", type=float, default=-0.03)
    parser.add_argument("--bear-source", default="base", choices=["base", "risk_on"])
    parser.add_argument("--normal-source", default="risk_on", choices=["base", "risk_on"])
    parser.add_argument("--allow-forward", action="store_true")
    return parser.parse_args()


def load_index_state(path, ma_window, crash_ret):
    states = load_index_states(path, ma_window=ma_window, crash_ret=crash_ret)
    return {
        row.date.strftime("%Y-%m-%d"): row.market_state
        for row in states.itertuples(index=False)
    }


def choose(source, base, risk_on):
    return base if source == "base" else risk_on


def main():
    args = parse_args()
    base_dates = load_alpha_dates(args.base_alpha)
    risk_on_dates = load_alpha_dates(args.risk_on_alpha)
    assert_dates = (
        assert_alpha_dates_within_forward
        if args.allow_forward
        else assert_alpha_dates_within_research
    )
    period = "forward" if args.allow_forward else "research"
    assert_dates(base_dates, context=f"market switch base {period} alpha")
    assert_dates(risk_on_dates, context=f"market switch risk-on {period} alpha")
    states = load_index_state(args.index_file, args.ma_window, args.crash_ret)
    out_path = Path(args.output_alpha)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {"base": 0, "risk_on": 0, "bear": 0, "crash": 0, "normal": 0}
    rows = 0
    def switched_rows():
        nonlocal rows
        for base, risk_on in iter_aligned_alpha_rows(args.base_alpha, args.risk_on_alpha):
            state = states.get(base["date"], "normal")
            source = args.bear_source if state in ("bear", "crash") else args.normal_source
            picked = dict(choose(source, base, risk_on))
            picked["market_alpha_switch"] = {
                "state": state,
                "source": source,
                "bear_source": args.bear_source,
                "normal_source": args.normal_source,
                "ma_window": int(args.ma_window),
                "crash_ret": float(args.crash_ret),
            }
            counts[source] += 1
            counts[state] += 1
            rows += 1
            yield picked

    write_alpha_rows(out_path, switched_rows())
    print(f"wrote={out_path} rows={rows} counts={counts}", flush=True)


if __name__ == "__main__":
    main()
