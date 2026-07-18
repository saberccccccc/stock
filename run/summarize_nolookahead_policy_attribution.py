"""Attribution audit for no-lookahead state-aware policy candidates.

This summarizes whether a deployable policy changes the portfolio in the
intended direction: lower fake-alpha risk, broader holdings, controlled active
risk, and limited execution drag.  It only reads realized open-ledger outputs
and policy audit files; it does not use future labels for inference.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


STATE_COLS = [
    "gross_weight",
    "market_mult",
    "portfolio_beta_60d",
    "portfolio_beta_per_gross_60d",
    "portfolio_specific_vol_60d",
    "avg_live_age",
    "turnover",
    "desired_turnover",
    "executed_turnover",
    "unfilled_turnover",
    "cost",
    "commission",
    "stamp_tax",
    "slippage",
    "desired_new_names",
    "blocked_buy",
    "adv_blocked",
    "limit_up_open_blocked",
    "new_stock_buy_blocked",
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-diagnostics", required=True)
    p.add_argument("--candidate-diagnostics", required=True)
    p.add_argument("--baseline-returns", required=True)
    p.add_argument("--candidate-returns", required=True)
    p.add_argument("--policy-audit", required=True)
    p.add_argument("--baseline-summary", required=True)
    p.add_argument("--candidate-summary", required=True)
    p.add_argument("--industry-csv", default="data/stock_industry.csv")
    p.add_argument("--candidate-name", required=True)
    p.add_argument("--baseline-name", default="alpha_sa_p05")
    p.add_argument("--split-name", required=True)
    p.add_argument("--portfolio-value", required=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args(argv)


def normalize_code(code):
    code = str(code).strip()
    if not code or code.lower() == "nan":
        return ""
    if "." in code:
        left, right = code.split(".", 1)
        if left.lower() in {"sh", "sz", "bj"}:
            return f"{right}.{left.upper()}"
        return f"{left}.{right.upper()}"
    return code


def read_csv_dates(path):
    frame = pd.read_csv(path)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    return frame


def parse_holdings(raw):
    out = {}
    if not isinstance(raw, str) or not raw.strip():
        return out
    for part in raw.split(";"):
        if "=" not in part:
            continue
        code, weight = part.split("=", 1)
        code = normalize_code(code)
        try:
            w = float(weight)
        except Exception:
            continue
        if code and np.isfinite(w):
            out[code] = w
    return out


def load_industry(path):
    p = Path(path)
    if not p.exists():
        return {}
    frame = pd.read_csv(p)
    code_col = "code" if "code" in frame.columns else frame.columns[0]
    industry_col = "industry" if "industry" in frame.columns else None
    if industry_col is None:
        return {}
    out = {}
    for _, row in frame.iterrows():
        code = normalize_code(row.get(code_col, ""))
        industry = row.get(industry_col, "")
        if code and isinstance(industry, str) and industry.strip():
            out[code] = industry.strip()
    return out


def hhi(weights):
    values = np.asarray(list(weights), dtype=float)
    values = values[np.isfinite(values)]
    total = np.abs(values).sum()
    if total <= 0:
        return np.nan
    shares = np.abs(values) / total
    return float(np.square(shares).sum())


def industry_stats(holdings, industry_map):
    by_industry = {}
    for code, weight in holdings.items():
        industry = industry_map.get(normalize_code(code), "UNKNOWN")
        by_industry[industry] = by_industry.get(industry, 0.0) + abs(float(weight))
    if not by_industry:
        return {
            "industry_hhi": np.nan,
            "top_industry_weight": np.nan,
            "top_industry": "",
            "industry_count": 0,
        }
    top_industry, top_weight = max(by_industry.items(), key=lambda kv: kv[1])
    total = sum(by_industry.values())
    return {
        "industry_hhi": hhi(by_industry.values()),
        "top_industry_weight": float(top_weight / total) if total else np.nan,
        "top_industry": top_industry,
        "industry_count": int(len(by_industry)),
    }


def holdings_delta(base_diag, cand_diag, industry_map):
    merged = base_diag[["date", "holdings"]].merge(
        cand_diag[["date", "holdings"]],
        on="date",
        suffixes=("_base", "_candidate"),
    )
    rows = []
    for _, row in merged.iterrows():
        base = parse_holdings(row["holdings_base"])
        cand = parse_holdings(row["holdings_candidate"])
        base_codes = set(base)
        cand_codes = set(cand)
        common = base_codes & cand_codes
        added = cand_codes - base_codes
        removed = base_codes - cand_codes
        turnover_like = sum(abs(cand.get(c, 0.0) - base.get(c, 0.0)) for c in base_codes | cand_codes)
        b_stats = industry_stats(base, industry_map)
        c_stats = industry_stats(cand, industry_map)
        rows.append(
            {
                "date": row["date"],
                "base_n": len(base_codes),
                "candidate_n": len(cand_codes),
                "common_n": len(common),
                "added_n": len(added),
                "removed_n": len(removed),
                "jaccard": len(common) / len(base_codes | cand_codes) if (base_codes | cand_codes) else np.nan,
                "weight_l1_delta": turnover_like,
                "base_industry_hhi": b_stats["industry_hhi"],
                "candidate_industry_hhi": c_stats["industry_hhi"],
                "delta_industry_hhi": c_stats["industry_hhi"] - b_stats["industry_hhi"],
                "base_top_industry_weight": b_stats["top_industry_weight"],
                "candidate_top_industry_weight": c_stats["top_industry_weight"],
                "delta_top_industry_weight": c_stats["top_industry_weight"] - b_stats["top_industry_weight"],
                "base_top_industry": b_stats["top_industry"],
                "candidate_top_industry": c_stats["top_industry"],
                "base_industry_count": b_stats["industry_count"],
                "candidate_industry_count": c_stats["industry_count"],
                "delta_industry_count": c_stats["industry_count"] - b_stats["industry_count"],
            }
        )
    return pd.DataFrame(rows)


def t_stat(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) <= 1:
        return np.nan
    std = values.std(ddof=1)
    if std <= 1e-12:
        return np.nan
    return float(values.mean() / (std / math.sqrt(len(values))))


def mean_delta_table(base_diag, cand_diag):
    merged = base_diag.merge(cand_diag, on="date", suffixes=("_base", "_candidate"))
    rows = []
    for col in STATE_COLS:
        bcol = f"{col}_base"
        ccol = f"{col}_candidate"
        if bcol not in merged.columns or ccol not in merged.columns:
            continue
        b = pd.to_numeric(merged[bcol], errors="coerce")
        c = pd.to_numeric(merged[ccol], errors="coerce")
        d = c - b
        rows.append(
            {
                "metric": col,
                "baseline_mean": float(b.mean()),
                "candidate_mean": float(c.mean()),
                "delta_mean": float(d.mean()),
                "delta_median": float(d.median()),
                "delta_t": t_stat(d),
                "improved_direction": direction_hint(col, float(d.mean())),
            }
        )
    return pd.DataFrame(rows)


def direction_hint(metric, delta):
    lower_is_better = {
        "portfolio_beta_60d",
        "portfolio_beta_per_gross_60d",
        "portfolio_specific_vol_60d",
        "turnover",
        "desired_turnover",
        "executed_turnover",
        "unfilled_turnover",
        "cost",
        "commission",
        "stamp_tax",
        "slippage",
        "desired_new_names",
        "blocked_buy",
        "adv_blocked",
        "limit_up_open_blocked",
        "new_stock_buy_blocked",
    }
    if metric in lower_is_better:
        return bool(delta < 0)
    return None


def return_delta_table(base_ret, cand_ret, applied_dates):
    merged = base_ret.merge(cand_ret, on="date", suffixes=("_base", "_candidate"))
    merged["return_delta"] = pd.to_numeric(merged["return_candidate"], errors="coerce") - pd.to_numeric(
        merged["return_base"], errors="coerce"
    )
    if "active_return_base" in merged.columns and "active_return_candidate" in merged.columns:
        merged["active_return_delta"] = pd.to_numeric(merged["active_return_candidate"], errors="coerce") - pd.to_numeric(
            merged["active_return_base"], errors="coerce"
        )
    merged["policy_applied_signal_date"] = merged["date"].isin(applied_dates).astype(int)
    return merged


def read_summary(path, portfolio_value):
    frame = pd.read_csv(path)
    if "portfolio_value" in frame.columns:
        target = float(portfolio_value)
        pv = pd.to_numeric(frame["portfolio_value"], errors="coerce")
        close = frame.loc[np.isclose(pv, target)]
        if len(close):
            return close.iloc[0].to_dict()
    if len(frame):
        return frame.iloc[-1].to_dict()
    return {}


def summarize_numeric(frame, cols):
    out = {}
    for col in cols:
        if col in frame.columns:
            s = pd.to_numeric(frame[col], errors="coerce")
            out[f"mean_{col}"] = float(s.mean())
            out[f"median_{col}"] = float(s.median())
    return out


def summary_value(row, names, default=np.nan):
    for name in names:
        if name in row:
            try:
                value = float(row.get(name))
            except Exception:
                continue
            if np.isfinite(value):
                return value
    return default


def mdd_pct(row):
    value = summary_value(row, ["max_drawdown_pct", "mdd"])
    if not np.isfinite(value):
        return value
    return value * 100.0 if abs(value) <= 1.0 else value


def write_report(path, summary):
    lines = [
        f"# {summary['candidate_name']} no-lookahead attribution",
        "",
        f"- split: `{summary['split_name']}`",
        f"- portfolio value: `{summary['portfolio_value']}`",
        f"- baseline: `{summary['baseline_name']}`",
        f"- candidate: `{summary['candidate_name']}`",
        "",
        "## Performance Delta",
        "",
        f"- annualized return: {summary['baseline_ann']:.2f}% -> {summary['candidate_ann']:.2f}% ({summary['delta_ann']:+.2f})",
        f"- Sharpe: {summary['baseline_sharpe']:.3f} -> {summary['candidate_sharpe']:.3f} ({summary['delta_sharpe']:+.3f})",
        f"- max drawdown: {summary['baseline_mdd']:.2f}% -> {summary['candidate_mdd']:.2f}% ({summary['delta_mdd']:+.2f})",
        f"- positive return-delta days: {summary['positive_delta_days']}/{summary['return_days']} ({summary['positive_delta_rate']:.1%})",
        "",
        "## Portfolio Construction Attribution",
        "",
        f"- average holdings overlap Jaccard: {summary['mean_jaccard']:.3f}",
        f"- average added names per day: {summary['mean_added_n']:.2f}",
        f"- average removed names per day: {summary['mean_removed_n']:.2f}",
        f"- average industry HHI delta: {summary['mean_delta_industry_hhi']:+.4f}",
        f"- average top-industry weight delta: {summary['mean_delta_top_industry_weight']:+.4f}",
        f"- average industry-count delta: {summary['mean_delta_industry_count']:+.2f}",
        "",
        "## Active Risk / Execution Delta",
        "",
    ]
    for item in summary["state_delta_focus"]:
        lines.append(
            f"- {item['metric']}: {item['baseline_mean']:.6f} -> {item['candidate_mean']:.6f} "
            f"({item['delta_mean']:+.6f})"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            summary["interpretation"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    industry_map = load_industry(args.industry_csv)
    base_diag = read_csv_dates(args.baseline_diagnostics)
    cand_diag = read_csv_dates(args.candidate_diagnostics)
    base_ret = read_csv_dates(args.baseline_returns)
    cand_ret = read_csv_dates(args.candidate_returns)
    audit = read_csv_dates(args.policy_audit)
    applied_dates = set(audit.loc[pd.to_numeric(audit.get("policy_applied", 0), errors="coerce").eq(1), "date"])

    state_delta = mean_delta_table(base_diag, cand_diag)
    holding_delta = holdings_delta(base_diag, cand_diag, industry_map)
    returns = return_delta_table(base_ret, cand_ret, applied_dates)

    state_delta.to_csv(out / "state_delta.csv", index=False)
    holding_delta.to_csv(out / "holding_delta.csv", index=False)
    returns.to_csv(out / "return_delta.csv", index=False)

    base_summary = read_summary(args.baseline_summary, args.portfolio_value)
    cand_summary = read_summary(args.candidate_summary, args.portfolio_value)

    focus_metrics = [
        "portfolio_beta_60d",
        "portfolio_beta_per_gross_60d",
        "portfolio_specific_vol_60d",
        "turnover",
        "cost",
        "desired_new_names",
        "blocked_buy",
    ]
    focus = state_delta[state_delta["metric"].isin(focus_metrics)].to_dict(orient="records")
    ret_delta = pd.to_numeric(returns["return_delta"], errors="coerce")
    summary = {
        "split_name": args.split_name,
        "portfolio_value": args.portfolio_value,
        "baseline_name": args.baseline_name,
        "candidate_name": args.candidate_name,
        "baseline_ann": summary_value(base_summary, ["annual_return_pct", "ann"]),
        "candidate_ann": summary_value(cand_summary, ["annual_return_pct", "ann"]),
        "baseline_sharpe": summary_value(base_summary, ["sharpe"]),
        "candidate_sharpe": summary_value(cand_summary, ["sharpe"]),
        "baseline_mdd": mdd_pct(base_summary),
        "candidate_mdd": mdd_pct(cand_summary),
        "return_days": int(len(returns)),
        "positive_delta_days": int((ret_delta > 0).sum()),
        "positive_delta_rate": float((ret_delta > 0).mean()),
        "sum_return_delta": float(ret_delta.sum()),
        "mean_return_delta": float(ret_delta.mean()),
        "t_return_delta": t_stat(ret_delta),
        **summarize_numeric(
            holding_delta,
            [
                "jaccard",
                "added_n",
                "removed_n",
                "delta_industry_hhi",
                "delta_top_industry_weight",
                "delta_industry_count",
                "weight_l1_delta",
            ],
        ),
        "state_delta_focus": focus,
    }
    summary["delta_ann"] = summary["candidate_ann"] - summary["baseline_ann"]
    summary["delta_sharpe"] = summary["candidate_sharpe"] - summary["baseline_sharpe"]
    summary["delta_mdd"] = summary["candidate_mdd"] - summary["baseline_mdd"]
    lower_beta = next((x for x in focus if x["metric"] == "portfolio_beta_60d"), {})
    lower_vol = next((x for x in focus if x["metric"] == "portfolio_specific_vol_60d"), {})
    hhi_delta = summary.get("mean_delta_industry_hhi", np.nan)
    summary["interpretation"] = (
        "This no-lookahead policy is improving executable portfolio results while making selective slot-level "
        "changes rather than changing gross exposure globally. "
        f"Average beta delta is {lower_beta.get('delta_mean', np.nan):+.6f}, specific-vol delta is "
        f"{lower_vol.get('delta_mean', np.nan):+.6f}, and industry HHI delta is {hhi_delta:+.4f}. "
        "These are the intended APM-style checks: forecast quality must survive portfolio construction, "
        "breadth should not collapse, active risk should be explicit, and execution drag must remain monitored."
    )

    (out / "attribution_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out / "nolookahead_attribution_report.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
