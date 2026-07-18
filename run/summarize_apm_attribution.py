"""Create APM-style attribution diagnostics from open-ledger outputs."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_ts_code(code):
    code = str(code).strip()
    if not code or code.lower() == "nan":
        return None
    lower = code.lower()
    if lower.startswith("sh.") or lower.startswith("sz."):
        return f"{code[3:9]}.{lower[:2].upper()}"
    if "." in code:
        left, right = code.split(".", 1)
        if left.isdigit():
            return f"{left.zfill(6)}.{right.upper()}"
    digits = "".join(ch for ch in code if ch.isdigit())
    if len(digits) >= 6:
        suffix = "SH" if digits[:1] in {"5", "6", "9"} else "SZ"
        return f"{digits[-6:]}.{suffix}"
    return code


def parse_holdings(raw):
    holdings = {}
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return holdings
    for part in str(raw).split(";"):
        if not part or "=" not in part:
            continue
        code, value = part.split("=", 1)
        code = normalize_ts_code(code)
        if code is None:
            continue
        try:
            weight = float(value)
        except ValueError:
            continue
        if abs(weight) > 1e-12:
            holdings[code] = weight
    return holdings


def load_industry_map(path):
    path = Path(path)
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if "code" not in frame or "industry" not in frame:
        return {}
    frame["code_norm"] = frame["code"].map(normalize_ts_code)
    frame = frame.dropna(subset=["code_norm", "industry"])
    return dict(zip(frame["code_norm"], frame["industry"].astype(str)))


class PriceFeatureCache:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self._cache = {}

    def get(self, code):
        code = normalize_ts_code(code)
        if code in self._cache:
            return self._cache[code]
        path = self.data_dir / f"{code}.csv"
        if not path.exists():
            self._cache[code] = None
            return None
        frame = pd.read_csv(path, usecols=["trade_date", "close", "money"])
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame = frame.set_index("trade_date").sort_index()
        close = frame["close"].astype(float)
        ret = close.pct_change()
        feature = pd.DataFrame(index=frame.index)
        feature["momentum20"] = close.shift(1) / close.shift(21) - 1.0
        feature["volatility60"] = ret.shift(1).rolling(60, min_periods=20).std() * np.sqrt(252)
        feature["liquidity20"] = np.log1p(
            frame["money"].astype(float).shift(1).rolling(20, min_periods=5).mean()
        )
        self._cache[code] = feature
        return feature

    def features_on(self, code, date):
        frame = self.get(code)
        if frame is None or frame.empty:
            return {}
        date = pd.Timestamp(date)
        pos = frame.index.searchsorted(date, side="right") - 1
        if pos < 0:
            return {}
        row = frame.iloc[pos]
        return {
            key: float(value)
            for key, value in row.items()
            if np.isfinite(value)
        }


def weighted_style_exposure(holdings, date, price_cache):
    gross = sum(abs(w) for w in holdings.values())
    if gross <= 1e-12:
        return {"momentum20": 0.0, "volatility60": 0.0, "liquidity20": 0.0}
    sums = {"momentum20": 0.0, "volatility60": 0.0, "liquidity20": 0.0}
    weights = {"momentum20": 0.0, "volatility60": 0.0, "liquidity20": 0.0}
    for code, weight in holdings.items():
        features = price_cache.features_on(code, date)
        for key in sums:
            if key not in features:
                continue
            aw = abs(weight)
            sums[key] += aw * features[key]
            weights[key] += aw
    return {
        key: sums[key] / weights[key] if weights[key] > 1e-12 else 0.0
        for key in sums
    }


def industry_weights(holdings, industry_map):
    result = {}
    for code, weight in holdings.items():
        industry = industry_map.get(normalize_ts_code(code), "UNKNOWN")
        result[industry] = result.get(industry, 0.0) + float(weight)
    return result


def _max_drawdown_from_returns(returns):
    returns = np.nan_to_num(np.asarray(returns, dtype=float), nan=0.0)
    if len(returns) == 0:
        return 0.0
    curve = np.concatenate(([1.0], np.cumprod(1.0 + returns)))
    peak = np.maximum.accumulate(curve)
    return float(np.max((peak - curve) / (peak + 1e-12)))


def _slice_metrics(frame, group_name, group_value):
    ret = frame["return"].astype(float).to_numpy()
    active = frame.get("active_return", frame["return"]).astype(float).to_numpy()
    active_std = float(np.std(active))
    return {
        "slice": group_name,
        "value": str(group_value),
        "days": int(len(frame)),
        "sum_return": float(np.sum(ret)),
        "sum_active_return": float(np.sum(active)),
        "mean_return": float(np.mean(ret)) if len(ret) else 0.0,
        "mean_active_return": float(np.mean(active)) if len(active) else 0.0,
        "information_ratio": float(np.mean(active) / (active_std + 1e-8) * np.sqrt(252))
        if len(active)
        else 0.0,
        "mdd": _max_drawdown_from_returns(ret),
        "avg_cost": float(frame.get("cost", pd.Series([0.0])).mean()),
        "avg_gross_weight": float(frame.get("gross_weight", pd.Series([0.0])).mean()),
        "avg_market_mult": float(frame.get("market_mult", pd.Series([1.0])).mean()),
        "avg_industry_hhi": float(frame.get("industry_abs_hhi", pd.Series([0.0])).mean()),
        "avg_top_industry_abs_weight": float(
            frame.get("top_industry_weight", pd.Series([0.0])).abs().mean()
        ),
    }


def build_slice_summary(returns_df, diag_df, industry_df, style_df):
    returns = returns_df.copy()
    diag = diag_df.copy()
    returns["date"] = pd.to_datetime(returns["date"])
    diag["date"] = pd.to_datetime(diag["date"])
    merged = returns.merge(
        diag[
            [
                col
                for col in (
                    "date",
                    "market_mult",
                    "gross_weight",
                    "cost",
                    "portfolio_beta_60d",
                )
                if col in diag.columns
            ]
        ],
        on="date",
        how="left",
    )
    if not industry_df.empty:
        ind = industry_df.copy()
        ind["date"] = pd.to_datetime(ind["date"])
        merged = merged.merge(ind, on="date", how="left")
    if not style_df.empty:
        style = style_df.copy()
        style["date"] = pd.to_datetime(style["date"])
        merged = merged.merge(style, on="date", how="left")

    merged["market_state"] = np.where(
        merged.get("market_mult", pd.Series(1.0, index=merged.index)).astype(float) < 0.8,
        "defensive",
        "normal",
    )
    hhi = merged.get("industry_abs_hhi", pd.Series(0.0, index=merged.index)).fillna(0.0)
    hhi_median = float(hhi.median()) if len(hhi) else 0.0
    merged["industry_concentration"] = np.where(
        hhi >= hhi_median,
        "high",
        "low",
    )

    rows = []
    for key in ("market_state", "industry_concentration"):
        for value, group in merged.groupby(key, dropna=False):
            rows.append(_slice_metrics(group, key, value))
    if "top_industry" in merged.columns:
        for value, group in merged.groupby("top_industry", dropna=False):
            if len(group) < 5:
                continue
            rows.append(_slice_metrics(group, "top_industry", value))
    return pd.DataFrame(rows), merged


def summarize_attribution(returns_df, diag_df, industry_map, data_dir):
    price_cache = PriceFeatureCache(data_dir)
    returns = returns_df.copy()
    diag = diag_df.copy()
    returns["date"] = pd.to_datetime(returns["date"])
    diag["date"] = pd.to_datetime(diag["date"])

    total_return = float((1.0 + returns["return"].astype(float)).prod() - 1.0)
    benchmark_return = float((1.0 + returns.get("benchmark_return", 0.0).astype(float)).prod() - 1.0)
    active_return = float((1.0 + returns.get("active_return", returns["return"]).astype(float)).prod() - 1.0)
    total_cost = float(diag.get("cost", pd.Series(dtype=float)).sum())
    total_commission = float(diag.get("commission", pd.Series(dtype=float)).sum())
    total_stamp_tax = float(diag.get("stamp_tax", pd.Series(dtype=float)).sum())
    total_slippage = float(diag.get("slippage", pd.Series(dtype=float)).sum())

    industry_rows = []
    style_rows = []
    for _, row in diag.iterrows():
        holdings = parse_holdings(row.get("holdings", ""))
        if not holdings:
            continue
        ind_w = industry_weights(holdings, industry_map)
        gross = sum(abs(w) for w in holdings.values())
        top_industry = max(ind_w.items(), key=lambda kv: abs(kv[1])) if ind_w else ("UNKNOWN", 0.0)
        industry_rows.append({
            "date": row["date"],
            "gross_weight": gross,
            "industry_count": len(ind_w),
            "top_industry": top_industry[0],
            "top_industry_weight": top_industry[1],
            "industry_abs_hhi": sum((abs(w) / gross) ** 2 for w in ind_w.values()) if gross > 0 else 0.0,
        })
        style = weighted_style_exposure(holdings, row["date"], price_cache)
        style["date"] = row["date"]
        style_rows.append(style)

    industry_df = pd.DataFrame(industry_rows)
    style_df = pd.DataFrame(style_rows)
    summary = {
        "days": int(len(returns)),
        "total_return": total_return,
        "benchmark_return": benchmark_return,
        "active_return": active_return,
        "total_cost": total_cost,
        "total_commission": total_commission,
        "total_stamp_tax": total_stamp_tax,
        "total_slippage": total_slippage,
        "cost_to_total_return": total_cost / max(abs(total_return), 1e-12),
        "avg_market_mult": float(diag.get("market_mult", pd.Series([1.0])).mean()),
        "avg_gross_weight": float(diag.get("gross_weight", pd.Series([0.0])).mean()),
        "avg_portfolio_beta_60d": float(diag.get("portfolio_beta_60d", pd.Series([0.0])).mean()),
        "avg_portfolio_beta_per_gross_60d": float(diag.get("portfolio_beta_per_gross_60d", pd.Series([0.0])).mean()),
        "avg_portfolio_specific_vol_60d": float(diag.get("portfolio_specific_vol_60d", pd.Series([0.0])).mean()),
        "avg_industry_count": float(industry_df["industry_count"].mean()) if not industry_df.empty else 0.0,
        "avg_top_industry_abs_weight": float(industry_df["top_industry_weight"].abs().mean()) if not industry_df.empty else 0.0,
        "avg_industry_abs_hhi": float(industry_df["industry_abs_hhi"].mean()) if not industry_df.empty else 0.0,
        "avg_momentum20": float(style_df["momentum20"].mean()) if not style_df.empty else 0.0,
        "avg_volatility60": float(style_df["volatility60"].mean()) if not style_df.empty else 0.0,
        "avg_liquidity20": float(style_df["liquidity20"].mean()) if not style_df.empty else 0.0,
    }
    slice_df, _ = build_slice_summary(returns, diag, industry_df, style_df)
    return summary, industry_df, style_df, slice_df


def write_markdown(summary, slice_df, output_path):
    lines = [
        "# APM Attribution Report",
        "",
        "This report is diagnostic-only. It does not change trades.",
        "",
        "## Return Decomposition",
        "",
        f"- total return: {summary['total_return']:.2%}",
        f"- benchmark return: {summary['benchmark_return']:.2%}",
        f"- compounded active return: {summary['active_return']:.2%}",
        f"- total explicit cost: {summary['total_cost']:.2%}",
        f"- cost / absolute total return: {summary['cost_to_total_return']:.2%}",
        "",
        "## Exposure Diagnostics",
        "",
        f"- average market multiplier: {summary['avg_market_mult']:.3f}",
        f"- average gross weight: {summary['avg_gross_weight']:.3f}",
        f"- average 60d portfolio beta: {summary['avg_portfolio_beta_60d']:.3f}",
        f"- average 60d beta per gross: {summary['avg_portfolio_beta_per_gross_60d']:.3f}",
        f"- average 60d specific volatility proxy: {summary['avg_portfolio_specific_vol_60d']:.3f}",
        "",
        "## Industry And Style",
        "",
        f"- average industry count: {summary['avg_industry_count']:.2f}",
        f"- average absolute top-industry weight: {summary['avg_top_industry_abs_weight']:.3f}",
        f"- average industry concentration HHI: {summary['avg_industry_abs_hhi']:.3f}",
        f"- average momentum20 exposure: {summary['avg_momentum20']:.3f}",
        f"- average volatility60 exposure: {summary['avg_volatility60']:.3f}",
        f"- average liquidity20 exposure: {summary['avg_liquidity20']:.3f}",
    ]
    if not slice_df.empty:
        lines.extend(["", "## Slice Diagnostics", ""])
        for slice_name in ("market_state", "industry_concentration"):
            subset = slice_df[slice_df["slice"] == slice_name]
            if subset.empty:
                continue
            lines.append(f"### {slice_name}")
            lines.append("")
            for _, row in subset.iterrows():
                lines.append(
                    f"- {row['value']}: days {int(row['days'])}, "
                    f"active sum {row['sum_active_return']:.2%}, "
                    f"IR {row['information_ratio']:.3f}, "
                    f"MDD {row['mdd']:.2%}"
                )
            lines.append("")
        top = (
            slice_df[slice_df["slice"] == "top_industry"]
            .sort_values(["days", "sum_active_return"], ascending=[False, False])
            .head(5)
        )
        if not top.empty:
            lines.append("### top_industry")
            lines.append("")
            for _, row in top.iterrows():
                lines.append(
                    f"- {row['value']}: days {int(row['days'])}, "
                    f"active sum {row['sum_active_return']:.2%}, "
                    f"IR {row['information_ratio']:.3f}"
                )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--returns-csv", required=True)
    parser.add_argument("--diagnostics-csv", required=True)
    parser.add_argument("--industry-csv", default="data/stock_industry.csv")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    returns_df = pd.read_csv(args.returns_csv)
    diag_df = pd.read_csv(args.diagnostics_csv)
    industry_map = load_industry_map(args.industry_csv)
    summary, industry_df, style_df, slice_df = summarize_attribution(
        returns_df,
        diag_df,
        industry_map,
        args.data_dir,
    )
    pd.DataFrame([summary]).to_csv(out_dir / "apm_attribution_summary.csv", index=False)
    industry_df.to_csv(out_dir / "apm_industry_exposure.csv", index=False)
    style_df.to_csv(out_dir / "apm_style_exposure.csv", index=False)
    slice_df.to_csv(out_dir / "apm_slice_summary.csv", index=False)
    write_markdown(summary, slice_df, out_dir / "apm_attribution_report.md")
    print(f"wrote APM attribution report to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
