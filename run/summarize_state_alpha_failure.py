"""Diagnose when open-ledger alpha works or fails by state slices.

This is diagnostic-only.  It does not change signals or portfolio weights.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_ts_code(code):
    code = str(code).strip()
    if not code or code.lower() == "nan":
        return None
    lower = code.lower()
    if lower.startswith(("sh.", "sz.", "bj.")):
        return f"{code[3:9]}.{lower[:2].upper()}"
    if "." in code:
        left, right = code.split(".", 1)
        if left.isdigit():
            return f"{left.zfill(6)}.{right.upper()}"
    digits = "".join(ch for ch in code if ch.isdigit())
    if len(digits) >= 6:
        suffix = "BJ" if digits[:2] in {"43", "83", "87", "88", "92"} else ("SH" if digits[:1] in {"5", "6", "9"} else "SZ")
        return f"{digits[-6:]}.{suffix}"
    return code


def parse_holdings(raw):
    out = {}
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return out
    for part in str(raw).split(";"):
        if "=" not in part:
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
            out[code] = weight
    return out


def load_industry_map(path):
    path = Path(path)
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if "code" not in frame.columns or "industry" not in frame.columns:
        return {}
    frame["code_norm"] = frame["code"].map(normalize_ts_code)
    frame = frame.dropna(subset=["code_norm", "industry"])
    return dict(zip(frame["code_norm"], frame["industry"].astype(str)))


class PriceFeatureCache:
    def __init__(self, data_dirs):
        self.data_dirs = [Path(p) for p in data_dirs]
        self._cache = {}

    def load(self, code):
        code = normalize_ts_code(code)
        if code in self._cache:
            return self._cache[code]
        for data_dir in self.data_dirs:
            path = data_dir / f"{code}.csv"
            if not path.exists():
                continue
            try:
                frame = pd.read_csv(path, usecols=["trade_date", "close", "money"])
            except Exception:
                continue
            frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
            frame = frame.set_index("trade_date").sort_index()
            close = pd.to_numeric(frame["close"], errors="coerce")
            ret = close.pct_change()
            feat = pd.DataFrame(index=frame.index)
            feat["holding_momentum20"] = close.shift(1) / close.shift(21) - 1.0
            feat["holding_volatility60"] = ret.shift(1).rolling(60, min_periods=20).std() * math.sqrt(252)
            feat["holding_liquidity20"] = np.log1p(
                pd.to_numeric(frame["money"], errors="coerce").shift(1).rolling(20, min_periods=5).mean()
            )
            self._cache[code] = feat
            return feat
        self._cache[code] = None
        return None

    def features_on(self, code, date):
        frame = self.load(code)
        if frame is None or frame.empty:
            return {}
        date = pd.Timestamp(date).normalize()
        pos = frame.index.searchsorted(date, side="right") - 1
        if pos < 0:
            return {}
        row = frame.iloc[pos]
        return {k: float(v) for k, v in row.items() if np.isfinite(v)}


def holding_state(row, industry_map, price_cache):
    holdings = parse_holdings(row.get("holdings", ""))
    gross = sum(abs(w) for w in holdings.values())
    result = {
        "industry_count": 0,
        "industry_hhi": np.nan,
        "top_industry": "",
        "top_industry_abs_weight": np.nan,
        "holding_momentum20": np.nan,
        "holding_volatility60": np.nan,
        "holding_liquidity20": np.nan,
    }
    if gross <= 1e-12:
        return result

    industry_weights = {}
    feature_sums = {"holding_momentum20": 0.0, "holding_volatility60": 0.0, "holding_liquidity20": 0.0}
    feature_weights = {key: 0.0 for key in feature_sums}
    date = row["date"]
    for code, weight in holdings.items():
        aw = abs(float(weight))
        industry = industry_map.get(normalize_ts_code(code), "UNKNOWN")
        industry_weights[industry] = industry_weights.get(industry, 0.0) + float(weight)
        features = price_cache.features_on(code, date)
        for key, value in features.items():
            if key not in feature_sums:
                continue
            feature_sums[key] += aw * value
            feature_weights[key] += aw

    if industry_weights:
        top_industry, top_weight = max(industry_weights.items(), key=lambda kv: abs(kv[1]))
        result["industry_count"] = len(industry_weights)
        result["industry_hhi"] = sum((abs(w) / gross) ** 2 for w in industry_weights.values())
        result["top_industry"] = str(top_industry)
        result["top_industry_abs_weight"] = abs(float(top_weight))
    for key in feature_sums:
        if feature_weights[key] > 1e-12:
            result[key] = feature_sums[key] / feature_weights[key]
    return result


def quantile_label(series, low_q=0.3, high_q=0.7):
    s = pd.to_numeric(series, errors="coerce")
    finite = s[np.isfinite(s)]
    if len(finite) < 10 or finite.nunique() <= 2:
        return pd.Series(np.where(s >= finite.median() if len(finite) else False, "high", "low"), index=series.index)
    low = float(finite.quantile(low_q))
    high = float(finite.quantile(high_q))
    return pd.Series(
        np.where(s <= low, "low", np.where(s >= high, "high", "mid")),
        index=series.index,
    )


def pressure_label(series):
    s = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        np.where(s >= 0.065, "stress_high", np.where(s >= 0.035, "stress_mid", "stress_low")),
        index=series.index,
    )


def trailing_active_label(series):
    s = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        np.where(s <= -0.04, "active_bad", np.where(s <= -0.02, "active_soft", "active_ok")),
        index=series.index,
    )


def slice_metrics(frame, field, value):
    ret = pd.to_numeric(frame["return"], errors="coerce").fillna(0.0).to_numpy()
    active = pd.to_numeric(frame["active_return"], errors="coerce").fillna(0.0).to_numpy()
    active_std = float(np.std(active))
    ret_std = float(np.std(ret))
    return {
        "slice": field,
        "value": str(value),
        "rows": int(len(frame)),
        "avg_return_bp": float(np.mean(ret) * 10000) if len(ret) else 0.0,
        "avg_active_bp": float(np.mean(active) * 10000) if len(active) else 0.0,
        "ann_return_pct": float(np.mean(ret) * 252 * 100) if len(ret) else 0.0,
        "ann_active_pct": float(np.mean(active) * 252 * 100) if len(active) else 0.0,
        "sharpe": float(np.mean(ret) / (ret_std + 1e-12) * math.sqrt(252)) if len(ret) else 0.0,
        "information_ratio": float(np.mean(active) / (active_std + 1e-12) * math.sqrt(252)) if len(active) else 0.0,
        "win_rate": float(np.mean(ret > 0)) if len(ret) else 0.0,
        "active_win_rate": float(np.mean(active > 0)) if len(active) else 0.0,
        "avg_gross_weight": float(pd.to_numeric(frame.get("gross_weight", 0.0), errors="coerce").mean()),
        "avg_beta_per_gross": float(pd.to_numeric(frame.get("portfolio_beta_per_gross_60d", 0.0), errors="coerce").mean()),
        "avg_industry_hhi": float(pd.to_numeric(frame.get("industry_hhi", np.nan), errors="coerce").mean()),
        "avg_holding_momentum20": float(pd.to_numeric(frame.get("holding_momentum20", np.nan), errors="coerce").mean()),
        "avg_global_us_hk_pressure": float(pd.to_numeric(frame.get("global_us_hk_pressure", np.nan), errors="coerce").mean()),
        "avg_prior_active_10d": float(pd.to_numeric(frame.get("prior_active_10d", np.nan), errors="coerce").mean()),
    }


def build_one_context(base_dir, candidate, split, portfolio_value, global_features, industry_map, price_cache):
    pv_tag = "0050w" if int(portfolio_value) == 500000 else "0100w"
    returns_path = base_dir / f"returns_pv{pv_tag}_target006_hold100.csv"
    diag_path = base_dir / f"diagnostics_pv{pv_tag}_target006_hold100.csv"
    if not returns_path.exists() or not diag_path.exists():
        return pd.DataFrame()
    returns = pd.read_csv(returns_path)
    diag = pd.read_csv(diag_path)
    returns["date"] = pd.to_datetime(returns["date"]).dt.normalize()
    diag["date"] = pd.to_datetime(diag["date"]).dt.normalize()
    state_rows = []
    for _, row in diag.iterrows():
        state = holding_state(row, industry_map, price_cache)
        state["date"] = row["date"]
        state_rows.append(state)
    holding = pd.DataFrame(state_rows)
    context_cols = [
        "date",
        "gross_weight",
        "market_mult",
        "turnover",
        "executed_turnover",
        "cost",
        "desired_new_names",
        "portfolio_beta_60d",
        "portfolio_beta_per_gross_60d",
        "portfolio_specific_vol_60d",
        "avg_live_age",
    ]
    context_cols = [c for c in context_cols if c in diag.columns]
    merged = returns.merge(diag[context_cols], on="date", how="left")
    merged = merged.merge(holding, on="date", how="left")
    if global_features is not None and not global_features.empty:
        merged = merged.merge(global_features, on="date", how="left")
    merged["candidate"] = candidate
    merged["split"] = split
    merged["portfolio_value"] = int(portfolio_value)
    merged = merged.sort_values("date")
    merged["prior_active_10d"] = (
        merged["active_return"].shift(1).rolling(10, min_periods=5).apply(lambda x: np.prod(1.0 + x) - 1.0)
    )
    merged["prior_return_10d"] = (
        merged["return"].shift(1).rolling(10, min_periods=5).apply(lambda x: np.prod(1.0 + x) - 1.0)
    )
    merged["global_pressure_bucket"] = pressure_label(merged.get("global_us_hk_pressure", pd.Series(np.nan, index=merged.index)))
    merged["global_defensive_bucket"] = pressure_label(merged.get("global_defensive_pressure", pd.Series(np.nan, index=merged.index)))
    merged["hk_pressure_bucket"] = pressure_label(merged.get("global_hk_risk_pressure", pd.Series(np.nan, index=merged.index)))
    merged["prior_active_bucket"] = trailing_active_label(merged["prior_active_10d"])
    for col, name in [
        ("industry_hhi", "industry_hhi_bucket"),
        ("top_industry_abs_weight", "top_industry_weight_bucket"),
        ("holding_momentum20", "holding_momentum_bucket"),
        ("holding_volatility60", "holding_vol_bucket"),
        ("portfolio_beta_per_gross_60d", "beta_bucket"),
        ("portfolio_specific_vol_60d", "specific_vol_bucket"),
        ("executed_turnover", "turnover_bucket"),
    ]:
        if col in merged:
            merged[name] = quantile_label(merged[col])
    return merged


def summarize_slices(context):
    fields = [
        "global_pressure_bucket",
        "global_defensive_bucket",
        "hk_pressure_bucket",
        "prior_active_bucket",
        "industry_hhi_bucket",
        "top_industry_weight_bucket",
        "holding_momentum_bucket",
        "holding_vol_bucket",
        "beta_bucket",
        "specific_vol_bucket",
        "turnover_bucket",
        "top_industry",
    ]
    rows = []
    group_keys = ["candidate", "portfolio_value", "split"]
    for keys, group in context.groupby(group_keys, dropna=False):
        base = dict(zip(group_keys, keys))
        for field in fields:
            if field not in group.columns:
                continue
            for value, sub in group.groupby(field, dropna=False):
                if len(sub) < 8:
                    continue
                row = {**base, **slice_metrics(sub, field, value)}
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_interactions(context):
    pairs = [
        ("global_pressure_bucket", "prior_active_bucket"),
        ("global_pressure_bucket", "industry_hhi_bucket"),
        ("global_pressure_bucket", "holding_momentum_bucket"),
        ("prior_active_bucket", "industry_hhi_bucket"),
        ("holding_momentum_bucket", "beta_bucket"),
    ]
    rows = []
    group_keys = ["candidate", "portfolio_value", "split"]
    for keys, group in context.groupby(group_keys, dropna=False):
        base = dict(zip(group_keys, keys))
        for left, right in pairs:
            if left not in group.columns or right not in group.columns:
                continue
            for values, sub in group.groupby([left, right], dropna=False):
                if len(sub) < 6:
                    continue
                value = f"{left}={values[0]} | {right}={values[1]}"
                row = {**base, **slice_metrics(sub, f"{left} x {right}", value)}
                rows.append(row)
    return pd.DataFrame(rows)


def write_report(slice_df, interaction_df, output):
    lines = [
        "# Alpha 失效状态分层诊断",
        "",
        "口径：只读取 baseline open-ledger 的日收益和日持仓诊断；不改变策略、不使用 forward 选参数。",
        "状态变量尽量使用交易前可知信息：全球隔夜压力、过去 10 日主动收益、持仓行业集中度、持仓动量/波动、组合 beta 等。",
        "",
        "## 最差单变量状态",
        "",
    ]
    focus = slice_df[
        slice_df["slice"].isin(
            [
                "global_pressure_bucket",
                "prior_active_bucket",
                "industry_hhi_bucket",
                "holding_momentum_bucket",
                "beta_bucket",
                "top_industry",
            ]
        )
    ].copy()
    if not focus.empty:
        bad = focus.sort_values(["ann_active_pct", "rows"], ascending=[True, False]).head(30)
        lines.append(bad[
            [
                "candidate",
                "portfolio_value",
                "split",
                "slice",
                "value",
                "rows",
                "ann_return_pct",
                "ann_active_pct",
                "information_ratio",
                "avg_global_us_hk_pressure",
                "avg_prior_active_10d",
                "avg_industry_hhi",
                "avg_holding_momentum20",
            ]
        ].to_markdown(index=False, floatfmt=".3f"))
    lines.extend(["", "## 最差交互状态", ""])
    if not interaction_df.empty:
        bad2 = interaction_df.sort_values(["ann_active_pct", "rows"], ascending=[True, False]).head(30)
        lines.append(bad2[
            [
                "candidate",
                "portfolio_value",
                "split",
                "slice",
                "value",
                "rows",
                "ann_return_pct",
                "ann_active_pct",
                "information_ratio",
                "avg_global_us_hk_pressure",
                "avg_prior_active_10d",
                "avg_industry_hhi",
                "avg_holding_momentum20",
            ]
        ].to_markdown(index=False, floatfmt=".3f"))
    lines.extend(
        [
            "",
            "## 初步读法",
            "",
            "- 如果某个状态在 2024 val、2025 test、2026 forward 都稳定为负，才值得做成风控或精排特征。",
            "- 如果只在 forward 为负而历史不负，先当作观察解释，不应直接降仓。",
            "- 如果是行业/动量/高 beta 交互状态变差，优先考虑精排或组合约束；如果是全局压力状态变差，才考虑市场风险预算。",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backtest-root", default="reports/continuous_global_active_20260704/backtests")
    parser.add_argument("--output-dir", default="reports/state_alpha_failure_20260704")
    parser.add_argument("--industry-csv", default="data/stock_industry.csv")
    parser.add_argument("--data-dirs", default="data/raw,data/forward_raw")
    parser.add_argument("--global-research", default="reports/continuous_global_active_20260704/global_features_research_us_hk.parquet")
    parser.add_argument("--global-forward", default="reports/continuous_global_active_20260704/global_features_forward_us_hk.parquet")
    parser.add_argument("--candidates", default="ens4_oldoolag1_eq,multi_downside_e19")
    parser.add_argument("--splits", default="val_2024,test_2025,forward_2026")
    parser.add_argument("--portfolio-values", default="500000,1000000")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    backtest_root = Path(args.backtest_root)
    industry_map = load_industry_map(args.industry_csv)
    price_cache = PriceFeatureCache([p.strip() for p in args.data_dirs.split(",") if p.strip()])

    global_research = pd.read_parquet(args.global_research)
    global_forward = pd.read_parquet(args.global_forward)
    keep_global = [
        "date",
        "global_defensive_pressure",
        "global_hk_risk_pressure",
        "global_us_hk_pressure",
        "global_us_risk_score",
        "global_tech_risk_score",
        "global_china_cross_market_score",
        "global_hk_market_score",
        "global_hk_tech_score",
    ]
    global_research = global_research[[c for c in keep_global if c in global_research.columns]].copy()
    global_forward = global_forward[[c for c in keep_global if c in global_forward.columns]].copy()
    global_research["date"] = pd.to_datetime(global_research["date"]).dt.normalize()
    global_forward["date"] = pd.to_datetime(global_forward["date"]).dt.normalize()

    contexts = []
    for candidate in [x.strip() for x in args.candidates.split(",") if x.strip()]:
        for split in [x.strip() for x in args.splits.split(",") if x.strip()]:
            global_features = global_forward if split == "forward_2026" else global_research
            base_dir = backtest_root / candidate / split / "baseline"
            for pv in [int(float(x.strip())) for x in args.portfolio_values.split(",") if x.strip()]:
                context = build_one_context(
                    base_dir,
                    candidate,
                    split,
                    pv,
                    global_features,
                    industry_map,
                    price_cache,
                )
                if not context.empty:
                    contexts.append(context)
    if not contexts:
        raise ValueError("no baseline contexts found")
    context = pd.concat(contexts, ignore_index=True)
    slice_df = summarize_slices(context)
    interaction_df = summarize_interactions(context)

    context.to_csv(output_dir / "daily_state_context.csv", index=False, encoding="utf-8-sig")
    slice_df.to_csv(output_dir / "state_slice_summary.csv", index=False, encoding="utf-8-sig")
    interaction_df.to_csv(output_dir / "state_interaction_summary.csv", index=False, encoding="utf-8-sig")
    write_report(slice_df, interaction_df, output_dir / "state_alpha_failure_report.md")
    print(f"wrote state alpha failure diagnostics to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
