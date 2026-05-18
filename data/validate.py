# -*- coding: utf-8 -*-
"""Daily data validation utilities."""
from datetime import datetime
from pathlib import Path

import pandas as pd


def date_coverage(data_dir, max_stale=500):
    """Check how many stock CSVs are not up-to-date. Returns (ok, msg)."""
    data_dir = Path(data_dir)
    today = datetime.today()
    stale = 0
    for csv_path in data_dir.glob("*.csv"):
        if not csv_path.stem[0].isdigit():
            continue
        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True, nrows=2)
            if df.empty:
                stale += 1
                continue
            last = df.index.max().to_pydatetime()
            if (today - last).days > 2:
                stale += 1
        except Exception:
            stale += 1
    ok = stale <= max_stale
    msg = f"日期覆盖: {stale} 只股票过期 (>2天), 阈值 {max_stale}"
    return ok, msg


def alpha_quality(df, max_bad_pct=0.05):
    """Check recommendation alpha for anomalies. Returns (ok, msg)."""
    if df is None or df.empty:
        return False, "alpha 质量: 无推荐数据"
    alpha = pd.to_numeric(df.get("alpha", []), errors="coerce")
    bad = alpha.isna().sum() + (alpha.abs() < 1e-12).sum()
    pct = bad / max(len(alpha), 1)
    ok = pct <= max_bad_pct
    msg = f"alpha 质量: {bad}/{len(alpha)} ({pct*100:.1f}%) 无效, 阈值 {max_bad_pct*100:.0f}%"
    return ok, msg


def position_stability(today_codes, prev_csv):
    """Check Jaccard similarity with previous day's recommendation. Returns (similarity, msg, warnings)."""
    warnings_list = []
    try:
        prev_df = pd.read_csv(prev_csv)
        prev_codes = set(prev_df["code"].tolist())
        today_set = set(today_codes)
        intersection = today_set & prev_codes
        union = today_set | prev_codes
        jaccard = len(intersection) / len(union) if union else 0
        msg = f"持仓相似度 (Jaccard): {jaccard*100:.1f}%"
        if jaccard < 0.3:
            warnings_list.append(f"⚠ 持仓大幅偏离预警: Jaccard={jaccard*100:.1f}%")
        if jaccard < 0.1:
            warnings_list.append("⚠ 严重偏离: 推荐结果与上一交易日几乎完全不同")
        return jaccard, msg, warnings_list
    except Exception:
        return 0, "持仓相似度: 无法读取前一日CSV", []


def validate_daily(data_dir, df):
    """Run all daily validations. Prints results, returns True if all pass."""
    all_ok = True
    print("\n========== 数据校验 ==========")
    ok, msg = date_coverage(data_dir)
    all_ok = all_ok and ok
    flag = "" if ok else " ⚠"
    print(f"  [{flag}] {msg}")

    ok, msg = alpha_quality(df)
    all_ok = all_ok and ok
    flag = "" if ok else " ⚠"
    print(f"  [{flag}] {msg}")

    return all_ok
