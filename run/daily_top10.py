#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""每日收盘后：更新数据 → 删缓存 → 跑 top_union_bottom_intersection 推荐 Top10 → 写入日志文件"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

PYTHON = sys.executable
LOG_FILE = PROJECT_ROOT / "recommendations" / "daily_top10_log.txt"
ERROR_LOG = PROJECT_ROOT / "errors.log"


def log_error(script, exc):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {script} | {type(exc).__name__}: {exc}\n")


def run(cmd, timeout=600):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 执行: {cmd[:80]}...")
    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT_ROOT)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                            timeout=timeout, env=env, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        safe_stderr = result.stderr[-200:].encode('ascii', errors='replace').decode('ascii')
        print(f"  警告: exit={result.returncode}, stderr={safe_stderr}")
    return result.stdout

def main():
    today = datetime.now().strftime("%Y-%m-%d")
    weekday = datetime.now().weekday()
    if weekday >= 5:
        print(f"{today} 非交易日（周末），跳过")
        return

    print(f"\n{'='*60}")
    print(f"  每日 Top10 推荐 — {today}")
    print(f"{'='*60}")

    # 1. 更新日线数据
    print("\n[1/3] 更新日线数据...")
    run(f"{PYTHON} data/update_daily.py --data-dir data/tracking_raw --api-batch 80 --api-sleep 1", timeout=600)

    # 2. 删除推理缓存
    print("\n[2/3] 清理推理缓存...")
    cache = PROJECT_ROOT / "cache" / "inference_matrices_cache.pkl"
    if cache.exists():
        cache.unlink()

    # 3. 跑推荐（输出CSV + 打印表）
    print("\n[3/3] 运行 Top10 推荐...")
    csv_path = PROJECT_ROOT / "recommendations" / f"daily_{today}.csv"
    output = run(
        f"{PYTHON} run/recommend_daily.py --as-of latest --predictor top_union_bottom_intersection --top-n 10 --output {csv_path} --data-dir data/tracking_raw",
        timeout=1200,
    )

    # 解析输出中的表格部分
    lines = output.split("\n")
    in_table = False
    table_lines = []
    header_line = " 排名      代码       名称      板块     Alpha      百分位"
    for line in lines:
        if "rank" in line.lower() and "code" in line.lower() and "alpha" in line.lower():
            table_lines.append(header_line)
            in_table = True
            continue
        if in_table:
            if line.strip() and (line.strip()[0].isdigit() or "主板" in line or "创业板" in line or "科创板" in line or "北交所" in line):
                table_lines.append(line)
            elif table_lines and not line.strip().startswith((" ", "\t", "1", "2", "3", "4", "5", "6", "7", "8", "9")):
                break

    # 数据校验
    try:
        from data.validate import validate_daily, position_stability
        import pandas as pd
        df = pd.read_csv(csv_path)
        validate_daily(PROJECT_ROOT / "data" / "raw", df)

        prev_files = sorted((PROJECT_ROOT / "recommendations").glob("daily_2*.csv"))
        if len(prev_files) >= 2:
            _, _, warnings_list = position_stability(df["code"].tolist(), prev_files[-2])
            for w in warnings_list:
                print(w)
    except Exception as e:
        log_error("daily_top10.validate", e)

    # 尝试从CSV构建更可靠的表
    import pandas as pd
    try:
        df = pd.read_csv(csv_path)
        table = build_table(df)
    except Exception:
        table = "\n".join(table_lines) if table_lines else "（无法解析输出）"
        table = f"日期: {today}\n{table}"

    # 写入日志文件
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(table)
        f.write("\n\n")

    print(f"\n已追加到日志: {LOG_FILE}")
    print(table)

    # 与上一个交易日对比，标记新进/持续
    prev_files = sorted((PROJECT_ROOT / "recommendations").glob("daily_2*.csv"))
    if len(prev_files) >= 2:
        try:
            prev_df = pd.read_csv(prev_files[-2])
            prev_codes = set(prev_df["code"].tolist())
            today_codes = set(df["code"].tolist())
            new_entries = today_codes - prev_codes
            persistent = today_codes & prev_codes
            dropped = prev_codes - today_codes
            if persistent:
                print(f"\n持续推荐 ({len(persistent)} 只): {', '.join(sorted(persistent))}")
            if new_entries:
                print(f"新进推荐 ({len(new_entries)} 只): {', '.join(sorted(new_entries))}")
            if dropped:
                print(f"退出推荐 ({len(dropped)} 只): {', '.join(sorted(dropped))}")
        except Exception:
            pass


SEP = "├──────┼───────────┼──────────────┼────────┼──────────┼──────────┤"


def build_table(df):
    """构建格式化表格"""
    today = df["date"].iloc[0] if "date" in df.columns else datetime.now().strftime("%Y-%m-%d")

    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"  Top10 每日推荐 | 日期: {today} | 策略: top_union_bottom_intersection")
    lines.append("=" * 80)

    if "board" in df.columns:
        board_counts = df["board"].value_counts()
        board_str = " | ".join(f"{b}: {c}" for b, c in board_counts.items())
        lines.append(f"  板块分布: {board_str}")

    lines.append("┌──────┬───────────┬──────────────┬────────┬──────────┬──────────┐")
    lines.append("│ 排名 │    代码   │     名称     │  板块  │  Alpha   │  百分位  │")
    lines.append(SEP)

    rows = []
    for _, row in df.iterrows():
        rank = int(row.get("rank", 0))
        code = str(row.get("code", ""))
        name = str(row.get("name", ""))
        board = str(row.get("board", ""))
        alpha = float(row.get("alpha", 0))
        pct = float(row.get("percentile", 0)) * 100
        rows.append(
            f"│ {rank:>4} │ {code:<9} │ {name:<12} │ {board:<6} │ {alpha:>7.4f} │ {pct:>7.2f}% │"
        )

    lines.append(("\n" + SEP + "\n").join(rows))
    lines.append("└──────┴───────────┴──────────────┴────────┴──────────┴──────────┘")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
