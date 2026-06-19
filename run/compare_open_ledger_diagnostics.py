"""Compare two open-ledger single-run diagnostic directories."""

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Compare open-ledger diagnostics")
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_summary(path):
    return pd.read_csv(Path(path) / "open_ledger_summary.csv")


def read_returns(path, portfolio_value):
    pv_w = int(round(float(portfolio_value) / 10000.0))
    matches = sorted(Path(path).glob(f"returns_pv{pv_w:04d}w_target006_hold100.csv"))
    if not matches:
        matches = sorted(Path(path).glob(f"returns_*target006_hold100.csv"))
    if not matches:
        raise FileNotFoundError(f"No returns file found in {path} for pv={portfolio_value}")
    df = pd.read_csv(matches[0])
    df["date"] = pd.to_datetime(df["date"])
    df["return"] = pd.to_numeric(df["return"], errors="coerce")
    return df


def read_diag(path, portfolio_value):
    pv_w = int(round(float(portfolio_value) / 10000.0))
    matches = sorted(Path(path).glob(f"diagnostics_pv{pv_w:04d}w_target006_hold100.csv"))
    if not matches:
        matches = sorted(Path(path).glob(f"diagnostics_*target006_hold100.csv"))
    if not matches:
        raise FileNotFoundError(f"No diagnostics file found in {path} for pv={portfolio_value}")
    df = pd.read_csv(matches[0])
    if "date" in df:
        df["date"] = pd.to_datetime(df["date"])
    for col in df.columns:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_sum = read_summary(args.base_dir)
    cand_sum = read_summary(args.candidate_dir)
    rows = []
    monthly_rows = []
    daily_rows = []
    for pv in sorted(base_sum["portfolio_value"].unique()):
        b = base_sum.loc[base_sum["portfolio_value"] == pv].iloc[0]
        c = cand_sum.loc[cand_sum["portfolio_value"] == pv].iloc[0]
        row = {"portfolio_value": pv}
        for col in [
            "ann",
            "sharpe",
            "mdd",
            "avg_turnover",
            "turnover_p95",
            "blocked_buy",
            "blocked_sell",
            "total_cost",
            "avg_names",
            "avg_gross_weight",
            "avg_market_mult",
        ]:
            if col in b.index and col in c.index:
                row[f"base_{col}"] = b[col]
                row[f"candidate_{col}"] = c[col]
                row[f"diff_{col}"] = c[col] - b[col]
        rows.append(row)

        br = read_returns(args.base_dir, pv)
        cr = read_returns(args.candidate_dir, pv)
        merged = br.merge(cr, on="date", suffixes=("_base", "_candidate"))
        merged["diff"] = merged["return_candidate"] - merged["return_base"]
        merged["portfolio_value"] = pv
        daily_rows.append(merged)
        monthly = merged.assign(month=merged["date"].dt.to_period("M").astype(str)).groupby("month").agg(
            base_return=("return_base", lambda x: (1.0 + x).prod() - 1.0),
            candidate_return=("return_candidate", lambda x: (1.0 + x).prod() - 1.0),
            diff_sum=("diff", "sum"),
            diff_mean=("diff", "mean"),
            days=("diff", "size"),
        ).reset_index()
        monthly["portfolio_value"] = pv
        monthly["candidate_minus_base_return"] = monthly["candidate_return"] - monthly["base_return"]
        monthly_rows.append(monthly)

        bd = read_diag(args.base_dir, pv)
        cd = read_diag(args.candidate_dir, pv)
        diag_cols = [c for c in ["turnover", "blocked_buy", "blocked_sell", "cost", "selected_n", "gross_weight"] if c in bd and c in cd]
        if "date" in bd and "date" in cd and diag_cols:
            dm = bd[["date"] + diag_cols].merge(cd[["date"] + diag_cols], on="date", suffixes=("_base", "_candidate"))
            for col in diag_cols:
                dm[f"diff_{col}"] = dm[f"{col}_candidate"] - dm[f"{col}_base"]
            dm["portfolio_value"] = pv
            dm.to_csv(out_dir / f"daily_diag_compare_pv{int(float(pv))}.csv", index=False)

    pd.DataFrame(rows).to_csv(out_dir / "summary_compare.csv", index=False)
    pd.concat(monthly_rows, ignore_index=True).to_csv(out_dir / "monthly_compare.csv", index=False)
    pd.concat(daily_rows, ignore_index=True).to_csv(out_dir / "daily_return_compare.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
