"""Post-process alpha JSONL to tilt toward defensive stocks in weak markets."""
import argparse, json, numpy as np, pandas as pd
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--tilt", type=float, default=0.3, help="Defense tilt in weak markets (0-1)")
    p.add_argument("--hs300-csv", default="data/raw/hs300_index.csv")
    args = p.parse_args()

    # Load alpha signal
    with open(args.input) as f:
        rows = [json.loads(line) for line in f]
    print(f"Loaded {len(rows)} dates")
    codes_set = set(); [codes_set.update(r["codes"]) for r in rows]
    codes = sorted(codes_set)
    print(f"Universe: {len(codes)} stocks")

    # Load HS300 for market state
    hs = pd.read_csv(args.hs300_csv)
    hs["date"] = pd.to_datetime(hs["date"])
    hs = hs.set_index("date").sort_index()
    hs_ret = hs["close"].pct_change()

    # Load close data for stocks
    data_dir = Path(args.data_dir)
    cd = {}
    for code in codes:
        p = data_dir / f"{code}.csv"
        if p.exists():
            df = pd.read_csv(p, usecols=["trade_date", "close"])
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            cd[code] = df.set_index("trade_date")["close"].astype(float)
    print(f"Loaded closes: {len(cd)} stocks")

    dates_all = pd.DatetimeIndex(sorted(set().union(*[s.index for s in cd.values()])))
    cdf = pd.DataFrame({c: s.reindex(dates_all) for c, s in cd.items()}, index=dates_all)
    cdf.index = pd.to_datetime(cdf.index)

    # For each date, detect weak market and adjust alpha
    output = []
    defense_dates = 0
    for r in rows:
        dt_str = r["date"][:10]; dt = pd.Timestamp(dt_str)
        codes_r = r["codes"]; alpha = np.array(r["alpha"], dtype=float)

        # Detect weak market: 5d return < -3% OR 20d return < -5%
        hs_5d = hs_ret.loc[dt] if dt in hs_ret.index else 0
        hs_20d = hs["close"].pct_change(20).loc[dt] if dt in hs["close"].pct_change(20).index else 0
        is_weak = (hs_5d < -0.03) or (hs_20d < -0.05)

        if is_weak:
            defense_dates += 1
            # Compute defense score for each stock
            def_scores = []
            for c in codes_r:
                if c not in cdf.columns: def_scores.append(0.5); continue
                close_s = cdf[c].dropna()
                close_before = close_s[close_s.index <= dt]
                if len(close_before) < 30:
                    def_scores.append(0.5); continue
                # Vol: 60d rolling std
                ret = close_before.pct_change().dropna()
                vol60 = ret.tail(60).std() if len(ret) >= 60 else ret.tail(min(len(ret),20)).std()*2
                # Beta: simplified regression vs HS300
                hs_local = hs_ret.reindex(ret.tail(60).index).dropna()
                aligned_ret = ret.tail(60).reindex(hs_local.index).dropna()
                hs_aligned = hs_local.reindex(aligned_ret.index).dropna()
                if len(aligned_ret) >= 20:
                    beta = np.nanmean((aligned_ret.values - aligned_ret.mean()) * (hs_aligned.values - hs_aligned.mean())) / max(np.nanvar(hs_aligned.values), 1e-12)
                else:
                    beta = 1.0
                # Score: low vol + low beta = high defense
                vol_s = max(0, min(1, 1 - vol60 / 0.05))
                beta_s = max(0, min(1, 1 - (beta - 1) / 1.5))
                def_scores.append(0.5 * vol_s + 0.5 * beta_s)
            def_scores = np.array(def_scores)
            # Apply tilt: alpha = (1-tilt)*alpha + tilt*defense
            alpha = (1 - args.tilt) * alpha + args.tilt * def_scores

        out = {"date": r["date"], "codes": codes_r, "alpha": alpha.tolist(), "n_stocks": len(codes_r)}
        output.append(out)

    with open(args.output, "w") as f:
        for out in output:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"Written {len(output)} dates to {args.output}")
    print(f"Defensive dates: {defense_dates}/{len(rows)} (tilt={args.tilt})")

if __name__ == "__main__":
    main()
