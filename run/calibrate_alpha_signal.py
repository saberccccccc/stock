"""Calibrate raw alpha scores to expected active return in bps/day.
Usage: python calibrate_alpha_signal.py raw.jsonl calibrated.jsonl --ic 0.05
"""
import argparse, json, numpy as np, pandas as pd
from pathlib import Path

def rolling_vol(close_series, window=60):
    ret = close_series.pct_change().dropna()
    return ret.rolling(window, min_periods=10).std()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="Raw alpha JSONL (date, codes, alpha)")
    p.add_argument("output", help="Output path")
    p.add_argument("--ic", type=float, default=0.05)
    p.add_argument("--vol-window", type=int, default=60)
    p.add_argument("--data-dir", default="data/raw")
    args = p.parse_args()

    print(f"Reading {args.input}...")
    dates_list = []
    all_codes = set()
    alpha_by_date = {}
    with open(args.input) as f:
        for line in f:
            row = json.loads(line)
            dt = row["date"][:10]
            codes = row["codes"]
            alpha = np.array(row["alpha"], dtype=float)
            alpha_by_date[dt] = {"codes": codes, "alpha": alpha}
            all_codes.update(codes)
            dates_list.append(dt)

    print(f"  Dates: {len(dates_list)}, Universe: {len(all_codes)} stocks")
    print(f"  Loading close data for {len(all_codes)} stocks...")

    # Load close prices
    data_dir = Path(args.data_dir)
    cd = {}
    loaded = 0
    for code in sorted(all_codes):
        path = data_dir / f"{code}.csv"
        if path.exists():
            try:
                df = pd.read_csv(path, usecols=["trade_date", "close"])
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                cd[code] = df.set_index("trade_date")["close"].astype(float)
                loaded += 1
            except:
                pass

    print(f"  Loaded {loaded}/{len(all_codes)} stocks")
    if not cd:
        print("ERROR: no stock data loaded")
        return

    # Rolling vol for all stocks
    print("  Computing rolling volatility...")
    dates = pd.DatetimeIndex(sorted(set().union(*[s.index for s in cd.values()])))
    cdf = pd.DataFrame({c: s.reindex(dates) for c, s in cd.items()}, index=dates)
    vol_df = cdf.apply(lambda col: rolling_vol(col, args.vol_window))

    # Calibrate each date
    print("  Calibrating alpha...")
    output_rows = []
    for dt_str in dates_list:
        dt = pd.Timestamp(dt_str)
        if dt not in vol_df.index:
            continue
        entry = alpha_by_date[dt_str]
        codes = entry["codes"]
        raw_alpha = entry["alpha"]

        # Get specific_vol for each stock
        vol_series = vol_df.loc[dt]
        spec_vols = np.array([vol_series.get(c, np.nan) for c in codes])

        if np.isnan(spec_vols).all():
            continue

        # Z-score raw alpha
        mu = np.nanmean(raw_alpha)
        sg = max(np.nanstd(raw_alpha), 1e-10)
        z = (raw_alpha - mu) / sg

        # Calibrate
        med_vol = np.nanmedian(spec_vols) if np.isfinite(np.nanmedian(spec_vols)) else 0.02
        fill_vol = np.where(np.isnan(spec_vols), med_vol, spec_vols)
        calibrated_bps = args.ic * z * fill_vol * 10000

        row = {
            "date": dt_str,
            "codes": codes,
            "alpha": raw_alpha.tolist(),
            "calibrated_bps": calibrated_bps.tolist(),
            "n_stocks": len(codes),
        }
        output_rows.append(row)

    with open(args.output, "w") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    all_bps = [b for row in output_rows for b in row["calibrated_bps"]]
    print(f"  Written {len(output_rows)} days -> {args.output}")
    print(f"  Calibrated BPS range: [{min(r['calibrated_bps']):.2f}, {max(r['calibrated_bps']):.2f}]")

if __name__ == "__main__":
    main()
