import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pathlib import Path
from backtest.risk_model import load_industry_map, compute_factor_exposures, FACTOR_COLS

def compute_factor_contrib(diagnostics_csv, returns_csv, output_dir, data_dir="data/raw", universe_size=200, n_dates=60):
    """Decompose portfolio active return into factor contributions."""
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    diag = pd.read_csv(diagnostics_csv)
    rcsv = pd.read_csv(returns_csv)
    mkt = pd.read_csv(data_dir + "/hs300_index.csv")
    mkt["date"] = pd.to_datetime(mkt["date"]); mkt = mkt.set_index("date").sort_index()
    mkret = mkt["close"].pct_change().dropna()
    ind_map, _, _ = load_industry_map("data/stock_industry.csv")

    univ = list(ind_map.keys())[:universe_size]
    cd, md = {}, {}
    for code in univ:
        p = Path(data_dir) / f"{code}.csv"
        if p.exists():
            df = pd.read_csv(p, usecols=["trade_date", "close", "money"])
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.set_index("trade_date").sort_index()
            cd[code] = df["close"].astype(float)
            md[code] = (df["money"].astype(float) * 1000.0)

    dates = pd.DatetimeIndex(sorted(set().union(*[s.index for s in cd.values()])))
    cdf = pd.DataFrame({c: s.reindex(dates) for c, s in cd.items()}, index=dates)
    mdf = pd.DataFrame({c: s.reindex(dates) for c, s in md.items()}, index=dates)
    ret_df = cdf.pct_change()
    fn4 = FACTOR_COLS[:4]

    all_fr, z_means, z_stds = [], [], []
    for i in range(min(n_dates, len(dates) - 100)):
        dt = dates[-100 + i]
        if dt not in ret_df.index: continue
        factors = compute_factor_exposures(dt, univ, None, cdf, mdf, mkret, ind_map)
        if factors.empty: continue
        cf = factors["code"].values
        rlist, xlist = [], []
        for j, c in enumerate(cf):
            if c in ret_df.columns and not pd.isna(ret_df.loc[dt, c]):
                rlist.append(ret_df.loc[dt, c])
                xlist.append(factors[fn4].fillna(0).values[j])
        if len(rlist) < 20: continue
        X = np.array(xlist)
        m = X.mean(axis=0); s = np.maximum(X.std(axis=0), 1e-10)
        X_z = (X - m) / s
        X_d = np.column_stack([np.ones(X_z.shape[0]), X_z])
        beta = np.linalg.lstsq(X_d, np.array(rlist), rcond=None)[0]
        all_fr.append(beta[1:])
        z_means.append(m); z_stds.append(s)

    if not all_fr: print("No factor returns!"); return
    avg_fr = np.mean(all_fr, axis=0)
    avg_m = np.mean(z_means, axis=0); avg_s = np.mean(z_stds, axis=0)

    contribs = {c: [] for c in fn4}; contribs["Specific"] = []; contribs["Total"] = []
    for _, row in diag.iterrows():
        dt_str = str(row["date"])[:10]; dt = pd.Timestamp(dt_str)
        h = row["holdings"]
        if not isinstance(h, str): continue
        cw = [(c.strip(), float(w)) for part in h.split(";") if "=" in part for c, w in [part.split("=")]]
        if len(cw) < 3: continue
        w = np.array([w for _, w in cw]); w = w / max(w.sum(), 1e-12)
        factors = compute_factor_exposures(dt, [c for c, _ in cw], None, cdf, mdf, mkret, ind_map)
        if factors.empty: continue
        merged = pd.DataFrame({"code": [c for c, _ in cw], "weight": w}).merge(factors, on="code", how="inner")
        if merged.empty: continue
        raw_exp = merged[fn4].fillna(0).values.T @ merged["weight"].values
        z_exp = (raw_exp - avg_m) / avg_s
        total_pred = z_exp @ avg_fr
        for i, name in enumerate(fn4): contribs[name].append(z_exp[i] * avg_fr[i])
        contribs["Total"].append(total_pred)
        ret_row = rcsv[rcsv["date"] == dt_str]
        if not ret_row.empty:
            contribs["Specific"].append(ret_row["active_return"].values[0] - total_pred)

    total_act = np.mean(contribs["Total"]) * 10000 * 252
    print(f"\nFactor contributions ({len(contribs[name])} dates):")
    for name in fn4:
        ann = np.mean(contribs[name]) * 10000 * 252
        print(f"  {name:15}: {ann:8.0f} bps/yr ({ann/max(abs(total_act),1)*100:5.1f}%)")
    if contribs["Specific"]:
        spec = np.mean(contribs["Specific"]) * 10000 * 252
        print(f"  {'Specific':15}: {spec:8.0f} bps/yr ({spec/max(abs(total_act),1)*100:5.1f}%)")
    print(f"  {'Total Active':15}: {total_act:8.0f} bps/yr")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--diagnostics-csv", required=True); p.add_argument("--returns-csv", required=True)
    p.add_argument("--output-dir", required=True); p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--universe-size", type=int, default=200); p.add_argument("--n-dates", type=int, default=60)
    args = p.parse_args()
    compute_factor_contrib(args.diagnostics_csv, args.returns_csv, args.output_dir, args.data_dir, args.universe_size, args.n_dates)
