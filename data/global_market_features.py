"""Build global overnight features aligned to A-share trading dates."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning


warnings.filterwarnings("ignore", category=PerformanceWarning)


FEATURE_SYMBOLS = {
    "GSPC": "sp500_idx",
    "IXIC": "nasdaq_comp_idx",
    "NDX": "nasdaq100_idx",
    "RUT": "russell2000_idx",
    "DJI": "dow_idx",
    "SOX": "sox_idx",
    "NYA": "nyse_idx",
    "MID": "sp400_idx",
    "W5000": "wilshire5000_idx",
    "SPY": "spy",
    "VTI": "vti",
    "QQQ": "qqq",
    "IWM": "iwm",
    "DIA": "dia",
    "MDY": "mdy",
    "IJR": "ijr",
    "IWF": "iwf",
    "IWD": "iwd",
    "IWO": "iwo",
    "IWN": "iwn",
    "MTUM": "mtum",
    "QUAL": "qual",
    "USMV": "usmv",
    "VLUE": "vlue",
    "SIZE": "size",
    "SMH": "smh",
    "SOXX": "soxx",
    "XSD": "xsd",
    "IGV": "igv",
    "FDN": "fdn",
    "SKYY": "skyy",
    "HACK": "hack",
    "BOTZ": "botz",
    "ROBO": "robo",
    "ARKW": "arkw",
    "ARKQ": "arkq",
    "XLK": "xlk",
    "XLF": "xlf",
    "XLV": "xlv",
    "XLY": "xly",
    "XLP": "xlp",
    "XLI": "xli",
    "XLE": "xle",
    "XLU": "xlu",
    "XLB": "xlb",
    "XLRE": "xlre",
    "XLC": "xlc",
    "KBE": "kbe",
    "KRE": "kre",
    "KIE": "kie",
    "IAI": "iai",
    "FINX": "finx",
    "IBB": "ibb",
    "XBI": "xbi",
    "IHE": "ihe",
    "IHI": "ihi",
    "TAN": "tan",
    "ICLN": "icln",
    "PBW": "pbw",
    "QCLN": "qcln",
    "DRIV": "driv",
    "LIT": "lit",
    "URA": "ura",
    "REMX": "remx",
    "XOP": "xop",
    "OIH": "oih",
    "XME": "xme",
    "COPX": "copx",
    "SLX": "slx",
    "GDX": "gdx",
    "ITB": "itb",
    "XHB": "xhb",
    "IYR": "iyr",
    "VNQ": "vnq",
    "IYT": "iyt",
    "JETS": "jets",
    "XRT": "xrt",
    "ITA": "ita",
    "PPA": "ppa",
    "MOO": "moo",
    "NVDA": "nvda",
    "AMD": "amd",
    "KWEB": "kweb",
    "FXI": "fxi",
    "MCHI": "mchi",
    "ASHR": "ashr",
    "CNYA": "cnya",
    "EWH": "ewh",
    "2800_HK": "hk_tracker",
    "3033_HK": "hstech_etf",
    "3067_HK": "hstech_ishares",
    "2822_HK": "a50_hk_etf",
    "3188_HK": "csi300_hk_etf",
    "3199_HK": "hk_china_etf",
    "BABA": "baba",
    "PDD": "pdd",
    "JD": "jd",
    "BIDU": "bidu",
    "NTES": "ntes",
    "0700_HK": "tencent_hk",
    "9988_HK": "alibaba_hk",
    "3690_HK": "meituan_hk",
    "9618_HK": "jd_hk",
    "9999_HK": "netease_hk",
    "1024_HK": "kuaishou_hk",
    "1810_HK": "xiaomi_hk",
    "1211_HK": "byd_hk",
    "0981_HK": "smic_hk",
    "2382_HK": "sunny_optical_hk",
    "1299_HK": "aia_hk",
    "2318_HK": "pingan_hk",
    "0939_HK": "ccb_hk",
    "3988_HK": "boc_hk",
    "0005_HK": "hsbc_hk",
    "0388_HK": "hkex_hk",
    "0386_HK": "sinopec_hk",
    "0883_HK": "cnooc_hk",
    "0857_HK": "petrochina_hk",
    "1109_HK": "cr_land_hk",
    "0688_HK": "coland_hk",
    "2007_HK": "country_garden_hk",
    "0016_HK": "shkp_hk",
    "2020_HK": "anta_hk",
    "2313_HK": "shenzhou_hk",
    "6862_HK": "haidilao_hk",
    "2269_HK": "wuxi_bio_hk",
    "6618_HK": "jd_health_hk",
    "HSI": "hsi_idx",
    "HSCE": "hsce_idx",
    "N225": "nikkei_idx",
    "KS11": "kospi_idx",
    "TWII": "taiwan_idx",
    "STI": "singapore_idx",
    "AXJO": "asx200_idx",
    "EWJ": "ewj",
    "EWY": "ewy",
    "EWT": "ewt",
    "INDA": "inda",
    "EWS": "ews",
    "EWA": "ewa",
    "EEM": "eem",
    "EFA": "efa",
    "FTSE": "ftse_idx",
    "GDAXI": "dax_idx",
    "FCHI": "cac40_idx",
    "STOXX50E": "stoxx50_idx",
    "VIX": "vix",
    "VVIX": "vvix",
    "MOVE": "move",
    "TNX": "tnx",
    "TYX": "tyx",
    "FVX": "fvx",
    "IRX": "irx",
    "TLT": "tlt",
    "IEF": "ief",
    "SHY": "shy",
    "HYG": "hyg",
    "LQD": "lqd",
    "EMB": "emb",
    "DX_Y_NYB": "dxy",
    "CNH_X": "usdcnh",
    "CNY_X": "usdcny",
    "HG_F": "copper",
    "CL_F": "oil",
    "BZ_F": "brent",
    "NG_F": "natural_gas",
    "GC_F": "gold",
    "SI_F": "silver",
    "PL_F": "platinum",
    "PA_F": "palladium",
    "ZC_F": "corn",
    "ZS_F": "soybean",
    "ZW_F": "wheat",
    "GLD": "gld",
    "SLV": "slv",
    "USO": "uso",
    "UNG": "ung",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-root", default="data/global")
    parser.add_argument("--a-index", default="data/raw/hs300_index.csv")
    parser.add_argument("--output", default="data/global/global_overnight_features.parquet")
    parser.add_argument("--csv-output", default="data/global/global_overnight_features.csv")
    parser.add_argument("--max-stale-days", type=int, default=7)
    return parser.parse_args()


def safe_symbol(symbol: str) -> str:
    return (
        symbol.replace("^", "")
        .replace("=", "_")
        .replace(".", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def read_a_dates(path: Path) -> pd.DatetimeIndex:
    frame = pd.read_csv(path)
    date_col = "date" if "date" in frame.columns else "trade_date"
    dates = pd.to_datetime(frame[date_col]).dropna().sort_values().drop_duplicates()
    return pd.DatetimeIndex(dates).normalize()


def read_symbol(global_root: Path, safe: str) -> pd.DataFrame:
    path = global_root / "raw" / "us_ohlcv" / f"{safe}.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    for column in ["open", "high", "low", "close", "adj_close", "volume"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.set_index("date")


def add_symbol_features(result: pd.DataFrame, global_root: Path, symbol: str, prefix: str, max_stale_days: int) -> dict:
    raw = read_symbol(global_root, safe_symbol(symbol))
    if raw.empty:
        result[f"global_{prefix}_missing"] = 1.0
        return {"symbol": symbol, "status": "missing"}
    close = raw["adj_close"].where(raw["adj_close"].notna(), raw["close"]).astype(float)
    daily = close.pct_change()
    features = pd.DataFrame(index=raw.index)
    features[f"global_{prefix}_ret_1d"] = daily
    features[f"global_{prefix}_ret_3d"] = close.pct_change(3)
    features[f"global_{prefix}_ret_5d"] = close.pct_change(5)
    features[f"global_{prefix}_vol_20d"] = daily.rolling(20, min_periods=10).std()
    features[f"global_{prefix}_ma20_gap"] = close / close.rolling(20, min_periods=10).mean() - 1.0
    if "volume" in raw.columns and raw["volume"].abs().sum() > 0:
        volume = raw["volume"].astype(float)
        features[f"global_{prefix}_volume_z20"] = (
            (volume - volume.rolling(20, min_periods=10).mean())
            / volume.rolling(20, min_periods=10).std().replace(0, np.nan)
        )
    aligned = features.reindex(result["us_session_date"])
    aligned.index = result.index
    for column in aligned.columns:
        result[column] = aligned[column].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    session_dates = pd.Series(result["us_session_date"].to_numpy(), index=result.index)
    valid_sessions = pd.Series(features.index, index=features.index)
    last_valid = valid_sessions.reindex(result["us_session_date"], method="ffill")
    last_valid.index = result.index
    stale_days = (session_dates - last_valid).dt.days
    result[f"global_{prefix}_stale_days"] = stale_days.fillna(999).clip(lower=0)
    result[f"global_{prefix}_missing"] = (
        result[f"global_{prefix}_stale_days"] > int(max_stale_days)
    ).astype(float)
    return {
        "symbol": symbol,
        "status": "ok",
        "rows": int(len(raw)),
        "date_start": str(raw.index.min().date()),
        "date_end": str(raw.index.max().date()),
    }


def build_session_map(a_dates: pd.DatetimeIndex, global_root: Path) -> pd.DataFrame:
    spy = read_symbol(global_root, "SPY")
    if spy.empty:
        raise ValueError("SPY data is required to infer US sessions")
    us_sessions = spy.index.sort_values()
    # For an A-share trading day D, use the latest completed US session whose
    # calendar date is strictly before D. This matches China morning availability.
    session_pos = np.searchsorted(us_sessions.to_numpy(), a_dates.to_numpy(), side="left") - 1
    valid = session_pos >= 0
    clipped = np.clip(session_pos, 0, len(us_sessions) - 1)
    mapped_values = us_sessions[clipped].to_numpy(dtype="datetime64[ns]")
    mapped_values[~valid] = np.datetime64("NaT")
    mapped = pd.DatetimeIndex(mapped_values)
    result = pd.DataFrame(index=a_dates)
    result.index.name = "date"
    result["us_session_date"] = mapped
    result["global_us_stale_days"] = (
        result.index.to_series() - result["us_session_date"]
    ).dt.days.fillna(999).clip(lower=0)
    return result


def load_safe_symbol_map(global_root: Path) -> dict[str, str]:
    path = global_root / "global_symbol_manifest.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if "safe_symbol" not in frame.columns or "symbol" not in frame.columns:
        return {}
    return dict(zip(frame["safe_symbol"].astype(str), frame["symbol"].astype(str)))


def add_composite_scores(result: pd.DataFrame):
    def col(name):
        return result[name] if name in result.columns else 0.0

    def available_mean(prefixes):
        values = []
        for prefix in prefixes:
            ret_col = f"global_{prefix}_ret_1d"
            missing_col = f"global_{prefix}_missing"
            if ret_col not in result.columns:
                continue
            series = result[ret_col].astype(float)
            if missing_col in result.columns:
                series = series.where(result[missing_col].astype(float) < 0.5)
            values.append(series)
        if not values:
            return pd.Series(0.0, index=result.index)
        frame = pd.concat(values, axis=1)
        return frame.mean(axis=1).fillna(0.0)

    sector_prefixes = ("xlk", "xlf", "xlv", "xly", "xlp", "xli", "xle", "xlu", "xlb", "xlre", "xlc")
    sector_ret_cols = [
        f"global_{prefix}_ret_1d"
        for prefix in sector_prefixes
        if f"global_{prefix}_ret_1d" in result.columns
    ]
    if sector_ret_cols:
        sector_returns = result[sector_ret_cols].astype(float).copy()
        for prefix in sector_prefixes:
            ret_col = f"global_{prefix}_ret_1d"
            missing_col = f"global_{prefix}_missing"
            if ret_col in sector_returns and missing_col in result.columns:
                sector_returns[ret_col] = sector_returns[ret_col].where(
                    result[missing_col].astype(float) < 0.5
                )
        available_counts = sector_returns.notna().sum(axis=1)
        result["global_us_sector_available_count"] = available_counts.astype(float)
        result["global_us_sector_breadth"] = (
            (sector_returns > 0).sum(axis=1) / available_counts.replace(0, np.nan)
        ).fillna(0.5)
        result["global_us_sector_avg_ret_1d"] = sector_returns.mean(axis=1).fillna(0.0)
    else:
        result["global_us_sector_available_count"] = 0.0
        result["global_us_sector_breadth"] = 0.5
        result["global_us_sector_avg_ret_1d"] = 0.0

    cyclical = available_mean(("xlf", "xly", "xli", "xle", "xlb"))
    defensive = available_mean(("xlv", "xlp", "xlu"))
    result["global_us_cyclical_vs_defensive"] = cyclical - defensive

    result["global_us_risk_score"] = (
        0.35 * col("global_spy_ret_1d")
        + 0.35 * col("global_qqq_ret_1d")
        + 0.15 * col("global_iwm_ret_1d")
        - 0.15 * col("global_vix_ret_1d")
    )
    result["global_tech_risk_score"] = (
        0.30 * col("global_qqq_ret_1d")
        + 0.30 * col("global_smh_ret_1d")
        + 0.20 * col("global_soxx_ret_1d")
        + 0.10 * col("global_nvda_ret_1d")
        + 0.10 * col("global_amd_ret_1d")
    )
    result["global_china_adr_score"] = (
        0.35 * col("global_kweb_ret_1d")
        + 0.25 * col("global_fxi_ret_1d")
        + 0.20 * col("global_mchi_ret_1d")
        + 0.10 * col("global_baba_ret_1d")
        + 0.10 * col("global_pdd_ret_1d")
    )
    result["global_hk_market_score"] = available_mean(
        ("hsi_idx", "hsce_idx", "hk_tracker", "ewh", "hk_china_etf")
    )
    result["global_hk_tech_score"] = available_mean(
        (
            "hstech_etf",
            "hstech_ishares",
            "tencent_hk",
            "alibaba_hk",
            "meituan_hk",
            "jd_hk",
            "netease_hk",
            "kuaishou_hk",
            "xiaomi_hk",
        )
    )
    result["global_hk_financial_score"] = available_mean(
        ("aia_hk", "pingan_hk", "ccb_hk", "boc_hk", "hsbc_hk", "hkex_hk")
    )
    result["global_hk_property_score"] = available_mean(
        ("cr_land_hk", "coland_hk", "country_garden_hk", "shkp_hk")
    )
    result["global_hk_ev_hardware_score"] = available_mean(
        ("byd_hk", "smic_hk", "sunny_optical_hk", "xiaomi_hk")
    )
    result["global_hk_energy_score"] = available_mean(
        ("sinopec_hk", "cnooc_hk", "petrochina_hk")
    )
    result["global_hk_consumer_health_score"] = available_mean(
        (
            "anta_hk",
            "shenzhou_hk",
            "haidilao_hk",
            "wuxi_bio_hk",
            "jd_health_hk",
        )
    )
    result["global_china_cross_market_score"] = (
        0.35 * result["global_china_adr_score"]
        + 0.30 * result["global_hk_market_score"]
        + 0.25 * result["global_hk_tech_score"]
        + 0.10 * available_mean(("a50_hk_etf", "csi300_hk_etf", "ashr", "cnya"))
    )
    result["global_hk_risk_pressure"] = -result["global_china_cross_market_score"]
    result["global_defensive_pressure"] = (
        -0.40 * result["global_us_risk_score"]
        -0.35 * result["global_tech_risk_score"]
        -0.25 * result["global_china_cross_market_score"]
        + 0.20 * col("global_vix_ret_1d")
        + 0.10 * (0.5 - result["global_us_sector_breadth"])
        - 0.10 * result["global_us_cyclical_vs_defensive"]
    )
    return result


def main():
    args = parse_args()
    global_root = Path(args.global_root)
    a_dates = read_a_dates(Path(args.a_index))
    result = build_session_map(a_dates, global_root)
    symbol_status = []
    safe_to_symbol = load_safe_symbol_map(global_root)
    for safe, prefix in FEATURE_SYMBOLS.items():
        original = safe_to_symbol.get(safe, safe)
        symbol_status.append(
            add_symbol_features(result, global_root, original, prefix, args.max_stale_days)
        )
    result = add_composite_scores(result)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result_reset = result.reset_index()
    result_reset.to_parquet(output, index=False)
    result_reset.to_csv(args.csv_output, index=False, encoding="utf-8")
    summary = {
        "rows": int(len(result_reset)),
        "date_start": str(result.index.min().date()),
        "date_end": str(result.index.max().date()),
        "columns": list(result_reset.columns),
        "symbols": symbol_status,
        "alignment_rule": "A-share date D uses latest US session date strictly before D.",
    }
    (output.parent / "global_overnight_features_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({k: summary[k] for k in ["rows", "date_start", "date_end", "alignment_rule"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
