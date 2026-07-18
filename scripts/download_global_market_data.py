"""Download global overnight market data for A-share risk overlays.

The data is cached as one CSV per Yahoo Finance symbol.  This script is
deliberately independent from model training: it only downloads and normalizes
OHLCV data so later feature builders can align the latest completed US session
to the next A-share trading day.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


DEFAULT_SYMBOLS = {
    "^GSPC": {"name": "S&P 500 Index", "group": "us_index"},
    "^IXIC": {"name": "Nasdaq Composite Index", "group": "us_index"},
    "^NDX": {"name": "Nasdaq 100 Index", "group": "us_index"},
    "^RUT": {"name": "Russell 2000 Index", "group": "us_index"},
    "^DJI": {"name": "Dow Jones Industrial Average", "group": "us_index"},
    "^SOX": {"name": "PHLX Semiconductor Index", "group": "us_index"},
    "^NYA": {"name": "NYSE Composite Index", "group": "us_index"},
    "^MID": {"name": "S&P MidCap 400 Index", "group": "us_index"},
    "^W5000": {"name": "Wilshire 5000 Total Market Index", "group": "us_index"},
    "SPY": {"name": "SPDR S&P 500 ETF", "group": "us_broad"},
    "VTI": {"name": "Vanguard Total Stock Market ETF", "group": "us_broad"},
    "QQQ": {"name": "Invesco QQQ Trust", "group": "us_tech"},
    "IWM": {"name": "iShares Russell 2000 ETF", "group": "us_small"},
    "DIA": {"name": "SPDR Dow Jones Industrial Average ETF", "group": "us_broad"},
    "MDY": {"name": "SPDR S&P MidCap 400 ETF", "group": "us_mid"},
    "IJR": {"name": "iShares Core S&P Small-Cap ETF", "group": "us_small"},
    "IWF": {"name": "iShares Russell 1000 Growth ETF", "group": "us_style"},
    "IWD": {"name": "iShares Russell 1000 Value ETF", "group": "us_style"},
    "IWO": {"name": "iShares Russell 2000 Growth ETF", "group": "us_style"},
    "IWN": {"name": "iShares Russell 2000 Value ETF", "group": "us_style"},
    "MTUM": {"name": "iShares MSCI USA Momentum Factor ETF", "group": "us_factor"},
    "QUAL": {"name": "iShares MSCI USA Quality Factor ETF", "group": "us_factor"},
    "USMV": {"name": "iShares MSCI USA Min Vol Factor ETF", "group": "us_factor"},
    "VLUE": {"name": "iShares MSCI USA Value Factor ETF", "group": "us_factor"},
    "SIZE": {"name": "iShares MSCI USA Size Factor ETF", "group": "us_factor"},
    "SMH": {"name": "VanEck Semiconductor ETF", "group": "semiconductor"},
    "SOXX": {"name": "iShares Semiconductor ETF", "group": "semiconductor"},
    "XSD": {"name": "SPDR S&P Semiconductor ETF", "group": "us_industry"},
    "IGV": {"name": "iShares Expanded Tech-Software ETF", "group": "us_industry"},
    "FDN": {"name": "First Trust Dow Jones Internet Index Fund", "group": "us_industry"},
    "SKYY": {"name": "First Trust Cloud Computing ETF", "group": "us_industry"},
    "HACK": {"name": "ETFMG Prime Cyber Security ETF", "group": "us_industry"},
    "BOTZ": {"name": "Global X Robotics & Artificial Intelligence ETF", "group": "us_industry"},
    "ROBO": {"name": "ROBO Global Robotics and Automation ETF", "group": "us_industry"},
    "ARKW": {"name": "ARK Next Generation Internet ETF", "group": "us_industry"},
    "ARKQ": {"name": "ARK Autonomous Technology & Robotics ETF", "group": "us_industry"},
    "XLK": {"name": "Technology Select Sector SPDR Fund", "group": "us_tech"},
    "XLF": {"name": "Financial Select Sector SPDR Fund", "group": "us_sector"},
    "XLV": {"name": "Health Care Select Sector SPDR Fund", "group": "us_sector"},
    "XLY": {"name": "Consumer Discretionary Select Sector SPDR Fund", "group": "us_sector"},
    "XLP": {"name": "Consumer Staples Select Sector SPDR Fund", "group": "us_sector"},
    "XLI": {"name": "Industrial Select Sector SPDR Fund", "group": "us_sector"},
    "XLE": {"name": "Energy Select Sector SPDR Fund", "group": "us_sector"},
    "XLU": {"name": "Utilities Select Sector SPDR Fund", "group": "us_sector"},
    "XLB": {"name": "Materials Select Sector SPDR Fund", "group": "us_sector"},
    "XLRE": {"name": "Real Estate Select Sector SPDR Fund", "group": "us_sector"},
    "XLC": {"name": "Communication Services Select Sector SPDR Fund", "group": "us_sector"},
    "KBE": {"name": "SPDR S&P Bank ETF", "group": "us_industry"},
    "KRE": {"name": "SPDR S&P Regional Banking ETF", "group": "us_industry"},
    "KIE": {"name": "SPDR S&P Insurance ETF", "group": "us_industry"},
    "IAI": {"name": "iShares U.S. Broker-Dealers & Securities Exchanges ETF", "group": "us_industry"},
    "FINX": {"name": "Global X FinTech ETF", "group": "us_industry"},
    "IBB": {"name": "iShares Biotechnology ETF", "group": "us_industry"},
    "XBI": {"name": "SPDR S&P Biotech ETF", "group": "us_industry"},
    "IHE": {"name": "iShares U.S. Pharmaceuticals ETF", "group": "us_industry"},
    "IHI": {"name": "iShares U.S. Medical Devices ETF", "group": "us_industry"},
    "TAN": {"name": "Invesco Solar ETF", "group": "us_industry"},
    "ICLN": {"name": "iShares Global Clean Energy ETF", "group": "us_industry"},
    "PBW": {"name": "Invesco WilderHill Clean Energy ETF", "group": "us_industry"},
    "QCLN": {"name": "First Trust NASDAQ Clean Edge Green Energy Index Fund", "group": "us_industry"},
    "DRIV": {"name": "Global X Autonomous & Electric Vehicles ETF", "group": "us_industry"},
    "LIT": {"name": "Global X Lithium & Battery Tech ETF", "group": "us_industry"},
    "URA": {"name": "Global X Uranium ETF", "group": "us_industry"},
    "REMX": {"name": "VanEck Rare Earth/Strategic Metals ETF", "group": "us_industry"},
    "XOP": {"name": "SPDR S&P Oil & Gas Exploration & Production ETF", "group": "us_industry"},
    "OIH": {"name": "VanEck Oil Services ETF", "group": "us_industry"},
    "XME": {"name": "SPDR S&P Metals & Mining ETF", "group": "us_industry"},
    "COPX": {"name": "Global X Copper Miners ETF", "group": "us_industry"},
    "SLX": {"name": "VanEck Steel ETF", "group": "us_industry"},
    "GDX": {"name": "VanEck Gold Miners ETF", "group": "us_industry"},
    "ITB": {"name": "iShares U.S. Home Construction ETF", "group": "us_industry"},
    "XHB": {"name": "SPDR S&P Homebuilders ETF", "group": "us_industry"},
    "IYR": {"name": "iShares U.S. Real Estate ETF", "group": "us_industry"},
    "VNQ": {"name": "Vanguard Real Estate ETF", "group": "us_industry"},
    "IYT": {"name": "iShares U.S. Transportation ETF", "group": "us_industry"},
    "JETS": {"name": "U.S. Global Jets ETF", "group": "us_industry"},
    "XRT": {"name": "SPDR S&P Retail ETF", "group": "us_industry"},
    "ITA": {"name": "iShares U.S. Aerospace & Defense ETF", "group": "us_industry"},
    "PPA": {"name": "Invesco Aerospace & Defense ETF", "group": "us_industry"},
    "MOO": {"name": "VanEck Agribusiness ETF", "group": "us_industry"},
    "NVDA": {"name": "NVIDIA", "group": "semiconductor_single"},
    "AMD": {"name": "AMD", "group": "semiconductor_single"},
    "KWEB": {"name": "KraneShares CSI China Internet ETF", "group": "china_adr"},
    "FXI": {"name": "iShares China Large-Cap ETF", "group": "china_adr"},
    "MCHI": {"name": "iShares MSCI China ETF", "group": "china_adr"},
    "ASHR": {"name": "Xtrackers Harvest CSI 300 China A-Shares ETF", "group": "china_adr"},
    "CNYA": {"name": "iShares MSCI China A ETF", "group": "china_adr"},
    "EWH": {"name": "iShares MSCI Hong Kong ETF", "group": "asia"},
    "2800.HK": {"name": "Tracker Fund of Hong Kong", "group": "hong_kong"},
    "3033.HK": {"name": "CSOP Hang Seng TECH Index ETF", "group": "hong_kong"},
    "3067.HK": {"name": "iShares Hang Seng TECH ETF", "group": "hong_kong"},
    "2822.HK": {"name": "CSOP FTSE China A50 ETF", "group": "hong_kong"},
    "3188.HK": {"name": "ChinaAMC CSI 300 Index ETF", "group": "hong_kong"},
    "3199.HK": {"name": "CSOP Hang Seng TECH/China ETF proxy", "group": "hong_kong"},
    "BABA": {"name": "Alibaba ADR", "group": "china_adr_single"},
    "PDD": {"name": "PDD Holdings ADR", "group": "china_adr_single"},
    "JD": {"name": "JD.com ADR", "group": "china_adr_single"},
    "BIDU": {"name": "Baidu ADR", "group": "china_adr_single"},
    "NTES": {"name": "NetEase ADR", "group": "china_adr_single"},
    "0700.HK": {"name": "Tencent Holdings", "group": "hong_kong_single"},
    "9988.HK": {"name": "Alibaba Group Holding HK", "group": "hong_kong_single"},
    "3690.HK": {"name": "Meituan", "group": "hong_kong_single"},
    "9618.HK": {"name": "JD.com HK", "group": "hong_kong_single"},
    "9999.HK": {"name": "NetEase HK", "group": "hong_kong_single"},
    "1024.HK": {"name": "Kuaishou Technology", "group": "hong_kong_single"},
    "1810.HK": {"name": "Xiaomi", "group": "hong_kong_single"},
    "1211.HK": {"name": "BYD Company", "group": "hong_kong_single"},
    "0981.HK": {"name": "SMIC", "group": "hong_kong_single"},
    "2382.HK": {"name": "Sunny Optical", "group": "hong_kong_single"},
    "1299.HK": {"name": "AIA Group", "group": "hong_kong_single"},
    "2318.HK": {"name": "Ping An Insurance", "group": "hong_kong_single"},
    "0939.HK": {"name": "China Construction Bank", "group": "hong_kong_single"},
    "3988.HK": {"name": "Bank of China", "group": "hong_kong_single"},
    "0005.HK": {"name": "HSBC Holdings", "group": "hong_kong_single"},
    "0388.HK": {"name": "Hong Kong Exchanges and Clearing", "group": "hong_kong_single"},
    "0386.HK": {"name": "Sinopec", "group": "hong_kong_single"},
    "0883.HK": {"name": "CNOOC", "group": "hong_kong_single"},
    "0857.HK": {"name": "PetroChina", "group": "hong_kong_single"},
    "1109.HK": {"name": "China Resources Land", "group": "hong_kong_single"},
    "0688.HK": {"name": "China Overseas Land", "group": "hong_kong_single"},
    "2007.HK": {"name": "Country Garden", "group": "hong_kong_single"},
    "0016.HK": {"name": "Sun Hung Kai Properties", "group": "hong_kong_single"},
    "2020.HK": {"name": "ANTA Sports", "group": "hong_kong_single"},
    "2313.HK": {"name": "Shenzhou International", "group": "hong_kong_single"},
    "6862.HK": {"name": "Haidilao", "group": "hong_kong_single"},
    "2269.HK": {"name": "Wuxi Biologics", "group": "hong_kong_single"},
    "6618.HK": {"name": "JD Health", "group": "hong_kong_single"},
    "^HSI": {"name": "Hang Seng Index", "group": "asia_index"},
    "^HSCE": {"name": "Hang Seng China Enterprises Index", "group": "asia_index"},
    "^N225": {"name": "Nikkei 225", "group": "asia_index"},
    "^KS11": {"name": "KOSPI Composite", "group": "asia_index"},
    "^TWII": {"name": "Taiwan Weighted Index", "group": "asia_index"},
    "^STI": {"name": "Straits Times Index", "group": "asia_index"},
    "^AXJO": {"name": "S&P/ASX 200", "group": "asia_index"},
    "EWJ": {"name": "iShares MSCI Japan ETF", "group": "asia"},
    "EWY": {"name": "iShares MSCI South Korea ETF", "group": "asia"},
    "EWT": {"name": "iShares MSCI Taiwan ETF", "group": "asia"},
    "INDA": {"name": "iShares MSCI India ETF", "group": "asia"},
    "EWS": {"name": "iShares MSCI Singapore ETF", "group": "asia"},
    "EWA": {"name": "iShares MSCI Australia ETF", "group": "asia"},
    "EEM": {"name": "iShares MSCI Emerging Markets ETF", "group": "global_equity"},
    "EFA": {"name": "iShares MSCI EAFE ETF", "group": "global_equity"},
    "^FTSE": {"name": "FTSE 100 Index", "group": "europe_index"},
    "^GDAXI": {"name": "DAX Performance Index", "group": "europe_index"},
    "^FCHI": {"name": "CAC 40 Index", "group": "europe_index"},
    "^STOXX50E": {"name": "EURO STOXX 50 Index", "group": "europe_index"},
    "^VIX": {"name": "CBOE Volatility Index", "group": "volatility"},
    "^VVIX": {"name": "CBOE VIX Volatility Index", "group": "volatility"},
    "^MOVE": {"name": "ICE BofA MOVE Index", "group": "volatility"},
    "^TNX": {"name": "US 10Y Treasury Yield Index", "group": "rates"},
    "^TYX": {"name": "US 30Y Treasury Yield Index", "group": "rates"},
    "^FVX": {"name": "US 5Y Treasury Yield Index", "group": "rates"},
    "^IRX": {"name": "US 13 Week Treasury Bill Index", "group": "rates"},
    "TLT": {"name": "iShares 20+ Year Treasury Bond ETF", "group": "rates"},
    "IEF": {"name": "iShares 7-10 Year Treasury Bond ETF", "group": "rates"},
    "SHY": {"name": "iShares 1-3 Year Treasury Bond ETF", "group": "rates"},
    "HYG": {"name": "iShares iBoxx High Yield Corporate Bond ETF", "group": "credit"},
    "LQD": {"name": "iShares iBoxx Investment Grade Corporate Bond ETF", "group": "credit"},
    "EMB": {"name": "iShares J.P. Morgan USD Emerging Markets Bond ETF", "group": "credit"},
    "DX-Y.NYB": {"name": "US Dollar Index", "group": "fx"},
    "CNH=X": {"name": "USD/CNH", "group": "fx"},
    "CNY=X": {"name": "USD/CNY", "group": "fx"},
    "HG=F": {"name": "Copper Futures", "group": "commodity"},
    "CL=F": {"name": "Crude Oil Futures", "group": "commodity"},
    "BZ=F": {"name": "Brent Crude Oil Futures", "group": "commodity"},
    "NG=F": {"name": "Natural Gas Futures", "group": "commodity"},
    "GC=F": {"name": "Gold Futures", "group": "commodity"},
    "SI=F": {"name": "Silver Futures", "group": "commodity"},
    "PL=F": {"name": "Platinum Futures", "group": "commodity"},
    "PA=F": {"name": "Palladium Futures", "group": "commodity"},
    "ZC=F": {"name": "Corn Futures", "group": "commodity"},
    "ZS=F": {"name": "Soybean Futures", "group": "commodity"},
    "ZW=F": {"name": "Wheat Futures", "group": "commodity"},
    "GLD": {"name": "SPDR Gold Shares", "group": "commodity_etf"},
    "SLV": {"name": "iShares Silver Trust", "group": "commodity_etf"},
    "USO": {"name": "United States Oil Fund", "group": "commodity_etf"},
    "UNG": {"name": "United States Natural Gas Fund", "group": "commodity_etf"},
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="data/global")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SYMBOLS),
        help="Comma-separated Yahoo symbols. Default is the first global risk basket.",
    )
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--auto-adjust", action="store_true")
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--test", type=int, default=None, help="Download only first N symbols.")
    return parser.parse_args()


def safe_symbol(symbol: str) -> str:
    return (
        symbol.replace("^", "")
        .replace("=", "_")
        .replace(".", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def normalize_yfinance_frame(frame: pd.DataFrame, symbol: str, source: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [
            "_".join(str(part) for part in col if str(part))
            for col in frame.columns.to_flat_index()
        ]
    frame = frame.reset_index()
    rename = {}
    for column in frame.columns:
        key = str(column).strip().lower().replace(" ", "_")
        if key in {"date", "datetime"}:
            rename[column] = "date"
        elif key in {"open", f"open_{symbol.lower()}"}:
            rename[column] = "open"
        elif key in {"high", f"high_{symbol.lower()}"}:
            rename[column] = "high"
        elif key in {"low", f"low_{symbol.lower()}"}:
            rename[column] = "low"
        elif key in {"close", f"close_{symbol.lower()}"}:
            rename[column] = "close"
        elif key in {"adj_close", "adjclose", f"adj_close_{symbol.lower()}"}:
            rename[column] = "adj_close"
        elif key in {"volume", f"volume_{symbol.lower()}"}:
            rename[column] = "volume"
    frame = frame.rename(columns=rename)
    required = ["date", "open", "high", "low", "close"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{symbol}: missing required columns {missing}; columns={list(frame.columns)}")
    if "adj_close" not in frame.columns:
        frame["adj_close"] = frame["close"]
    if "volume" not in frame.columns:
        frame["volume"] = 0
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
    for column in ["open", "high", "low", "close", "adj_close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "open", "high", "low", "close"])
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    frame["symbol"] = symbol
    frame["source"] = source
    frame["updated_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return frame[
        [
            "date",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
            "symbol",
            "source",
            "updated_at",
        ]
    ]


def merge_existing(path: Path, new_frame: pd.DataFrame) -> pd.DataFrame:
    if path.exists():
        old = pd.read_csv(path)
        old["date"] = pd.to_datetime(old["date"])
        combined = pd.concat([old, new_frame], ignore_index=True, sort=False)
        combined = combined.sort_values("date").drop_duplicates("date", keep="last")
        return combined
    return new_frame


def summarize_existing_cache(path: Path):
    if not path.exists():
        return None
    try:
        existing = pd.read_csv(path, usecols=["date"])
        if existing.empty:
            return None
        dates = pd.to_datetime(existing["date"], errors="coerce").dropna()
        if dates.empty:
            return None
        return {
            "rows": int(len(existing)),
            "date_start": str(dates.min().date()),
            "date_end": str(dates.max().date()),
        }
    except Exception:
        return None


def download_symbol(symbol: str, start: str, end: str | None, interval: str, auto_adjust: bool, repair: bool) -> pd.DataFrame:
    import yfinance as yf

    frame = yf.download(
        symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        repair=repair,
        progress=False,
        threads=False,
    )
    return normalize_yfinance_frame(frame, symbol=symbol, source="yfinance")


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    raw_dir = output_root / "raw" / "us_ohlcv"
    raw_dir.mkdir(parents=True, exist_ok=True)
    symbols = [symbol.strip() for symbol in args.symbols.split(",") if symbol.strip()]
    if args.test is not None:
        symbols = symbols[: int(args.test)]
    summary_rows = []
    manifest_rows = []
    for symbol in symbols:
        meta = DEFAULT_SYMBOLS.get(symbol, {"name": symbol, "group": "custom"})
        path = raw_dir / f"{safe_symbol(symbol)}.csv"
        start = args.start
        if path.exists() and not args.force:
            existing = pd.read_csv(path, usecols=["date"])
            if not existing.empty:
                latest = pd.to_datetime(existing["date"]).max()
                start = (latest - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        try:
            frame = download_symbol(
                symbol,
                start=start,
                end=args.end,
                interval=args.interval,
                auto_adjust=args.auto_adjust,
                repair=args.repair,
            )
            if frame.empty:
                raise ValueError("empty download")
            combined = merge_existing(path, frame)
            combined.to_csv(path, index=False, encoding="utf-8")
            status = "ok"
            error = ""
            rows = len(combined)
            date_start = str(pd.to_datetime(combined["date"]).min().date())
            date_end = str(pd.to_datetime(combined["date"]).max().date())
        except Exception as exc:  # noqa: BLE001 - keep batch downloads running
            cached = summarize_existing_cache(path)
            if cached is not None:
                status = "cached_stale"
                rows = cached["rows"]
                date_start = cached["date_start"]
                date_end = cached["date_end"]
            else:
                status = "error"
                rows = 0
                date_start = ""
                date_end = ""
            error = repr(exc)
            print(f"{symbol}: ERROR {error}", flush=True)
        else:
            print(f"{symbol}: {date_start}..{date_end} rows={rows}", flush=True)
        manifest_rows.append(
            {
                "symbol": symbol,
                "safe_symbol": safe_symbol(symbol),
                "name": meta["name"],
                "group": meta["group"],
                "source": "yfinance",
                "path": str(path),
            }
        )
        summary_rows.append(
            {
                "symbol": symbol,
                "status": status,
                "rows": rows,
                "date_start": date_start,
                "date_end": date_end,
                "path": str(path),
                "error": error,
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    summary = pd.DataFrame(summary_rows)
    manifest.to_csv(output_root / "global_symbol_manifest.csv", index=False, encoding="utf-8")
    summary_path = output_root / "download_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    (output_root / "download_summary.json").write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"saved summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
