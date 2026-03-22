#!/usr/bin/env python3
"""
Weekly multi-zone support/resistance scanner.

This script is a Python translation and extension of the uploaded Pine logic.
It preserves the core ideas:
- confirmed pivot-low support zones with ATR-based thickness
- S1 oversold support-touch signal
- S2 rebound confirmation after S1

And extends them with:
- multiple concurrent support zones
- multiple concurrent resistance zones
- R1 / R2 mirror logic
- regime-aware filtering using EMA200 (isBear)
- optional candle-pattern filters on initial zone-touch signals
- dynamic universes for Nasdaq-listed symbols and S&P 500 top-N by market cap

Notes on Pine parity:
- Pine's single var float / single box state has been replaced with a list of Zone objects.
- S2/R2 are re-armed whenever a new S1/R1 fires on the same zone, fixing the stale-fired-state issue.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd


# ==============================
# Configuration / data classes
# ==============================


@dataclass
class ScanConfig:
    pivot_len: int = 5
    rsi_len: int = 14
    rsi_s1: float = 35.0
    rsi_s2: float = 40.0
    rsi_r1: float = 65.0
    rsi_r2: float = 60.0
    vol_len: int = 20
    atr_len: int = 14
    atr_mult: float = 0.7
    ema_fast_len: int = 50
    ema_slow_len: int = 200
    s1_vol_mult: float = 1.0
    s2_vol_mult: float = 1.2
    r1_vol_mult: float = 1.0
    r2_vol_mult: float = 1.2
    confirm_window_bars: int = 15
    use_candle_filter: bool = True
    doji_body_ratio: float = 0.12
    wick_body_ratio: float = 2.0
    allow_s2_in_bear: bool = False
    allow_r2_in_bull: bool = True
    chart_bars: int = 180
    max_chart_zones_per_side: int = 10


@dataclass
class Zone:
    zone_id: str
    kind: Literal["support", "resistance"]
    pivot_index: int
    pivot_date: pd.Timestamp
    created_index: int
    created_date: pd.Timestamp
    pivot_price: float
    top: float
    bottom: float
    active: bool = True
    invalidated_index: Optional[int] = None
    invalidated_date: Optional[pd.Timestamp] = None
    touch_fired: bool = False
    confirm_fired: bool = False
    last_primary_signal_bar: Optional[int] = None
    primary_signal_count: int = 0
    secondary_signal_count: int = 0


# ==============================
# Argument parsing
# ==============================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-zone support/resistance scanner")
    parser.add_argument(
        "--universe",
        choices=["nasdaq_all", "nasdaq_top100", "sp500_top300", "custom_file"],
        default="nasdaq_all",
        help="Ticker universe selector.",
    )
    parser.add_argument("--tickers-file", default="", help="Used only when --universe custom_file")
    parser.add_argument("--top-n", type=int, default=100, help="Top-N market-cap cutoff for market-cap-ranked universe modes")
    parser.add_argument("--period", default="2y", help="History period for yfinance")
    parser.add_argument("--interval", default="1d", help="History interval for yfinance")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--chunk-size", type=int, default=80, help="Batch download chunk size")
    parser.add_argument("--pause-seconds", type=float, default=0.25, help="Pause between yfinance batch requests")
    parser.add_argument("--charts-limit", type=int, default=60, help="Maximum charts to generate")
    parser.add_argument(
        "--full-scan-limit",
        type=int,
        default=0,
        help="Maximum number of per-ticker full scan CSVs to write. 0 disables per-ticker files.",
    )
    parser.add_argument("--sp500-cap-workers", type=int, default=10, help="Thread count for S&P 500 market-cap ranking")
    parser.add_argument("--nasdaq-cap-workers", type=int, default=12, help="Thread count for Nasdaq market-cap ranking")
    parser.add_argument("--signal-start-date", default="", help="Optional YYYY-MM-DD filter start for exported signal hits")
    parser.add_argument("--signal-end-date", default="", help="Optional YYYY-MM-DD filter end for exported signal hits")

    # Logic parameters
    parser.add_argument("--pivot-len", type=int, default=5)
    parser.add_argument("--rsi-len", type=int, default=14)
    parser.add_argument("--rsi-s1", type=float, default=35.0)
    parser.add_argument("--rsi-s2", type=float, default=40.0)
    parser.add_argument("--rsi-r1", type=float, default=65.0)
    parser.add_argument("--rsi-r2", type=float, default=60.0)
    parser.add_argument("--vol-len", type=int, default=20)
    parser.add_argument("--atr-len", type=int, default=14)
    parser.add_argument("--atr-mult", type=float, default=0.7)
    parser.add_argument("--ema-fast-len", type=int, default=50)
    parser.add_argument("--ema-slow-len", type=int, default=200)
    parser.add_argument("--s1-vol-mult", type=float, default=1.0)
    parser.add_argument("--s2-vol-mult", type=float, default=1.2)
    parser.add_argument("--r1-vol-mult", type=float, default=1.0)
    parser.add_argument("--r2-vol-mult", type=float, default=1.2)
    parser.add_argument("--confirm-window-bars", type=int, default=15)
    parser.add_argument("--chart-bars", type=int, default=180)
    parser.add_argument("--max-chart-zones-per-side", type=int, default=10)
    parser.add_argument("--disable-candle-filter", action="store_true")
    parser.add_argument("--doji-body-ratio", type=float, default=0.12)
    parser.add_argument("--wick-body-ratio", type=float, default=2.0)
    parser.add_argument("--allow-s2-in-bear", action="store_true")
    parser.add_argument("--disable-r2-in-bull", action="store_true")
    return parser.parse_args()


# ==============================
# Core indicators
# ==============================


def rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / length, adjust=False).mean()


def compute_rsi(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = rma(gain, length)
    avg_loss = rma(loss, length)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def compute_atr(df: pd.DataFrame, length: int) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            (df["High"] - df["Low"]).abs(),
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return rma(tr, length)


def normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    out = out[required].dropna().copy()
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    out.index = idx
    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()


def enrich_indicators(df: pd.DataFrame, cfg: ScanConfig) -> pd.DataFrame:
    out = df.copy()
    out["RSI"] = compute_rsi(out["Close"], cfg.rsi_len)
    out["VolMA"] = out["Volume"].rolling(cfg.vol_len).mean()
    out["ATR"] = compute_atr(out, cfg.atr_len)
    out["EMA50"] = out["Close"].ewm(span=cfg.ema_fast_len, adjust=False).mean()
    out["EMA200"] = out["Close"].ewm(span=cfg.ema_slow_len, adjust=False).mean()
    out["isBear"] = out["Close"] < out["EMA200"]
    return out


# ==============================
# Candle pattern filters
# ==============================


def safe_div(a: float, b: float) -> float:
    if b == 0 or not np.isfinite(a) or not np.isfinite(b):
        return np.nan
    return a / b


def candle_metrics(row: pd.Series) -> dict[str, float]:
    o = float(row["Open"])
    h = float(row["High"])
    l = float(row["Low"])
    c = float(row["Close"])
    body = abs(c - o)
    rng = max(h - l, 1e-9)
    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)
    return {
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "body": body,
        "range": rng,
        "lower_wick": lower_wick,
        "upper_wick": upper_wick,
        "body_ratio": safe_div(body, rng),
    }


def is_doji(row: pd.Series, cfg: ScanConfig) -> bool:
    m = candle_metrics(row)
    return np.isfinite(m["body_ratio"]) and (m["body_ratio"] <= cfg.doji_body_ratio)


def is_bullish_reversal_candle(row: pd.Series, cfg: ScanConfig) -> bool:
    m = candle_metrics(row)
    hammer_like = (
        m["lower_wick"] >= cfg.wick_body_ratio * max(m["body"], 1e-9)
        and m["upper_wick"] <= max(m["body"], 1e-9) * 1.25
        and m["close"] >= m["open"]
    )
    doji_like = is_doji(row, cfg) and m["close"] >= (m["low"] + 0.4 * m["range"])
    return hammer_like or doji_like


def is_bearish_reversal_candle(row: pd.Series, cfg: ScanConfig) -> bool:
    m = candle_metrics(row)
    shooting_star_like = (
        m["upper_wick"] >= cfg.wick_body_ratio * max(m["body"], 1e-9)
        and m["lower_wick"] <= max(m["body"], 1e-9) * 1.25
        and m["close"] <= m["open"]
    )
    doji_like = is_doji(row, cfg) and m["close"] <= (m["high"] - 0.4 * m["range"])
    return shooting_star_like or doji_like


# ==============================
# Pivot detection
# ==============================


def _center_is_unique_min(values: np.ndarray, center: int) -> bool:
    center_value = values[center]
    min_value = np.nanmin(values)
    if not np.isfinite(center_value) or center_value != min_value:
        return False
    return int(np.nanargmin(values)) == center


def _center_is_unique_max(values: np.ndarray, center: int) -> bool:
    center_value = values[center]
    max_value = np.nanmax(values)
    if not np.isfinite(center_value) or center_value != max_value:
        return False
    return int(np.nanargmax(values)) == center


def pivot_low_confirmed_at_current_bar(lows: pd.Series, current_idx: int, pivot_len: int) -> tuple[bool, int | None, float | None]:
    candidate_idx = current_idx - pivot_len
    if candidate_idx - pivot_len < 0 or current_idx >= len(lows):
        return False, None, None
    window = lows.iloc[candidate_idx - pivot_len : candidate_idx + pivot_len + 1].to_numpy(dtype=float)
    if len(window) != (2 * pivot_len + 1):
        return False, None, None
    if _center_is_unique_min(window, pivot_len):
        return True, candidate_idx, float(window[pivot_len])
    return False, None, None


def pivot_high_confirmed_at_current_bar(highs: pd.Series, current_idx: int, pivot_len: int) -> tuple[bool, int | None, float | None]:
    candidate_idx = current_idx - pivot_len
    if candidate_idx - pivot_len < 0 or current_idx >= len(highs):
        return False, None, None
    window = highs.iloc[candidate_idx - pivot_len : candidate_idx + pivot_len + 1].to_numpy(dtype=float)
    if len(window) != (2 * pivot_len + 1):
        return False, None, None
    if _center_is_unique_max(window, pivot_len):
        return True, candidate_idx, float(window[pivot_len])
    return False, None, None


# ==============================
# Universe resolution
# ==============================


def normalize_ticker_symbol(symbol: str) -> str:
    return str(symbol).strip().upper().replace(".", "-")


def load_custom_tickers(path: str | Path) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    tickers: list[str] = []
    for line in lines:
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        for part in line.replace(" ", "").split(","):
            if part:
                tickers.append(normalize_ticker_symbol(part))
    seen: set[str] = set()
    ordered: list[str] = []
    for ticker in tickers:
        if ticker not in seen:
            seen.add(ticker)
            ordered.append(ticker)
    if not ordered:
        raise ValueError("No tickers found in custom ticker file")
    return ordered


def fetch_nasdaq_universe() -> list[str]:
    url = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
    df = pd.read_csv(url, sep="|", dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    df = df[df["Symbol"].notna()].copy()
    df = df[df["Symbol"] != "File Creation Time"]
    if "Test Issue" in df.columns:
        df = df[df["Test Issue"] == "N"]
    tickers = [normalize_ticker_symbol(s) for s in df["Symbol"].astype(str).tolist()]
    return tickers


def fetch_sp500_constituents() -> list[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    target = None
    for table in tables:
        cols = {str(c).strip() for c in table.columns}
        if "Symbol" in cols and ("Security" in cols or "GICS Sector" in cols):
            target = table
            break
    if target is None:
        raise ValueError("Could not find S&P 500 constituents table")
    tickers = [normalize_ticker_symbol(s) for s in target["Symbol"].astype(str).tolist()]
    return tickers


def fetch_market_cap_one(ticker: str) -> tuple[str, float]:
    import yfinance as yf

    cap = np.nan
    tk = yf.Ticker(ticker)
    try:
        # fast_info is usually cheaper than full info
        fi = getattr(tk, "fast_info", None)
        if fi:
            for key in ("marketCap", "market_cap"):
                try:
                    value = fi.get(key)
                except Exception:
                    value = None
                if value is not None and np.isfinite(value) and value > 0:
                    return ticker, float(value)
            try:
                shares = fi.get("shares") or fi.get("sharesOutstanding")
            except Exception:
                shares = None
            try:
                last_price = fi.get("lastPrice") or fi.get("last_price")
            except Exception:
                last_price = None
            if shares and last_price and np.isfinite(shares) and np.isfinite(last_price):
                return ticker, float(shares) * float(last_price)
    except Exception:
        pass

    try:
        info = tk.info  # type: ignore[name-defined]
        if isinstance(info, dict):
            for key in ("marketCap", "enterpriseValue"):
                value = info.get(key)
                if value is not None and np.isfinite(value) and value > 0:
                    return ticker, float(value)
            shares = info.get("sharesOutstanding")
            current_price = info.get("currentPrice") or info.get("regularMarketPrice")
            if shares and current_price and np.isfinite(shares) and np.isfinite(current_price):
                return ticker, float(shares) * float(current_price)
    except Exception:
        pass

    return ticker, cap


def rank_nasdaq_by_market_cap(top_n: int, workers: int) -> pd.DataFrame:
    constituents = fetch_nasdaq_universe()
    records: list[dict] = []
    with cf.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_market_cap_one, ticker): ticker for ticker in constituents}
        for idx, future in enumerate(cf.as_completed(futures), start=1):
            ticker = futures[future]
            try:
                symbol, market_cap = future.result()
            except Exception:
                symbol, market_cap = ticker, np.nan
            records.append({"ticker": symbol, "market_cap": market_cap})
            if idx % 100 == 0 or idx == len(constituents):
                print(f"[INFO] Nasdaq market-cap fetch progress: {idx}/{len(constituents)}", flush=True)

    ranked = pd.DataFrame(records)
    ranked = ranked.dropna(subset=["market_cap"]).sort_values("market_cap", ascending=False).reset_index(drop=True)
    ranked["rank_by_market_cap"] = np.arange(1, len(ranked) + 1)
    ranked = ranked.head(top_n).copy()
    ranked["universe"] = f"nasdaq_top{top_n}"
    return ranked


def rank_sp500_by_market_cap(top_n: int, workers: int) -> pd.DataFrame:
    constituents = fetch_sp500_constituents()
    records: list[dict] = []
    with cf.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_market_cap_one, ticker): ticker for ticker in constituents}
        for idx, future in enumerate(cf.as_completed(futures), start=1):
            ticker = futures[future]
            try:
                symbol, market_cap = future.result()
            except Exception:
                symbol, market_cap = ticker, np.nan
            records.append({"ticker": symbol, "market_cap": market_cap})
            if idx % 50 == 0 or idx == len(constituents):
                print(f"[INFO] S&P 500 market-cap fetch progress: {idx}/{len(constituents)}", flush=True)

    ranked = pd.DataFrame(records)
    ranked = ranked.dropna(subset=["market_cap"]).sort_values("market_cap", ascending=False).reset_index(drop=True)
    ranked["rank_by_market_cap"] = np.arange(1, len(ranked) + 1)
    ranked = ranked.head(top_n).copy()
    return ranked


def resolve_universe(args: argparse.Namespace) -> tuple[list[str], pd.DataFrame]:
    if args.universe == "custom_file":
        tickers = load_custom_tickers(args.tickers_file)
        meta = pd.DataFrame({"ticker": tickers})
        return tickers, meta

    if args.universe == "nasdaq_all":
        tickers = fetch_nasdaq_universe()
        meta = pd.DataFrame({"ticker": tickers})
        return tickers, meta

    if args.universe == "nasdaq_top100":
        ranked = rank_nasdaq_by_market_cap(top_n=args.top_n, workers=args.nasdaq_cap_workers)
        return ranked["ticker"].tolist(), ranked

    if args.universe == "sp500_top300":
        ranked = rank_sp500_by_market_cap(top_n=args.top_n, workers=args.sp500_cap_workers)
        return ranked["ticker"].tolist(), ranked

    raise ValueError(f"Unsupported universe: {args.universe}")


# ==============================
# Batch market data download
# ==============================


def split_chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def extract_history_from_download(raw: pd.DataFrame, ticker: str, requested_count: int) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()

    if requested_count == 1 and not isinstance(raw.columns, pd.MultiIndex):
        return normalize_history(raw)

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = list(raw.columns.get_level_values(0))
        level1 = list(raw.columns.get_level_values(1))

        if ticker in level0:
            sub = raw[ticker].copy()
            return normalize_history(sub)
        if ticker in level1:
            sub = raw.xs(ticker, axis=1, level=1).copy()
            return normalize_history(sub)

    return pd.DataFrame()


def fetch_histories_batch(tickers: list[str], period: str, interval: str, chunk_size: int, pause_seconds: float) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    import yfinance as yf

    histories: dict[str, pd.DataFrame] = {}
    errors: list[dict] = []

    for chunk_no, chunk in enumerate(split_chunks(tickers, chunk_size), start=1):
        print(f"[INFO] Downloading chunk {chunk_no} with {len(chunk)} tickers", flush=True)
        try:
            raw = yf.download(
                tickers=chunk,
                period=period,
                interval=interval,
                auto_adjust=False,
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception as exc:
            for ticker in chunk:
                errors.append({"ticker": ticker, "error": f"download failed: {exc}"})
            time.sleep(max(pause_seconds, 0.0))
            continue

        for ticker in chunk:
            try:
                hist = extract_history_from_download(raw, ticker, requested_count=len(chunk))
                if hist.empty:
                    errors.append({"ticker": ticker, "error": "empty history after batch download"})
                else:
                    histories[ticker] = hist
            except Exception as exc:
                errors.append({"ticker": ticker, "error": f"normalize failed: {exc}"})

        if pause_seconds > 0:
            time.sleep(pause_seconds)

    return histories, errors


# ==============================
# Zone helpers / signal logic
# ==============================


def make_zone_id(kind: str, pivot_date: pd.Timestamp, pivot_price: float, ordinal: int) -> str:
    prefix = "SUP" if kind == "support" else "RES"
    return f"{prefix}_{pivot_date.strftime('%Y%m%d')}_{ordinal:03d}_{pivot_price:.4f}"


def select_priority_zone(zones: list[Zone], close_price: float, kind: Literal["support", "resistance"]) -> Optional[Zone]:
    active = [z for z in zones if z.active]
    if not active:
        return None

    inside = [z for z in active if z.bottom <= close_price <= z.top]
    if inside:
        return min(inside, key=lambda z: abs(z.pivot_price - close_price))

    if kind == "support":
        preferred = [z for z in active if z.top <= close_price]
        if preferred:
            return min(preferred, key=lambda z: close_price - z.top)
    else:
        preferred = [z for z in active if z.bottom >= close_price]
        if preferred:
            return min(preferred, key=lambda z: z.bottom - close_price)

    return min(active, key=lambda z: abs(z.pivot_price - close_price))


def recent_chart_zones(zones: list[Zone], visible_start: pd.Timestamp, visible_end: pd.Timestamp, limit: int) -> list[Zone]:
    relevant = []
    for zone in zones:
        zone_end = zone.invalidated_date if zone.invalidated_date is not None else visible_end
        if zone_end >= visible_start and zone.created_date <= visible_end:
            relevant.append(zone)
    relevant.sort(key=lambda z: (z.active, z.created_date), reverse=True)
    return relevant[:limit]


def apply_signal_logic(df: pd.DataFrame, ticker: str, cfg: ScanConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = df.copy()

    base_cols = [
        "active_support_count",
        "active_resistance_count",
        "priority_support_top",
        "priority_support_bottom",
        "priority_support_price",
        "priority_support_pivot_date",
        "priority_support_zone_id",
        "priority_resistance_top",
        "priority_resistance_bottom",
        "priority_resistance_price",
        "priority_resistance_pivot_date",
        "priority_resistance_zone_id",
        "S1",
        "S2",
        "R1",
        "R2",
        "priceInSupportZone",
        "priceInResistanceZone",
        "S1_zone_ids",
        "S2_zone_ids",
        "R1_zone_ids",
        "R2_zone_ids",
    ]
    for col in base_cols:
        out[col] = "" if col.endswith("_zone_ids") or col.endswith("_zone_id") or col.endswith("_pivot_date") else False
    for col in [
        "active_support_count",
        "active_resistance_count",
        "priority_support_top",
        "priority_support_bottom",
        "priority_support_price",
        "priority_resistance_top",
        "priority_resistance_bottom",
        "priority_resistance_price",
    ]:
        out[col] = np.nan

    support_zones: list[Zone] = []
    resistance_zones: list[Zone] = []
    zone_records: list[dict] = []
    event_records: list[dict] = []
    zone_ordinal = 0

    for i in range(len(out)):
        idx = out.index[i]
        row = out.iloc[i]
        close_i = float(row["Close"])
        volume_i = float(row["Volume"])
        vol_ma_i = float(row["VolMA"]) if pd.notna(row["VolMA"]) else np.nan
        atr_i = float(row["ATR"]) if pd.notna(row["ATR"]) else np.nan
        rsi_i = float(row["RSI"]) if pd.notna(row["RSI"]) else np.nan
        rsi_prev = float(out["RSI"].iloc[i - 1]) if i > 0 and pd.notna(out["RSI"].iloc[i - 1]) else np.nan
        is_bear = bool(row["isBear"])

        # Confirmed pivots create new zones. This fixes the single-zone overwrite problem.
        low_ok, low_pivot_idx, low_pivot_val = pivot_low_confirmed_at_current_bar(out["Low"], i, cfg.pivot_len)
        if low_ok and np.isfinite(atr_i):
            zone_ordinal += 1
            support_zones.append(
                Zone(
                    zone_id=make_zone_id("support", out.index[low_pivot_idx], float(low_pivot_val), zone_ordinal),
                    kind="support",
                    pivot_index=int(low_pivot_idx),
                    pivot_date=out.index[low_pivot_idx],
                    created_index=i,
                    created_date=idx,
                    pivot_price=float(low_pivot_val),
                    top=float(low_pivot_val) + atr_i * cfg.atr_mult,
                    bottom=float(low_pivot_val) - atr_i * cfg.atr_mult,
                )
            )

        high_ok, high_pivot_idx, high_pivot_val = pivot_high_confirmed_at_current_bar(out["High"], i, cfg.pivot_len)
        if high_ok and np.isfinite(atr_i):
            zone_ordinal += 1
            resistance_zones.append(
                Zone(
                    zone_id=make_zone_id("resistance", out.index[high_pivot_idx], float(high_pivot_val), zone_ordinal),
                    kind="resistance",
                    pivot_index=int(high_pivot_idx),
                    pivot_date=out.index[high_pivot_idx],
                    created_index=i,
                    created_date=idx,
                    pivot_price=float(high_pivot_val),
                    top=float(high_pivot_val) + atr_i * cfg.atr_mult,
                    bottom=float(high_pivot_val) - atr_i * cfg.atr_mult,
                )
            )

        s1_ids: list[str] = []
        s2_ids: list[str] = []
        r1_ids: list[str] = []
        r2_ids: list[str] = []
        any_support_touch = False
        any_resistance_touch = False

        # ----- Support side -----
        for zone in support_zones:
            if zone.active and close_i < zone.bottom:
                zone.active = False
                zone.invalidated_index = i
                zone.invalidated_date = idx
                zone.touch_fired = False
                zone.confirm_fired = False

            if not zone.active:
                continue

            price_in_zone = zone.bottom <= close_i <= zone.top
            any_support_touch = any_support_touch or price_in_zone

            s1_vol_ok = np.isfinite(vol_ma_i) and volume_i > vol_ma_i * cfg.s1_vol_mult
            candle_ok = (not cfg.use_candle_filter) or is_bullish_reversal_candle(row, cfg)
            s1_raw = price_in_zone and s1_vol_ok and np.isfinite(rsi_i) and (rsi_i < cfg.rsi_s1) and candle_ok

            if s1_raw and not zone.touch_fired:
                zone.touch_fired = True
                zone.confirm_fired = False  # re-arm S2 whenever a new S1 fires on this zone
                zone.last_primary_signal_bar = i
                zone.primary_signal_count += 1
                s1_ids.append(zone.zone_id)
                event_records.append(
                    {
                        "ticker": ticker,
                        "date": idx.strftime("%Y-%m-%d"),
                        "signal_type": "S1",
                        "zone_id": zone.zone_id,
                        "zone_kind": zone.kind,
                        "zone_pivot_date": zone.pivot_date.strftime("%Y-%m-%d"),
                        "zone_price": round(zone.pivot_price, 6),
                        "zone_top": round(zone.top, 6),
                        "zone_bottom": round(zone.bottom, 6),
                        "close": round(close_i, 6),
                        "volume": int(volume_i),
                        "vol_ma": round(vol_ma_i, 6) if np.isfinite(vol_ma_i) else np.nan,
                        "rsi": round(rsi_i, 4),
                        "is_bear": is_bear,
                    }
                )

            if not price_in_zone:
                zone.touch_fired = False

            rsi_cross_up_s2 = np.isfinite(rsi_prev) and np.isfinite(rsi_i) and (rsi_prev <= cfg.rsi_s2) and (rsi_i > cfg.rsi_s2)
            s2_vol_ok = np.isfinite(vol_ma_i) and volume_i > vol_ma_i * cfg.s2_vol_mult
            s2_regime_ok = (not is_bear) or cfg.allow_s2_in_bear
            s2_window_ok = zone.last_primary_signal_bar is not None and (i - zone.last_primary_signal_bar <= cfg.confirm_window_bars)
            s2_raw = (
                zone.active
                and s2_window_ok
                and close_i > zone.bottom
                and s2_vol_ok
                and np.isfinite(rsi_i)
                and np.isfinite(rsi_prev)
                and (rsi_i > rsi_prev)
                and rsi_cross_up_s2
                and s2_regime_ok
            )
            if s2_raw and not zone.confirm_fired:
                zone.confirm_fired = True
                zone.secondary_signal_count += 1
                s2_ids.append(zone.zone_id)
                event_records.append(
                    {
                        "ticker": ticker,
                        "date": idx.strftime("%Y-%m-%d"),
                        "signal_type": "S2",
                        "zone_id": zone.zone_id,
                        "zone_kind": zone.kind,
                        "zone_pivot_date": zone.pivot_date.strftime("%Y-%m-%d"),
                        "zone_price": round(zone.pivot_price, 6),
                        "zone_top": round(zone.top, 6),
                        "zone_bottom": round(zone.bottom, 6),
                        "close": round(close_i, 6),
                        "volume": int(volume_i),
                        "vol_ma": round(vol_ma_i, 6) if np.isfinite(vol_ma_i) else np.nan,
                        "rsi": round(rsi_i, 4),
                        "is_bear": is_bear,
                    }
                )

        # ----- Resistance side -----
        for zone in resistance_zones:
            if zone.active and close_i > zone.top:
                zone.active = False
                zone.invalidated_index = i
                zone.invalidated_date = idx
                zone.touch_fired = False
                zone.confirm_fired = False

            if not zone.active:
                continue

            price_in_zone = zone.bottom <= close_i <= zone.top
            any_resistance_touch = any_resistance_touch or price_in_zone

            r1_vol_ok = np.isfinite(vol_ma_i) and volume_i > vol_ma_i * cfg.r1_vol_mult
            candle_ok = (not cfg.use_candle_filter) or is_bearish_reversal_candle(row, cfg)
            r1_raw = price_in_zone and r1_vol_ok and np.isfinite(rsi_i) and (rsi_i > cfg.rsi_r1) and candle_ok

            if r1_raw and not zone.touch_fired:
                zone.touch_fired = True
                zone.confirm_fired = False  # re-arm R2 whenever a new R1 fires on this zone
                zone.last_primary_signal_bar = i
                zone.primary_signal_count += 1
                r1_ids.append(zone.zone_id)
                event_records.append(
                    {
                        "ticker": ticker,
                        "date": idx.strftime("%Y-%m-%d"),
                        "signal_type": "R1",
                        "zone_id": zone.zone_id,
                        "zone_kind": zone.kind,
                        "zone_pivot_date": zone.pivot_date.strftime("%Y-%m-%d"),
                        "zone_price": round(zone.pivot_price, 6),
                        "zone_top": round(zone.top, 6),
                        "zone_bottom": round(zone.bottom, 6),
                        "close": round(close_i, 6),
                        "volume": int(volume_i),
                        "vol_ma": round(vol_ma_i, 6) if np.isfinite(vol_ma_i) else np.nan,
                        "rsi": round(rsi_i, 4),
                        "is_bear": is_bear,
                    }
                )

            if not price_in_zone:
                zone.touch_fired = False

            rsi_cross_down_r2 = np.isfinite(rsi_prev) and np.isfinite(rsi_i) and (rsi_prev >= cfg.rsi_r2) and (rsi_i < cfg.rsi_r2)
            r2_vol_ok = np.isfinite(vol_ma_i) and volume_i > vol_ma_i * cfg.r2_vol_mult
            r2_regime_ok = is_bear or cfg.allow_r2_in_bull
            r2_window_ok = zone.last_primary_signal_bar is not None and (i - zone.last_primary_signal_bar <= cfg.confirm_window_bars)
            r2_raw = (
                zone.active
                and r2_window_ok
                and close_i < zone.top
                and r2_vol_ok
                and np.isfinite(rsi_i)
                and np.isfinite(rsi_prev)
                and (rsi_i < rsi_prev)
                and rsi_cross_down_r2
                and r2_regime_ok
            )
            if r2_raw and not zone.confirm_fired:
                zone.confirm_fired = True
                zone.secondary_signal_count += 1
                r2_ids.append(zone.zone_id)
                event_records.append(
                    {
                        "ticker": ticker,
                        "date": idx.strftime("%Y-%m-%d"),
                        "signal_type": "R2",
                        "zone_id": zone.zone_id,
                        "zone_kind": zone.kind,
                        "zone_pivot_date": zone.pivot_date.strftime("%Y-%m-%d"),
                        "zone_price": round(zone.pivot_price, 6),
                        "zone_top": round(zone.top, 6),
                        "zone_bottom": round(zone.bottom, 6),
                        "close": round(close_i, 6),
                        "volume": int(volume_i),
                        "vol_ma": round(vol_ma_i, 6) if np.isfinite(vol_ma_i) else np.nan,
                        "rsi": round(rsi_i, 4),
                        "is_bear": is_bear,
                    }
                )

        active_support = [z for z in support_zones if z.active]
        active_resistance = [z for z in resistance_zones if z.active]
        priority_support = select_priority_zone(active_support, close_i, "support")
        priority_resistance = select_priority_zone(active_resistance, close_i, "resistance")

        out.at[idx, "active_support_count"] = len(active_support)
        out.at[idx, "active_resistance_count"] = len(active_resistance)
        out.at[idx, "S1"] = bool(s1_ids)
        out.at[idx, "S2"] = bool(s2_ids)
        out.at[idx, "R1"] = bool(r1_ids)
        out.at[idx, "R2"] = bool(r2_ids)
        out.at[idx, "priceInSupportZone"] = any_support_touch
        out.at[idx, "priceInResistanceZone"] = any_resistance_touch
        out.at[idx, "S1_zone_ids"] = ",".join(s1_ids)
        out.at[idx, "S2_zone_ids"] = ",".join(s2_ids)
        out.at[idx, "R1_zone_ids"] = ",".join(r1_ids)
        out.at[idx, "R2_zone_ids"] = ",".join(r2_ids)

        if priority_support is not None:
            out.at[idx, "priority_support_top"] = priority_support.top
            out.at[idx, "priority_support_bottom"] = priority_support.bottom
            out.at[idx, "priority_support_price"] = priority_support.pivot_price
            out.at[idx, "priority_support_pivot_date"] = priority_support.pivot_date.strftime("%Y-%m-%d")
            out.at[idx, "priority_support_zone_id"] = priority_support.zone_id

        if priority_resistance is not None:
            out.at[idx, "priority_resistance_top"] = priority_resistance.top
            out.at[idx, "priority_resistance_bottom"] = priority_resistance.bottom
            out.at[idx, "priority_resistance_price"] = priority_resistance.pivot_price
            out.at[idx, "priority_resistance_pivot_date"] = priority_resistance.pivot_date.strftime("%Y-%m-%d")
            out.at[idx, "priority_resistance_zone_id"] = priority_resistance.zone_id

    for zone in support_zones + resistance_zones:
        zone_records.append(
            {
                "ticker": ticker,
                "zone_id": zone.zone_id,
                "zone_kind": zone.kind,
                "pivot_date": zone.pivot_date.strftime("%Y-%m-%d"),
                "created_date": zone.created_date.strftime("%Y-%m-%d"),
                "invalidated_date": zone.invalidated_date.strftime("%Y-%m-%d") if zone.invalidated_date is not None else "",
                "pivot_price": round(zone.pivot_price, 6),
                "zone_top": round(zone.top, 6),
                "zone_bottom": round(zone.bottom, 6),
                "active": zone.active,
                "primary_signal_count": zone.primary_signal_count,
                "secondary_signal_count": zone.secondary_signal_count,
            }
        )

    events_df = pd.DataFrame(event_records)
    zones_df = pd.DataFrame(zone_records)
    return out, events_df, zones_df


# ==============================
# Reporting helpers
# ==============================


def build_summary_row(df: pd.DataFrame, events_df: pd.DataFrame, zones_df: pd.DataFrame, ticker: str) -> dict:
    last = df.iloc[-1]
    current_signals = [sig for sig in ["S1", "S2", "R1", "R2"] if bool(last[sig])]

    if events_df.empty:
        last_signal_type = ""
        last_signal_date = ""
        last_signal_zone_id = ""
    else:
        ev = events_df.iloc[-1]
        last_signal_type = str(ev["signal_type"])
        last_signal_date = str(ev["date"])
        last_signal_zone_id = str(ev["zone_id"])

    active_support = zones_df[(zones_df["zone_kind"] == "support") & (zones_df["active"] == True)] if not zones_df.empty else pd.DataFrame()
    active_resistance = zones_df[(zones_df["zone_kind"] == "resistance") & (zones_df["active"] == True)] if not zones_df.empty else pd.DataFrame()

    return {
        "ticker": ticker,
        "date": df.index[-1].strftime("%Y-%m-%d"),
        "close": round(float(last["Close"]), 6),
        "rsi": round(float(last["RSI"]), 4),
        "ema50": round(float(last["EMA50"]), 6),
        "ema200": round(float(last["EMA200"]), 6),
        "is_bear": bool(last["isBear"]),
        "volume": int(last["Volume"]),
        "active_support_count": int(last["active_support_count"]) if pd.notna(last["active_support_count"]) else 0,
        "active_resistance_count": int(last["active_resistance_count"]) if pd.notna(last["active_resistance_count"]) else 0,
        "priority_support_zone_id": str(last["priority_support_zone_id"] or ""),
        "priority_support_pivot_date": str(last["priority_support_pivot_date"] or ""),
        "priority_support_price": round(float(last["priority_support_price"]), 6) if pd.notna(last["priority_support_price"]) else np.nan,
        "priority_support_top": round(float(last["priority_support_top"]), 6) if pd.notna(last["priority_support_top"]) else np.nan,
        "priority_support_bottom": round(float(last["priority_support_bottom"]), 6) if pd.notna(last["priority_support_bottom"]) else np.nan,
        "priority_resistance_zone_id": str(last["priority_resistance_zone_id"] or ""),
        "priority_resistance_pivot_date": str(last["priority_resistance_pivot_date"] or ""),
        "priority_resistance_price": round(float(last["priority_resistance_price"]), 6) if pd.notna(last["priority_resistance_price"]) else np.nan,
        "priority_resistance_top": round(float(last["priority_resistance_top"]), 6) if pd.notna(last["priority_resistance_top"]) else np.nan,
        "priority_resistance_bottom": round(float(last["priority_resistance_bottom"]), 6) if pd.notna(last["priority_resistance_bottom"]) else np.nan,
        "latest_bar_signals": ",".join(current_signals),
        "last_signal_type": last_signal_type,
        "last_signal_date": last_signal_date,
        "last_signal_zone_id": last_signal_zone_id,
        "support_zone_total": len(active_support),
        "resistance_zone_total": len(active_resistance),
    }


def make_chart_candidates(summary_df: pd.DataFrame, charts_limit: int) -> list[str]:
    if summary_df.empty or charts_limit <= 0:
        return []

    work = summary_df.copy()
    work["has_latest_signal"] = work["latest_bar_signals"].astype(str).str.len() > 0
    work["signal_priority"] = work["latest_bar_signals"].fillna("").apply(lambda x: 0 if x == "" else len(str(x).split(",")))
    work = work.sort_values(
        ["has_latest_signal", "signal_priority", "active_support_count", "active_resistance_count", "ticker"],
        ascending=[False, False, False, False, True],
    )
    return work["ticker"].head(charts_limit).tolist()


def draw_zone_rectangles(ax, zones_df: pd.DataFrame, visible_start: pd.Timestamp, visible_end: pd.Timestamp, kind: str, limit: int) -> None:
    if zones_df.empty:
        return

    sub = zones_df[zones_df["zone_kind"] == kind].copy()
    if sub.empty:
        return

    sub["created_date"] = pd.to_datetime(sub["created_date"])
    sub["invalidated_date_clean"] = pd.to_datetime(sub["invalidated_date"].replace("", pd.NA), errors="coerce")

    zone_rows = []
    for _, row in sub.iterrows():
        end_date = row["invalidated_date_clean"] if pd.notna(row["invalidated_date_clean"]) else visible_end
        if end_date >= visible_start and row["created_date"] <= visible_end:
            zone_rows.append(row)
    zone_rows = sorted(zone_rows, key=lambda r: (bool(r["active"]), pd.Timestamp(r["created_date"])), reverse=True)[:limit]

    for row in zone_rows:
        start = max(row["created_date"], visible_start)
        end = min(row["invalidated_date_clean"] if pd.notna(row["invalidated_date_clean"]) else visible_end, visible_end)
        x0 = mdates.date2num(start)
        x1 = mdates.date2num(end)
        if x1 <= x0:
            x1 = x0 + 1
        width = x1 - x0
        rect = patches.Rectangle(
            (x0, float(row["zone_bottom"])),
            width,
            float(row["zone_top"]) - float(row["zone_bottom"]),
            linewidth=0.8,
            fill=True,
            alpha=0.14 if kind == "support" else 0.10,
        )
        ax.add_patch(rect)


def save_chart(df: pd.DataFrame, zones_df: pd.DataFrame, ticker: str, output_path: Path, cfg: ScanConfig) -> None:
    vis = df.tail(cfg.chart_bars).copy()
    if vis.empty:
        return

    visible_start = vis.index[0]
    visible_end = vis.index[-1]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(vis.index, vis["Close"], label="Close", linewidth=1.4)
    ax.plot(vis.index, vis["EMA50"], label="EMA50", linewidth=1.0)
    ax.plot(vis.index, vis["EMA200"], label="EMA200", linewidth=1.0)

    draw_zone_rectangles(ax, zones_df, visible_start, visible_end, "support", cfg.max_chart_zones_per_side)
    draw_zone_rectangles(ax, zones_df, visible_start, visible_end, "resistance", cfg.max_chart_zones_per_side)

    marker_specs = {"S1": "^", "S2": "P", "R1": "v", "R2": "X"}
    for signal_name, marker in marker_specs.items():
        mask = vis[signal_name].astype(bool)
        if mask.any():
            ax.scatter(vis.index[mask], vis.loc[mask, "Close"], marker=marker, s=70, label=signal_name)

    ax.set_title(f"{ticker} — Multi-zone Support / Resistance Scan")
    ax.set_ylabel("Price")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_window_bounds(start_date: str, end_date: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    start_ts = pd.Timestamp(start_date) if start_date else None
    end_ts = pd.Timestamp(end_date) if end_date else None
    return start_ts, end_ts


def format_date_column(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    return dt.dt.strftime("%Y-%m-%d").fillna("")


def build_signal_window_hits(events_df: pd.DataFrame, zones_df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    columns = [
        "ticker",
        "signal_date",
        "signal_type",
        "box_type",
        "box_start_date",
        "box_end_date",
        "zone_id",
        "zone_kind",
        "zone_pivot_date",
    ]
    if events_df.empty:
        return pd.DataFrame(columns=columns)

    work = events_df.copy()
    work["signal_date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["signal_date"])

    start_ts, end_ts = parse_window_bounds(start_date, end_date)
    if start_ts is not None:
        work = work[work["signal_date"] >= start_ts]
    if end_ts is not None:
        work = work[work["signal_date"] <= end_ts]

    if work.empty:
        return pd.DataFrame(columns=columns)

    if not zones_df.empty:
        zone_meta = zones_df[["ticker", "zone_id", "created_date", "invalidated_date"]].copy()
        zone_meta = zone_meta.rename(columns={"created_date": "box_start_date", "invalidated_date": "box_end_date"})
        work = work.merge(zone_meta, on=["ticker", "zone_id"], how="left")
    else:
        work["box_start_date"] = ""
        work["box_end_date"] = ""

    work["box_type"] = np.where(work["zone_kind"] == "support", "supportBox", "resistanceBox")
    work["signal_date"] = format_date_column(work["signal_date"])
    work["box_start_date"] = format_date_column(work["box_start_date"])
    work["box_end_date"] = format_date_column(work["box_end_date"])
    work["zone_pivot_date"] = format_date_column(work["zone_pivot_date"])

    work = work.sort_values(["signal_date", "ticker", "signal_type", "zone_id"], ascending=[True, True, True, True])
    return work[columns].reset_index(drop=True)


def build_structure_window_hits(zones_df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    columns = [
        "ticker",
        "box_type",
        "box_start_date",
        "box_end_date",
        "zone_id",
        "zone_kind",
        "zone_pivot_date",
        "zone_price",
        "zone_top",
        "zone_bottom",
        "active",
    ]
    if zones_df.empty:
        return pd.DataFrame(columns=columns)

    work = zones_df.copy()

    # Compatibility guard: zone catalogs store the pivot anchor as `pivot_price`,
    # while some downstream exports refer to the same field as `zone_price`.
    # Normalize to `zone_price` so structure exports don't fail when the source
    # dataframe was built with the older column name.
    if "zone_price" not in work.columns and "pivot_price" in work.columns:
        work["zone_price"] = work["pivot_price"]
    elif "zone_price" not in work.columns:
        work["zone_price"] = np.nan

    work["box_start_date"] = pd.to_datetime(work["created_date"], errors="coerce")
    work["box_end_date"] = pd.to_datetime(work["invalidated_date"].replace("", pd.NA), errors="coerce")
    work["zone_pivot_date"] = pd.to_datetime(work["pivot_date"], errors="coerce")

    start_ts, end_ts = parse_window_bounds(start_date, end_date)
    if start_ts is not None:
        work = work[(work["box_end_date"].isna()) | (work["box_end_date"] >= start_ts)]
    if end_ts is not None:
        work = work[work["box_start_date"] <= end_ts]

    if work.empty:
        return pd.DataFrame(columns=columns)

    work["box_type"] = np.where(work["zone_kind"] == "support", "supportBox", "resistanceBox")
    work["box_start_date"] = format_date_column(work["box_start_date"])
    work["box_end_date"] = format_date_column(work["box_end_date"])
    work["zone_pivot_date"] = format_date_column(work["zone_pivot_date"])
    work = work.sort_values(["box_start_date", "ticker", "box_type", "zone_id"], ascending=[True, True, True, True])
    return work[columns].reset_index(drop=True)


def build_window_scan_results(signal_window_hits: pd.DataFrame, structure_window_hits: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "ticker",
        "row_type",
        "item_type",
        "signal_date",
        "box_start_date",
        "box_end_date",
        "zone_id",
        "zone_kind",
        "zone_pivot_date",
    ]

    frames: list[pd.DataFrame] = []

    if not signal_window_hits.empty:
        sig = signal_window_hits.copy()
        sig["row_type"] = "signal"
        sig["item_type"] = sig["signal_type"]
        frames.append(sig[["ticker", "row_type", "item_type", "signal_date", "box_start_date", "box_end_date", "zone_id", "zone_kind", "zone_pivot_date"]])

    if not structure_window_hits.empty:
        box = structure_window_hits.copy()
        box["row_type"] = "structure"
        box["item_type"] = box["box_type"]
        box["signal_date"] = ""
        frames.append(box[["ticker", "row_type", "item_type", "signal_date", "box_start_date", "box_end_date", "zone_id", "zone_kind", "zone_pivot_date"]])

    if not frames:
        return pd.DataFrame(columns=columns)

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["ticker", "row_type", "signal_date", "box_start_date", "item_type", "zone_id"], ascending=[True, True, True, True, True, True]).reset_index(drop=True)
    return out[columns]


def write_markdown_summary(summary_df: pd.DataFrame, universe_meta: pd.DataFrame, output_path: Path, run_meta: dict) -> None:
    lines = [
        f"# Weekly Support / Resistance Scan — {run_meta['universe']}",
        "",
        f"- Universe: **{run_meta['universe']}**",
        f"- Period / interval: **{run_meta['period']} / {run_meta['interval']}**",
        f"- Requested tickers: **{run_meta['requested_tickers']}**",
        f"- Scanned successfully: **{len(summary_df)}**",
        f"- Run date (UTC): **{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M:%S')}**",
        f"- Signal export window: **{run_meta.get('signal_start_date') or '-'} ~ {run_meta.get('signal_end_date') or '-'}**",
        f"- Signal export rows: **{run_meta.get('signal_window_count', 0)}**",
        f"- Structure export rows: **{run_meta.get('structure_window_count', 0)}**",
        "",
        "## Latest-bar signals",
        "",
    ]

    latest_hits = summary_df[summary_df["latest_bar_signals"].astype(str) != ""].copy() if not summary_df.empty else pd.DataFrame()
    if latest_hits.empty:
        lines.append("No S1/S2/R1/R2 signals on the latest completed bar.")
    else:
        for _, row in latest_hits.sort_values(["latest_bar_signals", "ticker"], ascending=[False, True]).iterrows():
            lines.append(
                f"- **{row['ticker']}** | signals: `{row['latest_bar_signals']}` | close: `{row['close']}` | "
                f"supports: `{row['active_support_count']}` | resistances: `{row['active_resistance_count']}`"
            )

    lines.extend(["", "## Filtered signal window hits", ""])
    signal_window_hits_path = output_path.parent / "signal_window_hits.csv"
    if signal_window_hits_path.exists():
        signal_window_hits = pd.read_csv(signal_window_hits_path)
        if signal_window_hits.empty:
            lines.append("No S1/S2/R1/R2 hits in the requested signal window.")
        else:
            for _, row in signal_window_hits.iterrows():
                lines.append(
                    f"- **{row['ticker']}** | `{row['signal_type']}` | signal: `{row['signal_date']}` | "
                    f"box: `{row['box_type']}` | box start: `{row['box_start_date']}` | box end: `{row['box_end_date'] or '-'}`"
                )
    else:
        lines.append("signal_window_hits.csv not found.")

    lines.extend(["", "## Filtered structure window hits", ""])
    structure_window_hits_path = output_path.parent / "structure_window_hits.csv"
    if structure_window_hits_path.exists():
        structure_window_hits = pd.read_csv(structure_window_hits_path)
        if structure_window_hits.empty:
            lines.append("No supportBox / resistanceBox records overlap the requested window.")
        else:
            for _, row in structure_window_hits.iterrows():
                lines.append(
                    f"- **{row['ticker']}** | `{row['box_type']}` | box start: `{row['box_start_date']}` | "
                    f"box end: `{row['box_end_date'] or '-'}`"
                )
    else:
        lines.append("structure_window_hits.csv not found.")

    lines.extend(["", "## Last signal by ticker", ""])
    if summary_df.empty:
        lines.append("No successful ticker scans.")
    else:
        for _, row in summary_df.sort_values("ticker").iterrows():
            last_sig = row["last_signal_type"] if row["last_signal_type"] else "-"
            last_date = row["last_signal_date"] if row["last_signal_date"] else "-"
            lines.append(f"- **{row['ticker']}** → last signal: `{last_sig}` on `{last_date}`")

    if not universe_meta.empty and "rank_by_market_cap" in universe_meta.columns:
        lines.extend(["", "## S&P 500 market-cap selection snapshot", ""])
        for _, row in universe_meta.head(20).iterrows():
            lines.append(f"- `{int(row['rank_by_market_cap'])}` **{row['ticker']}** | market cap: `{int(row['market_cap'])}`")

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ==============================
# Main execution
# ==============================


def main() -> None:
    args = parse_args()
    cfg = ScanConfig(
        pivot_len=args.pivot_len,
        rsi_len=args.rsi_len,
        rsi_s1=args.rsi_s1,
        rsi_s2=args.rsi_s2,
        rsi_r1=args.rsi_r1,
        rsi_r2=args.rsi_r2,
        vol_len=args.vol_len,
        atr_len=args.atr_len,
        atr_mult=args.atr_mult,
        ema_fast_len=args.ema_fast_len,
        ema_slow_len=args.ema_slow_len,
        s1_vol_mult=args.s1_vol_mult,
        s2_vol_mult=args.s2_vol_mult,
        r1_vol_mult=args.r1_vol_mult,
        r2_vol_mult=args.r2_vol_mult,
        confirm_window_bars=args.confirm_window_bars,
        use_candle_filter=not args.disable_candle_filter,
        doji_body_ratio=args.doji_body_ratio,
        wick_body_ratio=args.wick_body_ratio,
        allow_s2_in_bear=args.allow_s2_in_bear,
        allow_r2_in_bull=not args.disable_r2_in_bull,
        chart_bars=args.chart_bars,
        max_chart_zones_per_side=args.max_chart_zones_per_side,
    )

    tickers, universe_meta = resolve_universe(args)
    requested_tickers = len(tickers)
    print(f"[INFO] Universe resolved: {args.universe} with {requested_tickers} tickers", flush=True)

    output_dir = Path(args.output_dir)
    charts_dir = output_dir / "charts"
    full_scans_dir = output_dir / "full_scans"
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    if args.full_scan_limit > 0:
        full_scans_dir.mkdir(parents=True, exist_ok=True)

    universe_meta.to_csv(output_dir / "universe_membership.csv", index=False)

    histories, download_errors = fetch_histories_batch(
        tickers=tickers,
        period=args.period,
        interval=args.interval,
        chunk_size=args.chunk_size,
        pause_seconds=args.pause_seconds,
    )

    summary_rows: list[dict] = []
    all_events: list[pd.DataFrame] = []
    all_zones: list[pd.DataFrame] = []
    error_rows: list[dict] = list(download_errors)

    min_bars_needed = max(cfg.ema_slow_len, cfg.atr_len, cfg.vol_len, cfg.rsi_len) + (2 * cfg.pivot_len) + 5

    successful_tickers: list[str] = []
    for ticker in tickers:
        raw = histories.get(ticker)
        if raw is None or raw.empty:
            continue
        try:
            if len(raw) < min_bars_needed:
                raise ValueError(f"Not enough bars. Needed at least {min_bars_needed}, got {len(raw)}")

            enriched = enrich_indicators(raw, cfg)
            scanned, events_df, zones_df = apply_signal_logic(enriched, ticker=ticker, cfg=cfg)

            summary_rows.append(build_summary_row(scanned, events_df, zones_df, ticker))
            if not events_df.empty:
                all_events.append(events_df)
            if not zones_df.empty:
                all_zones.append(zones_df)
            successful_tickers.append(ticker)

            if args.full_scan_limit > 0 and len(successful_tickers) <= args.full_scan_limit:
                scanned.to_csv(full_scans_dir / f"{ticker}_full_scan.csv", index_label="date")
        except Exception as exc:
            error_rows.append({"ticker": ticker, "error": str(exc)})
            print(f"[ERROR] {ticker}: {exc}", flush=True)

    summary_df = pd.DataFrame(summary_rows)
    events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    zones_df = pd.concat(all_zones, ignore_index=True) if all_zones else pd.DataFrame()
    errors_df = pd.DataFrame(error_rows)

    if not summary_df.empty:
        signal_sort = summary_df["latest_bar_signals"].fillna("").apply(lambda x: 0 if x == "" else len(str(x).split(",")))
        summary_df = summary_df.assign(_signal_score=signal_sort).sort_values(
            ["_signal_score", "active_support_count", "active_resistance_count", "ticker"],
            ascending=[False, False, False, True],
        ).drop(columns=["_signal_score"]).reset_index(drop=True)

    # Chart generation after summary ranking
    chart_tickers = make_chart_candidates(summary_df, args.charts_limit)
    if chart_tickers:
        zone_map = {ticker: zones_df[zones_df["ticker"] == ticker].copy() for ticker in chart_tickers if not zones_df.empty}
        for ticker in chart_tickers:
            try:
                raw = histories.get(ticker)
                if raw is None or raw.empty:
                    continue
                enriched = enrich_indicators(raw, cfg)
                scanned, _, _ = apply_signal_logic(enriched, ticker=ticker, cfg=cfg)
                save_chart(scanned, zone_map.get(ticker, pd.DataFrame()), ticker, charts_dir / f"{ticker}.png", cfg)
            except Exception as exc:
                error_rows.append({"ticker": ticker, "error": f"chart failed: {exc}"})
                print(f"[ERROR] Chart {ticker}: {exc}", flush=True)

    summary_df.to_csv(output_dir / "latest_summary.csv", index=False)
    events_df.to_csv(output_dir / "all_signal_events.csv", index=False)
    zones_df.to_csv(output_dir / "zones_catalog.csv", index=False)
    errors_df = pd.DataFrame(error_rows)
    errors_df.to_csv(output_dir / "errors.csv", index=False)

    if not summary_df.empty:
        latest_hits = summary_df[summary_df["latest_bar_signals"].astype(str) != ""]
        latest_hits.to_csv(output_dir / "latest_bar_signal_hits.csv", index=False)
    else:
        pd.DataFrame().to_csv(output_dir / "latest_bar_signal_hits.csv", index=False)

    signal_window_hits = build_signal_window_hits(events_df, zones_df, args.signal_start_date, args.signal_end_date)
    structure_window_hits = build_structure_window_hits(zones_df, args.signal_start_date, args.signal_end_date)
    window_scan_results = build_window_scan_results(signal_window_hits, structure_window_hits)

    signal_window_hits.to_csv(output_dir / "signal_window_hits.csv", index=False)
    structure_window_hits.to_csv(output_dir / "structure_window_hits.csv", index=False)
    window_scan_results.to_csv(output_dir / "window_scan_results.csv", index=False)

    write_markdown_summary(
        summary_df,
        universe_meta,
        output_dir / "summary.md",
        run_meta={
            "universe": args.universe,
            "period": args.period,
            "interval": args.interval,
            "requested_tickers": requested_tickers,
            "signal_start_date": args.signal_start_date,
            "signal_end_date": args.signal_end_date,
            "signal_window_count": len(signal_window_hits),
            "structure_window_count": len(structure_window_hits),
        },
    )

    print(f"[INFO] Done. Successful scans: {len(summary_df)} / {requested_tickers}", flush=True)
    print(f"[INFO] Results written to: {output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[ERROR] Interrupted by user", file=sys.stderr)
        raise
