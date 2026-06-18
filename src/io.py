"""Load client backtest input (ar_bt.xlsx format)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class BacktestInput:
    """Parsed OHLC + signals ready for VectorBT."""

    df: pd.DataFrame
    long_entries: np.ndarray
    long_exits: np.ndarray
    short_entries: np.ndarray
    short_exits: np.ndarray
    entry_price: np.ndarray
    exit_price: np.ndarray
    stop_loss_long: np.ndarray
    stop_loss_short: np.ndarray


COLUMN_MAP = {
    "date": "Time",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
}


def _parse_datetime(series: pd.Series) -> pd.DatetimeIndex:
    s = series.astype(str).str.strip()
    dt = pd.to_datetime(s, format="%Y.%m.%d %H:%M", errors="coerce")
    if dt.isna().any():
        dt = pd.to_datetime(s, errors="coerce")
    if dt.isna().any():
        raise ValueError(f"Failed to parse {dt.isna().sum()} date values")
    return pd.DatetimeIndex(dt)


def _signal_mask(series: pd.Series, token: str) -> np.ndarray:
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        return series.astype(str).str.strip().str.upper().eq(token.upper()).to_numpy(dtype=bool)
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(bool).to_numpy()


def load_ar_bt_xlsx(
    path: str | Path,
    *,
    sheet_name: str = "data",
    start_date: str | None = None,
    end_date: str | None = None,
    cache_parquet: bool = True,
) -> BacktestInput:
    """
    Load ar_bt.xlsx (sheet 'data').

    Columns: date, open, high, low, close,
              entry_buy, exit_buy, entry_sell, exit_sell,
              stoploss_buy, stoploss_sell, takeprofit_buy, takeprofit_sell,
              tsl_buy, tsl_sell
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    cache_path = path.with_suffix(".pkl")
    if cache_parquet and cache_path.exists() and cache_path.stat().st_mtime >= path.stat().st_mtime:
        raw = pd.read_pickle(cache_path)
    else:
        raw = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
        if cache_parquet:
            raw.to_pickle(cache_path)

    raw = raw.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in raw.columns})

    required = ["Time", "Open", "High", "Low", "Close"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    raw["Time"] = _parse_datetime(raw["Time"])
    raw = raw.sort_values("Time").drop_duplicates(subset=["Time"], keep="last")
    raw = raw.set_index("Time")

    if start_date:
        raw = raw[raw.index >= pd.Timestamp(start_date)]
    if end_date:
        raw = raw[raw.index <= pd.Timestamp(end_date)]

    if raw.empty:
        raise ValueError("No rows after date filtering")

    n = len(raw)
    close = raw["Close"].to_numpy(dtype=np.float64)

    long_entries = _signal_mask(raw.get("entry_buy", pd.Series(False, index=raw.index)), "BUY")
    long_exits = _signal_mask(raw.get("exit_buy", pd.Series(False, index=raw.index)), "EXIT")
    short_entries = _signal_mask(raw.get("entry_sell", pd.Series(False, index=raw.index)), "SELL")
    short_exits = _signal_mask(raw.get("exit_sell", pd.Series(False, index=raw.index)), "EXIT")

    entry_price = np.full(n, np.nan, dtype=np.float64)
    exit_price = np.full(n, np.nan, dtype=np.float64)

    entry_price[long_entries | short_entries] = close[long_entries | short_entries]
    exit_price[long_exits | short_exits] = close[long_exits | short_exits]

    stop_loss_long = raw.get("stoploss_buy", pd.Series(np.nan, index=raw.index)).to_numpy(dtype=np.float64)
    stop_loss_short = raw.get("stoploss_sell", pd.Series(np.nan, index=raw.index)).to_numpy(dtype=np.float64)

    return BacktestInput(
        df=raw,
        long_entries=long_entries,
        long_exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        entry_price=entry_price,
        exit_price=exit_price,
        stop_loss_long=stop_loss_long,
        stop_loss_short=stop_loss_short,
    )
