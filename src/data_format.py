"""Shared OHLC CSV format for fetch, backtest, and optimization."""

from pathlib import Path

import pandas as pd

OHLC_COLUMNS = ["Time", "Open", "High", "Low", "Close"]
OPTIONAL_COLUMNS = ["Volume"]
REQUIRED_PRICE_COLUMNS = ["Open", "High", "Low", "Close"]


def build_csv_stem(symbol: str, timeframe: str) -> str:
    """File stem used across the project, e.g. XAUUSD_1min."""
    symbol = symbol.strip().upper()
    timeframe = timeframe.strip().lower()
    return f"{symbol}_{timeframe}"


def build_csv_filename(symbol: str, timeframe: str) -> str:
    return f"{build_csv_stem(symbol, timeframe)}.csv"


def timeframe_label_from_fetch_config(config: dict) -> str:
    """
    Resolve output timeframe label for filenames (must match backtesting config).

    Prefer explicit TIMEFRAME. If TO_DAILY is true, defaults to 'daily'.
    Otherwise derive from BAR_INTERVAL + BAR_TYPE (e.g. 1 + minute -> 1min).
    """
    if config.get("TIMEFRAME"):
        return str(config["TIMEFRAME"]).strip().lower()

    if config.get("TO_DAILY"):
        return "daily"

    interval = int(config["BAR_INTERVAL"])
    bar_type = str(config["BAR_TYPE"]).strip().lower()

    if bar_type == "minute":
        return "1min" if interval == 1 else f"{interval}min"
    if bar_type == "hour":
        return "1h" if interval == 1 else f"{interval}h"
    if bar_type == "day":
        return "daily"

    return f"{interval}{bar_type}"


def normalize_ohlc_dataframe(df: pd.DataFrame, *, to_daily: bool = False) -> pd.DataFrame:
    """
    Standardize InsightSentry / CSV data to project format.

    - Columns: Time, Open, High, Low, Close [, Volume]
    - Time: UTC datetime, sorted ascending, duplicates removed (last wins)
    """
    rename_map = {
        "time": "Time",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    out = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}).copy()

    if "Time" not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex) or out.index.name in ("Time", "time"):
            out = out.reset_index()
            if out.columns[0] != "Time":
                out = out.rename(columns={out.columns[0]: "Time"})

    if "Time" not in out.columns:
        raise ValueError("DataFrame must have a Time column or datetime index")

    if pd.api.types.is_numeric_dtype(out["Time"]):
        out["Time"] = pd.to_datetime(out["Time"], unit="s", utc=True)
    else:
        out["Time"] = pd.to_datetime(out["Time"], utc=True)
    out = out.sort_values("Time")

    for col in REQUIRED_PRICE_COLUMNS:
        if col not in out.columns:
            raise ValueError(f"Missing required column: {col}")
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if "Volume" in out.columns:
        out["Volume"] = pd.to_numeric(out["Volume"], errors="coerce")

    out = out.dropna(subset=REQUIRED_PRICE_COLUMNS)
    out = out.drop_duplicates(subset=["Time"], keep="last")

    if to_daily:
        out = out.set_index("Time")
        daily = out.resample("1D").agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                **({"Volume": "sum"} if "Volume" in out.columns else {}),
            }
        ).dropna(subset=REQUIRED_PRICE_COLUMNS)
        out = daily.reset_index()

    cols = [c for c in OHLC_COLUMNS + OPTIONAL_COLUMNS if c in out.columns]
    return out[cols].reset_index(drop=True)


def save_ohlc_csv(df: pd.DataFrame, path: str | Path, *, to_daily: bool = False) -> Path:
    """Write standardized OHLC CSV (Time as column)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_ohlc_dataframe(df, to_daily=to_daily)
    normalized.to_csv(path, index=False)
    return path


def read_ohlc_csv(path: str | Path) -> pd.DataFrame:
    """Read project OHLC CSV into normalized in-memory format (Time column, UTC)."""
    df = pd.read_csv(path)
    return normalize_ohlc_dataframe(df)
