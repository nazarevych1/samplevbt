import os
from pathlib import Path

from src.warnings_config import configure_warnings

configure_warnings()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.io as pio
import vectorbt as vbt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from src.data_format import build_csv_stem, read_ohlc_csv


def _bool_series(series: pd.Series) -> pd.Series:
    """Coerce to bool; NaN becomes False (no pandas downcast warning)."""
    return series.where(series.notna(), False).astype(bool)


def _shift_no_lookahead(events: pd.Series) -> pd.Series:
    """Shift one HTF bar; incomplete bar -> False."""
    return _bool_series(events.shift(1))


def resolve_data_file(data_dir: str | Path, symbol: str, timeframe: str) -> Path:
    """Find CSV: {SYMBOL}_{timeframe}.csv (e.g. XAUUSD_1min.csv, XAUUSD_daily.csv)."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir.resolve()}")

    symbol = symbol.strip().upper()
    timeframe = timeframe.strip().lower()
    if not symbol:
        raise ValueError("SYMBOL must not be empty")
    if not timeframe:
        raise ValueError("timeframe must not be empty")

    expected_stem = build_csv_stem(symbol, timeframe)
    csv_files = sorted(data_dir.glob("*.csv"))
    matches = [f for f in csv_files if f.stem.upper() == expected_stem.upper()]

    if not matches:
        symbol_files = [f.name for f in csv_files if f.stem.upper().startswith(f"{symbol}_")]
        if symbol_files:
            raise FileNotFoundError(
                f"No CSV for '{expected_stem}.csv' in '{data_dir.resolve()}'. "
                f"Files for {symbol}: {', '.join(symbol_files)}"
            )
        available = [f.name for f in csv_files]
        if available:
            raise FileNotFoundError(
                f"No CSV for '{expected_stem}.csv'. Available: {', '.join(available)}"
            )
        raise FileNotFoundError(
            f"No CSV files in '{data_dir.resolve()}'. Cannot load {expected_stem}.csv"
        )

    if len(matches) > 1:
        names = ", ".join(f.name for f in matches)
        raise FileNotFoundError(f"Multiple files match '{expected_stem}': {names}")

    return matches[0]


def _read_ohlc_csv(path: Path, config: dict) -> pd.DataFrame:
    df = read_ohlc_csv(path).set_index("Time")

    if config["CONVERT_TO_NY_TIMEZONE"]:
        df.index = df.index.tz_convert("America/New_York")
    else:
        df.index = pd.to_datetime(df.index, utc=True)

    return df[config["BACKTESTING_START_DATE"] : config["BACKTESTING_END_DATE"]]


def load_backtest_datasets(
    config: dict, data_dir: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    """
    Load execution (low TF) and signals (high TF) datasets for the same symbol.
    """
    symbol = config["SYMBOL"]
    execution_tf = config["EXECUTION_TIMEFRAME"]
    signals_tf = config["SIGNALS_TIMEFRAME"]

    execution_path = resolve_data_file(data_dir, symbol, execution_tf)
    signals_path = resolve_data_file(data_dir, symbol, signals_tf)

    execution_df = _read_ohlc_csv(execution_path, config)
    signals_df = _read_ohlc_csv(signals_path, config)

    if execution_df.empty:
        raise ValueError(f"Execution dataset is empty: {execution_path.name}")
    if signals_df.empty:
        raise ValueError(f"Signals dataset is empty: {signals_path.name}")

    return execution_df, signals_df, execution_path, signals_path


def map_htf_events_to_ltf(
    htf_events: pd.Series, ltf_index: pd.DatetimeIndex
) -> np.ndarray:
    """
    Map sparse HTF event flags onto LTF index.
    htf_events must already be shifted (no lookahead) — each True marks an HTF bar
    where the event is known after that bar closes. The event is applied on the
    first LTF bar at or after that timestamp.
    """
    htf_events = _bool_series(htf_events)
    ltf_flags = np.zeros(len(ltf_index), dtype=bool)

    for ts in htf_events.index[htf_events]:
        pos = ltf_index.searchsorted(ts, side="left")
        if pos < len(ltf_index):
            ltf_flags[pos] = True

    return ltf_flags


def generate_sma_cross_signals_htf(
    signals_df: pd.DataFrame,
    sma_fast: int,
    sma_slow: int,
    allow_short: bool,
) -> dict[str, pd.Series]:
    """SMA crossover signals on the signals (higher) timeframe."""
    if sma_fast >= sma_slow:
        raise ValueError(
            f"SMA_FAST ({sma_fast}) must be less than SMA_SLOW ({sma_slow})."
        )

    close = signals_df["Close"]
    fast_ma = close.rolling(window=sma_fast, min_periods=sma_fast).mean()
    slow_ma = close.rolling(window=sma_slow, min_periods=sma_slow).mean()

    bullish_cross = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
    bearish_cross = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))

    long_entries = _bool_series(bullish_cross)
    long_exits = _bool_series(bearish_cross)

    if allow_short:
        short_entries = _bool_series(bearish_cross)
        short_exits = _bool_series(bullish_cross)
    else:
        short_entries = pd.Series(False, index=signals_df.index)
        short_exits = pd.Series(False, index=signals_df.index)

    # Use only completed HTF bars (avoid lookahead on LTF execution).
    long_entries = _shift_no_lookahead(long_entries)
    long_exits = _shift_no_lookahead(long_exits)
    short_entries = _shift_no_lookahead(short_entries)
    short_exits = _shift_no_lookahead(short_exits)

    return {
        "long_entries": long_entries,
        "long_exits": long_exits,
        "short_entries": short_entries,
        "short_exits": short_exits,
    }


def map_sma_signals_to_execution(
    execution_df: pd.DataFrame,
    htf_signals: dict[str, pd.Series],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project HTF SMA signals onto the execution timeframe index."""
    ltf_index = execution_df.index

    return (
        map_htf_events_to_ltf(htf_signals["long_entries"], ltf_index),
        map_htf_events_to_ltf(htf_signals["long_exits"], ltf_index),
        map_htf_events_to_ltf(htf_signals["short_entries"], ltf_index),
        map_htf_events_to_ltf(htf_signals["short_exits"], ltf_index),
    )


def save_backtesting_results(pf, output_dir: str = "output"):
    os.makedirs(output_dir, exist_ok=True)

    stats_df = pf.stats().to_frame()

    with PdfPages(f"{output_dir}/portfolio_report.pdf") as pdf:
        fig, ax = plt.subplots(figsize=(8.5, len(stats_df) * 0.4))
        ax.axis("off")
        table = ax.table(
            cellText=stats_df.values,
            colLabels=stats_df.columns,
            rowLabels=stats_df.index,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

    fig = pf.plot()
    pio.write_html(fig, file=f"{output_dir}/portfolio_plot.html", auto_open=False)


def backtest_with_vectorbt(
    long_entries_arr: np.ndarray,
    long_exits_arr: np.ndarray,
    short_entries_arr: np.ndarray,
    short_exits_arr: np.ndarray,
    df: pd.DataFrame,
    init_balance: float,
    fees: float,
    slippage: float,
    size: float,
    size_type: str,
    freq: str = "1min",
):
    return vbt.Portfolio.from_signals(
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        entries=long_entries_arr,
        exits=long_exits_arr,
        short_entries=short_entries_arr,
        short_exits=short_exits_arr,
        size=size,
        size_type=size_type,
        fees=fees,
        slippage=slippage,
        init_cash=init_balance,
        freq=freq,
    )


def backtest_strategy(
    execution_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    config: dict,
):
    htf_signals = generate_sma_cross_signals_htf(
        signals_df=signals_df,
        sma_fast=int(config["SMA_FAST"]),
        sma_slow=int(config["SMA_SLOW"]),
        allow_short=config.get("ALLOW_SHORT", True),
    )

    long_entries, long_exits, short_entries, short_exits = map_sma_signals_to_execution(
        execution_df, htf_signals
    )

    freq = config.get("EXECUTION_FREQ") or config.get("DATA_FREQ", "1min")

    return backtest_with_vectorbt(
        long_entries_arr=long_entries,
        long_exits_arr=long_exits,
        short_entries_arr=short_entries,
        short_exits_arr=short_exits,
        df=execution_df,
        init_balance=config["INIT_BALANCE"],
        fees=config["FEES"],
        slippage=config["SLIPPAGE"],
        freq=freq,
        size=config["POSITION_SIZE"],
        size_type=config["POSITION_SIZE_TYPE"],
    )


def optimize(
    execution_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    config: dict,
) -> list:
    results = []

    step = config["OPTIMIZATION_STEP"]
    start = config["OPTIMIZATION_START"]
    end = config["OPTIMIZATION_END"]

    n1_values = np.arange(start, end + step, step)
    n2_values = np.arange(start, end + step, step)

    new_config = config.copy()

    for n1 in tqdm(n1_values):
        n1 = int(n1)
        for n2 in tqdm(n2_values, leave=False):
            n2 = int(n2)

            if n1 >= n2:
                continue

            new_config["SMA_FAST"] = n1
            new_config["SMA_SLOW"] = n2

            pf = backtest_strategy(
                execution_df=execution_df,
                signals_df=signals_df,
                config=new_config,
            )

            pf_stats = pf.stats().to_dict()
            pf_stats["Total Return [%]"] = float(pf_stats.get("Total Return [%]", 0) or 0)
            pf_stats["Open Trade PnL"] = float(pf_stats.get("Open Trade PnL", 0) or 0)
            pf_stats["End Value"] = float(pf_stats.get("End Value", 1) or 1)

            pf_stats["Total Return [%]"] = (
                0 if np.isnan(pf_stats["Total Return [%]"]) else pf_stats["Total Return [%]"]
            )
            pf_stats["Open Trade PnL"] = (
                0 if np.isnan(pf_stats["Open Trade PnL"]) else pf_stats["Open Trade PnL"]
            )
            pf_stats["Total Return [%]"] = pf_stats["Total Return [%]"] + (
                pf_stats["Open Trade PnL"] / pf_stats["End Value"] * 100
            )

            stats = {"N1": n1, "N2": n2}
            stats.update(pf_stats)
            results.append(stats)

    return results
