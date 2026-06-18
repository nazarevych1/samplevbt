"""VectorBT backtest from pre-built signal arrays."""

from __future__ import annotations

import numpy as np
import vectorbt as vbt

from src.io import BacktestInput


def _combined_price(entry_price: np.ndarray, exit_price: np.ndarray) -> np.ndarray:
    price = np.full(len(entry_price), np.nan, dtype=np.float64)
    price[~np.isnan(entry_price)] = entry_price[~np.isnan(entry_price)]
    price[~np.isnan(exit_price)] = exit_price[~np.isnan(exit_price)]
    return price


def _merge_stop_loss_exits(
    high: np.ndarray,
    low: np.ndarray,
    long_entries: np.ndarray,
    long_exits: np.ndarray,
    short_entries: np.ndarray,
    short_exits: np.ndarray,
    stop_loss_long: np.ndarray,
    stop_loss_short: np.ndarray,
    exit_price: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Close a position when either an exit signal fires or price crosses the stop.

    Long:  Low <= stoploss_buy
    Short: High >= stoploss_sell
    """
    n = len(high)
    long_out = long_exits.copy()
    short_out = short_exits.copy()
    price_out = exit_price.copy()

    in_long = False
    in_short = False

    for i in range(n):
        if long_entries[i]:
            in_long = True

        if in_long:
            sl = stop_loss_long[i]
            sl_hit = np.isfinite(sl) and low[i] <= sl
            if sl_hit or long_exits[i]:
                long_out[i] = True
                if sl_hit and not long_exits[i]:
                    price_out[i] = sl
                in_long = False

        if short_entries[i]:
            in_short = True

        if in_short:
            sl = stop_loss_short[i]
            sl_hit = np.isfinite(sl) and high[i] >= sl
            if sl_hit or short_exits[i]:
                short_out[i] = True
                if sl_hit and not short_exits[i]:
                    price_out[i] = sl
                in_short = False

    return long_out, short_out, price_out


def run_backtest(data: BacktestInput, config: dict) -> vbt.Portfolio:
    df = data.df
    freq = config.get("FREQ", "3min")

    long_exits = data.long_exits
    short_exits = data.short_exits
    exit_price = data.exit_price

    if config.get("USE_STOP_LOSS", False):
        long_exits, short_exits, exit_price = _merge_stop_loss_exits(
            high=df["High"].to_numpy(dtype=np.float64),
            low=df["Low"].to_numpy(dtype=np.float64),
            long_entries=data.long_entries,
            long_exits=data.long_exits,
            short_entries=data.short_entries,
            short_exits=data.short_exits,
            stop_loss_long=data.stop_loss_long,
            stop_loss_short=data.stop_loss_short,
            exit_price=data.exit_price,
        )

    price = _combined_price(data.entry_price, exit_price)

    return vbt.Portfolio.from_signals(
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        price=price,
        entries=data.long_entries,
        exits=long_exits,
        short_entries=data.short_entries,
        short_exits=short_exits,
        size=config["POSITION_SIZE"],
        size_type=config.get("POSITION_SIZE_TYPE", "amount"),
        fees=config.get("FEES", 0),
        slippage=config.get("SLIPPAGE", 0),
        init_cash=config.get("INIT_BALANCE", 100_000),
        freq=freq,
    )
