"""Walk-forward optimization: optimize in-sample, validate out-of-sample per fold."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.functions import backtest_strategy, optimize


# Friendly names -> VectorBT stats() column titles
METRIC_ALIASES: dict[str, str] = {
    "sharpe_ratio": "Sharpe Ratio",
    "sharpe": "Sharpe Ratio",
    "total_return": "Total Return [%]",
    "profit_factor": "Profit Factor",
    "max_drawdown": "Max Drawdown [%]",
    "calmar_ratio": "Calmar Ratio",
    "win_rate": "Win Rate [%]",
    "total_trades": "Total Trades",
}


@dataclass
class WfoFold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def resolve_metric_column(metric: str, available_columns: list[str]) -> str:
    """Map config metric name to a column present in optimization results."""
    if metric in available_columns:
        return metric

    alias = METRIC_ALIASES.get(metric.strip().lower())
    if alias and alias in available_columns:
        return alias

    lower_map = {c.lower(): c for c in available_columns}
    key = metric.strip().lower()
    if key in lower_map:
        return lower_map[key]

    raise ValueError(
        f"Unknown WFO metric '{metric}'. Use one of: {sorted(set(METRIC_ALIASES) | set(METRIC_ALIASES.values()))} "
        f"or an exact stats column. Available: {available_columns[:12]}..."
    )


def generate_wfo_folds(config: dict) -> list[WfoFold]:
    """
    Build rolling train/test windows from WFO config.

    Expected under config['WFO']:
      START_DATE, END_DATE (optional; fall back to BACKTESTING_*)
      TRAIN_MONTHS, TEST_MONTHS, STEP_MONTHS
      ANCHORED_TRAIN (optional, default false): if true, train always starts at START_DATE
    """
    wfo = config["WFO"]
    start = pd.Timestamp(wfo.get("START_DATE", config["BACKTESTING_START_DATE"]), tz="UTC")
    end = pd.Timestamp(wfo.get("END_DATE", config["BACKTESTING_END_DATE"]), tz="UTC")

    train_months = int(wfo["TRAIN_MONTHS"])
    test_months = int(wfo["TEST_MONTHS"])
    step_months = int(wfo["STEP_MONTHS"])
    anchored = bool(wfo.get("ANCHORED_TRAIN", False))

    if train_months <= 0 or test_months <= 0 or step_months <= 0:
        raise ValueError("TRAIN_MONTHS, TEST_MONTHS, and STEP_MONTHS must be positive")

    folds: list[WfoFold] = []
    fold_id = 0
    cursor = start

    while True:
        train_start = start if anchored else cursor
        train_end = train_start + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)

        if test_end > end:
            break

        folds.append(
            WfoFold(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        fold_id += 1
        cursor = cursor + pd.DateOffset(months=step_months)

    if not folds:
        raise ValueError(
            f"No WFO folds fit between {start.date()} and {end.date()} with "
            f"train={train_months}m, test={test_months}m, step={step_months}m."
        )

    return folds


def _slice_period(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Slice [start, end) — end exclusive."""
    idx = df.index
    if idx.tz is None:
        start = start.tz_localize(None)
        end = end.tz_localize(None)
    return df[(idx >= start) & (idx < end)]


def select_best_params(
    optimization_results: list[dict],
    metric: str,
    direction: str,
) -> dict:
    """Pick best parameter row from in-sample optimization results."""
    if not optimization_results:
        raise ValueError("No optimization results to select from")

    results_df = pd.DataFrame(optimization_results)
    metric_col = resolve_metric_column(metric, results_df.columns.tolist())

    series = pd.to_numeric(results_df[metric_col], errors="coerce")
    if series.notna().sum() == 0:
        raise ValueError(f"Metric '{metric_col}' has no valid values in optimization results")

    direction = direction.strip().lower()
    if direction == "max":
        best_idx = series.idxmax()
    elif direction == "min":
        best_idx = series.idxmin()
    else:
        raise ValueError("WFO_SELECTION.DIRECTION must be 'max' or 'min'")

    return results_df.loc[best_idx].to_dict()


def run_walk_forward(
    execution_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run walk-forward optimization.

    Returns:
        folds_df: per-fold IS selection + OOS metrics
        summary_df: one-row aggregate over OOS folds
    """
    folds = generate_wfo_folds(config)
    selection = config["WFO_SELECTION"]
    metric = selection["METRIC"]
    direction = selection.get("DIRECTION", "max")

    min_signals_bars = int(config["WFO"].get("MIN_SIGNALS_BARS", 0))

    fold_rows: list[dict] = []

    for fold in folds:
        train_exec = _slice_period(execution_df, fold.train_start, fold.train_end)
        train_sig = _slice_period(signals_df, fold.train_start, fold.train_end)
        test_exec = _slice_period(execution_df, fold.test_start, fold.test_end)
        test_sig = _slice_period(signals_df, fold.test_start, fold.test_end)

        if train_exec.empty or test_exec.empty:
            continue
        if min_signals_bars and len(train_sig) < min_signals_bars:
            continue

        opt_results = optimize(train_exec, train_sig, config)
        if not opt_results:
            continue

        best = select_best_params(opt_results, metric, direction)
        sma_fast = int(best["N1"])
        sma_slow = int(best["N2"])

        test_config = config.copy()
        test_config["SMA_FAST"] = sma_fast
        test_config["SMA_SLOW"] = sma_slow

        pf_oos = backtest_strategy(test_exec, test_sig, test_config)
        oos_stats = pf_oos.stats().to_dict()

        metric_col = resolve_metric_column(metric, list(best.keys()))
        row = {
            "fold": fold.fold_id,
            "train_start": fold.train_start,
            "train_end": fold.train_end,
            "test_start": fold.test_start,
            "test_end": fold.test_end,
            "SMA_FAST": sma_fast,
            "SMA_SLOW": sma_slow,
            "is_metric": metric,
            "is_metric_value": best.get(metric_col),
            "train_exec_bars": len(train_exec),
            "test_exec_bars": len(test_exec),
        }
        for k, v in oos_stats.items():
            row[f"oos_{k}"] = v
        fold_rows.append(row)

    if not fold_rows:
        raise RuntimeError("WFO produced no valid folds. Check windows, data range, or MIN_SIGNALS_BARS.")

    folds_df = pd.DataFrame(fold_rows)

    oos_return_cols = [c for c in folds_df.columns if c == "oos_Total Return [%]"]
    summary = {
        "folds": len(folds_df),
        "metric": metric,
        "direction": direction,
    }
    if oos_return_cols:
        returns = pd.to_numeric(folds_df["oos_Total Return [%]"], errors="coerce")
        summary["oos_total_return_mean"] = returns.mean()
        summary["oos_total_return_sum"] = returns.sum()
    if "oos_Sharpe Ratio" in folds_df.columns:
        sharpe = pd.to_numeric(folds_df["oos_Sharpe Ratio"], errors="coerce")
        summary["oos_sharpe_mean"] = sharpe.mean()

    summary_df = pd.DataFrame([summary])
    return folds_df, summary_df
