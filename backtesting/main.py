import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

sys.path.insert(0, PROJECT_ROOT)
from src.warnings_config import configure_warnings

configure_warnings()

from src.functions import backtest_strategy, load_backtest_datasets, save_backtesting_results

import yaml

if __name__ == "__main__":
    print(">>> Starting script...")
    with open("config.yaml", "r") as fp:
        config = yaml.safe_load(fp)

    data_dir = os.path.join(PROJECT_ROOT, "data")
    execution_df, signals_df, execution_path, signals_path = load_backtest_datasets(
        config, data_dir
    )

    print(f">>> Symbol: {config['SYMBOL']}")
    print(f">>> Execution: {execution_path.name} ({len(execution_df)} bars)")
    print(f">>> Signals:   {signals_path.name} ({len(signals_df)} bars)")
    print(
        f">>> SMA {config['SMA_FAST']}/{config['SMA_SLOW']} on {config['SIGNALS_TIMEFRAME']} "
        f"-> trade on {config['EXECUTION_TIMEFRAME']}"
    )

    pf = backtest_strategy(execution_df, signals_df, config)

    print(">>> Saving backtest results...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_backtesting_results(pf, output_dir=OUTPUT_DIR)
    pf.trades.records_readable.to_csv(os.path.join(OUTPUT_DIR, "trades.csv"))
    print(f">>> Output: {OUTPUT_DIR}")

    print(">>> Done.")
