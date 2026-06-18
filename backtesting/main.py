import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

sys.path.insert(0, PROJECT_ROOT)
from src.warnings_config import configure_warnings

configure_warnings()

from src.backtester import run_backtest
from src.io import load_ar_bt_xlsx
from src.reports import save_backtesting_results

import pandas as pd
import yaml


def _resolve_path(path: str) -> str:
    p = os.path.normpath(os.path.join(os.path.dirname(__file__), path))
    if not os.path.isabs(path):
        return p
    return path


if __name__ == "__main__":
    print(">>> VBTSample backtest")
    with open("config.yaml", "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)

    input_path = _resolve_path(config["INPUT_FILE"])
    start = config.get("BACKTESTING_START_DATE")
    end = config.get("BACKTESTING_END_DATE")

    print(f">>> Input: {input_path}")
    data = load_ar_bt_xlsx(
        input_path,
        sheet_name=config.get("INPUT_SHEET", "data"),
        start_date=start,
        end_date=end,
        cache_parquet=config.get("CACHE_PARQUET", True),
    )

    n_long_in = int(data.long_entries.sum())
    n_long_out = int(data.long_exits.sum())
    n_short_in = int(data.short_entries.sum())
    print(f">>> Bars: {len(data.df)} | Long entries: {n_long_in} | Long exits: {n_long_out}")
    print(f">>> Date range: {data.df.index.min()} -> {data.df.index.max()}")

    pf = run_backtest(data, config)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_backtesting_results(pf, output_dir=OUTPUT_DIR)
    pf.trades.records_readable.to_csv(os.path.join(OUTPUT_DIR, "trades.csv"))

    signals_df = pd.DataFrame(
        {
            "Time": data.df.index,
            "long_entry": data.long_entries,
            "long_exit": data.long_exits,
            "short_entry": data.short_entries,
            "short_exit": data.short_exits,
            "entry_price": data.entry_price,
            "exit_price": data.exit_price,
            "stoploss_long": data.stop_loss_long,
            "stoploss_short": data.stop_loss_short,
        }
    )
    signals_df = signals_df[
        signals_df[["long_entry", "long_exit", "short_entry", "short_exit"]].any(axis=1)
    ]
    signals_df.to_csv(os.path.join(OUTPUT_DIR, "signals_extracted.csv"), index=False)

    print(f">>> Trades: {len(pf.trades.records_readable)}")
    print(f">>> Output: {OUTPUT_DIR}")
    print(">>> Done.")
