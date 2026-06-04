import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

sys.path.insert(0, PROJECT_ROOT)
from src.warnings_config import configure_warnings

configure_warnings()

from src.functions import load_backtest_datasets
from src.wfo import METRIC_ALIASES, generate_wfo_folds, run_walk_forward

import yaml


def print_folds_preview(config: dict) -> None:
    folds = generate_wfo_folds(config)
    print(f">>> WFO folds: {len(folds)}")
    for fold in folds:
        print(
            f"    [{fold.fold_id}] train {fold.train_start.date()} -> {fold.train_end.date()} | "
            f"test {fold.test_start.date()} -> {fold.test_end.date()}"
        )


if __name__ == "__main__":
    print(">>> Starting WFO...")
    with open("config.yaml", "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)

    data_dir = os.path.join(PROJECT_ROOT, "data")
    execution_df, signals_df, execution_path, signals_path = load_backtest_datasets(
        config, data_dir
    )

    print(f">>> Symbol: {config['SYMBOL']}")
    print(f">>> Execution: {execution_path.name} ({len(execution_df)} bars)")
    print(f">>> Signals:   {signals_path.name} ({len(signals_df)} bars)")

    wfo = config["WFO"]
    sel = config["WFO_SELECTION"]
    print(
        f">>> Windows: train={wfo['TRAIN_MONTHS']}m, test={wfo['TEST_MONTHS']}m, "
        f"step={wfo['STEP_MONTHS']}m"
    )
    print(f">>> Selection: {sel['METRIC']} ({sel.get('DIRECTION', 'max')})")
    print(f">>> Supported metric aliases: {', '.join(sorted(METRIC_ALIASES.keys()))}")

    print_folds_preview(config)

    print("\n>>> Running walk-forward optimization...")
    folds_df, summary_df = run_walk_forward(execution_df, signals_df, config)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    folds_path = os.path.join(OUTPUT_DIR, "wfo_folds.csv")
    summary_path = os.path.join(OUTPUT_DIR, "wfo_summary.csv")

    folds_df.to_csv(folds_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"\n>>> Saved: {folds_path}")
    print(f">>> Saved: {summary_path}")
    print("\n>>> OOS summary:")
    print(summary_df.to_string(index=False))
    print(f"\n>>> Output: {OUTPUT_DIR}")
    print(">>> Done.")
