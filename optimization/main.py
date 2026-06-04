import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.warnings_config import configure_warnings

configure_warnings()

from src.functions import load_backtest_datasets, optimize

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml


def create_heatmap(results_df, metric_name, output_dir="optimization_output", figsize=(30, 30)):
    try:
        print(f"\n>>> Creating heatmap for: {metric_name}")

        if metric_name not in results_df.columns:
            print(f"WARNING: '{metric_name}' not found in results. Skipping...")
            return False

        heatmap_data = results_df.pivot_table(
            index="N1", columns="N2", values=metric_name, aggfunc="mean"
        )

        if heatmap_data.empty or heatmap_data.isna().all().all():
            print(f"WARNING: No data for '{metric_name}'. Skipping...")
            return False

        valid_values = heatmap_data.stack().dropna()
        print(f"  Valid values: {len(valid_values)}")
        print(f"  Min: {valid_values.min():.2f}, Max: {valid_values.max():.2f}, Mean: {valid_values.mean():.2f}")

        plt.figure(figsize=figsize)
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            cbar_kws={"label": metric_name},
        )
        plt.title(f"{metric_name} Heatmap (SMA_FAST x SMA_SLOW)")
        plt.xlabel("SMA_SLOW (N2)")
        plt.ylabel("SMA_FAST (N1)")
        plt.tight_layout()

        os.makedirs(output_dir, exist_ok=True)
        filepath = f"{output_dir}/{metric_name}.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  Saved: {filepath}")
        return True

    except Exception as e:
        print(f"ERROR creating heatmap for '{metric_name}': {e}")
        plt.close()
        return False


if __name__ == "__main__":
    print(">>> Starting script...")
    with open("config.yaml", "r") as fp:
        config = yaml.safe_load(fp)

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    execution_df, signals_df, execution_path, signals_path = load_backtest_datasets(
        config, data_dir
    )

    print(f"\n>>> Execution: {execution_path.name} ({len(execution_df)} bars)")
    print(f">>> Signals:   {signals_path.name} ({len(signals_df)} bars)")

    print("\n>>> Running optimization (SMA_FAST x SMA_SLOW on signals TF)...")
    optimization_results_list = optimize(
        execution_df=execution_df,
        signals_df=signals_df,
        config=config,
    )

    print(f"\n>>> Optimization complete. Results: {len(optimization_results_list)}")

    results_df = pd.DataFrame(optimization_results_list)
    results_df = results_df.dropna()

    if results_df.empty:
        print("ERROR: No valid results after dropping NaN values!")
        sys.exit(1)

    os.makedirs("optimization_output", exist_ok=True)
    results_df.to_csv("optimization_output/optimization_results.csv", index=False)
    print("\n>>> Saved: optimization_output/optimization_results.csv")

    metrics = [
        "Profit Factor",
        "Sharpe Ratio",
        "Total Trades",
        "Total Return [%]",
        "Max Drawdown [%]",
        "Win Rate [%]",
        "Avg Winning Trade [%]",
        "Avg Losing Trade [%]",
    ]

    successful = sum(create_heatmap(results_df, metric) for metric in metrics)
    print(f"\n>>> COMPLETE: {successful}/{len(metrics)} heatmaps created")
