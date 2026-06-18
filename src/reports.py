"""Standard VectorBT backtest report outputs."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import plotly.io as pio
from matplotlib.backends.backend_pdf import PdfPages


def save_backtesting_results(pf, output_dir: str = "output") -> None:
    os.makedirs(output_dir, exist_ok=True)

    stats_df = pf.stats().to_frame()

    with PdfPages(f"{output_dir}/portfolio_report.pdf") as pdf:
        fig, ax = plt.subplots(figsize=(8.5, max(4, len(stats_df) * 0.4)))
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
