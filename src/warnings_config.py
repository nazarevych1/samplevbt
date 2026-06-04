"""Suppress noisy warnings for CLI scripts (pandas, vectorbt, etc.)."""

import warnings


def configure_warnings() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    try:
        import pandas as pd

        pd.set_option("future.no_silent_downcasting", True)
    except Exception:
        pass
