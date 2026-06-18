# VBTSample

VectorBT backtesting framework that reads **client Excel input** (`ar_bt.xlsx` format), extracts OHLC, signals, prices, and stop-loss levels, and produces standard backtest reports.

## Project structure

```
vbtsample/
├── backtesting/
│   ├── config.yaml      # paths, portfolio settings
│   ├── main.py          # entry point
│   └── output/          # generated reports (gitignored)
├── data/
│   └── ar_bt.xlsx       # client input (not in git; copy your file here)
├── src/
│   ├── io.py            # load & parse ar_bt.xlsx
│   ├── backtester.py    # VectorBT portfolio simulation
│   ├── reports.py       # PDF / HTML export
│   └── warnings_config.py
├── pyproject.toml
├── README.md
└── RELEASE_NOTES.txt
```

## Input format (`ar_bt.xlsx`, sheet `data`)

| Column | Description |
|--------|-------------|
| `date` | Timestamp (`YYYY.MM.DD HH:MM`) |
| `open`, `high`, `low`, `close` | OHLC prices |
| `entry_buy` | `BUY` on long entry bars |
| `exit_buy` | `EXIT` on long exit bars |
| `entry_sell` / `exit_sell` | Short signals (if used) |
| `stoploss_buy` / `stoploss_sell` | Stop-loss price levels |
| `takeprofit_buy` / `takeprofit_sell` | Take-profit levels |
| `tsl_buy` / `tsl_sell` | Trailing-stop flags |

## Setup

```bash
poetry install
```

Copy your input file to `data/ar_bt.xlsx` or set `INPUT_FILE` in `backtesting/config.yaml`.

## Run backtest

```bash
cd backtesting
poetry run python main.py
```

## Outputs (`backtesting/output/`)

| File | Description |
|------|-------------|
| `portfolio_report.pdf` | VectorBT statistics table |
| `portfolio_plot.html` | Interactive equity / trade chart |
| `trades.csv` | Trade log |
| `signals_extracted.csv` | Parsed entry/exit/price/SL rows |

## Performance

- First run parses Excel (~30–60s for large files) and caches a `.pkl` file beside the input for faster re-runs.
- Set `CACHE_PARQUET: true` in config (default).
- Use `BACKTESTING_START_DATE` / `BACKTESTING_END_DATE` to limit the simulation window.

## Configuration (`backtesting/config.yaml`)

```yaml
INPUT_FILE: ../data/ar_bt.xlsx
INPUT_SHEET: data
FREQ: 3min
INIT_BALANCE: 100000
POSITION_SIZE: 1000
POSITION_SIZE_TYPE: amount
USE_STOP_LOSS: false   # true: exit on EXIT signal OR stop-loss cross (Low/High vs stoploss_*)
```
