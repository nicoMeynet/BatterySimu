# Battery Simulator

Battery simulation for a 3-phase house using Home Assistant power history data.

The project compares:
- baseline (no battery)
- simulated battery behavior
- financial impact (CHF)
- monthly and seasonal performance

## What is in this repo

- `battery_sim.py`: main simulation engine
- `config/*.json`: battery and tariff scenarios
- `dataset/`: input CSV files (phase A/B/C power history)
- `out/`: generated outputs (`.csv` and `.json`)
- `battery_comparison_month.ipynb`: monthly comparison notebook
- `battery_comparison_season.ipynb`: seasonal comparison notebook
- `Makefile`: setup and batch run shortcuts

## Requirements

- Python 3
- Packages in `requirements.txt`:
  - `pandas`
  - `tabulate`
  - `matplotlib`

## Setup

```bash
make venv
source venv/bin/activate
```

The Makefile also appends `ulimit -n 2048` (configurable) to `venv/bin/activate`.

## Run simulations

### Run one configuration

```bash
python battery_sim.py \
  dataset/2025/2025_history_phase_a_1dec2024-1dec2025.csv \
  dataset/2025/2025_history_phase_b_1dec2024-1dec2025.csv \
  dataset/2025/2025_history_phase_c_1dec2024-1dec2025.csv \
  --config config/config_Zendure2400_5760kwh.json
```

### Run all configured scenarios

```bash
make simulate_all
```

Current scenarios run by `make simulate_all`:
- `config/config_Zendure2400_noBattery.json`
- `config/config_Zendure2400_2880kwh.json`
- `config/config_Zendure2400_5760kwh.json`
- `config/config_Zendure2400_8640kwh.json`
- `config/config_Zendure2400_11520kwh.json`
- `config/config_Zendure2400_14400kwh.json`

## Inputs

`battery_sim.py` expects 3 CSV files (phase A, B, C) with this schema:
- `entity_id`
- `state` (W)
- `last_changed` (timestamp)

Typical source: Home Assistant export from Shelly 3EM power sensors.

The script:
- rounds timestamps to minute resolution
- groups duplicate timestamps by mean power
- fills missing timestamps by linear interpolation

## Outputs

For each run, outputs are written to `out/` using the config file name:
- `out/<config-name>.csv`: minute-level simulation table
- `out/<config-name>.json`: structured report (global + monthly + seasonal metrics)

Example:
- `--config config/config_Zendure2400_5760kwh.json`
- output files:
  - `out/config_Zendure2400_5760kwh.csv`
  - `out/config_Zendure2400_5760kwh.json`

Console output also includes injected/consumed deltas, gain/amortization, battery status, and charge/discharge usage summaries.

## Analyze results

After generating JSON files in `out/`, open:
- `battery_comparison_month.ipynb`
- `battery_comparison_season.ipynb`

Both notebooks are preconfigured to read:
- `out/config_Zendure2400_noBattery.json`
- `out/config_Zendure2400_2880kwh.json`
- `out/config_Zendure2400_5760kwh.json`
- `out/config_Zendure2400_8640kwh.json`
- `out/config_Zendure2400_11520kwh.json`
- `out/config_Zendure2400_14400kwh.json`

## Build a PDF from notebook graphs

Export your notebook charts as image files (for example PNG), then run:

```bash
python generate_pdf_report.py \
  --monthly out/graphs/monthly/*.png \
  --seasonal out/graphs/seasonal/*.png \
  --output out/battery_graph_report.pdf \
  --title "Battery Simulation Graph Report" \
  --subtitle "Monthly and seasonal comparison charts"
```

Notes:
- `--monthly` and `--seasonal` accept one or more image paths.
- Output is a multi-page PDF with cover page, monthly section, and seasonal section.
- Captions are inferred from each image file name.

## Configuration format

Each JSON config contains:
- `battery`
  - `capacity_Wh_per_phase`
  - `cost_chf`
  - `max_charge_power_watts`
  - `max_discharge_power_watts`
  - `charge_efficiency`
  - `discharge_efficiency`
  - `max_cycles`
  - `initial_soc_percent_per_phase`
  - `soc_min_pct`
  - `soc_max_pct`
- `tariff`
  - `peak.tariff_consume`
  - `peak.tariff_inject`
  - `peak.days` (weekday indexes)
  - `peak.hours` (hour list)
  - `off_peak.tariff_consume`
  - `off_peak.tariff_inject`

Use `config/config_Zendure2400_2880kwh.json` as a template for new scenarios.

## License

Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0): http://creativecommons.org/licenses/by-nc/4.0/
