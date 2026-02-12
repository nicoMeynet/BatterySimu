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
- `out/`: generated outputs (`.csv`, `.json`, `.pdf`, and exported graph images)
- `battery_comparison_month.ipynb`: monthly comparison notebook
- `battery_comparison_season.ipynb`: seasonal comparison notebook
- `battery_comparison_global.ipynb`: global (full-range) comparison notebook
- `generate_pdf_report.py`: builds a PDF report from exported monthly/seasonal notebook graphs
- `generate_recommendation.py`: generates a recommendation from the PDF report via local Ollama
- `Makefile`: setup and batch run shortcuts

## Requirements

- Python 3
- Packages in `requirements.txt`:
  - `pandas`
  - `tabulate`
  - `matplotlib`
  - `nbconvert`
  - `pypdf`

## Setup

```bash
make venv
source venv/bin/activate
```

The Makefile also appends `ulimit -n 2048` (configurable) to `venv/bin/activate`.

## Step-by-step procedure

### a) Simulation

Description:
- Run battery simulation scenarios from real 3-phase household data.
- Compare each scenario against a no-battery baseline under the same tariff assumptions.

Input:
- 3 CSV files (phase A/B/C) with columns:
  - `entity_id`
  - `state` (W)
  - `last_changed` (timestamp)
- One or more config files in `config/*.json`.

Generated data:
- `out/<config-name>.csv` (minute-level simulation table)
- `out/<config-name>.json` (global/monthly/seasonal structured results)

Commands:
```bash
# One scenario
python battery_sim.py \
  dataset/2025/2025_history_phase_a_1dec2024-1dec2025.csv \
  dataset/2025/2025_history_phase_b_1dec2024-1dec2025.csv \
  dataset/2025/2025_history_phase_c_1dec2024-1dec2025.csv \
  --config config/config_Zendure2400_5760kwh.json

# All scenarios from Makefile
make simulate_all
```

Example output files:
- `out/config_Zendure2400_5760kwh.csv`
- `out/config_Zendure2400_5760kwh.json`

### b) Notebook analysis and graph export

Description:
- Execute global, monthly, and seasonal notebooks on simulation JSON outputs.
- Auto-export charts as images for reporting.

Input:
- Simulation JSON files in `out/*.json` (generated in step a).
- Notebooks:
  - `battery_comparison_global.ipynb`
  - `battery_comparison_month.ipynb`
  - `battery_comparison_season.ipynb`

Generated data:
- `out/global/*.png` (global graphs)
- `out/month/*.png` (monthly graphs)
- `out/season/*.png` (seasonal graphs)

Commands:
```bash
# Execute notebooks and export graphs
make run_notebooks
```

Example output files:
- `out/month/01_monthly_net_financial_gain_vs_no_battery.png`
- `out/season/01_seasonal_net_financial_gain_vs_no_battery.png`

Optional manual run (if you want only global):
```bash
MPLCONFIGDIR=/tmp/matplotlib venv/bin/python -m nbconvert \
  --to notebook --execute --inplace battery_comparison_global.ipynb
```

Global notebook output files:
- `out/global/01_global_energy_reduction_kwh.png`
- `out/global/05_global_battery_status_heatmap.png`

### c) PDF generation

Description:
- Build a consolidated PDF report from exported graph images.
- Include intro/methodology/scope/data-requirements sections and configuration cards.
- If global graphs are available in `out/global`, they are included before monthly and seasonal sections.

Input:
- Graph images from step b:
  - `out/global/*.png` (optional, included first if present)
  - `out/month/*.png`
  - `out/season/*.png`
- Scenario config files:
  - `config/*.json`

Generated data:
- `out/battery_graph_report.pdf` (default)

Commands:
```bash
# Default
make pdf_report

# Optional custom output file
make pdf_report PDF_REPORT_OUTPUT=out/my_report.pdf
```

Example output file:
- `out/battery_graph_report.pdf`

Note:
- `make pdf_report` automatically includes recommendation text from `out/recommendation.md` when that file exists.

### d) AI recommendation from PDF (local Ollama)

Description:
- Use a local LLM via Ollama to read the generated PDF report and produce a battery configuration recommendation.

Input:
- PDF report from step c (default: `out/battery_graph_report.pdf`)
- Local Ollama model (default: `llama3.1:70b-instruct-q4_K_M`)

Generated data:
- `out/recommendation.md` (default)

Setup:
```bash
# Python deps (inside venv)
venv/bin/python -m pip install -r requirements.txt

# Install Ollama (macOS/Homebrew)
brew install ollama
```

Start Ollama server:
```bash
ollama serve
```

In another terminal, prepare/check model:
```bash
ollama pull llama3.1:70b-instruct-q4_K_M
ollama list
```

Commands:
```bash
# Default recommendation flow
make recommend

# Optional custom model/input/output
make recommend \
  OLLAMA_MODEL=mistral \
  OLLAMA_TEMPERATURE=0.2 \
  OLLAMA_TOP_P=0.9 \
  OLLAMA_NUM_CTX=32768 \
  RECOMMEND_INPUT_PDF=out/my_report.pdf \
  RECOMMEND_OUTPUT=out/my_recommendation.md
```

Troubleshooting (`make recommend`):

If you get:
```text
Error: Ollama API error at http://127.0.0.1:11434/api/generate:
{"error":"model runner has unexpectedly stopped ... resource limitations ..."}
```

Use this checklist:
```bash
# 1) If this fails with "address already in use", Ollama is already running
ollama serve

# 2) Server connectivity
curl http://127.0.0.1:11434/api/tags

# 3) Direct model test
ollama run llama3.1:70b-instruct-q4_K_M "say hello"
```

If step 3 fails with the same 500 error, the model is too heavy for available resources (or context is too large).  
Use a smaller model and/or lower context:
```bash
make recommend OLLAMA_MODEL=llama3.1:8b-instruct-q4_K_M OLLAMA_NUM_CTX=8192
```

Optional (keep 70B but reduce context):
```bash
make recommend OLLAMA_MODEL=llama3.1:70b-instruct-q4_K_M OLLAMA_NUM_CTX=4096
```

Example output file:
- `out/recommendation.md`

To embed recommendation in PDF:
```bash
make recommend
make pdf_report
```

### End-to-end shortcut

```bash
make simulate_all
make run_notebooks
make pdf_report
make recommend
make pdf_report
```

Or:

```bash
make simulate_all run_notebooks pdf_report recommend pdf_report
```

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

Copyright (c) 2026 Nicolas Meynet.

This project is available under Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0):  
http://creativecommons.org/licenses/by-nc/4.0/

Usage summary:
- Personal and non-commercial use is allowed.
- For commercial/company/product usage, contact: `nicolas@meynet.ch`

See `LICENSE` for details.
