# Battery Simulator

Battery simulation for a 3-phase house using Home Assistant power history data.

The project compares:
- baseline (no battery)
- simulated battery behavior
- financial impact (CHF)
- monthly and seasonal performance

## What is in this repo

- `battery_sim.py`: main simulation engine
- `config/config_*.json`: battery scenarios (battery-only config)
- `config/energy_tariff.json`: global tariff configuration shared by all scenarios
- `config/kpi_scoring.json`: graph-KPI thresholds used for deterministic battery sizing decisions (required by `compute_kpi_summary.py`)
- `doc/pdf_report/`: mandatory PDF report text snippets (`default_intro.md`, `default_scope.md`, etc.)
- `dataset/`: input CSV files (phase A/B/C power history)
- `out/`: generated outputs (simulations, graph images, PDF report, LLM recommendations)
- `battery_comparison_month.ipynb`: monthly comparison notebook
- `battery_comparison_season.ipynb`: seasonal comparison notebook
- `battery_comparison_global.ipynb`: global (full-range) comparison notebook
- `battery_kpi_recommendation.ipynb`: KPI vote-summary notebook (exports the KPI recommendation graph used in the PDF)
- `compute_kpi_summary.py`: computes deterministic graph-KPI decision summaries from simulation JSON outputs
- `generate_pdf_report.py`: builds a PDF report from exported global/seasonal/monthly notebook graphs
- `generate_recommendation.py`: generates a recommendation from KPI summary Markdown via local Ollama
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
make help
make venv
source venv/bin/activate
```

The Makefile also appends `ulimit -n 2048` (configurable) to `venv/bin/activate`.

## Makefile commands (important)

Start with:

```bash
make help
```

Main commands:
- `make help`: show available commands and suggested workflow
- `make venv`: create/update the virtual environment and install requirements
- `make activate`: print the activation command
- `make clean`: delete all generated content under `out/`
- `make simulate_all`: run simulations for all battery configs in `config/config_*.json`
- `make kpi_summary`: compute deterministic graph-KPI decision outputs (JSON + Markdown) from `out/simulation_json/*.json`
- `make run_notebooks`: execute global/monthly/seasonal/KPI notebooks and export graphs
- `make pdf_report`: generate the PDF report from exported graphs and simulation outputs
- `make recommend`: generate an LLM recommendation markdown file from `out/kpi_summary/kpi_summary.md` via Ollama

## File structure

### `config/`

- `config/config_*.json`: battery-specific settings only (`battery`)
- `config/energy_tariff.json`: shared tariff settings (`tariff`)
- `config/kpi_scoring.json`: graph-KPI thresholds (`thresholds.graph_kpis.*`)

### `out/`

- `out/simulation_csv/`: simulation CSV outputs (`config_*.csv`)
- `out/simulation_json/`: simulation JSON outputs (`config_*.json`)
- `out/kpi_summary/`: graph-KPI decision outputs for decision support (`kpi_summary.json`, `kpi_summary.md`)
- `out/kpi_images/`: KPI notebook exported graph(s) used in the PDF AI/final recommendation section
- `out/month_images/`: exported monthly notebook graphs
- `out/season_images/`: exported seasonal notebook graphs
- `out/global_images/`: exported global notebook graphs
- `out/simulation_llm_recommendation/`: recommendation markdown files (`recommendation_*.md`)
- `out/battery_graph_report.pdf`: generated PDF report (default)

## Step-by-step procedure

Recommended KPI-first workflow order:
1. `Simulation`
2. `KPI summary`
3. `AI recommendation` (Markdown from KPI summary only)
4. `Notebook analysis and graph export`
5. `PDF generation`

### a) Simulation

Description:
- Run battery simulation scenarios from real 3-phase household data.
- Compare each scenario against a no-battery baseline under the same tariff assumptions.

Input:
- 3 CSV files (phase A/B/C) with columns:
  - `entity_id`
  - `state` (power in `W`, signed)
  - `last_changed` (timestamp)
- One or more battery config files in `config/config_*.json`.
- Global tariff file in `config/energy_tariff.json`.

CSV dataset format (exact):
- One CSV file per phase (`A`, `B`, `C`)
- Comma-separated with header (UTF-8 text)
- Expected column order in the provided datasets:
  - `entity_id,state,last_changed`
- `entity_id`
  - String sensor identifier from Home Assistant (for example `sensor.shellyem3_..._channel_a_power`)
  - Dropped by the simulator after loading (used only as source metadata)
- `state`
  - Numeric power measurement in **watts (`W`)**
  - Decimal values are accepted
  - The simulator expects **power**, not current (`A`)
  - Sign convention used by the simulator:
    - positive value => grid import / consumption from grid
    - negative value => grid export / injection to grid
- `last_changed`
  - Timestamp parsed by pandas (`parse_dates=["last_changed"]`)
  - ISO 8601 timestamps are recommended (your datasets use UTC `Z`, e.g. `2024-12-01T00:00:00.000Z`)

Example (real format):
```csv
entity_id,state,last_changed
sensor.shellyem3_244cab435bf4_channel_a_power,147.39145558806499,2024-11-30T23:00:00.000Z
sensor.shellyem3_244cab435bf4_channel_a_power,153.50503759636666,2024-12-01T00:00:00.000Z
```

Pre-processing behavior inside `battery_sim.py`:
- Timestamps are rounded down to the minute (`floor("min")`)
- Duplicate rows within the same minute are averaged (`groupby(timestamp).mean()`)
- Missing minutes are inserted and linearly interpolated per phase
- Non-numeric `state` values are coerced to `NaN` and dropped

If your source dataset is current (`A`) instead of power (`W`):
- Convert it to power first before running the simulator (the script does not convert current to power automatically).

Generated data:
- `out/simulation_csv/<config-name>.csv` (minute-level simulation table)
- `out/simulation_json/<config-name>.json` (global/monthly/seasonal structured results)

Commands:
```bash
# One scenario
python battery_sim.py \
  dataset/2025/2025_history_phase_a_1dec2024-1dec2025.csv \
  dataset/2025/2025_history_phase_b_1dec2024-1dec2025.csv \
  dataset/2025/2025_history_phase_c_1dec2024-1dec2025.csv \
  --config config/config_Zendure2400_5760Wh.json

# All scenarios from Makefile
make simulate_all
```

Example output files:
- `out/simulation_csv/config_Zendure2400_5760Wh.csv`
- `out/simulation_json/config_Zendure2400_5760Wh.json`

### b) KPI summary (deterministic graph-KPI decision matrix)

Description:
- Compute deterministic graph-based battery sizing KPIs from simulation JSON outputs.
- Produce a graph-KPI-only decision summary (JSON + Markdown) used by the LLM and PDF.
- Each KPI encodes a concrete sizing rule (step-delta or constraint-cap), not a hidden weighted score.
- Use `config/kpi_scoring.json` to tune graph-KPI thresholds without changing code.

Input:
- Simulation JSON files in `out/simulation_json/*.json` (generated in step a).
- KPI scoring config file (required): `config/kpi_scoring.json`

Generated data:
- `out/kpi_summary/kpi_summary.json` (machine/LLM-friendly graph-KPI summary)
- `out/kpi_summary/kpi_summary.md` (human-readable graph-KPI summary, also embedded in the PDF appendix)
- Outputs record the applied KPI config source and resolved thresholds for traceability.

Commands:
```bash
# Default (config file is required)
make kpi_summary

# Use a custom KPI scoring config file
make kpi_summary KPI_CONFIG=config/my_kpi_scoring.json

# Optional CLI overrides (legacy overrides of config values)
make kpi_summary \
  KPI_MARGINAL_GAIN_THRESHOLD=15 \
  KPI_SEASONAL_MARGINAL_GAIN_THRESHOLD=10
```

Example output files:
- `out/kpi_summary/kpi_summary.json`
- `out/kpi_summary/kpi_summary.md`

### c) Notebook analysis and graph export

Description:
- Execute global, monthly, seasonal, and KPI notebooks on simulation JSON outputs / KPI summary outputs.
- Auto-export charts as images for reporting (including the KPI recommendation graph used in the PDF).

Input:
- Simulation JSON files in `out/simulation_json/*.json` (generated in step a).
- KPI summary files from step b:
  - `out/kpi_summary/kpi_summary.json`
  - `out/kpi_summary/kpi_summary.md`
- Notebooks:
  - `battery_comparison_global.ipynb`
  - `battery_comparison_month.ipynb`
  - `battery_comparison_season.ipynb`
  - `battery_kpi_recommendation.ipynb`

Generated data:
- `out/global_images/*.png` (global graphs)
- `out/month_images/*.png` (monthly graphs)
- `out/season_images/*.png` (seasonal graphs)
- `out/kpi_images/*.png` (KPI recommendation graph export)

Commands:
```bash
# Execute notebooks and export graphs
make run_notebooks
```

Example output files:
- `out/month_images/01_monthly_net_financial_gain_vs_no_battery.png`
- `out/season_images/01_seasonal_net_financial_gain_vs_no_battery.png`
- `out/kpi_images/01_graph_kpi_consensus_best_battery.png`

Optional manual run (if you want only global):
```bash
MPLCONFIGDIR=/tmp/matplotlib venv/bin/python -m nbconvert \
  --to notebook --execute --inplace battery_comparison_global.ipynb
```

Global notebook output files:
- `out/global_images/01_global_energy_reduction_kwh.png`
- `out/global_images/05_global_battery_status_heatmap.png`

### d) PDF generation

Description:
- Build a consolidated PDF report from exported graph images.
- Include intro/scope/methodology/configuration sections, graph sections, KPI summary, recommendation sections, and appendices.
- Include the project `LICENSE` text as a dedicated section when `LICENSE` is present.
- Include AI recommendation text (if present), KPI recommendation graph (if exported), and `kpi_summary.md` (compact summary + detailed appendix).
- Graph section order is: global, seasonal, monthly (rendered as renamed analysis sections in the PDF).
- Current PDF chapter order (TOC): `Introduction`, `Report Scope`, `Input Data Overview`, `Simulation Methodology`, `Tariff Model Configuration`, `Battery Scenarios Evaluated`, `Global (Full-Year) Analysis`, `Seasonal Analysis`, `Monthly Analysis`, `Sizing KPI Summary (Decision Matrix)`, `Final Recommended Configuration`, `Detailed KPI Appendix`, `Data Requirements & Assumptions`, `License`.

Input:
- Graph images from step c:
  - `out/global_images/*.png` (optional, included first if present)
  - `out/season_images/*.png`
  - `out/month_images/*.png`
  - `out/kpi_images/01_graph_kpi_consensus_best_battery.png` (optional, used in Final Recommended Configuration)
- KPI summary markdown from step b (optional but recommended):
  - `out/kpi_summary/kpi_summary.md`
- AI recommendation markdown from step e (optional):
  - `out/simulation_llm_recommendation/recommendation_ollama.md`
- Scenario config files:
  - `config/config_*.json`

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
- `make pdf_report` automatically includes recommendation text from `out/simulation_llm_recommendation/recommendation_ollama.md` when that file exists.
- `make pdf_report` automatically includes the KPI recommendation graph from `out/kpi_images/01_graph_kpi_consensus_best_battery.png` when that file exists.
- `make pdf_report` automatically includes `out/kpi_summary/kpi_summary.md` as:
  - `Sizing KPI Summary (Decision Matrix)` (single compact page)
  - `Detailed KPI Appendix` (full markdown with rendered tables)
- `make pdf_report` automatically includes the `LICENSE` file as a PDF section when that file exists.
- Files in `doc/pdf_report/` are mandatory for PDF generation (`default_intro.md`, `default_scope.md`, etc.).

### e) AI recommendation from KPI summary Markdown (local Ollama)

Description:
- Use a local LLM via Ollama to produce a battery configuration recommendation.
- `make recommend` uses `out/kpi_summary/kpi_summary.md` only (no PDF context, no KPI JSON context).
- The script trims KPI markdown prompt content to fit `OLLAMA_NUM_CTX` and saves raw model output for debugging on failures.

Input:
- KPI summary markdown file from step b:
  - `out/kpi_summary/kpi_summary.md`
- Local Ollama model (Makefile default is `gemma3:12b`)

Generated data:
- `out/simulation_llm_recommendation/recommendation_ollama.md` (default)

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
ollama pull gemma3:12b
ollama list
```

Commands:
```bash
# Default recommendation flow (KPI summary markdown only)
make recommend

# You can run this before notebooks/PDF if `make kpi_summary` has already been run.

# Optional custom model/input/output
make recommend \
  OLLAMA_MODEL=mistral \
  OLLAMA_TEMPERATURE=0.2 \
  OLLAMA_TOP_P=0.9 \
  OLLAMA_NUM_CTX=32768 \
  KPI_SUMMARY_MD=out/kpi_summary/kpi_summary.md \
  RECOMMEND_OUTPUT=out/simulation_llm_recommendation/recommendation_custom.md
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

If JSON parsing fails, inspect the saved raw model output file (written next to the recommendation output), for example:
- `out/simulation_llm_recommendation/recommendation_ollama.md.raw_model.txt`

Example output file:
- `out/simulation_llm_recommendation/recommendation_ollama.md`

To embed recommendation in PDF (and include the KPI recommendation graph + KPI appendix):
```bash
make recommend
make run_notebooks
make pdf_report
```

### End-to-end shortcut

```bash
make simulate_all
make kpi_summary
make recommend
make run_notebooks
make pdf_report
```

Or:

```bash
make simulate_all kpi_summary recommend run_notebooks pdf_report
```

## Configuration format

Battery scenario files (`config/config_*.json`) contain:
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

Global tariff file (`config/energy_tariff.json`) contains:
- `tariff`
  - `peak.tariff_consume`
  - `peak.tariff_inject`
  - `peak.days` (weekday indexes)
  - `peak.hours` (hour list)
  - `off_peak.tariff_consume`
  - `off_peak.tariff_inject`

Use `config/config_Zendure2400_2880Wh.json` as a template for new scenarios.

KPI scoring file (`config/kpi_scoring.json`) contains graph-KPI thresholds only:
- `thresholds.graph_kpis.global_energy_reduction_kwh.consumed_reduction_increment_pct_points_min`
- `thresholds.graph_kpis.global_energy_financial_impact_chf.bill_offset_increment_pct_points_min`
- `thresholds.graph_kpis.global_rentability_overview.amortization_years_max`
- `thresholds.graph_kpis.global_battery_utilization.pct_max_cycles_per_year_max`
- `thresholds.graph_kpis.global_battery_status_heatmap.empty_pct_max`
- `thresholds.graph_kpis.seasonal_power_saturation_at_max_limit.power_saturation_pct_any_season_max`
- `thresholds.graph_kpis.monthly_structural_evening_energy_undersizing_peak_period.evening_undersize_pct_per_month_max`
- `thresholds.graph_kpis.monthly_structural_evening_energy_undersizing_peak_period.max_months_above_threshold`

These thresholds drive the 7 graph KPI rules used in:
- `out/kpi_summary/kpi_summary.json`
- `out/kpi_summary/kpi_summary.md`
- `battery_kpi_recommendation.ipynb` (KPI vote graph)

## License

Copyright (c) 2026 Nicolas Meynet.

This project is available under Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0):  
http://creativecommons.org/licenses/by-nc/4.0/

Usage summary:
- Personal and non-commercial use is allowed.
- For commercial/company/product usage, contact: `nicolas@meynet.ch`

See `LICENSE` for details.
