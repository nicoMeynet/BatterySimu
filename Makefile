# -- Makefile (macOS/Homebrew friendly) --

# Choisis explicitement le Python Homebrew si dispo, sinon python3
PYTHON := $(shell command -v /opt/homebrew/bin/python3.12 || command -v python3)
VENV_DIR := venv
REQ_FILE := requirements.txt
ULIMIT_VALUE ?= 2048
MONTHLY_GRAPHS_DIR ?= out/month
SEASONAL_GRAPHS_DIR ?= out/season
GLOBAL_GRAPHS_DIR ?= out/global
PDF_REPORT_OUTPUT ?= out/battery_graph_report.pdf
PDF_REPORT_TITLE ?= Battery Simulation Graph Report
PDF_REPORT_SUBTITLE ?= Monthly and seasonal comparison charts
PDF_REPORT_INTRO ?= This report evaluates multiple residential battery configurations to determine the optimal storage size for the household. The analysis is based on real 3-phase grid measurements collected prior to battery installation, and each scenario is compared against an identical no-battery baseline for consistency. The objective is to support a data-driven investment decision by balancing financial return, energy adequacy, and power limitations while selecting the smallest robust configuration.
MONTH_NOTEBOOK ?= battery_comparison_month.ipynb
SEASON_NOTEBOOK ?= battery_comparison_season.ipynb
GLOBAL_NOTEBOOK ?= battery_comparison_global.ipynb
NOTEBOOK_TIMEOUT ?= -1
NOTEBOOK_MPLCONFIGDIR ?= /tmp/matplotlib
RECOMMEND_INPUT_PDF ?= out/battery_graph_report.pdf
RECOMMEND_OUTPUT ?= out/recommendation.md
# Ollama model selection (uncomment one if you want to switch default)
# Recommended for MacBook Pro 48GB RAM: good quality/speed/stability balance
# OLLAMA_MODEL ?= qwen3:14b
# OLLAMA_MODEL ?= llama3.1:latest
# OLLAMA_MODEL ?= mixtral:8x7b-instruct-v0.1-q4_K_M
# OLLAMA_MODEL ?= gemma3:27b
OLLAMA_MODEL ?= gemma3:12b
# OLLAMA_MODEL ?= gpt-oss:20b
# OLLAMA_MODEL ?= mistral:7b-instruct
# OLLAMA_MODEL ?= llama3.2:3b
# OLLAMA_MODEL ?= llama3.1:8b-instruct-q4_K_M
# OLLAMA_MODEL ?= qwen3:30b-a3b-instruct-2507-q4_K_M
# OLLAMA_MODEL ?= qwen2.5:14b-instruct
# OLLAMA_MODEL ?= qwen2.5:32b
# OLLAMA_MODEL ?= qwen3:30b
# OLLAMA_MODEL ?= deepseek-r1:14b
# OLLAMA_MODEL ?= openhermes:latest
# OLLAMA_MODEL ?= qwen:14b
# OLLAMA_MODEL ?= zephyr:7b
# OLLAMA_MODEL ?= deepseek-coder:6.7b
# OLLAMA_MODEL ?= dolphin-mistral:latest
# OLLAMA_MODEL ?= phi3:mini
# OLLAMA_MODEL ?= mistral-small3.1:24b
# OLLAMA_MODEL ?= devstral:latest
# OLLAMA_MODEL ?= gemma3:latest
# OLLAMA_MODEL ?= qwen3:32b
OLLAMA_TEMPERATURE ?= 0.2
OLLAMA_TOP_P ?= 0.9
OLLAMA_NUM_CTX ?= 8192
RECOMMENDATION_FILE ?= out/recommendation.md

DATASETS := \
	dataset/2025/2025_history_phase_a_1dec2024-1dec2025.csv \
	dataset/2025/2025_history_phase_b_1dec2024-1dec2025.csv \
	dataset/2025/2025_history_phase_c_1dec2024-1dec2025.csv

CONFIGS := $(sort $(wildcard config/*.json))

SIMULATION_JSONS := $(patsubst config/%.json,out/%.json,$(CONFIGS))

.PHONY: all
all: venv activate

.PHONY: venv
venv:
	# (Re)crée le venv avec le bon Python
	$(PYTHON) -m venv $(VENV_DIR)
	# Upgrades de base (toujours via -m pip)
	$(VENV_DIR)/bin/python -m pip install --upgrade pip setuptools wheel
ifneq ("$(wildcard $(REQ_FILE))","")
	# Installe les deps projet
	$(VENV_DIR)/bin/python -m pip install -r $(REQ_FILE)
endif
	# Ajoute/maj ulimit dans activate (sans dupliquer)
	@if ! grep -q 'ulimit -n $(ULIMIT_VALUE)' $(VENV_DIR)/bin/activate ; then \
		printf '\n# Auto-set by Makefile\nulimit -n $(ULIMIT_VALUE)\n' >> $(VENV_DIR)/bin/activate ; \
		echo "ulimit set to $(ULIMIT_VALUE) inside the virtual environment."; \
	else \
		echo "ulimit already set to $(ULIMIT_VALUE) in activate."; \
	fi

.PHONY: activate
activate:
	@echo "Run the following command to activate the virtual environment:"
	@echo "source $(VENV_DIR)/bin/activate"


.PHONY: simulate_all
simulate_all:
	@echo "Running battery simulation for all configurations..."
	@for cfg in $(CONFIGS); do \
		echo "----------------------------------------"; \
		echo "Running with $$cfg"; \
		$(VENV_DIR)/bin/python battery_sim.py $(DATASETS) --config $$cfg; \
	done

.PHONY: run_notebooks
run_notebooks:
	@echo "Executing notebooks to refresh graphs/images..."
	@$(VENV_DIR)/bin/python -c "import nbconvert" >/dev/null 2>&1 || { \
		echo "nbconvert is not installed in $(VENV_DIR)."; \
		echo "Install it with: $(VENV_DIR)/bin/python -m pip install nbconvert"; \
		exit 1; \
	}
	@echo "Cleaning previous exported graph images (month/season/global)..."
	@mkdir -p "$(MONTHLY_GRAPHS_DIR)" "$(SEASONAL_GRAPHS_DIR)" "$(GLOBAL_GRAPHS_DIR)"
	@find "$(MONTHLY_GRAPHS_DIR)" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' -o -name '*.svg' \) -delete
	@find "$(SEASONAL_GRAPHS_DIR)" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' -o -name '*.svg' \) -delete
	@find "$(GLOBAL_GRAPHS_DIR)" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' -o -name '*.svg' \) -delete
	@mkdir -p "$(NOTEBOOK_MPLCONFIGDIR)"
	@MPLCONFIGDIR="$(NOTEBOOK_MPLCONFIGDIR)" $(VENV_DIR)/bin/python -m nbconvert \
		--to notebook \
		--execute \
		--inplace \
		--ExecutePreprocessor.timeout=$(NOTEBOOK_TIMEOUT) \
		"$(MONTH_NOTEBOOK)"
	@MPLCONFIGDIR="$(NOTEBOOK_MPLCONFIGDIR)" $(VENV_DIR)/bin/python -m nbconvert \
		--to notebook \
		--execute \
		--inplace \
		--ExecutePreprocessor.timeout=$(NOTEBOOK_TIMEOUT) \
		"$(SEASON_NOTEBOOK)"
	@if [ -f "$(GLOBAL_NOTEBOOK)" ]; then \
		MPLCONFIGDIR="$(NOTEBOOK_MPLCONFIGDIR)" $(VENV_DIR)/bin/python -m nbconvert \
			--to notebook \
			--execute \
			--inplace \
			--ExecutePreprocessor.timeout=$(NOTEBOOK_TIMEOUT) \
			"$(GLOBAL_NOTEBOOK)"; \
	else \
		echo "Global notebook not found: $(GLOBAL_NOTEBOOK) (skipping)."; \
	fi

.PHONY: pdf_report
pdf_report:
	@echo "Building PDF report from notebook graph exports..."
	@if [ ! -d "$(MONTHLY_GRAPHS_DIR)" ]; then \
		echo "Monthly graphs directory does not exist: $(MONTHLY_GRAPHS_DIR)"; \
		echo "Create it and export notebook figures there (png/jpg/jpeg/svg)."; \
		exit 1; \
	fi
	@if [ ! -d "$(SEASONAL_GRAPHS_DIR)" ]; then \
		echo "Seasonal graphs directory does not exist: $(SEASONAL_GRAPHS_DIR)"; \
		echo "Create it and export notebook figures there (png/jpg/jpeg/svg)."; \
		exit 1; \
	fi
	@monthly_files=$$(find "$(MONTHLY_GRAPHS_DIR)" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' -o -name '*.svg' \) | sort); \
	if [ -z "$$monthly_files" ]; then \
		echo "No monthly images found in $(MONTHLY_GRAPHS_DIR)"; \
		exit 1; \
	fi; \
	seasonal_files=$$(find "$(SEASONAL_GRAPHS_DIR)" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' -o -name '*.svg' \) | sort); \
	if [ -z "$$seasonal_files" ]; then \
		echo "No seasonal images found in $(SEASONAL_GRAPHS_DIR)"; \
		exit 1; \
	fi; \
	global_args=""; \
	if [ -d "$(GLOBAL_GRAPHS_DIR)" ]; then \
		global_files=$$(find "$(GLOBAL_GRAPHS_DIR)" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' -o -name '*.svg' \) | sort); \
		if [ -n "$$global_files" ]; then \
			echo "Including global images from $(GLOBAL_GRAPHS_DIR)"; \
			global_args="--global $$global_files"; \
		else \
			echo "No global images found in $(GLOBAL_GRAPHS_DIR); continuing without global section."; \
		fi; \
	else \
		echo "Global graphs directory does not exist: $(GLOBAL_GRAPHS_DIR) (continuing without global section)."; \
	fi; \
	$(VENV_DIR)/bin/python generate_pdf_report.py \
		$$global_args \
		--configs $(CONFIGS) \
		--simulation-jsons $(SIMULATION_JSONS) \
		--recommendation-file "$(RECOMMENDATION_FILE)" \
		--monthly $$monthly_files \
		--seasonal $$seasonal_files \
		--output "$(PDF_REPORT_OUTPUT)" \
		--title "$(PDF_REPORT_TITLE)" \
		--subtitle "$(PDF_REPORT_SUBTITLE)" \
		--intro "$(PDF_REPORT_INTRO)"

.PHONY: recommend
recommend:
	@echo "Generating recommendation from PDF with local Ollama..."
	@$(VENV_DIR)/bin/python generate_recommendation.py \
		"$(RECOMMEND_INPUT_PDF)" \
		--model "$(OLLAMA_MODEL)" \
		--temperature "$(OLLAMA_TEMPERATURE)" \
		--top-p "$(OLLAMA_TOP_P)" \
		--num-ctx "$(OLLAMA_NUM_CTX)" \
		--output "$(RECOMMEND_OUTPUT)"
