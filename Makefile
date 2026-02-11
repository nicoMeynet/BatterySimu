# -- Makefile (macOS/Homebrew friendly) --

# Choisis explicitement le Python Homebrew si dispo, sinon python3
PYTHON := $(shell command -v /opt/homebrew/bin/python3.12 || command -v python3)
VENV_DIR := venv
REQ_FILE := requirements.txt
ULIMIT_VALUE ?= 2048

DATASETS := \
	dataset/2025/2025_history_phase_a_1dec2024-1dec2025.csv \
	dataset/2025/2025_history_phase_b_1dec2024-1dec2025.csv \
	dataset/2025/2025_history_phase_c_1dec2024-1dec2025.csv

CONFIGS := \
	config/config_Zendure2400_11520kwh.json \
	config/config_Zendure2400_14400kwh.json \
	config/config_Zendure2400_2880kwh.json \
	config/config_Zendure2400_5760kwh.json \
	config/config_Zendure2400_8640kwh.json \
	config/config_Zendure2400_noBattery.json

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
