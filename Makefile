# Makefile for Uncertainty-Aware Grid-Edge Control Research Repository

PYTHON = python

.PHONY: help setup data train sensitivities test smoke experiments figures report clean

help:
	@echo "Available commands:"
	@echo "  make setup          - Install dependencies in editable development mode"
	@echo "  make data           - Fetch feeder, load data, weather data, and prepare datasets"
	@echo "  make train          - Train probabilistic forecasting models and calibrate residuals"
	@echo "  make sensitivities  - Estimate voltage sensitivities on IEEE 123 feeder via OpenDSS"
	@echo "  make test           - Run full pytest test suite"
	@echo "  make smoke          - Run multi-hour closed loop smoke test"
	@echo "  make experiments    - Run full 3x5x3 experiment matrix and ablation studies"
	@echo "  make figures        - Generate publication figures and statistical tables"
	@echo "  make clean          - Remove temporary files and caches"

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

data:
	$(PYTHON) scripts/fetch_feeder.py
	$(PYTHON) scripts/fetch_load_data.py
	$(PYTHON) scripts/fetch_weather_data.py
	$(PYTHON) scripts/prepare_data.py

train:
	$(PYTHON) scripts/train_forecasts.py
	$(PYTHON) scripts/calibrate_forecasts.py

sensitivities:
	$(PYTHON) scripts/estimate_sensitivities.py

test:
	$(PYTHON) -m pytest tests -v

smoke:
	$(PYTHON) scripts/run_experiment.py --controller rule_based --pv-penetration 0.50 --uncertainty nominal --steps 6 --run-id smoke_test

experiments:
	$(PYTHON) scripts/run_experiment_matrix.py --steps 72

figures:
	$(PYTHON) scripts/generate_report_assets.py

clean:
	rm -rf .pytest_cache build dist *.egg-info
