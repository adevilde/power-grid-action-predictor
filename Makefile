# -----------------------------
# Makefile for power_grid_pred
# -----------------------------

PYTHONPATH := src
UV_PYTHON := .venv/bin/python

# Default values for optional arguments
FORCE ?= False
EPISODES ?= 2
ACTIONS ?= 100

# ----- Main commands -----

generate_data:
	PYTHONPATH=$(PYTHONPATH) $(UV_PYTHON) -m power_grid_pred.generate_data --force=$(FORCE) --n_episodes=$(EPISODES) --n_actions=$(ACTIONS)

preprocess:
	PYTHONPATH=$(PYTHONPATH) $(UV_PYTHON) -m power_grid_pred.preprocess --force=$(FORCE)

train_model:
	PYTHONPATH=$(PYTHONPATH) $(UV_PYTHON) -m power_grid_pred.model

viz:
	PYTHONPATH=$(PYTHONPATH) $(UV_PYTHON) -m power_grid_pred.visualization

test:
	PYTHONPATH=$(PYTHONPATH) $(UV_PYTHON) -m pytest --cov=power_grid_pred --cov-report=term-missing tests

# ----- Dev helpers -----

init:
	uv venv --python 3.12 .venv
	uv pip install --python $(UV_PYTHON) -r requirements.txt
	uv pip install --python $(UV_PYTHON) -e .

clean:
	rm -rf data/raw/*.parquet
	rm -rf data/raw/*.npy
	rm -rf data/processed/*.parquet
	rm -rf data/processed/*.joblib
	rm -rf data/figures/*.png
	find . -name "__pycache__" -type d -exec rm -rf {} +

.PHONY: init generate_data preprocess train_model viz test clean
