# -----------------------------
# Makefile for power_grid_pred
# -----------------------------

PYTHON := python
PYTHONPATH := src

# Default values for optional arguments
FORCE ?= False
EPISODES ?= 2
ACTIONS ?= 100

# ----- Main commands -----

generate_data:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m power_grid_pred.generate_data --force=$(FORCE) --n_episodes=$(EPISODES) --n_actions=$(ACTIONS)

preprocess:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m power_grid_pred.preprocess --force=$(FORCE)

train_model:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m power_grid_pred.model

viz:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m power_grid_pred.visualization

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest --cov=power_grid_pred --cov-report=term-missing tests

# ----- Dev helpers -----

install:
	pip install -e . -r requirements.txt

clean:
	rm -rf data/raw/*.parquet
	rm -rf data/raw/*.npy
	rm -rf data/processed/*.parquet
	rm -rf data/processed/figures/*.png
	rm -rf data/processed/*.joblib
	find . -name "__pycache__" -type d -exec rm -rf {} +

.PHONY: generate_data preprocess train model viz test install clean
