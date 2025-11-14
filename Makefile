# -----------------------------
# Makefile for power_grid_pred
# -----------------------------

PYTHON := python
PYTHONPATH := src

# ----- Main commands -----

generate_data:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m power_grid_pred.generate_data --force=$(FORCE)

preprocess:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m power_grid_pred.preprocess

viz:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m power_grid_pred.viz

train_model:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m power_grid_pred.model

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest --cov=power_grid_pred --cov-report=term-missing tests

# ----- Dev helpers (optional) -----

install:
	pip install -e . -r requirements.txt

clean:
	rm -rf data/raw/*.parquet
	rm -rf data/processed/*.parquet
	rm -rf data/processed/*.png
	rm -rf data/processed/*.joblib
	find . -name "__pycache__" -type d -exec rm -rf {} +

.PHONY: generate_data dataset viz train test install clean
