.PHONY: help install train evaluate test clean docker-build docker-run

help:
	@echo "Alzheimer's Disease Classification Model - Available Commands"
	@echo "=============================================================="
	@echo "make install          - Install Python dependencies"
	@echo "make generate-data    - Generate synthetic dataset"
	@echo "make train            - Run complete training pipeline"
	@echo "make evaluate         - Evaluate trained models"
	@echo "make test             - Run test suite"
	@echo "make clean            - Remove generated files and cache"
	@echo "make docker-build     - Build Docker image"
	@echo "make docker-run       - Run Docker container"

install:
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"

generate-data:
	python data/raw/generate_data.py
	@echo "✓ Synthetic dataset generated"

train: generate-data
	python scripts/train_pipeline.py
	@echo "✓ Training pipeline completed"

evaluate:
	python scripts/evaluate_model.py
	@echo "✓ Model evaluation completed"

test:
	pytest tests/ -v --tb=short
	@echo "✓ Tests completed"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	rm -f data/processed/*.csv
	rm -f models/*.pkl models/*.json
	rm -f results/metrics/*.json results/predictions/*.csv results/plots/*.png
	@echo "✓ Cleaned up generated files"

docker-build:
	docker build -f docker/Dockerfile -t alzheimers-classifier:latest .
	@echo "✓ Docker image built"

docker-run: docker-build
	docker run -it --rm -v $(PWD):/app alzheimers-classifier:latest make train
	@echo "✓ Docker container executed"

venv:
	python -m venv venv
	@echo "✓ Virtual environment created. Run: source venv/bin/activate"

requirements:
	pip freeze > requirements.txt
	@echo "✓ Requirements frozen"
