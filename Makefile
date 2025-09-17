# LogBERT Anomaly Detection System
# Makefile for common development and deployment tasks

.PHONY: help setup venv install install-dev test train eval serve clean docker-build docker-train docker-train-mps docker-serve docker-serve-alt docker-dev docker-stop docker-clean docs lint format uv-sync uv-update

# Default target
help:
	@echo "LogBERT Anomaly Detection System"
	@echo "================================="
	@echo ""
	@echo "Available targets:"
	@echo "  setup        - Initial project setup with UV"
	@echo "  venv         - Create virtual environment with UV"
	@echo "  install      - Install dependencies with UV"
	@echo "  install-dev  - Install dev dependencies with UV"
	@echo "  test         - Run tests"
	@echo "  train        - Train LogBERT model"
	@echo "  eval         - Evaluate model performance"
	@echo "  serve        - Start inference service"
	@echo "  pipeline     - Run complete training pipeline"
	@echo "  docker-train-mps - Run training in Docker with Apple Silicon MPS"
	@echo "  docker-build - Build Docker images"
	@echo "  docker-train - Run training in Docker"
	@echo "  docker-serve - Run inference service in Docker (port 8000)"
	@echo "  docker-serve-alt - Run inference service on port 8001"
	@echo "  docker-dev   - Start interactive development container"
	@echo "  docker-stop  - Stop all LogBERT containers"
	@echo "  docker-clean - Clean up Docker resources"
	@echo "  clean        - Clean up generated files"
	@echo "  lint         - Run code linting"
	@echo "  format       - Format code with black"
	@echo "  docs         - Generate documentation"

# Configuration
CONFIG_FILE := configs/config.yaml
DATA_DIR := data
MODELS_DIR := models
LOGS_DIR := logs

# Setup project
setup:
	@echo "Setting up LogBERT project..."
	mkdir -p $(DATA_DIR)/{raw,normalized,datasets,evaluation} $(MODELS_DIR) $(LOGS_DIR)
	@echo "Checking for UV package manager..."
	@command -v uv >/dev/null 2>&1 || { echo "UV not found. Installing UV..."; curl -LsSf https://astral.sh/uv/install.sh | sh; }
	$(MAKE) venv
	$(MAKE) install
	@echo "Setup complete!"

# Create virtual environment with UV
venv:
	@echo "Creating virtual environment with UV..."
	uv venv --python 3.9
	@echo "Virtual environment created! Activate with: source .venv/bin/activate"

# Install dependencies with UV
install:
	@echo "Installing dependencies with UV..."
	uv pip install -r requirements.txt
	@echo "Dependencies installed!"

# Install development dependencies with UV
install-dev: install
	@echo "Installing development dependencies with UV..."
	uv pip install -r requirements-dev.txt
	@echo "Development dependencies installed!"

# Run tests
test:
	@echo "Running tests..."
	pytest tests/ -v --tb=short
	@echo "Tests completed!"

# Data extraction and normalization
extract-data:
	@echo "Extracting and normalizing data..."
	python3 -m normalization.drain_miner --config $(CONFIG_FILE) --input $(DATA_DIR)/raw --output $(DATA_DIR)/normalized
	@echo "Data extraction complete!"

# Build datasets
build-dataset:
	@echo "Building windowed datasets..."
	python3 -m dataset.build_windows --config $(CONFIG_FILE) --input $(DATA_DIR)/normalized --output $(DATA_DIR)/datasets
	@echo "Dataset building complete!"

# Train MLM model
train-mlm:
	@echo "Training MLM model..."
	python3 -m models.train_mlm --config $(CONFIG_FILE) --dataset $(DATA_DIR)/datasets --output $(MODELS_DIR)/mlm
	@echo "MLM training complete!"

# Train InfoNCE model (optional)
train-infonce:
	@echo "Training InfoNCE model..."
	python3 -m models.train_infonce --config $(CONFIG_FILE) --dataset $(DATA_DIR)/datasets --mlm-path $(MODELS_DIR)/mlm --output $(MODELS_DIR)/infonce
	@echo "InfoNCE training complete!"

# Train all models
train: train-mlm train-infonce
	@echo "All model training complete!"

# Inject synthetic campaigns for evaluation
inject-campaigns:
	@echo "Injecting synthetic attack campaigns..."
	python -m eval.inject_campaigns --config $(CONFIG_FILE) --input $(DATA_DIR)/datasets/test_windows.parquet --output $(DATA_DIR)/evaluation
	@echo "Campaign injection complete!"

# Generate weak labels
generate-labels:
	@echo "Generating weak labels..."
	python -m eval.weak_labels --config $(CONFIG_FILE) --input $(DATA_DIR)/evaluation/test_with_campaigns.parquet --output $(DATA_DIR)/evaluation/weak_labels.parquet
	@echo "Weak label generation complete!"

# Run model evaluation
eval: inject-campaigns generate-labels
	@echo "Running comprehensive evaluation..."
	python -m eval.compute_metrics --config $(CONFIG_FILE) --predictions $(DATA_DIR)/evaluation/predictions.parquet --campaigns $(DATA_DIR)/evaluation/campaigns.json --weak-labels $(DATA_DIR)/evaluation/weak_labels.parquet --output $(DATA_DIR)/evaluation/results.json
	@echo "Evaluation complete!"

# Start inference service
serve:
	@echo "Starting LogBERT inference service..."
	uvicorn inference_service.app:app --host 0.0.0.0 --port 8000 --reload
	@echo "Service running at http://localhost:8000"

# Run complete training pipeline with Prefect
pipeline:
	@echo "Running complete LogBERT training pipeline..."
	python -m flows.prefect_flow
	@echo "Pipeline complete!"

# Quick start for file-based data
quickstart-files:
	@echo "LogBERT Quickstart (Files Mode)"
	@echo "=============================="
	@if [ ! -f $(CONFIG_FILE) ]; then echo "Error: $(CONFIG_FILE) not found!"; exit 1; fi
	@echo "1. Extracting and normalizing logs..."
	$(MAKE) extract-data
	@echo "2. Building datasets..."
	$(MAKE) build-dataset  
	@echo "3. Training MLM model..."
	$(MAKE) train-mlm
	@echo "4. Running evaluation..."
	$(MAKE) eval
	@echo "5. Starting inference service..."
	@echo "   Run 'make serve' in another terminal"
	@echo ""
	@echo "Quickstart complete! 🎉"
	@echo "Check $(DATA_DIR)/evaluation/results.json for metrics"

# Quick start for OpenSearch data  
quickstart-opensearch:
	@echo "LogBERT Quickstart (OpenSearch Mode)"
	@echo "===================================="
	@echo "Not yet implemented - please configure OpenSearch in $(CONFIG_FILE)"

# Docker targets
docker-build:
	@echo "Building Docker images..."
	docker build --target training -t logbert:training .
	docker build --target inference -t logbert:inference .
	docker build --target development -t logbert:dev .
	@echo "Docker images built!"

docker-train: docker-build
	@echo "Running training in Docker..."
	docker run --rm -v $(PWD)/data:/home/app/data -v $(PWD)/models:/home/app/models -v $(PWD)/configs:/home/app/configs logbert:training
	@echo "Docker training complete!"

docker-train-mps:
	@echo "Running training in Docker with Apple Silicon MPS support..."
	chmod +x ./run_docker_train.sh
	./run_docker_train.sh
	@echo "Docker MPS training complete!"

docker-serve: docker-build
	@echo "Starting inference service in Docker..."
	@echo "Checking for port conflicts..."
	@docker ps --filter "publish=8000" --format "table {{.ID}}\t{{.Image}}\t{{.Ports}}" | grep -q 8000 && echo "Port 8000 in use, stopping containers..." && docker stop $$(docker ps --filter "publish=8000" -q) || true
	docker run --rm -p 8000:8000 -v $(PWD)/models:/home/app/models -v $(PWD)/configs:/home/app/configs logbert:inference
	@echo "Docker service stopped!"

docker-serve-alt: docker-build
	@echo "Starting inference service on alternative port 8001..."
	docker run --rm -p 8001:8000 -v $(PWD)/models:/home/app/models -v $(PWD)/configs:/home/app/configs logbert:inference
	@echo "Docker service stopped! (Available at http://localhost:8001)"

docker-dev: docker-build
	@echo "Starting development container..."
	@echo "Checking for port conflicts..."
	@docker ps --filter "publish=8000" --format "table {{.ID}}\t{{.Image}}\t{{.Ports}}" | grep -q 8000 && echo "Port 8000 in use, stopping containers..." && docker stop $$(docker ps --filter "publish=8000" -q) || true
	docker run --rm -it -v $(PWD):/home/app -p 8000:8000 -p 4200:4200 logbert:dev
	@echo "Development container stopped!"

docker-stop:
	@echo "Stopping all LogBERT containers..."
	@docker ps -a --filter "ancestor=logbert:training" --filter "ancestor=logbert:inference" --filter "ancestor=logbert:dev" -q | xargs -r docker stop
	@echo "All LogBERT containers stopped!"

docker-clean:
	@echo "Cleaning up Docker containers and networks..."
	@docker container prune -f
	@docker network prune -f
	@echo "Docker cleanup complete!"

# Code quality
lint:
	@echo "Running code linting..."
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	@echo "Linting complete!"

format:
	@echo "Formatting code..."
	black . --line-length=100
	@echo "Code formatted!"

type-check:
	@echo "Running type checking..."
	mypy . --ignore-missing-imports
	@echo "Type checking complete!"

quality: lint format type-check
	@echo "All quality checks complete!"

# Documentation
docs:
	@echo "Generating documentation..."
	@echo "Documentation generation not yet implemented"

# Clean up
clean:
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/ dist/ *.egg-info/
	@echo "Cleanup complete!"

clean-data:
	@echo "Cleaning data directories..."
	rm -rf $(DATA_DIR)/normalized/* $(DATA_DIR)/datasets/* $(DATA_DIR)/evaluation/*
	@echo "Data cleanup complete!"

clean-models:
	@echo "Cleaning model directories..."
	rm -rf $(MODELS_DIR)/*
	@echo "Model cleanup complete!"

clean-all: clean clean-data clean-models
	@echo "Full cleanup complete!"

# Monitoring and health checks
health-check:
	@echo "Checking system health..."
	@if [ -f $(CONFIG_FILE) ]; then echo "✓ Config file exists"; else echo "✗ Config file missing"; fi
	@if [ -d $(DATA_DIR) ]; then echo "✓ Data directory exists"; else echo "✗ Data directory missing"; fi
	@if [ -d $(MODELS_DIR) ]; then echo "✓ Models directory exists"; else echo "✗ Models directory missing"; fi
	@python -c "import torch; print('✓ PyTorch available')" || echo "✗ PyTorch not available"
	@python -c "import transformers; print('✓ Transformers available')" || echo "✗ Transformers not available"
	@python -c "import prefect; print('✓ Prefect available')" || echo "✗ Prefect not available"
	@echo "Health check complete!"

# Show configuration
show-config:
	@echo "Current configuration:"
	@echo "====================="
	@if [ -f $(CONFIG_FILE) ]; then cat $(CONFIG_FILE); else echo "No config file found at $(CONFIG_FILE)"; fi

# Show status
status:
	@echo "LogBERT System Status"
	@echo "===================="
	@echo "Config file: $(CONFIG_FILE)"
	@echo "Data directory: $(DATA_DIR)"
	@echo "Models directory: $(MODELS_DIR)"
	@echo "Logs directory: $(LOGS_DIR)"
	@echo ""
	@echo "Data files:"
	@ls -la $(DATA_DIR)/ 2>/dev/null || echo "  No data directory"
	@echo ""
	@echo "Model files:"
	@ls -la $(MODELS_DIR)/ 2>/dev/null || echo "  No models directory"
	@echo ""
	@$(MAKE) health-check

# Performance profiling
profile-train:
	@echo "Profiling training performance..."
	python -m cProfile -o train_profile.prof -m models.train_mlm --config $(CONFIG_FILE)
	@echo "Training profile saved to train_profile.prof"

profile-serve:
	@echo "Profiling inference performance..."
	# This would require additional tooling for FastAPI profiling
	@echo "Inference profiling not yet implemented"

# Backup and restore
backup:
	@echo "Creating backup..."
	tar -czf logbert_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz \
		$(CONFIG_FILE) $(DATA_DIR) $(MODELS_DIR) $(LOGS_DIR) \
		--exclude="*.pyc" --exclude="__pycache__" --exclude=".git"
	@echo "Backup created!"

# Development shortcuts with UV
dev-setup: setup venv install-dev
	@echo "Development environment ready with UV!"

dev-setup-fast: venv install-dev
	@echo "Fast development setup (skip directories) with UV!"

dev-train: extract-data build-dataset train-mlm
	@echo "Quick development training complete!"

dev-test: test lint
	@echo "Development testing complete!"

# UV-specific commands
uv-sync:
	@echo "Syncing dependencies with UV..."
	uv pip sync requirements.txt
	@echo "Dependencies synced!"

uv-update:
	@echo "Updating all dependencies with UV..."
	uv pip install --upgrade -r requirements.txt
	@echo "Dependencies updated!"

# Deployment helpers
deploy-check:
	@echo "Checking deployment readiness..."
	@$(MAKE) health-check
	@$(MAKE) test
	@echo "Deployment check complete!"

# Show help for key targets
help-quickstart:
	@echo "LogBERT Quickstart Guide (with UV)"
	@echo "=================================="
	@echo ""
	@echo "1. First time setup:"
	@echo "   make setup      # Installs UV if needed, creates venv, installs deps"
	@echo ""
	@echo "2. OR create environment manually:"
	@echo "   make venv       # Create virtual environment"
	@echo "   source .venv/bin/activate  # Activate environment"
	@echo "   make install    # Install dependencies"
	@echo ""
	@echo "2. For file-based logs:"
	@echo "   - Place log files in data/raw/"
	@echo "   - Run: make quickstart-files"
	@echo ""
	@echo "3. For OpenSearch logs:"
	@echo "   - Configure OpenSearch in configs/config.yaml"
	@echo "   - Run: make quickstart-opensearch"
	@echo ""
	@echo "4. Start inference service:"
	@echo "   make serve"
	@echo ""
	@echo "5. Check results:"
	@echo "   cat data/evaluation/results.json"

# Version information
version:
	@echo "LogBERT Anomaly Detection System"
	@echo "Version: 1.0.0"
	@echo "python: $(shell python --version)"
	@echo "PyTorch: $(shell python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
	@echo "Transformers: $(shell python3 -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo 'Not installed')"