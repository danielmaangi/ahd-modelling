# HIV Disease Prediction - Makefile
# Uses uv for fast Python package management

.PHONY: help setup install clean train api test docker deploy

# Default target
help:
	@echo "🏥 HIV Disease Prediction - Available Commands"
	@echo "=============================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  setup          - Complete project setup with uv"
	@echo "  install        - Install dependencies with uv"
	@echo "  clean          - Clean up generated files"
	@echo ""
	@echo "Development Commands:"
	@echo "  train          - Run the complete training pipeline"
	@echo "  api            - Start the FastAPI development server"
	@echo "  test           - Run all tests"
	@echo ""
	@echo "Deployment Commands:"
	@echo "  docker         - Build Docker image"
	@echo "  deploy         - Deploy to Kubernetes"
	@echo ""
	@echo "Utility Commands:"
	@echo "  format         - Format code with black and isort"
	@echo "  lint           - Run linting with flake8"
	@echo "  check          - Run all code quality checks"

# Setup the project
setup:
	@echo "🚀 Setting up HIV Disease Prediction project..."
	@./setup.sh

# Install dependencies with uv
install:
	@echo "📦 Installing dependencies with uv..."
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	@uv venv --python 3.9
	@uv pip install -r requirements.txt
	@echo "✅ Dependencies installed successfully"

# Clean up generated files
clean:
	@echo "🧹 Cleaning up..."
	@rm -rf .venv/
	@rm -rf __pycache__/
	@rm -rf src/__pycache__/
	@rm -rf src/*/__pycache__/
	@rm -rf src/*/*/__pycache__/
	@rm -rf .pytest_cache/
	@rm -rf *.egg-info/
	@rm -rf build/
	@rm -rf dist/
	@rm -rf data/processed/*
	@rm -rf data/synthetic/*
	@rm -rf models/*
	@rm -rf logs/*
	@rm -rf reports/*
	@rm -rf plots/*
	@echo "✅ Cleanup completed"

# Run the training pipeline
train:
	@echo "🤖 Starting training pipeline..."
	@python scripts/train_complete_pipeline.py --n-samples 1000 --create-deployment
	@echo "✅ Training completed"

# Start the API server
api:
	@echo "🚀 Starting FastAPI server..."
	@uvicorn src.api.app:app --host 0.0.0.0 --port 5000 --reload

# Run tests
test:
	@echo "🧪 Running tests..."
	@python -m pytest tests/ -v --tb=short
	@echo "✅ Tests completed"

# Format code
format:
	@echo "🎨 Formatting code..."
	@python -m black src/ scripts/ tests/ --line-length 88
	@python -m isort src/ scripts/ tests/ --profile black
	@echo "✅ Code formatted"

# Run linting
lint:
	@echo "🔍 Running linting..."
	@python -m flake8 src/ scripts/ tests/ --max-line-length 88 --extend-ignore E203,W503
	@echo "✅ Linting completed"

# Run all code quality checks
check: format lint test
	@echo "✅ All quality checks passed"

# Build Docker image
docker:
	@echo "🐳 Building Docker image..."
	@docker build -t hiv-prediction-api:latest .
	@echo "✅ Docker image built successfully"

# Deploy to Kubernetes (requires kubectl)
deploy:
	@echo "☸️ Deploying to Kubernetes..."
	@if ! command -v kubectl >/dev/null 2>&1; then \
		echo "❌ kubectl not found. Please install kubectl first."; \
		exit 1; \
	fi
	@kubectl apply -f kubernetes/
	@echo "✅ Deployment completed"

# Quick start - setup and train
quickstart: setup train
	@echo "🎉 Quick start completed!"
	@echo "Run 'make api' to start the API server"

# Development setup
dev-setup: install
	@echo "🛠️ Setting up development environment..."
	@uv pip install black isort flake8 pytest pytest-cov
	@echo "✅ Development environment ready"

# Generate synthetic data only
generate-data:
	@echo "📊 Generating synthetic data..."
	@python -c "from src.data.data_generator import HIVDataGenerator; g = HIVDataGenerator(); d = g.generate_dataset(1000); g.save_dataset(d, 'data/synthetic/hiv_data.csv')"
	@echo "✅ Synthetic data generated (CD4 count used for target creation only, not as input feature)"

# Run API tests
test-api:
	@echo "🔌 Testing API endpoints..."
	@python tests/test_api.py
	@echo "✅ API tests completed"

# Show project status
status:
	@echo "📊 Project Status"
	@echo "================"
	@echo "Python version: $(shell python --version)"
	@echo "uv version: $(shell uv --version 2>/dev/null || echo 'Not installed')"
	@echo "Virtual env: $(shell echo $$VIRTUAL_ENV || echo 'Not activated')"
	@echo ""
	@echo "Project structure:"
	@find . -name "*.py" -path "./src/*" | wc -l | xargs echo "Python files:"
	@find . -name "*.yaml" -o -name "*.yml" | wc -l | xargs echo "Config files:"
	@echo ""
	@echo "Data files:"
	@ls -la data/ 2>/dev/null || echo "No data directory"
	@echo ""
	@echo "Model files:"
	@ls -la models/ 2>/dev/null || echo "No models directory"

# Install development dependencies
install-dev:
	@echo "🛠️ Installing development dependencies..."
	@uv pip install black isort flake8 pytest pytest-cov jupyter notebook
	@echo "✅ Development dependencies installed"

# Start Jupyter notebook
notebook:
	@echo "📓 Starting Jupyter notebook..."
	@jupyter notebook notebooks/

# Run performance tests
perf-test:
	@echo "⚡ Running performance tests..."
	@python tests/performance_test.py
	@echo "✅ Performance tests completed"

# Update dependencies
update:
	@echo "🔄 Updating dependencies..."
	@uv pip install --upgrade -r requirements.txt
	@echo "✅ Dependencies updated"

# Show help for uv commands
uv-help:
	@echo "📦 UV Commands Reference"
	@echo "======================="
	@echo "uv venv                    - Create virtual environment"
	@echo "uv pip install <package>  - Install package"
	@echo "uv pip install -r req.txt - Install from requirements"
	@echo "uv pip list               - List installed packages"
	@echo "uv pip freeze             - Show installed packages with versions"
	@echo "uv pip uninstall <pkg>    - Uninstall package"
	@echo "uv --help                 - Show uv help"
