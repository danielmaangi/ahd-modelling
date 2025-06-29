#!/bin/bash
"""
HIV Disease Prediction - Setup Script

This script sets up the HIV disease prediction project using uv for fast package management.
"""

set -e  # Exit on any error

echo "🏥 HIV Disease Prediction - Project Setup"
echo "========================================"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    echo "✅ uv installed successfully"
else
    echo "✅ uv is already installed"
fi

# Create virtual environment
echo "🐍 Creating virtual environment..."
uv venv

# Activate virtual environment
echo "🔄 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi

# Install dependencies
echo "📚 Installing dependencies with uv..."
uv pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/{raw,processed,synthetic}
mkdir -p models
mkdir -p logs
mkdir -p reports
mkdir -p plots

# Set executable permissions for scripts
echo "🔧 Setting executable permissions..."
chmod +x scripts/*.py

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "   source .venv/Scripts/activate"
else
    echo "   source .venv/bin/activate"
fi
echo ""
echo "2. Run the training pipeline:"
echo "   python scripts/train_complete_pipeline.py --n-samples 1000"
echo ""
echo "3. Start the API server:"
echo "   uvicorn src.api.app:app --host 0.0.0.0 --port 5000 --reload"
echo ""
echo "4. View API documentation:"
echo "   http://localhost:5000/docs"
echo ""
echo "Happy predicting! 🚀"
