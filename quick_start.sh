#!/bin/bash
# Quick start script for SRD2026

echo "=================================================="
echo "  SRD2026 Quick Start Setup"
echo "=================================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Install package in development mode
echo ""
echo "Installing SRD2026 package..."
pip install -e .

# Install development dependencies (optional)
echo ""
read -p "Install development dependencies (pytest, jupyter, etc.)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Installing development dependencies..."
    pip install -r requirements-dev.txt
fi

echo ""
echo "=================================================="
echo "  Setup Complete!"
echo "=================================================="
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
echo ""
echo "Next steps:"
echo "  1. Prepare your data"
echo "  2. Update experiments/config.yaml with your settings"
echo "  3. Run training: python scripts/train.py --config experiments/config.yaml"
echo ""
