#!/bin/bash
# Setup script for Unified AI System

echo "Setting up Unified AI System..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Create data directories
mkdir -p data/models
mkdir -p data/logs
mkdir -p data/checkpoints

# Initialize database
python scripts/init_db.py

# Run tests
pytest tests/

echo "✓ Setup completed successfully!"
