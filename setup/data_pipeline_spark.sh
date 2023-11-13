#!/bin/bash

# Setup Script for Data Gathering with Conda Environment

ENV_NAME="FinMAF_env"

# Check for Conda
if ! command -v conda &> /dev/null
then
    echo "Conda could not be found, please install it."
    exit 1
fi

# Check if the environment already exists
if conda info --envs | grep -q $ENV_NAME; then
    echo "Conda environment '$ENV_NAME' already exists. Skipping creation."
else
    # Create Conda environment
    echo "Creating Conda environment..."
    conda create -n $ENV_NAME python=3.11 -y
fi

# Activate environment
echo "Activating the environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Install Python packages from requirements.txt
echo "Installing Python packages..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p ../data/historical/hist ../data/historical/stocks ../data/historical/etfs

echo "Setup complete. Conda environment '$ENV_NAME' is ready. Activate it using 'conda activate $ENV_NAME' and run your script."
