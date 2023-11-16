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

    # Activate environment
    echo "Activating the environment..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate $ENV_NAME

    # Install Python packages from requirements.txt
    echo "Installing Python packages..."
    pip install -r requirements.txt
fi

# Define the directories
dir1="../data/historical"
dir2="../logs"

echo "Creating directories..."

# Function to create directory if it doesn't exist
create_dir_if_not_exists() {
    local dir=$1
    if [ ! -d "$dir" ]; then
        echo "Creating directory: $dir"
        mkdir -p "$dir"
    else
        echo "Directory already exists: $dir"
    fi
}

# Create directories
create_dir_if_not_exists "$dir1"
create_dir_if_not_exists "$dir2"

echo "Setup complete. Conda environment '$ENV_NAME' is ready. Activate it using 'conda activate $ENV_NAME' and run your script."
