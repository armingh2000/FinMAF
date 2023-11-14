#!/bin/bash

# Cleanup Script for Reverting Setup Changes

# ENV_NAME="FinMAF_env"

# # Remove Conda environment
# if conda info --envs | grep -q $ENV_NAME; then
#     echo "Removing Conda environment '$ENV_NAME'..."
#     conda remove --name $ENV_NAME --all -y
# else
#     echo "Conda environment '$ENV_NAME' does not exist. No need to remove."
# fi

# Remove created directories
echo "Removing directories..."
rm -rf ../data/historical/hist ../data/historical/stocks ../data/historical/etfs

echo "Cleanup complete. Conda environment '$ENV_NAME' and created directories have been removed."
