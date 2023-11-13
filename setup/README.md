# Setup Instructions for FinMAF Environment

This directory contains scripts for setting up the environment required for running the FinMAF project.

## What does `data_pipeline_spark.sh` do?

- Checks if Conda is installed and available.
- Creates a Conda environment named `FinMAF_env` with Python 3.11 (if it doesn't already exist).
- Activates the `FinMAF_env` environment.
- Installs Python packages required for running data_pipeline/spark/ directory listed in `requirements.txt`.
- Creates necessary directories for storing data.

## What do `clean_filename.sh` scripts do?
- They revert the changes made by `filename.sh`.

## How to Use a Setup Script

1. **Ensure Conda is Installed**: The script requires Conda to be installed on your system. If you don't have Conda, please install it from [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/individual).

2. **move to setup directory**:

    ```bash
    cd setup
    ```

3. **Make the Script Executable**: 

    ```bash
    chmod +x filename.sh
    ```

4. **Run the Setup Script**:

    ```bash
    ./filename.sh
    ```

5. **Activate the Conda Environment**:

    ```bash
    conda activate FinMAF_env
    ```

## How to Revert Changes Made by a Setup Script
1. **move to setup directory**:

    ```bash
    cd setup
    ```

2. **Make the Script Executable**: 

    ```bash
    chmod +x clean_filename.sh
    ```

3. **Run the Setup Script**:

    ```bash
    ./clean_filename.sh
    ```

## Troubleshooting

- If you encounter any permission issues, make sure you have the necessary permissions to create directories and files in the specified paths.
- For any issues related to Conda, refer to the [Conda documentation](https://docs.conda.io/projects/conda/en/latest/).

---

For more information or assistance, please open an issue in this repository.
