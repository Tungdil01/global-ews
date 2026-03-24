#!/bin/bash

# setup_ewstools.sh – create local conda environment .env and install ewstools + jupyter

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

ENV_DIR="$PROJECT_ROOT/.env"

# Ensure conda is available (assumes Miniconda is installed; if not, run_model_test.sh installs it)
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true

# Create local conda environment if it doesn't exist
if [ ! -d "$ENV_DIR" ]; then
    echo "Creating local conda environment at $ENV_DIR with Python 3.9"
    conda create --prefix "$ENV_DIR" python=3.9 -y
else
    echo "Local conda environment already exists at $ENV_DIR"
fi

# Activate the environment
conda activate "$ENV_DIR"

# Upgrade pip and install Jupyter + ewstools
pip install --upgrade pip
pip install jupyter ewstools

# Verify ewstools
python -c "import ewstools" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: ewstools is not installed correctly."
    exit 1
fi
echo "ewstools installed and verified successfully."

# Register Jupyter kernel
python -m ipykernel install --user --name="$(basename $ENV_DIR)" --display-name "Python (.env)"

echo "Setup complete. To use: conda activate $ENV_DIR"
