#!/usr/bin/env bash
set -uo pipefail

# run_model.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

ENV_NAME="ewstools-py39"
MINICONDA_DIR="$HOME/miniconda3"
MINICONDA_SH="Miniconda3-latest-Linux-x86_64.sh"
MINICONDA_URL="https://repo.anaconda.com/miniconda/$MINICONDA_SH"

# Install Miniconda if not present (to have a consistent Python 3.9)
if [ ! -x "$MINICONDA_DIR/bin/conda" ]; then
    echo "Miniconda not found. Installing to $MINICONDA_DIR"
    wget -q "$MINICONDA_URL" -O "/tmp/$MINICONDA_SH"
    bash "/tmp/$MINICONDA_SH" -b -p "$MINICONDA_DIR"
    rm -f "/tmp/$MINICONDA_SH"
    # add to PATH for future shells
    if ! grep -q 'miniconda3/bin' "$HOME/.bashrc" 2>/dev/null; then
        echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> "$HOME/.bashrc"
    fi
fi

# Ensure conda is available in this shell
if [ -f "$MINICONDA_DIR/etc/profile.d/conda.sh" ]; then
    source "$MINICONDA_DIR/etc/profile.d/conda.sh"
else
    export PATH="$MINICONDA_DIR/bin:$PATH"
fi

# Create a fresh environment (remove any existing one first)
if conda info --envs | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Removing existing Conda environment '$ENV_NAME' to create a fresh one."
    conda env remove -n "$ENV_NAME" -y
fi

echo "Creating Conda environment '$ENV_NAME' with Python 3.9"
conda create -n "$ENV_NAME" python=3.9 -y

# Activate the environment
conda activate "$ENV_NAME"

# Install required packages (will only install missing/updated ones)
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "No requirements.txt found at project root"
fi

# Ensure ewstools is installed
echo "Installing ewstools..."
pip install ewstools || echo "Warning: pip install ewstools returned a non-zero exit code"

# List of Python scripts to run
scripts=(
    "main.py"
    "remove_duplicates.py"
    "remove_duplicates_2.py"
    "species_status_assessment.py"
    "cropping.py"
    "time_series.py"
    "biomes.py"
    "comparison_bayesian.py"
    "data_plot.py"
    "main_plot_species.py"
    "transition_year.py"
)

echo "Running Python scripts in environment '$ENV_NAME'..."
for script in "${scripts[@]}"; do
    if [ -f "$script" ]; then
        echo "Running $script ..."
        python "$script" || echo "Warning: $script failed, continuing to next."
    else
        echo "Warning: $script not found, skipping."
    fi
done

echo "Run completed. Environment '$ENV_NAME' remains for future use."
