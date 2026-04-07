#!/usr/bin/env bash
set -euo pipefail

# run_ewstools.sh
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

MINICONDA_DIR="$HOME/miniconda3"
MINICONDA_SH="Miniconda3-latest-Linux-x86_64.sh"
MINICONDA_URL="https://repo.anaconda.com/miniconda/$MINICONDA_SH"

if [ -x "$MINICONDA_DIR/bin/conda" ]; then
    echo "Miniconda found at $MINICONDA_DIR"
else
    echo "Miniconda not found. Installing Miniconda to $MINICONDA_DIR"
    wget -q "$MINICONDA_URL" -O "/tmp/$MINICONDA_SH"
    bash "/tmp/$MINICONDA_SH" -b -p "$MINICONDA_DIR"
    rm -f "/tmp/$MINICONDA_SH"
    if ! grep -q 'miniconda3/bin' "$HOME/.bashrc" 2>/dev/null; then
        echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> "$HOME/.bashrc"
    fi
    source "$HOME/.bashrc" || true
fi

# Ensure conda is available in this shell
if [ -f "$MINICONDA_DIR/etc/profile.d/conda.sh" ]; then
    source "$MINICONDA_DIR/etc/profile.d/conda.sh"
else
    export PATH="$MINICONDA_DIR/bin:$PATH"
fi

# Create conda environment if it does not exist
if conda info --envs | awk '{print $1}' | grep -qx "ewstools-py39"; then
    echo "Conda environment 'ewstools-py39' already exists"
else
    echo "Creating conda environment 'ewstools-py39' with Python 3.9"
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    conda create -n ewstools-py39 python=3.9 -y
fi

# Call setup script so user need not run it manually
SETUP_SCRIPT="$PROJECT_ROOT/setup_ewstools.sh"
if [ -f "$SETUP_SCRIPT" ]; then
    echo "Running setup_ewstools.sh..."
    bash "$SETUP_SCRIPT"
else
    echo "Error: setup_ewstools.sh not found in $PROJECT_ROOT"
    exit 1
fi

# Activate environment, ensure dependencies, run tutorial
conda activate ewstools-py39

##echo "Installing R packages for logistic regression analysis..."
##conda install -n ewstools-py39 -c conda-forge r-logistf r-pROC r-jsonlite r-lme4 r-broom.mixed -y

#if [ -f "requirements.txt" ]; then
#    echo "Installing dependencies from requirements.txt..."
#    pip install --upgrade pip
##    # First install numpy and pandas with pip to ensure compatibility
##    pip install --force-reinstall numpy==1.24.3 pandas==2.0.3
#    # Then install the rest of requirements
#    pip install -r requirements.txt
#else
#    echo "No requirements.txt found at project root"
#fi

#echo "Installing ewstools (pip install ewstools)..."
#pip install ewstools || echo "Warning: pip install ewstools returned a non-zero exit code"

if [ -d "tutorials" ]; then
    cd tutorials
#    echo "Running remove_duplicates.py..."
#    python remove_duplicates.py
#    echo "Running remove_duplicates_2.py..."
#    python remove_duplicates_2.py
#    echo "Running main.py..."
#    python main.py
#    echo "Running species_status_assessment.py..."
#    python species_status_assessment.py
#    echo "Running cropping.py..."
#    python cropping.py
#    echo "Running time_series.py..."
#    python time_series.py
#    echo "Running biomes.py..."
#    python biomes.py
    echo "Running comparison_bayesian.py..."
    python comparison_bayesian.py
#    echo "Running data_plot.py..."
#    python data_plot.py
    echo "Running main_plot_species.py..."
    python main_plot_species.py
    echo "Running transition_year.py..."
    python transition_year.py
else
    echo "Error: tutorials directory not found in $PROJECT_ROOT"
    conda deactivate
    exit 1
fi

conda deactivate
echo "run_ewstools.sh completed."
