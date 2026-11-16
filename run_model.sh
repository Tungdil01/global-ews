#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# --- BioTIME Cache Check ---
BIOTIME_PKL="codes/csv/datasets/biotime_1f9effe9.pkl"
BIOTIME_URL="https://github.com/Tungdil01/global-ews/releases/download/data-biotime-v1/biotime_1f9effe9.pkl"
if [ ! -f "$BIOTIME_PKL" ]; then
    echo "======================================================================"
    echo "BioTIME cache missing. Auto-downloading (~627 MB) from GitHub Release..."
    echo "======================================================================"
    mkdir -p "$(dirname "$BIOTIME_PKL")"
    wget -q --show-progress -O "$BIOTIME_PKL" "$BIOTIME_URL" || { echo "Download failed. Please download manually."; exit 1; }
fi

run_python_script() { local script="$1"; echo -e "\033[1;31mRunning $script...\033[0m"; python "$script"; }
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
if conda info --envs | awk '{print $1}' | grep -qx "ewstools-py310"; then
    echo "Conda environment 'ewstools-py310' already exists"
else
    echo "Creating conda environment 'ewstools-py310' with Python 3.10"
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    conda create -n ewstools-py310 python=3.10 -y
fi

# Activate environment, ensure dependencies, run tutorial
conda activate ewstools-py310
pip install --upgrade pip

# Run setup_ewstools.sh only when needed:
#   - SETUP_EWSTOOLS=1 forces a rerun
#   - setup_ewstools.sh changed since last successful run
#   - no sentinel exists yet
SETUP_SCRIPT="$PROJECT_ROOT/setup_ewstools.sh"
if [ -f "$SETUP_SCRIPT" ]; then
    SETUP_STAMP="$PROJECT_ROOT/.setup_ok"
    SETUP_HASH="$(sha256sum "$SETUP_SCRIPT" | awk '{print $1}')"
    if [ "${SETUP_EWSTOOLS:-0}" = "1" ] \
       || [ ! -f "$SETUP_STAMP" ] \
       || [ "$(cat "$SETUP_STAMP")" != "$SETUP_HASH" ]; then
        echo "Running setup_ewstools.sh..."
        bash "$SETUP_SCRIPT"
        echo "$SETUP_HASH" > "$SETUP_STAMP"
    else
        echo "setup_ewstools.sh unchanged; skipping."
    fi
else
    echo "Error: setup_ewstools.sh not found in $PROJECT_ROOT"
    exit 1
fi

# Install R + R bridge only when needed:
#   - INSTALL_R_DEPS=1 forces a reinstall
#   - no sentinel exists yet
#   - rpy2 cannot be imported, or the R packages brglm2/pmlr are missing (self-heal)
# Note: pmlr is no longer on CRAN, so it is installed from GitHub (jshinb/pmlr).
R_STAMP="$PROJECT_ROOT/.r_deps_ok"
r_deps_ok() {
    python - <<'PY' >/dev/null 2>&1
import rpy2.robjects.packages as rp
rp.importr('brglm2')
rp.importr('pmlr')
PY
}
if [ "${INSTALL_R_DEPS:-0}" = "1" ] \
   || [ ! -f "$R_STAMP" ] \
   || ! r_deps_ok; then
    echo "Installing R and rpy2 (r-base, r-brglm2, rpy2) + pmlr (from GitHub)..."
    conda install -c conda-forge r-base=4.5 r-brglm2 -y
    Rscript -e 'if (!requireNamespace("remotes", quietly=TRUE)) install.packages("remotes", repos="https://cloud.r-project.org"); if (!requireNamespace("pmlr", quietly=TRUE)) remotes::install_github("jshinb/pmlr", upgrade="never")'
    pip install rpy2
    r_deps_ok
    touch "$R_STAMP"
else
    echo "R dependencies unchanged; skipping install."
fi

# Install dependencies only when needed:
#   - INSTALL_DEPS=1 forces a reinstall
#   - requirements.txt changed since last successful install
#   - no sentinel exists yet
#   - pip check reports broken requirements (self-heal)
if [ -f "requirements.txt" ]; then
    REQ_STAMP="$PROJECT_ROOT/.deps_ok"
    REQ_HASH="$(sha256sum requirements.txt | awk '{print $1}')"
    if [ "${INSTALL_DEPS:-0}" = "1" ] \
       || [ ! -f "$REQ_STAMP" ] \
       || [ "$(cat "$REQ_STAMP")" != "$REQ_HASH" ] \
       || ! pip check >/dev/null 2>&1; then
        echo "Installing dependencies from requirements.txt..."
        pip install -r requirements.txt
        pip check
        echo "$REQ_HASH" > "$REQ_STAMP"
    else
        echo "Dependencies unchanged; skipping install."
    fi
else
    echo "No requirements.txt found at project root"
    exit 1
fi

printf "\n\n\n\n----------------------------------------------------------------------\n\n\n\n"
if [ -d "codes" ]; then
    cd codes
    scripts=(
        "list_repeated.py"
        "only_year_1900.py"
        "list_threatened.py"
        "main_orchestrator.py"
        "build_short.py"
        "species_status_assessment.py"
        "biomes.py"
        "comparison_logistic_frequentist.py"
        "transition_year.py"
        "data_plot.py"
        "plot_dl_figure.py"
        "count.py"
        "data_plot_count.py"
    )
    for script in "${scripts[@]}"; do
        run_python_script "$script"
    done
else
    echo "Error: code directory not found in $PROJECT_ROOT"
    conda deactivate
    exit 1
fi

echo "Running img/convert_png.sh..."
bash img/convert_png.sh
conda deactivate
echo "run_model.sh completed."
