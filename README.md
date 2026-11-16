# Global Early Warning Signals (EWS)

Code for reproducibility that computes early warning signals for ecological time series data.

Try it out in your browser: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Tungdil01/global-ews/HEAD)

## Development Environment

We use Python 3.10. A `run_model.sh` script is provided to set up a Conda environment (`ewstools-py310`) with all dependencies and run the analyses.

## Required Packages

For the exact package versions, see [requirements.txt](https://github.com/Tungdil01/global-ews/blob/main/requirements.txt).

## BioTIME cache

`code/csv/datasets/biotime_1f9effe9.pkl` (~627 MB) exceeds GitHub's file limit and is distributed as a Release asset. Download it and place it in `code/csv/datasets/` before running `run_model.sh`:

```bash
wget -P code/csv/datasets/ https://github.com/Tungdil01/global-ews/releases/download/data-biotime-v1/biotime_1f9effe9.pkl
