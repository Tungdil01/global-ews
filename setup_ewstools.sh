#!/bin/bash

# Create and activate a virtual environment
python -m venv .env
source .env/bin/activate

# Upgrade pip and install Jupyter
pip install --upgrade pip
pip install jupyter

# Install ewstools
pip install ewstools

# Verify installation of ewstools
python -c "import ewstools" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: ewstools is not installed correctly. Please check compatibility or dependencies."
    exit 1
fi
echo "ewstools installed and verified successfully."

# Ensure Jupyter is using the correct Python kernel
python -m ipykernel install --user --name=.env --display-name "Python (.env)"

# Launch Jupyter Notebook
#jupyter notebook --ip='*' --NotebookApp.token='' --NotebookApp.password=''
