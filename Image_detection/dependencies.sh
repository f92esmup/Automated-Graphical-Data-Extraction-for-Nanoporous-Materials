#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status
set -x  # Print commands and their arguments as they are executed

echo "Creating a virtual environment in the user's home directory..."
python3.9 -m venv ~/imagedetection

echo "Activating the virtual environment..."
source ~/imagedetection/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu117/torch_stable.html

echo "Installing mmcv-full..."
mim install mmcv-full

echo "Creating directories..."
mkdir -p ./weights/

echo "Updating apt-get and installing necessary packages..."
apt-get update && apt-get install -y wget unzip python3-pip

echo "Installing gdown..."
pip3 install gdown

if [ ! -d ./weights/work_dirs ]; then
    echo "Downloading weights..."
    gdown --id 1n9UtHgfOA6H8cxp4Y44fG7OdXbVJzMnJ -O ./weights/work_dirs.zip

    echo "Unzipping weights..."
    unzip ./weights/work_dirs.zip -d ./weights/
    rm ./weights/work_dirs.zip
else
    echo "Weights already downloaded."
fi

if [ ! -f ./weights/weights.pth ]; then
    echo "Downloading additional weights..."
    gdown --id 1cIWM7lTisd1GajDR98IymDssvvLAKH1n -O ./weights/weights.pth
else
    echo "Additional weights already downloaded."
fi

echo "Setup complete!"