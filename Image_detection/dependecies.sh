#!/bin/bash

# Create a virtual environment
python3.9 -m venv imagedetection

# Activate the virtual environment
source imagedetection/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies from requirements.txt
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu117/torch_stable.html

# Install mmcv-full
mim install mmcv-full

# Crea las carpetas input y output
mkdir -p ./weights/


# Add weights
apt-get update && apt-get install -y wget unzip python3-pip
pip3 install gdown
gdown --id 1n9UtHgfOA6H8cxp4Y44fG7OdXbVJzMnJ -O ./weights/work_dirs.zip
unzip ./weights/work_dirs.zip -d ./weights/
rm ./weights/work_dirs.zip
gdown --id 1cIWM7lTisd1GajDR98IymDssvvLAKH1n -O ./weights/weights.pth

###AUN TENGO QUE CORREGIRLO. NO ESTÁ BIEN DEL TODO#####
