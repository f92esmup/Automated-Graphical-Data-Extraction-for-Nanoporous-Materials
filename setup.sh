#!/bin/bash

# Crear un entorno virtual de Python
python3 -m venv venv

# Activar el entorno virtual
source venv/bin/activate

# Instalar las dependencias de requirements.txt
pip install -r requirements.txt

# Instalar openmim
pip install -U openmim

# Actualizar setuptools
pip install --upgrade setuptools

# Instalar mmcv-full usando openmim
mim install mmcv-full

# Ejecutar el archivo Download_weights_configs.py
python Download_weights_configs.py

echo "Setup completado con éxito"