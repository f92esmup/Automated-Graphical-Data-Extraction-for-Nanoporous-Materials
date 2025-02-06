# CICProject

0. Realizar las anotaciones y reentrenar el modelo de Florence 2 base ft.

1. Utilizar gemini o algun modelo de interpretación de visión para clasificar las gráficas.

2. Hacer una función que interprete el texto extraido (o mandar el documento completo, ya vamos viendo). Se debe llevar a cabo las siguientes tareas:
    - Interpretación de propiedades que se mencionen en el texto
    - Esas propiedades relacionarlas con la gráfica a la que se refiere.
    - Usar al información del texto también para ayudar a la clasificación de la gráfica.

He incluido que la extracción de texto y clasificación se pueda desactivar.

## Docker
Crear la imagen:
```bash
docker build -t cicproject .
```

Crear el contenedor:
```bash
sudo docker run -v ./data:/CICProject/data --name cicproject-container -d cicproject
```
-d: datached mode, ejecuta el contenedor en segundo plano
poner docker run --rm -v ...  elimina el contenedor una vez ha terminado su ejecución.


## Instalación
Para instalar las utilidades necesarias, ejecute el siguiente comando:
```bash
sudo apt-get install poppler-utils
```
Para instalar las dependencias necesarias, ejecute el siguiente comando:
```bash
pip install --upgrade pip

pip install torch torchvision torchaudio

pip install -U openmim

pip install --upgrade setuptools

mim install mmcv-full 



pip install -q transformers timm einops peft
pip install Pillow
pip install supervision


pip install bibtexparser
pip install PyPDF2
pip install arxiv
pip install beautifulsoup4
pip install undetected_chromedriver
pip install crossref_commons
pip install pyChainedProxy
pip install terminaltables
pip install bresenham
pip install pdf2image
pip install pandas
pip install google.generativeai
pip install pycocotools
```