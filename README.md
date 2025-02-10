# CICProject
## Docker
Build the image:
```bash
docker build -t cicproject .
```

Create the container:
```bash
sudo docker run -v ./data:/CICProject/data --name cicproject-container -d cicproject
```
-d: detached mode, runs the container in the background  
using docker run --rm -v ... will remove the container once its execution finishes.

## Installation
To install the required utilities, run:
```bash
sudo apt-get install poppler-utils
```
To install the necessary dependencies, run:
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