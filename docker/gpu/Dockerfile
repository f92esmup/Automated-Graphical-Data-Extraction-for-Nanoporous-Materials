# Usa una imagen base con CUDA y cuDNN preinstalados
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# Establece la zona horaria
ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Instala Git y otras dependencias
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y poppler-utils \
    && apt-get install -y libgl1-mesa-glx \
    && apt-get install -y python3-pip \
    && apt-get install -y libglib2.0-0

#### REQUERIMIENTOS DE LA APLICACIÓN ####
# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install required Python packages with CUDA support
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117 \
    && pip install -q transformers timm einops peft \
    && pip install Pillow \
    && pip install supervision \
    && pip install bibtexparser \
    && pip install PyPDF2 \
    && pip install arxiv \
    && pip install beautifulsoup4 \
    && pip install undetected_chromedriver \
    && pip install crossref_commons \
    && pip install pyChainedProxy \
    && pip install terminaltables \
    && pip install bresenham \
    && pip install pdf2image \
    && pip install pandas \
    && pip install google.generativeai \
    && pip install -U openmim \
    && pip install --upgrade setuptools \
    && mim install mmcv-full --extra-index-url https://download.openmmlab.com/mmcv/dist/cu117/torch1.12.0/index.html \
    && pip install pycocotools

###########################################

# Copia los archivos de la aplicación en el contenedor
COPY . /CICProject

# Establece el directorio de trabajo al directorio copiado
WORKDIR /CICProject

# Downloads the necessary weights and configuration files by running the Download_weights_configs.py script
RUN python Download_weights_configs.py

# Especifica el comando para ejecutar tu aplicación
CMD ["python3", "main.py"]
