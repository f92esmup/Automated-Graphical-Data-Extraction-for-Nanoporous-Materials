# Use a base Python image
FROM python:3.11-slim

# Set the timezone
ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    ca-certificates \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libgdbm-dev \
    libdb5.3-dev \
    libbz2-dev \
    libexpat1-dev \
    liblzma-dev \
    tk-dev \
    libffi-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Add NVIDIA package repositories
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.2.89-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu1804_10.2.89-1_amd64.deb && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub && \
    apt-get update && \
    apt-get install -y cuda

# Install cuDNN
RUN apt-get install -y libcudnn7


# Instala otras dependencias
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y poppler-utils \
    && apt-get install -y libgl1-mesa-glx





#### REQUERIMIENTOS DE LA APLICACIÓN ####
# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install required Python packages
RUN pip install torch torchvision torchaudio \
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
    && mim install mmcv-full \
    && pip install pycocotools

###########################################



# Copia los archivos de la aplicación en el contenedor
COPY . /CICProject
#git clone https://f92esmup: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@github.com/f92esmup/CICProject.git

# Establece el directorio de trabajo al directorio copiado
WORKDIR /CICProject

# Downloads the necessary weights and configuration files by running the Download_weights_configs.py script
RUN python Download_weights_configs.py

# Especifica el comando para ejecutar tu aplicación
CMD ["python", "main.py"]