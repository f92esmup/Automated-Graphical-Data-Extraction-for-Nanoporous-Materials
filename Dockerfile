# Usa una imagen base con soporte para PyTorch y CUDA
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

# Set the timezone
ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Instala las dependencias necesarias
RUN apt-get update && apt-get install -y \
    wget \
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
    curl \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Instala Rust y Cargo
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Descarga e instala Python 3.11
RUN wget https://www.python.org/ftp/python/3.11.0/Python-3.11.0.tgz \
    && tar xzf Python-3.11.0.tgz \
    && cd Python-3.11.0 \
    && ./configure --enable-optimizations \
    && make altinstall \
    && cd .. \
    && rm -rf Python-3.11.0 Python-3.11.0.tgz

# Establece Python 3.11 como predeterminado
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.11 1

# Crea un entorno virtual
RUN python3 -m venv /opt/venv

# Activa el entorno virtual y actualiza pip
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip

# Instala los paquetes de Python requeridos en el entorno virtual
RUN pip install torch torchvision torchaudio
RUN pip install -q transformers timm einops peft
RUN pip install Pillow
RUN pip install supervision
RUN pip install bibtexparser
RUN pip install PyPDF2
RUN pip install arxiv
RUN pip install beautifulsoup4
RUN pip install undetected_chromedriver
RUN pip install crossref_commons
RUN pip install pyChainedProxy
RUN pip install terminaltables
RUN pip install bresenham
RUN pip install pdf2image
RUN pip install pandas
RUN pip install -U openmim
RUN pip install --upgrade setuptools
RUN mim install mmcv-full
RUN pip install pycocotools

# Copia los archivos de la aplicación al contenedor
COPY . /CICProject

# Establece el directorio de trabajo
WORKDIR /CICProject

# Descarga los pesos y archivos de configuración necesarios ejecutando el script Download_weights_configs.py
RUN python Download_weights_configs.py


RUN pip install google.generativeai

# Especifica el comando para ejecutar tu aplicación
CMD ["python", "main.py"]