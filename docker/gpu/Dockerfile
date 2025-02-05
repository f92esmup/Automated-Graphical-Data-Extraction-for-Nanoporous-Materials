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
    libffi-dev \ 
    && rm -rf /var/lib/apt/lists/*

#Update GCC and G++
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y gcc-9 g++-9 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 60
# Instala Rust y Cargo
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Descarga e instala Python 3.9
RUN wget https://www.python.org/ftp/python/3.9.7/Python-3.9.7.tgz \
    && tar xzf Python-3.9.7.tgz \
    && cd Python-3.9.7 \
    && ./configure --enable-optimizations \
    && make altinstall \
    && cd .. \
    && rm -rf Python-3.9.7 Python-3.9.7.tgz

# Establece Python 3.9 como predeterminado
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.9 1

# Upgrade pip to the latest version
RUN python3.9 -m pip install --upgrade pip

# Instala los paquetes de Python requeridos en el entorno virtual usando python3.9 explícitamente
RUN python3.9 -m pip install torch torchvision torchaudio \
    && python3.9 -m pip install -q transformers timm einops peft \
    && python3.9 -m pip install Pillow \
    && python3.9 -m pip install supervision \
    && python3.9 -m pip install bibtexparser \
    && python3.9 -m pip install PyPDF2 \
    && python3.9 -m pip install arxiv \
    && python3.9 -m pip install beautifulsoup4 \
    && python3.9 -m pip install undetected_chromedriver \
    && python3.9 -m pip install crossref_commons \
    && python3.9 -m pip install pyChainedProxy \
    && python3.9 -m pip install terminaltables \
    && python3.9 -m pip install bresenham \
    && python3.9 -m pip install pdf2image \
    && python3.9 -m pip install pandas \
    && python3.9 -m pip install -U openmim \
    && python3.9 -m pip install --upgrade setuptools \
    && python3.9 -m mim install mmcv-full \
    && python3.9 -m pip install pycocotools

RUN python3.9 -m pip install google.generativeai

# Copia los archivos de la aplicación al contenedor
COPY . /CICProject

# Establece el directorio de trabajo
WORKDIR /CICProject

# Descarga los pesos y archivos de configuración necesarios ejecutando el script Download_weights_configs.py
RUN python3.9 Download_weights_configs.py

# Especifica el comando para ejecutar tu aplicación
CMD ["python3.9", "main.py"]