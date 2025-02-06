# Usa una imagen base con soporte para PyTorch y CUDA
FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

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
    lsb-release \
    poppler-utils \ 
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

# Establece Python 3.9 como predeterminado y asegúrate de que pip esté disponible
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.9 1 \
    && ln -s /usr/local/bin/pip3.9 /usr/local/bin/pip

# Asegúrate de que pip esté disponible en el PATH
ENV PATH="/usr/local/bin:${PATH}"

# Instala pip explícitamente
RUN wget https://bootstrap.pypa.io/get-pip.py && python3.9 get-pip.py && rm get-pip.py

# Instala lsb-release antes de actualizar pip
RUN apt-get update && apt-get install -y lsb-release

# Upgrade pip to the latest version using python3.9
RUN python3.9 -m pip install --upgrade pip

# Instala distro después de actualizar pip
RUN python3.9 -m pip install distro

# Instala los paquetes de Python requeridos en el entorno virtual usando pip
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu122 \
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
    && pip install -U openmim \
    && pip install --upgrade setuptools \
    && pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu122/torch1.9.0/index.html \
    && pip install pycocotools

RUN pip install google.generativeai

# Copia los archivos de la aplicación al contenedor
COPY . /CICProject

# Establece el directorio de trabajo
WORKDIR /CICProject

# Descarga los pesos y archivos de configuración necesarios ejecutando el script Download_weights_configs.py
RUN python3.9 Download_weights_configs.py

# Especifica el comando para ejecutar tu aplicación
CMD ["python3.9", "main.py"]