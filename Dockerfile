# Use a base image with PyTorch and CUDA support
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

# Install Python 3.9 or later
RUN apt-get update && apt-get install -y python3.9 python3.9-distutils \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Set the timezone
ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install Git, build-essential (for C compiler), and other dependencies
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y poppler-utils \
    && apt-get install -y libgl1-mesa-glx \
    && apt-get install -y curl \
    && apt-get install -y build-essential

# Install Rust and Cargo
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Upgrade pip to the latest version
RUN python3 -m pip install --upgrade pip

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

# Copy the application files into the container
COPY . /CICProject

# Set the working directory to the copied directory
WORKDIR /CICProject

# Download the necessary weights and configuration files by running the Download_weights_configs.py script
RUN python Download_weights_configs.py

# Specify the command to run your application
CMD ["python", "main.py"]