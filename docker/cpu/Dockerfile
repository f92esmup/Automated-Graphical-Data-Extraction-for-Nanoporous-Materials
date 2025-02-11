# Use a base image with Git installed
FROM python:3.11

# Set the timezone
ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install Git and other dependencies
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y poppler-utils \
    && apt-get install -y libgl1-mesa-glx

#### APPLICATION REQUIREMENTS ####
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

# Copy application files into the container
COPY . /CICProject

# Set the working directory to the copied directory
WORKDIR /CICProject

# Downloads the necessary weights and configuration files by running the Download_weights_configs.py script
RUN python Download_weights_configs.py

# Specify the command to run your application
CMD ["python", "main.py"]