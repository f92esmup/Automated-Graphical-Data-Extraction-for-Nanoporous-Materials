# Usa una imagen base con Git instalado
FROM python:3.11

# Establece la zona horaria
ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
# Instala Git y otras dependencias
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git \
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



# Clone the GitHub repository using a personal access token
RUN git clone https://f92esmup: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@github.com/f92esmup/CICProject.git

# Establece el directorio de trabajo al repositorio clonado
WORKDIR /CICProject

# Downloads the necessary weights and configuration files by running the Download_weights_configs.py script
RUN python Download_weights_configs.py

# Especifica el comando para ejecutar tu aplicación
CMD ["python", "main.py"]