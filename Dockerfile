# Usa una imagen base con Git instalado
FROM python:3.11

# Establece la zona horaria
ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Instala Git
RUN apt-get update && apt-get install -y git

# Clone the GitHub repository using a personal access token
RUN git clone https://f92esmup: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@github.com/f92esmup/CICProject.git

# Establece el directorio de trabajo al repositorio clonado
WORKDIR /CICProject

# Instala PyTorch
RUN pip install torch torchvision torchaudio

# Install OpenMIM and MMDetection dependencies
RUN pip install -U openmim \
    && mim install mmengine \
    && mim install mmcv-full \
    && git clone https://github.com/open-mmlab/mmdetection.git /CICProject/Image_detection/mmdetection

RUN pip install -e /CICProject/Image_detection/mmdetection

# Installs the required Python packages listed in the requirements.txt file
RUN pip install -r requirements.txt

# Set the Hugging Face token as an environment variable
ENV HUGGINGFACE_API_KEY=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

# Downloads the necessary weights and configuration files by running the Download_weights_configs.py script
RUN python Download_weights_configs.py

# Especifica el comando para ejecutar tu aplicación
CMD ["python", "main.py"]