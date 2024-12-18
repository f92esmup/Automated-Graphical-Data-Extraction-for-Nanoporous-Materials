# Usa una imagen base de NVIDIA que soporte CUDA
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Establece la zona horaria
ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Instala Python 3.9.20
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.9 python3.9-distutils python3-pip && \
    apt-get update && \
    apt-get install -y libgl1-mesa-glx

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Crea las carpetas input y output
RUN mkdir -p /app/data /app/data/images /app/data/Line_output /app/Image_detection /app/Image_detection/weights/

# Copia el archivo requirements.txt al contenedor
COPY ./requirements.txt .

# Instala las dependencias listadas en requirements.txt
RUN pip install -r requirements.txt -f https://download.pytorch.org/whl/cu117/torch_stable.html
#RUN pip install --no-cache-dir -r requirements.txt
RUN mim install mmcv-full

# Copia el resto del código de la aplicación al contenedor
COPY . ./Image_detection
RUN mv ./Image_detection/app.py .

# Añadimos los pesos:
RUN apt-get update && apt-get install -y wget unzip python3-pip && \
    pip3 install gdown && \
    gdown --id 1n9UtHgfOA6H8cxp4Y44fG7OdXbVJzMnJ -O /app/Image_detection/weights/work_dirs.zip && \
    unzip /app/Image_detection/weights/work_dirs.zip -d /app/Image_detection/weights/ && \
    rm /app/Image_detection/weights/work_dirs.zip && \
    gdown --id 1cIWM7lTisd1GajDR98IymDssvvLAKH1n -O /app/Image_detection/weights/weights.pth

# Especifica el comando por defecto para ejecutar tu aplicación
#Set the ENTRYPOINT to run the Python script
ENTRYPOINT ["python3", "app.py"]

# Set the CMD to provide default arguments
CMD ["--device=cuda:0"]

# Uso de bind mounts. Para correr el contenedor con bind mounts, se debe correr el siguiente comando:
# docker run -v /ruta/a/input_data:/app/input -v /ruta/a/output_data:/app/output my_conda_app