# Usa una imagen base con Git instalado
FROM python:3.11

# Establece la zona horaria
ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Instala Git
RUN apt-get update && apt-get install -y git

# Clona el repositorio usando el token de acceso personal
#RUN git clone https://github.com/f92esmup/CICProject.git /app

# Copia el contenido del directorio actual al contenedor en /app
COPY . /app

# Establece el directorio de trabajo al repositorio clonado
WORKDIR /app

# Instala las dependencias de Python


# Añadimos los pesos:
RUN apt-get update && apt-get install -y wget unzip python3-pip && \
    pip3 install gdown && \
    gdown --id 1n9UtHgfOA6H8cxp4Y44fG7OdXbVJzMnJ -O /app/Image_detection/weights/work_dirs.zip && \
    unzip /app/Image_detection/weights/work_dirs.zip -d /app/Image_detection/weights/ && \
    rm /app/Image_detection/weights/work_dirs.zip && \
    gdown --id 1cIWM7lTisd1GajDR98IymDssvvLAKH1n -O /app/Image_detection/weights/weights.pth

# Especifica el comando para ejecutar tu aplicación
CMD ["python", "main.py"]