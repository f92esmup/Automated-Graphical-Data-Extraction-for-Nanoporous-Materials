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

# Installs the required Python packages listed in the requirements.txt file
RUN pip install -r requirements.txt

# Downloads the necessary weights and configuration files by running the Download_weights&configs.py script
RUN python Download_weights_configs.py

# Especifica el comando para ejecutar tu aplicación
CMD ["python", "main.py"]