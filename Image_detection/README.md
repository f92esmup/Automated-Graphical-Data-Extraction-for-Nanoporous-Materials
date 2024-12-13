# Environment Setup Guide

## Table of Contents
1. [Conda Environment Setup](#conda-environment-setup)
2. [Docker Environment Setup](#docker-environment-setup)
3. [Python venv Environment Setup](#python-venv-environment-setup)
4. [Instalación de Controladores de NVIDIA en el Host](#instalación-de-controladores-de-nvidia-en-el-host)
5. [Instalación del NVIDIA Container Toolkit](#instalación-del-nvidia-container-toolkit)

## Conda Environment Setup
- Step-by-step instructions on how to set up the environment using Conda.
```bash
conda env create -f environment.yml
```

## Docker Environment Setup

- Step-by-step instructions on how to set up the environment using Docker.

Crear la imagen:
```bash
docker build -t imagedetection .
```

Crear el contenedor:
```bash
sudo docker run -v ./data/images:/app/data/images -v ./data/Line_output:/app/data/Line_output --name imagedetection-container -d imagedetection
```
-d: datached mode, ejecuta el contenedor en segundo plano
poner docker run --rm -v ...  elimina el contenedor una vez ha terminado su ejecución.


Para entrar en la terminal sin contenedor corriendo:
```bash
sudo docker run -v ./data/images:/app/input -v ./data/Line_output:/app/output --name imagedetection-container -it imagedetection /bin/bash
```

Para entrar en la terminal con el contenedor corriendo:
```bash
sudo docker exec -it imagedetection-container /bin/bash
```

Limpiar caché de Docker:
```bash
docker system prune -a --volumes
```

## Python venv Environment Setup
-   Puedes ejecutar el archivo dependencies.sh haciendo:
```bash
cd Image_detection

chmod +x dependencies.sh

./dependencies.sh
```

    
O

- Step-by-step instructions on how to set up the environment using Python's venv.

Se usa python 3.9.20
pero hay que 
```bash
 python3.9 -m venv imagedetection
```
```bash
source imagedetection/bin/activate
```

```bash
pip install --upgrade pip
```

```bash
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu117/torch_stable.html
```


```bash
mim install mmcv-full
```
[CHECK] device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


## Instalación de Controladores de NVIDIA en el Host (Linux)

Actualizar el Sistema:
```bash
sudo apt update
sudo apt upgrade
```

Agregar el Repositorio de NVIDIA:
```bash
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
```

Instalar los Controladores de NVIDIA:
```bash
ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
```

Reiniciar el Sistema:
```bash
sudo reboot
```

Verificar la Instalación:
```bash
nvidia-smi
```

## Instalación del NVIDIA Container Toolkit (Linux)

Configurar el Repositorio de NVIDIA:
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```

Actualizar el Sistema:
```bash
sudo apt update
```

Instalar el NVIDIA Container Toolkit:
```bash
sudo apt install -y nvidia-docker2
```

Reiniciar el Servicio de Docker:
```bash
sudo systemctl restart docker
```

Verificación de la Configuración:
```bash
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

Uso en Contenedores:
```bash
docker run --gpus all my-gpu-application
```

## Instalación de Controladores de NVIDIA en el Host (Windows)

Descargar los Controladores: Ve al sitio web de NVIDIA y descarga los controladores más recientes para tu GPU.

Instalar los Controladores: Ejecuta el instalador descargado y sigue las instrucciones en pantalla para instalar los controladores.

Reiniciar el Sistema: Es posible que necesites reiniciar tu sistema después de la instalación.

Verificar la Instalación: Abre el Panel de Control de NVIDIA para verificar que los controladores estén instalados correctamente.

## Instalación del NVIDIA Container Toolkit (Windows)

Instalar Docker Desktop: Descarga e instala Docker Desktop para Windows.

Habilitar WSL 2: Asegúrate de que WSL 2 esté habilitado en tu sistema.


dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
Descarga e instala el paquete de actualización del kernel de Linux para WSL 2 desde el sitio web de Microsoft.
wsl --set-default-version 2


Instalar el NVIDIA Container Toolkit: Abre una terminal de WSL 2 y sigue los pasos para instalar el NVIDIA Container Toolkit en el entorno de WSL 2.

Configurar Docker Desktop para Usar WSL 2: Abre Docker Desktop y habilita la opción "Use the WSL 2 based engine" y la integración con tu distribución de WSL 2.

Verificación de la Configuración: Ejecuta un contenedor de prueba desde la terminal de WSL 2:
```bash
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

Uso en Contenedores: Cuando ejecutes contenedores que necesiten acceso a la GPU, utiliza el flag `--gpus`:
```bash
docker run --gpus all my-gpu-application
```
