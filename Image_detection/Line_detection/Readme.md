# LineFormer Environment Setup

Este documento describe cómo configurar el entorno necesario para ejecutar el proyecto **LineFormer**. Incluye instrucciones para instalar las herramientas requeridas, configurar las dependencias mediante `conda`, `pip`, y `mim`, y solucionar posibles problemas de compatibilidad.

---

## Prerrequisitos

Antes de configurar el entorno, asegúrate de cumplir con los siguientes requisitos:

1. **Instalación de Miniconda:**  
   Descarga Miniconda desde su sitio oficial y sigue las instrucciones de instalación. Miniconda te permitirá gestionar fácilmente las dependencias del proyecto.

2. **Repositorio del proyecto:**  
   Si estás utilizando un repositorio de código, descárgalo o clónalo en tu máquina local.

---

## Pasos para la configuración del entorno

### 1. Crear el entorno base

El proyecto utiliza `conda` para gestionar las dependencias principales, como Python, NumPy, SciPy, y PyTorch. Estas dependencias están definidas en el archivo `environment.yaml`.

Para configurar el entorno:

- Utiliza el archivo `environment.yaml` para crear un entorno llamado `LineFormer`.
- Este archivo asegura que todas las versiones de las bibliotecas sean compatibles entre sí y facilita la replicación del entorno en cualquier máquina.

### 2. Activar el entorno

Una vez creado, activa el entorno llamado `LineFormer`. Esto cargará todas las configuraciones necesarias, asegurando que puedas trabajar dentro del entorno adecuado.

### 3. Instalar dependencias adicionales con `mim`

El proyecto requiere `mmcv` y `mmcv-full`, que no están incluidos en `conda`. Estos paquetes se instalan utilizando `mim`, una herramienta de OpenMMLab.

Los pasos incluyen:

- Descargar e instalar `mim` desde su fuente oficial.
- Utilizar `mim` para instalar `mmcv-full`, asegurándote de que sea compatible con la versión de PyTorch en tu entorno.

Es importante verificar la tabla de compatibilidad de OpenMMLab para evitar errores de instalación.

### 4. Complementar con paquetes de `pip`

Aunque la mayoría de las dependencias están en `conda` o `mim`, algunos paquetes se instalan directamente con `pip`. Estas dependencias están listadas en el archivo `requirements.txt`. Asegúrate de instalarlas después de completar las configuraciones anteriores.

---

## Verificación del entorno

Una vez completada la configuración:

1. Asegúrate de que todas las bibliotecas estén instaladas correctamente.
2. Verifica que `mmcv` esté funcional importándolo en un script básico. Esto garantiza que las dependencias específicas de OpenMMLab estén correctamente configuradas.
3. Si estás utilizando GPU, confirma que las versiones de PyTorch y CUDA sean compatibles.

---

## Restaurar el entorno en otra máquina

Si necesitas replicar este entorno en una máquina diferente:

1. Instala Miniconda y clona el repositorio del proyecto.
2. Crea el entorno utilizando el archivo `environment.yaml`. Esto descargará las dependencias principales.
3. Reinstala los paquetes gestionados con `mim`.
4. Completa la instalación con los paquetes adicionales desde el archivo `requirements.txt`.

---

## Problemas comunes y cómo solucionarlos

1. **Errores al importar `mmcv`:**  
   Esto ocurre si `mmcv` no es compatible con la versión de PyTorch. Consulta la tabla de compatibilidad oficial para instalar la versión correcta.

2. **Problemas con CUDA:**  
   Si usas GPU, verifica que la versión de PyTorch instalada sea compatible con tu versión de CUDA. Considera reinstalar PyTorch si es necesario.

3. **Conflictos entre dependencias:**  
   Asegúrate de no mezclar `pip` y `conda` para instalar el mismo paquete. Prioriza `conda` siempre que sea posible.

---

## Notas adicionales

- Documenta cualquier cambio que realices en las dependencias para mantener el entorno actualizado.
- Realiza pruebas en un entorno limpio antes de compartir el archivo `environment.yaml` o `requirements.txt` con otros.

Con estos pasos, tendrás el entorno configurado y listo para ejecutar el proyecto **LineFormer** de manera eficiente y reproducible.
