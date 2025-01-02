import itertools  # Funciones eficientes para trabajar con iteradores en bucles.
import numpy as np  # Biblioteca fundamental para computación científica.
from scipy.interpolate import CubicSpline, interp1d  # Funciones para interpolación de datos.
import os
import sys

# Configura el PYTHONPATH para importar `mmdet`
project_root = './Line_detection'
sys.path.append(os.path.join(project_root, 'mmdetection'))

# Importaciones de mmdet y utils
from mmdetection.mmdet.apis import inference_detector, init_detector
import utils  # Módulo personalizado con funciones útiles para trabajar con líneas en imágenes.

class LineInference:
    """
    Clase para realizar inferencia y procesar los resultados de detección de instancias en imágenes.
    """

    def __init__(self, config, ckpt, device='cpu'):
        """
        Inicializa el modelo de detección con el archivo de configuración y punto de control.
        
        Parámetros:
        - config: Ruta al archivo de configuración del modelo.
        - ckpt: Ruta al archivo de punto de control del modelo (checkpoint).
        - device: Dispositivo en el que se ejecutará el modelo (por ejemplo, 'cuda:0' o 'cpu').
        """
        # Declara la variable 'model' como global para accederla fuera de la función
        self.model = init_detector(config, ckpt, device=device)

    def do_instance(self, img, score_thr=0.3):
        """
        Realiza la detección de instancias en la imagen de entrada usando el modelo proporcionado.

        Parámetros:
        - img: Imagen de entrada en formato de array de numpy.
        - score_thr: Umbral de puntuación para filtrar las detecciones.

        Retorna:
        - inst_masks: Máscaras de instancias filtradas según el umbral de puntuación.
        """
        # Realiza la inferencia sobre la imagen de entrada usando el modelo
        result = inference_detector(self.model, img)
        
        # Procesa los resultados de detección usando el umbral de puntuación especificado y devuelve los resultados filtrados
        return self.parse_result(result, score_thr)

    @staticmethod
    def parse_result(result, score_thresh=0.3):
        """
        Procesa los resultados de detección y filtra máscaras según el umbral de puntuación.

        Parámetros:
        - result: Resultados de detección devueltos por el modelo.
        - score_thresh: Umbral de puntuación para filtrar máscaras.

        Retorna:
        - inst_masks: Máscaras de instancias filtradas.
        """
        # Asigna el resultado a la variable linea_data
        linea_data = result

        # Extrae los cuadros delimitadores (bounding boxes) y máscaras del resultado
        bbox, masks = linea_data[0][0], linea_data[1][0]

        # Filtra las máscaras basándose en el umbral de puntuación
        # Solo incluye máscaras donde la puntuación del cuadro delimitador correspondiente es mayor que el umbral de puntuación
        inst_masks = list(itertools.compress(masks, ((bbox[:, 4] > score_thresh).tolist())))
        
        # Retorna las máscaras de instancias filtradas
        return inst_masks

    @staticmethod
    def interpolate(line_ds, inter_type='linear'):
        """
        Interpola una serie de datos.

        Parámetros:
        - line_ds: Serie de datos predicha.
        - inter_type: Tipo de interpolación ('linear' o 'cubic_spline').

        Retorna:
        - inter_line_ds: Lista de objetos de interpolación para cada línea en la máscara.
        """
        # Inicializa listas para almacenar coordenadas x e y
        x, y = [], []

        # Extrae coordenadas x e y de la serie de datos
        for pt in line_ds:
            x.append(pt['x'])
            y.append(pt['y'])

        # Elimina duplicados
        unique_x, unique_y = [], []
        for i in range(len(x)):
            if x.count(x[i]) == 1:
                unique_x.append(int(x[i]))
                unique_y.append(int(y[i]))

        # Si hay menos de 2 coordenadas x únicas, retorna la serie de datos original
        if len(unique_x) < 2:
            return line_ds

        # Realiza la interpolación
        if inter_type == 'linear':
            # Interpolación lineal
            inter = interp1d(unique_x, unique_y)
        elif inter_type == 'cubic_spline':
            # Interpolación de spline cúbico
            inter = CubicSpline(unique_x, unique_y)

        # Inicializa la lista para almacenar la serie de datos interpolada
        inter_line_ds = []
        x_min, x_max = min(unique_x), max(unique_x)

        # Genera puntos interpolados para cada x en el rango de x_min a x_max
        for x in range(x_min, x_max + 1):
            inter_line_ds.append({"x": x, "y": int(inter(x))})

        # Retorna la serie de datos interpolada
        return inter_line_ds

    def get_dataseries(self, img, mask_kp_sample_interval=10,inter_type='linear' ,return_masks=False):
        """
        Extrae series de datos de líneas de una imagen de gráficos.

        Parámetros:
        - img: Imagen de gráfico en formato de array de numpy (3 canales).
        - mask_kp_sample_interval: Intervalo para muestrear puntos de la máscara de línea predicha para obtener la serie de datos.
        - return_masks: Si es True, retorna las máscaras junto con las series de datos.

        Retorna:
        - pred_ds: Series de datos de líneas en el formato especificado (lista de líneas, cada una como lista de puntos {x:, y:}).
        - inst_masks (opcional): Máscaras de instancias si return_masks es True.
        """
        # Obtiene máscaras de instancias usando la inferencia
        inst_masks = self.do_instance(img, score_thr=0.3)
        
        # Convierte las máscaras de instancias a formato uint8 y las multiplica por 255
        inst_masks = [line_mask.astype(np.uint8) * 255 for line_mask in inst_masks]

        # Extrae la serie de datos de la línea a partir de las máscaras
        pred_ds = []
        for line_mask in inst_masks:
            # Obtiene el rango x de la línea de la máscara
            x_range = utils.get_xrange(line_mask)
            
            # Obtiene puntos clave (keypoints) de la máscara de línea usando el intervalo especificado
            line_ds = utils.get_kp(line_mask, interval=mask_kp_sample_interval, x_range=x_range, get_num_lines=False, get_center=True)
            
            # Realiza la interpolación en la serie de datos de línea
            line_ds = self.interpolate(line_ds, inter_type=inter_type)

            # Añade la serie de datos interpolada a la lista de series de datos predichas
            pred_ds.append(line_ds)

        # Retorna la serie de datos predicha y, opcionalmente, las máscaras de instancias
        return (pred_ds, inst_masks) if return_masks else pred_ds
