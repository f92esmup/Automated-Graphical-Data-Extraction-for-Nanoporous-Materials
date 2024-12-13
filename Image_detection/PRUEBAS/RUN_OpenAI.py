import warnings
warnings.filterwarnings('ignore')

import cv2
import sys
import numpy as np
import argparse
import os

sys.path.append('./CE_detection/ChartDete')
from mmdet.apis import init_detector, inference_detector

from util.utils import filter_lines, draw_lines, line_intersection

from google.cloud import vision
import skimage.io as io  # Cambiar la importación de io a skimage.io


# Configuración de argparse para recibir los argumentos
parser = argparse.ArgumentParser(description="Run inference on input images with a pretrained DETR model.")
parser.add_argument('--weights',default='./CE_detection/ChartDete/docker/work_dirs/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss/checkpoint.pth' , type=str, help="Path to the model weights file.")
parser.add_argument('--config',default='./CE_detection/ChartDete/docker/work_dirs/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss.py' , type=str, help="Path to the model configuration file.")
parser.add_argument('--input_path',default='./data/images',  type=str, help="Path to the directory containing input images.")
parser.add_argument('--output_path',default='./data/CE_output', type=str, help="Path to the directory where output will be saved.")
parser.add_argument('--device', default='cpu', type=str, help="Device to run the model on (cpu or cuda).")
args = parser.parse_args()


# Ejecutar inferencia y procesamiento en cada imagen del directorio de entrada
input_dir = args.input_path
output_dir = args.output_path
weights = args.weights
config = args.config

for image_name in os.listdir(input_dir):
    image_path = os.path.join(input_dir, image_name)
    output_path = os.path.join(output_dir, f"axes_{image_name}")






    # Ejecutar inferencia en la imagen
    model = init_detector(config, weights, device='cpu')
    result = inference_detector(model, image_path)
    # Guardar los resultados de la visualización en archivos de imagen (EL OUTPUT HAY QUE ARREGLARLO)
    #model.show_result(image_path, result, out_file='./data/CE_output/sample_result.png')
    # Extraer las regiones de interés (ROIs)
    X_label_ROI, Y_label_ROI, plot_area_ROI, _,  x_numbers_ROI, y_numbers_ROI, title_ROI, x_ticks_ROI, y_ticks_ROI, legend_points_ROI, legend_text_ROI, _,  legend_area_ROI, _ , _ , y_area_ROI, x_area_ROI, _ = result

    
    # Imprimir las ROIs para depuración
    # Filter ROIs by confidence
    def filter_rois_by_confidence(rois, threshold=0.9):
        filtered_rois = [roi for roi in rois if roi[-1] > threshold]
        return filtered_rois

    filtered_x_numbers_ROI = filter_rois_by_confidence(x_numbers_ROI)
    filtered_y_numbers_ROI = filter_rois_by_confidence(y_numbers_ROI)
    filtered_x_label_ROI = filter_rois_by_confidence(X_label_ROI)
    filtered_y_label_ROI = filter_rois_by_confidence(Y_label_ROI)

    # Initialize Google Cloud Vision client
    # Set up the Google Cloud Vision client
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './CE_detection/ChartDete/cedetection-21c906290fe8.json'
    client = vision.ImageAnnotatorClient()

    def detect_text_in_roi(image_path, roi):
        # Crop the ROI from the image
        image = io.imread(image_path)  # Usar skimage.io.imread para leer la imagen
        cropped_image = image[int(roi[1]):int(roi[3]), int(roi[0]):int(roi[2])]
        success, encoded_image = cv2.imencode('.png', cropped_image)
        content = encoded_image.tobytes()

        # Perform text detection
        image = vision.Image(content=content)
        response = client.text_detection(image=image)
        texts = response.text_annotations

        if texts:
            return texts[0].description
        else:
            return None

    # Detect text in each filtered ROI
    x_numbers_texts = [detect_text_in_roi(image_path, roi) for roi in filtered_x_numbers_ROI]
    y_numbers_texts = [detect_text_in_roi(image_path, roi) for roi in filtered_y_numbers_ROI]

    # Detect text in filtered X_label_ROI and Y_label_ROI
    x_label_texts = [detect_text_in_roi(image_path, roi) for roi in filtered_x_label_ROI]
    y_label_texts = [detect_text_in_roi(image_path, roi) for roi in filtered_y_label_ROI]

    def filter_and_sort_numbers(texts):
        numbers = []
        for text in texts:
            if text is not None:
                try:
                    number = float(text)
                    numbers.append(number)
                except ValueError:
                    continue
        return sorted(numbers)

    # Filter and sort the detected texts
    x_numbers_sorted = filter_and_sort_numbers(x_numbers_texts)
    y_numbers_sorted = filter_and_sort_numbers(y_numbers_texts)

    # Print detected texts for debugging
    print(f"X Numbers Texts: {x_numbers_sorted}")
    print(f"Y Numbers Texts: {y_numbers_sorted}")
    print(f"X Label Texts: {x_label_texts}")
    print(f"Y Label Texts: {y_label_texts}")

    ########################    ########################





    ########################ESTA ES OTRA PARTE ########################    
    ########################ESTA ES OTRA PARTE ########################
    # Cargar la imagen procesada para detección de bordes y líneas
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar detección de bordes
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detectar líneas con la Transformada de Hough
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # Filtrar líneas y encontrar verticales y horizontales (si es aplicable)
    height, width = image.shape[:2]
    vertical_line, parallel_vertical_line, horizontal_line, parallel_horizontal_line = filter_lines(lines, width, height)

    # Calcular los puntos de intersección entre las líneas seleccionadas
    intersection_points = []
    if vertical_line and horizontal_line:
        intersection_points.append(line_intersection(vertical_line, horizontal_line))
    if vertical_line and parallel_horizontal_line:
        intersection_points.append(line_intersection(vertical_line, parallel_horizontal_line))
    if parallel_vertical_line and horizontal_line:
        intersection_points.append(line_intersection(parallel_vertical_line, horizontal_line))
    if parallel_vertical_line and parallel_horizontal_line:
        intersection_points.append(line_intersection(parallel_vertical_line, parallel_horizontal_line))

    # Dibujar las líneas y las secciones en la imagen
    processed_image = draw_lines(image, lines, vertical_lines=[vertical_line, parallel_vertical_line], horizontal_lines=[horizontal_line, parallel_horizontal_line], intersection_points=intersection_points)

    # Guardar la imagen con las líneas detectadas en el directorio de salida
    cv2.imwrite(output_path, processed_image)
    print(f"Processed {image_name}, results saved to {output_path}.")








    



# Ejemplo de lista de datos para enfocarnos en la tarea de calcular la escala
example_data = [
    {'pixel_coord': [369.692535, 261.592926], 'value': None},
    {'pixel_coord': [158.525894, 261.573151], 'value': None},
    {'pixel_coord': [342.640778, 261.603394], 'value': None},
    # Agregar más ejemplos según sea necesario...
]

# Cálculo de la escala de los datos basados en las ROIs y conversión a coordenadas reales
def calculate_scale(plot_area, x_numbers_roi, y_numbers_roi):
    x_scale = None
    y_scale = None

    return x_scale, y_scale

# Pasos para Mejorar la Escala:
# 1. Asegurarse de que los valores de las etiquetas numéricas estén correctamente identificados y filtrados para evitar valores atípicos que distorsionen la escala.
# 2. Utilizar interpolación lineal si hay puntos faltantes en los valores de los ejes para una mejor estimación de la escala.
# 3. Verificar que la plot_area esté correctamente identificada, ajustando manualmente si es necesario para que corresponda solo al área útil de la gráfica.
# 4. Considerar el uso de una escala logarítmica si los valores de los ejes lo sugieren, ya que esto podría impactar la precisión de la conversión.
# 5. Validar la coherencia de la escala calculada probando con varios puntos de prueba dentro del área de la gráfica.

# Aplicar la escala a los datos de ejemplo



"""
··························································
result[0] --> Creo que intenta ser el X label | SI |

result[1] --> Y label | SI |

result[2] --> Plot area | SI |

result[3] --> Ni  idea 

result[4] --> numeros de x | No todos (aunque suficientes) | 

result[5] --> numeros de y | SI | 

result[6] --> ¿Titulo? | no hay titulo en la imagen1 | 

result[7] --> x-ticks | SI | 

result[8] --> Y-ticks | SI  |

result[9] --> Elemenos de la leyenda (formato de puntos) | no tiene leyenda como tal | 

result[10] -->  Texto leyenda. |  | 

result[11] -->  Ni idea 

result[12] --> leyenda area  |  | 

result[13] --> Ni idea 

result[14] --> Ni idea

result[15] --> AREA DE Y | SI | 

result[16] --> AREA DE X | SI | 

result[17] --> Ni idea

"""









#######OBTENCIÓN DE LOS NUMEROS DE LOS EJES#######
"""
Esté código extrae y definie los números de los ejes. 
Su problema principal es tener que declarar dos hiperparámetros correctamente para que realice la función correctamente.

import pytesseract
import easyocr
import re
import sys

# Función para extraer texto de una sección usando OCR
def extract_text_tesseract(image):
    # Verificar si la imagen no está vacía
    if image is None or image.size == 0:
        return ""  # Retorna una cadena vacía si el ROI está vacío

    # Escalar el ROI
    scale_percent = 150  # Escalar al 150% del tamaño original
    new_width = int(image.shape[1] * scale_percent / 100)
    new_height = int(image.shape[0] * scale_percent / 100)
    roi_resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Convertir a escala de grises
    roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
    #NO USAR roi_sharp, empeora los resultados
    # Aplicar un umbral adaptativo para binarizar la imagen
    roi_thresh = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Aplicar OCR de Tesseract en la ROI
    text = pytesseract.image_to_string(roi_thresh, config='--psm 6')  # Usar PSM 7 para una sola línea de texto
    return text

# Función para limpiar el texto y obtener solo números, incluyendo negativos y decimales
def clean_text(text):
    # Capturar números enteros, decimales y negativos
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return list(map(float, numbers)) if numbers else []

# Función para dividir el eje en secciones y aplicar OCR con mejor configuración
def extract_ticks_by_section(image, line, axis, num_sections, margin):
    height, width = image.shape[:2]
    tick_values = []
    
    if axis == "x":
        x1, y1, x2, y2, _ = line
        section_width = (x2 - x1) // num_sections

        for i in range(num_sections):
            section_x1 = x1 + i * section_width
            section_x2 = section_x1 + section_width
            roi = image[max(0, y1 - margin):min(height, y2 + margin), section_x1:section_x2]
            
            # Extraer texto y limpiar
            text = extract_text_tesseract(roi)
            cleaned_numbers = clean_text(text)
            tick_values.extend(cleaned_numbers)

    elif axis == "y":
        # Asegurar que y1 es la coordenada superior y y2 la inferior
        x1, y1, x2, y2, _ = line
        invert=False
        if y1 > y2:  # Intercambiar si es necesario
            y1, y2 = y2, y1
            invert=True
    
        # Calcular la altura de cada sección
        section_height = (y2 - y1) // num_sections

        for i in range(num_sections):
            section_y1 = y1 + i * section_height
            section_y2 = section_y1 + section_height
            roi = image[section_y1:section_y2, max(0, x1 - margin):min(width, x2 + margin)]
        
            # Extraer texto y limpiar
            text = extract_text_tesseract(roi)
            cleaned_numbers = clean_text(text)
            tick_values.extend(cleaned_numbers)
        if invert:  # Intercambiar si es necesario
            # Invertir el orden si es necesario
            tick_values = tick_values[::-1]  # Esto invierte la lista de valores

    # Filtrar valores duplicados
    tick_values = list(dict.fromkeys(tick_values))  # Elimina duplicados y preserva el orden

    return tick_values


def get_optimal_parameters(image, line, is_horizontal,length_factor=100):
    # Dimensiones de la imagen
    height, width = image.shape[:2]
    x1, y1, x2, y2, _ = line  # La longitud no se utiliza en este cálculo
    
    # Calcular num_sections y margin en función de la orientación y del tamaño de la imagen
    if is_horizontal:  # Línea horizontal
        line_length = x2 - x1
        num_sections = max(3, line_length // length_factor)  # Ajuste empírico en función de la longitud del eje
        margin = int(width * 0.06)  # Margen como 5% del ancho de la imagen
    else:  # Línea vertical
        line_length = y2 - y1
        num_sections = max(3, line_length // length_factor)  # Ajuste empírico en función de la longitud del eje
        margin = int(height * 0.06)  # Margen como 5% de la altura de la imagen

    return num_sections, margin






# Inicializar el lector de EasyOCR
reader = easyocr.Reader(['en'])

# Definir listas para almacenar posiciones y valores de ticks en X y Y
tick_positions_x = []
tick_values_x = []
tick_positions_y = []
tick_values_y = []

# Usar solo las líneas de los ejes detectadas y filtradas
# vertical_line y horizontal_line deben ser las coordenadas de los ejes que detectamos previamente

# Aplicar el OCR en secciones en los ejes
num_sections_x, margin_x = get_optimal_parameters(image,horizontal_line, is_horizontal=True)
num_sections_y, margin_y = get_optimal_parameters(image,vertical_line, is_horizontal=False)
print()
print(f'secciones X: {num_sections_x} Y margen X {margin_x}')
print()
print(f'secciones Y: {num_sections_y} Y margen Y {margin_y}')
print()

tick_values_x = extract_ticks_by_section(image, horizontal_line,"x" , 4,20)
tick_values_y = extract_ticks_by_section(image, vertical_line,"y" ,3, 20)



print("Valores en X:", tick_values_x)
print("Valores en Y:", tick_values_y)
#SEGUIR CON LA CONVERSACIÓOON

#AHORA MISMO HE COMPLETADO CASI TODOS LOS OBJETIVOS. ME FALTA CALIBRAR LOS PUNTOS Y 
#LO MÁS IMPORTANTE ES SABER COMO OPTIMIZAR LOS PARÁMETROS DE LA DETECCION DE NÚMEROS DE
#LA GRÁFICA.


#MI IDEA AHORA ES USAR CE_DETECTION COMO CAJA NEGRA, ES DECIR LA USO PARA DETECTAR BORDES Y LO DEMAS LO COJO.

################################################## 
# Podria optimizarlo más adelante para mejorar el programa.

"""

