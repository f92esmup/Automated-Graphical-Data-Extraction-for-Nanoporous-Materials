import warnings
warnings.filterwarnings('ignore')

import cv2
import sys
import numpy as np
import argparse
import os
import subprocess
sys.path.append('./CE_detection/ChartDete')
from mmdet.apis import init_detector, inference_detector

#from util.utils import filter_lines, draw_lines, line_intersection

import google.generativeai as genai

import skimage.io as io 
import tempfile

subprocess.run("clear" if os.name == "posix" else "cls", shell=True)

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

###########funciones####################
def filter_rois_by_confidence(rois, threshold=0.6):
    filtered_rois = [roi for roi in rois if roi[-1] > threshold]
    return filtered_rois

def detect_text_in_roi(image_path, roi):
    """
    Detecta texto en una región de interés (ROI) de una imagen utilizando la API de Google AI.

    Args:
      image_path: Ruta a la imagen.
      roi: Tupla que define la ROI (xmin, ymin, xmax, ymax).

    Returns:
      El texto detectado en la ROI o None si no se detecta texto.
    """
    # Recorta la ROI de la imagen
    image = io.imread(image_path)
    cropped_image = image[int(roi[1]):int(roi[3]), int(roi[0]):int(roi[2])]

    # Guarda la imagen recortada en un archivo temporal
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        tmp_file_path = tmp_file.name
        cv2.imwrite(tmp_file_path, cropped_image)

    # Creamos el modelo de Gemini
    generation_config = {
      "temperature": 1,
      "top_p": 0.95,
      "top_k": 40,
      "max_output_tokens": 8192,
      "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
      model_name="gemini-1.5-pro",
      generation_config=generation_config,
      system_instruction="You will be provided with a single image, from which you must extract the text within it. Be very concise."
    )

    # Iniciamos un chat con el modelo y enviamos la imagen
    chat_session = model.start_chat(history=[{"role": "user", "parts": [genai.upload_file(tmp_file_path, mime_type="image/png")]}])
    response = chat_session.send_message("Extract the text from the image.")

    
    os.remove(tmp_file_path)

    # Devolvemos el texto detectado
    if response.text:
        return response.text[0]
    else:
        return None

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

def draw_and_save_rois(image_path, output_path,rois_dict , thickness=2):
    # Load the image
    image = cv2.imread(image_path)

    # Draw each ROI on the image with different colors for each group
    for group, rois in rois_dict.items():
        color = rois_dict[group]['color']
        for roi in rois['rois']:
            start_point = (int(roi[0]), int(roi[1]))
            end_point = (int(roi[2]), int(roi[3]))
            image = cv2.rectangle(image, start_point, end_point, color, thickness)
    
    # Save the image with ROIs drawn
    cv2.imwrite(output_path, image)




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
weights_path = args.weights
config_path = args.config

for image_name in os.listdir(input_dir):
    image_path = os.path.join(input_dir, image_name)
    output_path = os.path.join(output_dir, f"axes_{image_name}")






    # Ejecutar inferencia en la imagen
    model = init_detector(config_path, weights_path, device='cpu')
    result = inference_detector(model, image_path)
    # Guardar los resultados de la visualización en archivos de imagen (EL OUTPUT HAY QUE ARREGLARLO)
    #model.show_result(image_path, result, out_file='./data/CE_output/sample_result.png')
    # Extraer las regiones de interés (ROIs)
    X_label_ROI, Y_label_ROI, plot_area_ROI, _,  x_numbers_ROI, y_numbers_ROI, title_ROI, x_ticks_ROI, y_ticks_ROI, legend_points_ROI, legend_text_ROI, _,  legend_area_ROI, _ , _ , y_area_ROI, x_area_ROI, _ = result

    
    # Imprimir las ROIs para depuración
    # Filter ROIs by confidence

    filtered_x_numbers_ROI = filter_rois_by_confidence(x_numbers_ROI)
    filtered_y_numbers_ROI = filter_rois_by_confidence(y_numbers_ROI)
    filtered_x_label_ROI = filter_rois_by_confidence(X_label_ROI)
    filtered_y_label_ROI = filter_rois_by_confidence(Y_label_ROI)
    filtered_title_ROI = filter_rois_by_confidence(title_ROI)
    filtered_x_ticks_ROI = filter_rois_by_confidence(x_ticks_ROI)
    filtered_y_ticks_ROI = filter_rois_by_confidence(y_ticks_ROI)
    filtered_legend_points_ROI = filter_rois_by_confidence(legend_points_ROI)
    filtered_legend_text_ROI = filter_rois_by_confidence(legend_text_ROI)
    filtered_legend_area_ROI = filter_rois_by_confidence(legend_area_ROI)
    filtered_y_area_ROI = filter_rois_by_confidence(y_area_ROI)
    filtered_x_area_ROI = filter_rois_by_confidence(x_area_ROI)
    
     # Define colors for each group
    rois_dict = {
        'x_numbers': {'rois': filtered_x_numbers_ROI, 'color': (0, 255, 0)},
        'y_numbers': {'rois': filtered_y_numbers_ROI, 'color': (255, 0, 0)},
        'x_label': {'rois': filtered_x_label_ROI, 'color': (0, 0, 255)},
        'y_label': {'rois': filtered_y_label_ROI, 'color': (255, 255, 0)},
        'title': {'rois': filtered_title_ROI, 'color': (255, 0, 255)},
        'x_ticks': {'rois': filtered_x_ticks_ROI, 'color': (0, 255, 255)},
        'y_ticks': {'rois': filtered_y_ticks_ROI, 'color': (128, 0, 128)},
        'legend_points': {'rois': filtered_legend_points_ROI, 'color': (128, 128, 0)},
        'legend_text': {'rois': filtered_legend_text_ROI, 'color': (0, 128, 128)},
        'legend_area': {'rois': filtered_legend_area_ROI, 'color': (128, 128, 128)},
        'y_area': {'rois': filtered_y_area_ROI, 'color': (64, 64, 64)},
        'x_area': {'rois': filtered_x_area_ROI, 'color': (192, 192, 192)},
    }


    draw_and_save_rois(image_path, output_path,rois_dict)


    # Detect text in each filtered ROI
    x_numbers_texts = [detect_text_in_roi(image_path, roi) for roi in filtered_x_numbers_ROI]
    y_numbers_texts = [detect_text_in_roi(image_path, roi) for roi in filtered_y_numbers_ROI]

    # Detect text in filtered X_label_ROI and Y_label_ROI
    x_label_texts = [detect_text_in_roi(image_path, roi) for roi in filtered_x_label_ROI]
    y_label_texts = [detect_text_in_roi(image_path, roi) for roi in filtered_y_label_ROI]

 
    # Filter and sort the detected texts
    x_numbers_sorted = filter_and_sort_numbers(x_numbers_texts)
    y_numbers_sorted = filter_and_sort_numbers(y_numbers_texts)

    # Print detected texts for debugging
    print(f"X Numbers Texts: {x_numbers_sorted}")
    print(f"Y Numbers Texts: {y_numbers_sorted}")
    print(f"X Label Texts: {x_label_texts}")
    print(f"Y Label Texts: {y_label_texts}")

    ########################    ########################
    
    print(f"Processed {image_name}, results saved to {output_path}.")



