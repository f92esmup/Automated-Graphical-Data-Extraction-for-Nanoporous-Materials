from utils import *
from BB_Inference import LineInference
import sys
import warnings
warnings.filterwarnings("ignore")

def main():
    """
    Función principal para ejecutar el modelo en cada imagen del directorio especificado.
    """
    # Obtener los caminos de las imágenes
    try:
        images_path = get_images_path(dir='./data/images')
        if not images_path:
            raise ValueError("No images found in the specified directory")
    except ValueError as e:
        print(e, "Please, check the path and try again", sep='\n')
        sys.exit()

    # Crear una instancia de LineInference con configuración y punto de control especificados
    inference = LineInference(config="./Line_detection/config.py", ckpt="./weights/weights.pth", device="cpu")

    # Procesar cada imagen
    for i, image_path in enumerate(images_path):
        # Procesa la imagen y obtiene la serie de datos de línea
        line_dataseries = process_image(inference, image_path)
        
        #Aqui quiero incluir el reescalado de line_dataseries, de esta forma se guardará correctamente en su csv

        line_dataseries_escal= line_dataseries #Ahora mismo no hace nada.



        # Guarda y grafica la serie de datos
        save_and_plot_data(line_dataseries_escal, image_path, save_path='./data/Line_output/',plot=True)

        # Dibuja las líneas en la imagen (opcional)
        #draw_lines_on_image(image_path, line_dataseries, output_dir='./data/Line_output/')

if __name__ == "__main__":
    main()


