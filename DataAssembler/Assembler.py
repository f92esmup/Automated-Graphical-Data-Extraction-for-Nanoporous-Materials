import pandas as pd
import os

def run_assembler():
    # Crear un DataFrame vacío con las columnas especificadas
    columns = ['graph', 'properties', 'paper', 'errors', 'confidence_score']
    df = pd.DataFrame(columns=columns)

    # Recorrer los directorios de imágenes
    data_dir = './data/imagenes'
    for paper_dir in os.listdir(data_dir):
        paper_path = os.path.join(data_dir, paper_dir)
        if os.path.isdir(paper_path):
            for image in os.listdir(paper_path):
                image_path = os.path.join(paper_path, image)
                if os.path.isfile(image_path):
                    df = df.append({
                        'graph': image,
                        'properties': None,
                        'paper': f"{paper_dir}.pdf",
                        'errors': None,
                        'confidence_score': None
                    }, ignore_index=True)

    # Guardar el DataFrame en un archivo CSV
    df.to_csv('./data/dataset.csv', index=False)

if __name__ == "__main__":
    run_assembler()