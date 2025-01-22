import pandas as pd
import os
from AI.Gemini.Imageproperties import GraphDocumentAnalysis
from tqdm import tqdm

def run_assembler():
    analyzer = GraphDocumentAnalysis(debug=False)
    # Crear un DataFrame vacío con las columnas especificadas
    columns = ['graph', 'paper', 'errors', 'confidence_score', 'properties']
    df = pd.DataFrame(columns=columns)

    # Recorrer los directorios de imágenes y PDFs
    data_dir = './data/papers'
    rows = []
    files_list = [os.path.join(root, file) for root, dirs, files in os.walk(data_dir) for file in files if file.endswith(".pdf")]
    for document_path in tqdm(files_list, desc="Processing files"):
        image_folder = os.path.join(os.path.dirname(document_path), os.path.splitext(os.path.basename(document_path))[0])
        if os.path.isdir(image_folder):
            for image_file in os.listdir(image_folder):
                image_path = os.path.join(image_folder, image_file)
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    properties = analyzer.analyze_graph_and_document(image_path, document_path)
                    rows.append({
                        'graph': image_file,
                        'paper': os.path.basename(document_path),
                        'errors': None,
                        'confidence_score': None,
                        'properties': properties
                    })

    # Convertir la lista de filas en un DataFrame y concatenar
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    # Guardar el DataFrame en un archivo CSV
    df.to_csv('./data/dataset.csv', index=False)

if __name__ == "__main__":
    run_assembler()