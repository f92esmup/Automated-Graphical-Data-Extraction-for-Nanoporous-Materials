import pandas as pd
import os
from AI.Gemini.Imageproperties import GraphDocumentAnalysis
from tqdm import tqdm

def run_assembler():
    analyzer = GraphDocumentAnalysis(debug=False)
    # Crear un DataFrame vacío con las columnas especificadas
    #columns = ['graph', 'paper', 'errors', 'confidence_score', 'properties']
    columns = ['graph', 'paper', 'errors', 'properties']
    df = pd.DataFrame(columns=columns)

    # Recorrer los directorios de imágenes y PDFs
    data_dir = './data/papers'
    rows = []
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]

    for pdf_file in tqdm(pdf_files, desc="Processing files"):
        pdf_name = os.path.splitext(pdf_file)[0]
        pdf_path = os.path.join(data_dir, pdf_file)
        image_folder = os.path.join(data_dir, pdf_name)
        
        if os.path.isdir(image_folder):
            for image_file in os.listdir(image_folder):
                image_path = os.path.join(image_folder, image_file)
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    properties = analyzer.analyze_graph_and_document(image_path, pdf_path)
                    #errors = ---
                    rows.append({
                        'graph': image_file,
                        'paper': pdf_file,
                        'errors': None,
                        #'confidence_score': None,
                        'properties': properties
                    })

    # Convertir la lista de filas en un DataFrame y concatenar
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    # Guardar el DataFrame en un archivo CSV
    df.to_csv('./data/dataset.csv', index=False)

if __name__ == "__main__":
    run_assembler()