import pandas as pd
import os
from AI.Gemini.Imageproperties import GraphDocumentAnalysis

def analyze_graphs(df):
    analyzer = GraphDocumentAnalysis(debug=False)
    for index, row in df.iterrows():
        image_path = os.path.join('./data/images', row['paper'].replace('.pdf', ''), row['graph'])
        document_path = os.path.join('./data/papers', row['paper'])
        if os.path.isfile(image_path) and os.path.isfile(document_path):
            analysis_result = analyzer.analyze_graph_and_document(image_path, document_path)
            df.at[index, 'properties'] = analysis_result

def run_assembler():
    # Crear un DataFrame vacío con las columnas especificadas
    columns = ['graph', 'properties', 'paper', 'errors', 'confidence_score']
    df = pd.DataFrame(columns=columns)

    # Recorrer los directorios de imágenes
    data_dir = './data/images'
    rows = []
    for paper_dir in os.listdir(data_dir):
        paper_path = os.path.join(data_dir, paper_dir)
        if os.path.isdir(paper_path):
            for image in os.listdir(paper_path):
                image_path = os.path.join(paper_path, image)
                if os.path.isfile(image_path):
                    rows.append({
                        'graph': image,
                        'properties': None,
                        'paper': f"{paper_dir}.pdf",
                        'errors': None,
                        'confidence_score': None
                    })

    # Convertir la lista de filas en un DataFrame y concatenar
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    analyze_graphs(df)
    # Guardar el DataFrame en un archivo CSV
    df.to_csv('./data/dataset.csv', index=False)

if __name__ == "__main__":
    run_assembler()