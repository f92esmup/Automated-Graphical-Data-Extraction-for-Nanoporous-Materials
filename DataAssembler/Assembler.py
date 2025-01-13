import pandas as pd

# Crear un DataFrame vacío con las columnas especificadas
columns = ['graph', 'properties', 'paper', 'errors', 'confidence_score']
df = pd.DataFrame(columns=columns)

# Guardar el DataFrame en un archivo CSV
df.to_csv('./data/dataset.csv', index=False)