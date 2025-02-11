# setup
import tiktoken
import PyPDF2
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import google.generativeai as genai

def calcular_costo(ruta_pdf, costo_por_token, prompt=None, encoding=None, modelo='gpt'):
    # Read the PDF file
    with open(ruta_pdf, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        texto_pdf = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            texto_pdf += page.extract_text()

    if encoding is None:
        encoding = tiktoken.get_encoding("cl100k_base")

    if modelo == 'gpt':
        # Tokenize the PDF text
        tokens_pdf = encoding.encode(texto_pdf)

        # Calculate the total cost for the PDF text
        costo_total_pdf = len(tokens_pdf) * costo_por_token

        # Return the number of pages in the PDF
        numero_paginas = len(reader.pages)

        return costo_total_pdf, len(tokens_pdf), numero_paginas

    elif modelo == 'gemini':
        file = genai.upload_file(path=ruta_pdf, mime_type='application/pdf')

        # Tokenize the prompt and the PDF file
        tokens_prompt = encoding.encode(prompt)
        tokens_file = encoding.encode(texto_pdf)
        total_tokens = len(tokens_prompt) + len(tokens_file)

        costo_total_pdf = total_tokens * costo_por_token

        # Return the number of pages in the PDF
        numero_paginas = len(reader.pages)

        return costo_total_pdf, total_tokens, numero_paginas

def calcular_costo_para_todos_archivos(carpeta, costo_por_token, encoding=None, modelo='gpt', prompt=None):
    if encoding is None:
        encoding = tiktoken.get_encoding("cl100k_base")
    resultados = []
    # Iterate over all files in the folder
    for archivo in os.listdir(carpeta):
        ruta_archivo = os.path.join(carpeta, archivo)
        # Check if the file is a PDF
        if archivo.endswith('.pdf'):
            if modelo == 'gpt':
                costo_total, tokens, numero_paginas = calcular_costo(ruta_archivo, costo_por_token, encoding=encoding, modelo=modelo)
            elif modelo == 'gemini':
                costo_total, tokens, numero_paginas = calcular_costo(ruta_archivo, costo_por_token, prompt=prompt, encoding=encoding, modelo=modelo)
            # Append the results to the list
            resultados.append({
                'archivo': archivo,
                'tipo': 'pdf',
                'numero_paginas': numero_paginas,
                'numero_tokens': tokens,
                'costo_total': costo_total
            })
    return resultados

def realizar_regresion(df, x_col, y_col, costo_por_token):
    # Prepare the data
    X = df[[x_col]].values
    y = df[y_col].values

    # Create the linear regression model
    modelo = LinearRegression()

    # Fit the model
    modelo.fit(X, y)

    # Get the regression coefficients
    pendiente = modelo.coef_[0]
    intercepto = modelo.intercept_

    # Calculate the R² value
    r2 = modelo.score(X, y)

    # Predict the values
    y_pred = modelo.predict(X)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', alpha=0.6, label='Datos reales')
    plt.plot(X, y_pred, color='red', linewidth=2, label='Regresión lineal')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True)
    plt.show()

    # Create a dictionary with the regression values
    valores_regresion = {
        'Relación tokens/página': [int(pendiente)],
        'Relación coste/página': [pendiente * costo_por_token],
        'Intercepto': [intercepto],
        'R²': [r2]
    }

    # Convert the dictionary to a pandas DataFrame
    df_valores_regresion = pd.DataFrame(valores_regresion)

    # Add units to the cells
    df_valores_regresion['Relación coste/página'] = df_valores_regresion['Relación coste/página'].apply(lambda x: f"$ {np.ceil(x * 100) / 100:.2f}")

    # Display the DataFrame
    return df_valores_regresion.style.set_properties(**{'text-align': 'center'}).set_table_styles(
        [{'selector': 'th', 'props': [('text-align', 'center')]}]).hide(axis='index')

def convertir_resultados_a_dataframe(resultados):
    df_resultados = pd.DataFrame(resultados)
    # Sort the DataFrame by the number of tokens
    outupt = df_resultados.sort_values(by='numero_tokens', ascending=False).style.set_properties(**{'text-align': 'center'}).set_table_styles(
        [{'selector': 'th', 'props': [('text-align', 'center')]},
         {'selector': 'th.col_heading', 'props': [('color', 'black')]},
         {'selector': 'th.col_heading', 'props': [('font-weight', 'bold'), ('background-color', '#d0f0c0')]}
         ]
    ).hide(axis='index')
    return df_resultados, outupt

carpeta_pdfs = './data/papers'

# Define the text and the cost per token
texto = "Este es un ejemplo de texto para tokenizar."
#costo_por_token_gpt = 0.0001
costo_por_token_gemini = 0.30 / 1000000

# Load the tokenization model
#encoding_gpt4 = tiktoken.get_encoding("cl100k_base") # Es el modelo de tokenización para los modelos más recientes.

# Set your API key
api_key = 'AIzaSyDFuwrnPunjaEG5WlzjycQ75km-w2MFsgc'
genai.configure(api_key=api_key)
model = genai.GenerativeModel("models/gemini-1.5-flash")

# Pass the text first.
# Results is a list of dictionaries, where each dictionary is the information of a pdf.
#resultados_gpt = calcular_costo_para_todos_pdfs(carpeta_pdfs, costo_por_token_gpt, encoding= encoding_gpt4)
# Represent using pandas.
#df_resultados_gpt, table = convertir_resultados_a_dataframe(resultados_gpt)
#table
# Call the function with the current data
#realizar_regresion(df_resultados_gpt, 'numero_paginas', 'numero_tokens', costo_por_token_gpt)

# Now we should pass the image to check the total cost.
prompt = """You will be provided with a single image, from which you must extract the following information and format it as a JSON object:
                - title
                - x_label (use HTML structure for subscript, e.g., N<sub>L</sub>)
                - x_numbers
                - y_label (use HTML structure for subscript, e.g., Ω [k<sub>B</sub>T])
                - y_numbers
                - legend
                - x_units
                - y_units (use HTML structure for subscript, e.g., k<sub>B</sub>T)

                The x_numbers and y_numbers should be arrays of numbers.

                Do not include any additional text or explanations, only the JSON object."""

resultados_gemini = calcular_costo_para_todos_archivos(carpeta_pdfs, costo_por_token_gemini, prompt=prompt, modelo='gemini')
df_resultados_gemini, table = convertir_resultados_a_dataframe(resultados_gemini)
print(table)

# Call the function with the current data
realizar_regresion(df_resultados_gemini[df_resultados_gemini['tipo'] == 'pdf'], 'numero_paginas', 'numero_tokens', costo_por_token_gemini)

# Calculate the average number of pages
numero_medio_paginas = df_resultados_gemini[df_resultados_gemini['tipo'] == 'pdf']['numero_paginas'].mean()
# Calculate the average cost per PDF
coste_medio_pdf = df_resultados_gemini[df_resultados_gemini['tipo'] == 'pdf']['costo_total'].mean()
# Calculate the average cost per page
coste_medio_por_pagina = coste_medio_pdf / numero_medio_paginas

# Calculate the total cost for all PDFs
coste_total_pdf = df_resultados_gemini[df_resultados_gemini['tipo'] == 'pdf']['costo_total'].sum()
coste_total = coste_total_pdf

# Create a DataFrame with the calculated values
df = pd.DataFrame({
    'numero_medio_paginas': [numero_medio_paginas],
    'coste_medio_pdf': [coste_medio_pdf],
    'coste_medio_por_pagina': [coste_medio_por_pagina],
    'coste_total_pdf': [coste_total_pdf],
    'coste_total': [coste_total]
})

# Configure pandas to display all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print(df)
