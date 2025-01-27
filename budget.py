# set up
import tiktoken
import PyPDF2
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import google.generativeai as genai
from PIL import Image

def calcular_costo(ruta_pdf, costo_por_token, prompt=None, encoding=None, modelo='gpt'):
    # Leer el archivo PDF
    with open(ruta_pdf, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        texto_pdf = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            texto_pdf += page.extract_text()

    if encoding is None:
        encoding = tiktoken.get_encoding("cl100k_base")

    if modelo == 'gpt':
        # Tokenizar el texto del PDF
        tokens_pdf = encoding.encode(texto_pdf)

        # Calcular el costo total para el texto del PDF
        costo_total_pdf = len(tokens_pdf) * costo_por_token

        # Devolver el número de páginas del PDF
        numero_paginas = len(reader.pages)

        return costo_total_pdf, len(tokens_pdf), numero_paginas

    elif modelo == 'gemini':
        file = genai.upload_file(path=ruta_pdf, mime_type='application/pdf')

        # Tokenizar el prompt y el archivo PDF
        tokens_prompt = encoding.encode(prompt)
        tokens_file = encoding.encode(texto_pdf)
        total_tokens = len(tokens_prompt) + len(tokens_file)

        costo_total_pdf = total_tokens * costo_por_token

        # Devolver el número de páginas del PDF
        numero_paginas = len(reader.pages)

        return costo_total_pdf, total_tokens, numero_paginas

def calcular_costo_imagen(ruta_imagen, costo_por_token, prompt, encoding=None):
    if encoding is None:
        encoding = tiktoken.get_encoding("cl100k_base")

    with open(ruta_imagen, 'rb') as img_file:
        img_data = img_file.read()

    file = genai.upload_file(path=ruta_imagen, mime_type='image/jpeg')

    tokens_prompt = encoding.encode(prompt)
    tokens_file = encoding.encode(img_data)
    total_tokens = len(tokens_prompt) + len(tokens_file)

    costo_total_imagen = total_tokens * costo_por_token

    return costo_total_imagen, total_tokens

def calcular_costo_para_todos_archivos(carpeta, costo_por_token, encoding=None, modelo='gpt', prompt=None):
    if encoding is None:
        encoding = tiktoken.get_encoding("cl100k_base")
    resultados = []
    for archivo in os.listdir(carpeta):
        ruta_archivo = os.path.join(carpeta, archivo)
        if archivo.endswith('.pdf'):
            if modelo == 'gpt':
                costo_total, tokens, numero_paginas = calcular_costo(ruta_archivo, costo_por_token, encoding=encoding, modelo=modelo)
            elif modelo == 'gemini':
                costo_total, tokens, numero_paginas = calcular_costo(ruta_archivo, costo_por_token, prompt=prompt, encoding=encoding, modelo=modelo)
            resultados.append({
                'archivo': archivo,
                'tipo': 'pdf',
                'numero_paginas': numero_paginas,
                'numero_tokens': tokens,
                'costo_total': costo_total
            })
        elif archivo.endswith(('.png', '.jpg', '.jpeg')):
            costo_total, tokens = calcular_costo_imagen(ruta_archivo, costo_por_token, prompt, encoding=encoding)
            resultados.append({
                'archivo': archivo,
                'tipo': 'imagen',
                'numero_paginas': 0,  # No consideramos las imágenes como páginas
                'numero_tokens': tokens,
                'costo_total': costo_total
            })
    return resultados

def realizar_regresion(df, x_col, y_col, costo_por_token):
    # Preparar los datos
    X = df[[x_col]].values
    y = df[y_col].values

    # Crear el modelo de regresión lineal
    modelo = LinearRegression()

    # Ajustar el modelo
    modelo.fit(X, y)

    # Obtener los coeficientes de la regresión
    pendiente = modelo.coef_[0]
    intercepto = modelo.intercept_

    # Calcular el valor de R²
    r2 = modelo.score(X, y)

    # Predecir los valores
    y_pred = modelo.predict(X)

    # Graficar los resultados
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', alpha=0.6, label='Datos reales')
    plt.plot(X, y_pred, color='red', linewidth=2, label='Regresión lineal')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True)
    plt.show()

    # Crear un diccionario con los valores de la regresión lineal
    valores_regresion = {
        'Relación tokens/página': [int(pendiente)],
        'Relación coste/página': [pendiente * costo_por_token],
        'Intercepto': [intercepto],
        'R²': [r2]
    }

    # Convertir el diccionario a un DataFrame de pandas
    df_valores_regresion = pd.DataFrame(valores_regresion)

    # Añadir unidades a las celdas
    df_valores_regresion['Relación coste/página'] = df_valores_regresion['Relación coste/página'].apply(lambda x: f"$ {np.ceil(x * 100) / 100:.2f}")

    # Mostrar el DataFrame
    return df_valores_regresion.style.set_properties(**{'text-align': 'center'}).set_table_styles(
        [{'selector': 'th', 'props': [('text-align', 'center')]}]).hide(axis='index')

def convertir_resultados_a_dataframe(resultados):
    df_resultados = pd.DataFrame(resultados)
    outupt = df_resultados.sort_values(by='numero_tokens', ascending=False).style.set_properties(**{'text-align': 'center'}).set_table_styles(
        [{'selector': 'th', 'props': [('text-align', 'center')]},
         {'selector': 'th.col_heading', 'props': [('color', 'black')]},
         {'selector': 'th.col_heading', 'props': [('font-weight', 'bold'), ('background-color', '#d0f0c0')]}
         ]
    ).hide(axis='index')
    return df_resultados, outupt

carpeta_pdfs = './data/papers'

# Definir el texto y el costo por token
texto = "Este es un ejemplo de texto para tokenizar."
#costo_por_token_gpt = 0.0001
costo_por_token_gemini = 0.30 / 1000000

"""
# Cargar el modelo de tokenización
encoding_gpt4 = tiktoken.get_encoding("cl100k_base") # Es el modelo de tokenización para los modelos más recientes.
"""
# Set your API key
api_key = 'AIzaSyDFuwrnPunjaEG5WlzjycQ75km-w2MFsgc'
genai.configure(api_key=api_key)
model = genai.GenerativeModel("models/gemini-1.5-flash")

"""
 gpt-o4
 Pasamos primero el texto.
 Resultados es una lista de diccionarios, donde cada diccionario es la información de un pdf.
 resultados_gpt = calcular_costo_para_todos_pdfs(carpeta_pdfs, costo_por_token_gpt, encoding= encoding_gpt4)
# Representamos usando pandas.
 df_resultados_gpt, table = convertir_resultados_a_dataframe(resultados_gpt)
 table
# Llamar a la función con los datos actuales
 realizar_regresion(df_resultados_gpt, 'numero_paginas', 'numero_tokens', costo_por_token_gpt)
"""
# AHora deberiamos pasar la imagen para comprobar el coste total.
# Gemini-1.5-flash

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

# Llamar a la función con los datos actuales
realizar_regresion(df_resultados_gemini[df_resultados_gemini['tipo'] == 'pdf'], 'numero_paginas', 'numero_tokens', costo_por_token_gemini)

numero_medio_paginas = df_resultados_gemini[df_resultados_gemini['tipo'] == 'pdf']['numero_paginas'].mean()
coste_medio_pdf = df_resultados_gemini[df_resultados_gemini['tipo'] == 'pdf']['costo_total'].mean()
coste_medio_por_pagina = coste_medio_pdf / numero_medio_paginas

coste_total_pdf = df_resultados_gemini[df_resultados_gemini['tipo'] == 'pdf']['costo_total'].sum()
coste_total_imagen = df_resultados_gemini[df_resultados_gemini['tipo'] == 'imagen']['costo_total'].sum()
coste_total = coste_total_pdf + coste_total_imagen

df = pd.DataFrame({
    'numero_medio_paginas': [numero_medio_paginas],
    'coste_medio_pdf': [coste_medio_pdf],
    'coste_medio_por_pagina': [coste_medio_por_pagina],
    'coste_total_pdf': [coste_total_pdf],
    'coste_total_imagen': [coste_total_imagen],
    'coste_total': [coste_total]
})
print(df)
