import requests
import xml.etree.ElementTree as ET
import os

def download_ieee_papers(query, max_results=1, start_year=None, end_year=None):
    # Construir la consulta con filtros adicionales
    args = []

    base_url = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
    api_key = "445cefmjypbfptjgnzgtwzpt"  # Reemplaza con tu clave de API de IEEE Xplore

    params = {
        "apikey": api_key,
        "querytext": query,
        "max_records": max_results,
        "start_year": start_year,
        "end_year": end_year,
        "format": "json"  # Solicitar la respuesta en formato JSON
    }

    response = requests.get(base_url, params=params)

    # Imprimir la respuesta para depuración
    print("Status Code:", response.status_code)
    print("Response Text:", response.text)

    if response.status_code == 403:
        print("Error: Developer Inactive. Verifica que tu clave de API está activa.")
        return

    if response.status_code != 200:
        print("Error en la solicitud:", response.status_code)
        return

    try:
        data = response.json()
    except requests.exceptions.JSONDecodeError as e:
        print("Error al decodificar JSON:", e)
        return

    for article in data.get('articles', []):
        name = article.get('title', None)
        doi = article.get('doi', None)
        pdf_url = article.get('pdf_url', None)
        year = article.get('publication_year', None)
        journal = article.get('publication_title', None)
        authors = ', '.join(author.get('full_name', None) for author in article.get('authors', {}).get('authors', []))
        abstract = article.get('abstract', None)

        args.append([name, doi, pdf_url, year, journal, authors, abstract])

        # Mostrar los metadatos de forma bonita
        print(f"Title: {name}")
        print(f"DOI: {doi}")
        print(f"PDF URL: {pdf_url}")
        print(f"Year: {year}")
        print(f"Journal: {journal}")
        print(f"Authors: {authors}")
        print(f"Abstract: {abstract}")
        print()

        # Descargar el PDF
        if pdf_url:
            pdf_response = requests.get(pdf_url)
            if pdf_response.status_code == 200:
                pdf_filename = f"{doi.replace('/', '_')}.pdf"
                with open(pdf_filename, 'wb') as pdf_file:
                    pdf_file.write(pdf_response.content)
                print(f"PDF descargado: {pdf_filename}")
            else:
                print(f"Error al descargar el PDF: {pdf_response.status_code}")

    return args

if __name__ == "__main__":
    query = "Nanoporous materials"
    start_year = 2020
    end_year = 2023
    author = None
    results = download_ieee_papers(query, start_year=start_year, end_year=end_year)
    print(results)
