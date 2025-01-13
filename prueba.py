import requests

def download_ieee_papers(query, max_results=1, start_year=None, end_year=None):
    # Construir la consulta con filtros adicionales
    args = []

    base_url = "http://ieeexploreapi.ieee.org/api/v1/search/articles"
    api_key = " xd94qvvrbkhnuckszr544znp"  # Reemplaza con tu clave de API de IEEE Xplore

    params = {
        "apikey": api_key,
        "querytext": query,
        "max_records": max_results,
        "start_year": start_year,
        "end_year": end_year,
        "format": "json"
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    for article in data.get('articles', []):
        name = article.get('title', 'N/A')
        doi = article.get('doi', 'N/A')
        pdf_url = article.get('pdf_url', 'N/A')
        pdf_name = pdf_url.split('/')[-1] + ".pdf" if pdf_url != 'N/A' else 'N/A'
        year = article.get('publication_year', 'N/A')
        journal = article.get('publication_title', 'N/A')
        authors = ', '.join(author.get('full_name', 'N/A') for author in article.get('authors', {}).get('authors', []))
        abstract = article.get('abstract', 'N/A')

        args.append([name, doi, pdf_name, year, journal, authors, abstract])

        # Descargar el PDF si la URL está disponible
        if pdf_url != 'N/A':
            pdf_response = requests.get(pdf_url)
            with open(f'./data/papers2/{pdf_name}', 'wb') as pdf_file:
                pdf_file.write(pdf_response.content)

    return args

if __name__ == "__main__":
    query = "Nanoporous materials"
    start_year = 2020
    end_year = 2023
    author = None
    download_ieee_papers(query, start_year=start_year, end_year=end_year)
