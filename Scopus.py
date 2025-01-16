import requests
import pandas as pd

def download_scopus_papers(query, max_results=1, start_year=None, end_year=None):
    base_url = "https://api.elsevier.com/content/search/scopus"
    api_key = "8a51251f45eceafbd0ebfa005f9b7709"  # Reemplaza con tu clave de API de Scopus

    params = {
        "apikey": api_key,
        "query": query,
        "count": max_results,
        "date": f"{start_year}-{end_year}" if start_year and end_year else None,
        "httpAccept": "application/json"
    }

    response = requests.get(base_url, params=params)
    
    # Print the response text for debugging
    #print(response.text)
    
    try:
        data = response.json()
    except requests.exceptions.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return

    papers = []
    for entry in data.get('search-results', {}).get('entry', []):
        paper = {
            'title': entry.get('dc:title', 'N/A'),
            'doi': entry.get('prism:doi', 'N/A'),
            'year': entry.get('prism:coverDate', 'N/A').split('-')[0],
            'url': entry.get('link', [{}])[2].get('@href', 'N/A') if len(entry.get('link', [])) > 2 else 'N/A',
            'publicationName': entry.get('prism:publicationName', 'N/A'),
            'creator': entry.get('dc:creator', 'N/A'),
            'affiliation': ', '.join([affil.get('affilname', 'N/A') for affil in entry.get('affiliation', [])]),
            'abstract': entry.get('dc:description', 'N/A')
        }
        papers.append(paper)
    
    for paper in papers:
        print(f"Title: {paper['title']}")
        print(f"DOI: {paper['doi']}")
        print(f"Year: {paper['year']}")
        print(f"URL: {paper['url']}")
        print(f"Publication Name: {paper['publicationName']}")
        print(f"Creator: {paper['creator']}")
        print(f"Affiliation: {paper['affiliation']}")
        print(f"Abstract: {paper['abstract']}")
        print("-" * 40)

# Example usage
download_scopus_papers("Nanoporous materials", max_results=5, start_year=2020, end_year=2025)
