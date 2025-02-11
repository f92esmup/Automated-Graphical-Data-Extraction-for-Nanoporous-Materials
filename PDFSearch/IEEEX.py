import requests
import os

def download_ieee_papers(query, dwn_dir, max_results=1, start_year=None, end_year=None):
    if max_results is None:
        max_results = 5

    args = []

    base_url = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
    api_key = "445cefmjypbfptjgnzgtwzpt"  # Replace with your IEEE Xplore API key

    params = {
        "apikey": api_key,
        "querytext": query,
        "max_records": max_results,
        "start_year": start_year,
        "end_year": end_year,
        "format": "json"  # Request the response in JSON format
    }

    response = requests.get(base_url, params=params)

    # Print the response for debugging
    print("Status Code:", response.status_code)
    print("Response Text:", response.text)

    if response.status_code == 403:
        print("Error: Developer Inactive. Verify that your API key is active.")
        return

    if response.status_code != 200:
        print("Request error:", response.status_code)
        return

    try:
        data = response.json()
    except requests.exceptions.JSONDecodeError as e:
        print("Error decoding JSON:", e)
        return

    for article in data.get('articles', []):
        paper_info = {
            "name": article.get('title', None),
            "doi": article.get('doi', None),
            "pdf_name": f"{article.get('doi', '').replace('/', '_')}.pdf" if article.get('doi') else None,
            "year": article.get('publication_year', None),
            "journal": article.get('publication_title', None),
            "authors": ', '.join(author.get('full_name', None) for author in article.get('authors', {}).get('authors', [])),
            "abstract": article.get('abstract', None)
        }

        args.append(paper_info)

        # Display metadata in a readable format
        print(f"Title: {paper_info['name']}")
        print(f"DOI: {paper_info['doi']}")
        print(f"PDF Name: {paper_info['pdf_name']}")
        print(f"Year: {paper_info['year']}")
        print(f"Journal: {paper_info['journal']}")
        print(f"Authors: {paper_info['authors']}")
        print(f"Abstract: {paper_info['abstract']}")
        print()

        # Download the PDF
        pdf_url = article.get('pdf_url', None)
        if pdf_url:
            pdf_response = requests.get(pdf_url)
            if pdf_response.status_code == 200:
                pdf_filename = os.path.join(dwn_dir, paper_info['pdf_name'])
                with open(pdf_filename, 'wb') as pdf_file:
                    pdf_file.write(pdf_response.content)
                print(f"PDF downloaded: {pdf_filename}")
            else:
                print(f"Error downloading the PDF: {pdf_response.status_code}")

    return args

if __name__ == "__main__":
    query = "Nanoporous materials"
    start_year = 2020
    end_year = 2023
    dwn_dir = "./downloads"
    results = download_ieee_papers(query, dwn_dir, start_year=start_year, end_year=end_year)
    print(results)
