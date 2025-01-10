import arxiv

def download_arxiv_papers(query, max_results=1, start_year=None, end_year=None):
    # Construir la consulta con filtros adicionales
    args = {}

    if start_year or end_year:
        query_parts = [query]
        if start_year:
            query_parts.append(f"submittedDate:[{start_year}0101 TO {end_year}1231]")
        query = " AND ".join(query_parts)

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    client = arxiv.Client()
    for result in client.results(search):
        name = result.title
        doi = result.doi if result.doi else "N/A"
        pdf_name = result.pdf_url.split('/')[-1] + ".pdf"
        year = result.published.year
        journal = result.journal_ref if result.journal_ref else "N/A"
        authors = ', '.join(author.name for author in result.authors)
        abstract = result.summary

        args.append([name, doi, pdf_name, year, journal, authors, abstract])

        result.download_pdf(dirpath='./data/papers2/')

    return args

if __name__ == "__main__":
    query = "Nanoporous materials"
    start_year = 2020
    end_year = 2023
    author = None
    download_arxiv_papers(query, start_year=start_year, end_year=end_year, author=author)
