import os
import sys
from PDFSearch.PyPaperBot.PyPaperBot.__main__ import main

def download_papers(query, scholar_pages, download_dir):
    # Ensure the download directory exists
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    
    # Set the parameters
    args = [
        '--query', query,
        '--scholar-pages', str(scholar_pages),
        '--dwn-dir', download_dir,
        '--restrict', '0',  # Download only BibTeX (which includes the abstract)
        '--use-doi-as-filename', 'False'  # Optional: Use DOI as filename
    ]
    
    # Call the main function with the arguments
    sys.argv = [''] + args
    main()

if __name__ == "__main__":
    query = "Machine learning"
    scholar_pages = 1
    download_dir = "./data/papers2/"
    
    download_papers(query, scholar_pages, download_dir)
