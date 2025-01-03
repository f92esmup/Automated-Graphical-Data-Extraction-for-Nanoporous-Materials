import os
import pandas as pd
import requests
from PDFSearch.PyPaperBot.__main__ import PyPaperBot

def get_results_csv(query, scholar_pages, download_dir):
    # Create an instance of PyPaperBot and set the parameters
    bot = PyPaperBot(query=query, scholar_pages=scholar_pages, dwn_dir=download_dir, restrict=0)
    # Call the main method
    bot.main()

def download_pdfs(download_dir, dois):
    # Create an instance of PyPaperBot to download PDFs
    bot = PyPaperBot(dwn_dir=download_dir, restrict=1, DOIs=dois)
    # Call the main method
    bot.main()

query = "Machine learning"
scholar_pages = "1-1"
download_dir = "./data/papers2/"
get_results_csv(query, scholar_pages, download_dir)

# Assuming the DOIs are stored in a CSV file
dois = pd.read_csv(os.path.join(download_dir, "result.csv"))["DOI"].tolist()
download_pdfs(download_dir, dois)
