import os
import pandas as pd
import requests
from PDFSearch.PyPaperBot.__main__ import PyPaperBot

def get_results_csv(query, scholar_pages, download_dir):
    # Create an instance of PyPaperBot and set the parameters
    bot = PyPaperBot(query=query, scholar_pages=scholar_pages, dwn_dir=download_dir, restrict=1)
    # Call the main method
    bot.main()
    del bot

if __name__ == "__main__":

    query = "Nanoporus materials"
    scholar_pages = "1-1"
    download_dir = "./data/papers2/"
    get_results_csv(query, scholar_pages, download_dir)
