import os
import pandas as pd
import requests
from PDFSearch.PyPaperBot.__main__ import PyPaperBot

def main():
    query = "Machine learning"
    scholar_pages = "1-3"
    dwn_dir = "./data/papers2/"

    bot = PyPaperBot(query=query, scholar_pages=scholar_pages, dwn_dir=dwn_dir, restrict=1)
    bot.main()

if __name__ == "__main__":
    main()
