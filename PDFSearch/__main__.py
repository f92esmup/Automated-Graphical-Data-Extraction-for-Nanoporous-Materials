# -*- coding: utf-8 -*-

import sys
import os
import time
import requests
from .Paper import Paper
from .PapersFilters import filterJurnals, filter_min_date, similarStrings
from .Downloader import downloadPapers, download_arxiv_papers
from .Scholar import ScholarPapersInfo
from .Crossref import getPapersInfoFromDOIs
from .proxy import proxy
from .__init__ import __version__
from urllib.parse import urljoin

class PyPaperBot:
    def __init__(self, query=None, scholar_results=10, scholar_pages=1, dwn_dir=None, proxy_list=None, min_date=None, 
                 num_limit=None, num_limit_type=None, filter_jurnal_file=None, restrict=None, DOIs=None, SciHub_URL=None, 
                 chrome_version=None, cites=None, use_doi_as_filename=False, SciDB_URL=None, skip_words=None, single_proxy=None, doi_file=None, description=None):
        # Query to make on Google Scholar or Google Scholar page link
        self.query = query
        # Number of scholar results to be downloaded when --scholar-pages=1
        self.scholar_results = scholar_results
        # Number or range of Google Scholar pages to inspect. Each page has a maximum of 10 papers
        self.scholar_pages = scholar_pages
        # Directory path in which to save the result
        self.dwn_dir = dwn_dir
        # Proxies to be used. Please specify the protocol to be used.
        self.proxy_list = proxy_list or []
        # Minimal publication year of the paper to download
        self.min_date = min_date
        # Maximum number of papers to download sorted by year or citations
        self.num_limit = num_limit
        # Type of limit: 0 for year, 1 for citations
        self.num_limit_type = num_limit_type
        # CSV file path of the journal filter
        self.filter_jurnal_file = filter_jurnal_file
        # 0: Download only Bibtex, 1: Download only papers PDF
        self.restrict = restrict
        # DOI of the paper to download (this option uses only SciHub to download)
        self.DOIs = DOIs
        # Mirror for downloading papers from sci-hub. If not set, it is selected automatically
        self.SciHub_URL = SciHub_URL
        # First three digits of the chrome version installed on your machine. If provided, selenium will be used for scholar search.
        self.chrome_version = chrome_version
        # Paper ID (from scholar address bar when you search cites) if you want to get only citations of that paper
        self.cites = cites
        # If provided, files are saved using the unique DOI as the filename rather than the default paper title
        self.use_doi_as_filename = use_doi_as_filename
        # Mirror for downloading papers from Annas Archive (SciDB). If not set, https://annas-archive.se is used
        self.SciDB_URL = SciDB_URL
        # List of comma separated words i.e. "word1,word2 word3,word4". Articles containing any of this word in the title or google scholar summary will be ignored
        self.skip_words = skip_words
        # Use a single proxy. Recommended if using --proxy gives errors.
        self.single_proxy = single_proxy
        # File .txt containing the list of paper's DOIs to download
        self.doi_file = doi_file

        self.description = description
    def checkVersion(self):
        try:
            print("PyPaperBot v" + __version__)
            response = requests.get('https://pypi.org/pypaperbot/json')
            latest_version = response.json()['info']['version']
            if latest_version != __version__:
                print("NEW VERSION AVAILABLE!\nUpdate with 'pip install PyPaperBot —upgrade' to get the latest features!\n")
        except:
            pass

    def start(self):
        if self.SciDB_URL is not None and "/scidb" not in self.SciDB_URL:
            self.SciDB_URL = urljoin(self.SciDB_URL, "/scidb/")

        to_download = []
        if self.DOIs is None:
            print("Query: {}".format(self.query))
            print("Cites: {}".format(self.cites))
            if isinstance(self.scholar_pages, str):
                try:
                    split = self.scholar_pages.split('-')
                    if len(split) == 1:
                        self.scholar_pages = range(1, int(split[0]) + 1)
                    elif len(split) == 2:
                        start_page, end_page = [int(x) for x in split]
                        self.scholar_pages = range(start_page, end_page + 1)
                    else:
                        raise ValueError
                except Exception:
                    print(
                        r"Error: Invalid format for --scholar-pages option. Expected: %d or %d-%d, got: " + self.scholar_pages)
                    sys.exit()
            to_download = ScholarPapersInfo(self.query, self.scholar_pages, self.restrict, self.min_date, self.scholar_results, self.chrome_version, self.cites, self.skip_words)
        else:
            print("Downloading papers from DOIs\n")
            num = 1
            i = 0
            while i < len(self.DOIs):
                DOI = self.DOIs[i]
                print("Searching paper {} of {} with DOI {}".format(num, len(self.DOIs), DOI))
                papersInfo = getPapersInfoFromDOIs(DOI, self.restrict)
                papersInfo.use_doi_as_filename = self.use_doi_as_filename
                to_download.append(papersInfo)

                num += 1
                i += 1

        if self.restrict != 0 and to_download:
            if self.filter_jurnal_file is not None:
                to_download = filterJurnals(to_download, self.filter_jurnal_file)

            if self.min_date is not None:
                to_download = filter_min_date(to_download, self.min_date)

            if self.num_limit_type is not None and self.num_limit_type == 0:
                to_download.sort(key=lambda x: int(x.year) if x.year is not None else 0, reverse=True)

            if self.num_limit_type is not None and self.num_limit_type == 1:
                to_download.sort(key=lambda x: int(x.cites_num) if x.cites_num is not None else 0, reverse=True)

            result = download_arxiv_papers(self.query, max_results=self.num_limit, start_year=self.min_date, end_year=None)
            downloadPapers(to_download, self.dwn_dir, self.num_limit, self.SciHub_URL, self.SciDB_URL)
            
        Paper.generateReport(result, to_download, self.dwn_dir + "search.csv", self.dwn_dir, self.description)
        #Paper.generateBibtex(to_download, self.dwn_dir + "bibtex.bib")

    def main(self):
        if self.single_proxy is not None:
            os.environ['http_proxy'] = self.single_proxy
            os.environ['HTTP_PROXY'] = self.single_proxy
            os.environ['https_proxy'] = self.single_proxy
            os.environ['HTTPS_PROXY'] = self.single_proxy
            print("Using proxy: ", self.single_proxy)
        else:
            proxy(self.proxy_list)

        if self.query is None and self.DOIs is None and self.cites is None:
            print("Error, provide at least one of the following arguments: --query, --doi, or --cites")
            sys.exit()

        if (self.query is not None and self.DOIs is not None) or (self.query is not None and self.cites is not None) or (
                self.DOIs is not None and self.cites is not None):
            print("Error: Only one option between '--query', '--doi' and '--cites' can be used")
            sys.exit()

        if self.dwn_dir is None:
            print("Error, provide the directory path in which to save the results")
            sys.exit()

        if self.scholar_results != 10 and self.scholar_pages != 1:
            print("Scholar results best applied along with --scholar-pages=1")

        dwn_dir = self.dwn_dir.replace('\\', '/')
        if dwn_dir[-1] != '/':
            dwn_dir += "/"
        if not os.path.exists(dwn_dir):
            os.makedirs(dwn_dir, exist_ok=True)

        if self.num_limit is not None and self.num_limit_type is not None:
            print("Error: Only one option between '--max-dwn-year' and '--max-dwn-cites' can be used ")
            sys.exit()

        if self.query is not None or self.cites is not None:
            if self.scholar_pages:
                try:
                    split = self.scholar_pages.split('-')
                    if len(split) == 1:
                        self.scholar_pages = range(1, int(split[0]) + 1)
                    elif len(split) == 2:
                        start_page, end_page = [int(x) for x in split]
                        self.scholar_pages = range(start_page, end_page + 1)
                    else:
                        raise ValueError
                except Exception:
                    print(
                        r"Error: Invalid format for --scholar-pages option. Expected: %d or %d-%d, got: " + self.scholar_pages)
                    sys.exit()
            else:
                print("Error: with --query provide also --scholar-pages")
                sys.exit()
        else:
            self.scholar_pages = 0

        if self.DOIs is None and self.doi_file is not None:
            self.DOIs = []
            f = self.doi_file.replace('\\', '/')
            with open(f) as file_in:
                for line in file_in:
                    if line[-1] == '\n':
                        self.DOIs.append(line[:-1])
                    else:
                        self.DOIs.append(line)

        self.start()

if __name__ == "__main__":
    bot = PyPaperBot(query="Machine learning", scholar_pages="1-3", dwn_dir="./data/papers2/", doi_file="path/to/doi_file.txt")
    bot.checkVersion()
    bot.main()