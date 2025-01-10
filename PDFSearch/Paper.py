# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 21:43:30 2020

@author: Vito
"""
import bibtexparser
import re
import pandas as pd
import urllib.parse

import google.generativeai as genai
import PyPDF2

class Paper:


    def __init__(self,title=None, scholar_link=None, scholar_page=None, cites=None, link_pdf=None, year=None, authors=None):        
        self.title = title
        self.scholar_page = scholar_page
        self.scholar_link = scholar_link
        self.pdf_link = link_pdf
        self.year = year
        self.authors = authors

        self.jurnal = None
        self.cites_num = None
        self.bibtex = None
        self.DOI = None
        self.abstract = None  # Add abstract attribute

        self.downloaded = False
        self.downloadedFrom = 0  # 1-SciHub 2-scholar
        
        self.use_doi_as_filename = False # if True, the filename will be the DOI

    def getFileName(self):
            try:
                if self.use_doi_as_filename:
                    return urllib.parse.quote(self.DOI, safe='') + ".pdf"
                else:
                    return re.sub(r'[^\w\-_. ]', '_', self.title) + ".pdf"
            except:
                return "none.pdf"

    def setBibtex(self, bibtex):
        x = bibtexparser.loads(bibtex, parser=None)
        x = x.entries

        self.bibtex = bibtex

        try:
            if "year" in x[0]:
                self.year = x[0]["year"]
            if 'author' in x[0]:
                self.authors = x[0]["author"]
            self.jurnal = x[0]["journal"].replace("\\", "") if "journal" in x[0] else None
            if self.jurnal is None:
                self.jurnal = x[0]["publisher"].replace("\\", "") if "publisher" in x[0] else None
            if 'abstract' in x[0]:
                self.abstract = x[0]["abstract"]  # Extract abstract if available
        except:
            pass

    def canBeDownloaded(self):
        return self.DOI is not None or self.scholar_link is not None

    def generateReport(papers, path, dwn_dir, Description):
        # Define the column names
        columns = ["Name", "Scholar Link", "Bibtex", "PDF Name",
                   "Year", "Scholar page", "Journal", "Downloaded",
                   "Downloaded from", "Authors", "Abstract", "Description"]

        # Prepare data to populate the DataFrame
        data = []
        for p in papers:
            pdf_name = p.getFileName() if p.downloaded else ""
            bibtex_found = p.bibtex is not None

            # Determine download source
            dwn_from = ""
            if p.downloadedFrom == 1:
                dwn_from = "SciDB"
            elif p.downloadedFrom == 2:
                dwn_from = "SciHub"
            elif p.downloadedFrom == 3:
                dwn_from = "Scholar"

            # Extract two pages if abstract is not found
            if p.abstract is None or p.abstract.lower() == 'abstract not found':
                try:
                    pdf_path = dwn_dir + p.getFileName()
                    p.abstract = Paper.extract_text_from_first_two_pages(pdf_path)
                except:
                    pass
            


            # Append row data as a dictionary
            data.append({
                "Name": p.title,
                "Scholar Link": p.scholar_link,
                "DOI": p.DOI,
                #"Bibtex": bibtex_found,
                "PDF Name": pdf_name,
                "Year": p.year,
                "Scholar page": p.scholar_page,
                "Journal": p.jurnal,
                "Downloaded": p.downloaded,
                "Downloaded from": dwn_from,
                "Authors": p.authors,
                "Abstract": p.abstract,  # Include abstract in the report
                "Description": Paper.filter_with_gemini(p.abstract, Description)
            })

    

        # Create a DataFrame and write to CSV
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(path, index=False, encoding='utf-8')

    def generateBibtex(papers, path):
        content = ""
        for p in papers:
            if p.bibtex is not None:
                content += p.bibtex + "\n"

        relace_list = ["\ast", "*", "#"]
        for c in relace_list:
            content = content.replace(c, "")

        f = open(path, "w", encoding="latin-1", errors="ignore")
        f.write(str(content))
        f.close()
    
    def filter_with_gemini(text, Description):
    # Create the model
        generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
        }

        model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        generation_config=generation_config,
        system_instruction= "Your task is to evaluate whether a given text matches exactly with the following description:" + Description + "You must respond only with 'YES' if the text fully matches the description or 'NO' if it does not. Do not include explanations, additional comments, or modify the response format."
        )

        chat_session = model.start_chat(
        history=[
        ]
        )

        response = chat_session.send_message(text)

        return response.text.strip().lower() == 'yes'
    
    def extract_text_from_first_two_pages(pdf_path):
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            num_pages = min(2, len(reader.pages))
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text += page.extract_text()
        return text