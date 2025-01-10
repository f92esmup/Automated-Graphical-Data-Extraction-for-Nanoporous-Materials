from PDFSearch.__main__ import PyPaperBot
import google.generativeai as genai
import os

# Set the API key in the environment variables
os.environ["GEMINI_API_KEY"] = 'AIzaSyDFuwrnPunjaEG5WlzjycQ75km-w2MFsgc'
api_key = os.environ["GEMINI_API_KEY"]
genai.configure(api_key=api_key)


def get_pdfs(query, scholar_pages, download_dir, Description):
    # Create an instance of PyPaperBot and set the parameters
    bot = PyPaperBot(query=query, scholar_pages=scholar_pages, dwn_dir=download_dir, restrict=1, description=Description)
    # Call the main method
    bot.main()
    del bot

if __name__ == "__main__":

    query = "Nanoporus materials"
    scholar_pages = "1-1"
    download_dir = "./data/papers2/"
    Description = "The document should focus on the processes of liquid intrusion and extrusion in confined media, either from a theoretical or experimental perspective. It may include analysis of physical properties such as wettability, hydrophobicity, surface tension, and bubble nucleation. The document should also discuss technological applications such as energy storage, liquid separation, or chromatography, as well as implications for biological or bioinspired systems. Relevant theoretical models could include confined classical nucleation theories (cCNT), experimental methods such as liquid porosimetry or calorimetry, and atomistic or DFT-based simulations. Keywords should include terms like 'intrusion-extrusion', 'wetting-drying', 'hydrophobicity-lyophobicity', 'nucleation', and 'nanoporous materials.'"
    
    get_pdfs(query, scholar_pages, download_dir, Description)

