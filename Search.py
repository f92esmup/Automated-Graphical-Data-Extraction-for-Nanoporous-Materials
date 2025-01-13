from PDFSearch.__main__ import PyPaperBot
import google.generativeai as genai
import os

# Set the API key in the environment variables
os.environ["GEMINI_API_KEY"] = 'AIzaSyDFuwrnPunjaEG5WlzjycQ75km-w2MFsgc'
api_key = os.environ["GEMINI_API_KEY"]
genai.configure(api_key=api_key)


def get_pdfs(params):
    # Check if the output directory exists, if not, create it
    output_dir = params.get("dwn_dir", "./data/papers/")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create an instance of PyPaperBot and set the parameters
    bot = PyPaperBot(**params)
    # Call the main method
    bot.main()
    del bot

if __name__ == "__main__":
    params = {
        "query": "Nanoporous materials",
        "scholar_results": 10,
        "scholar_pages": "1-1",
        "dwn_dir": "./data/papers/",
        "proxy_list": None,
        "min_date": None,
        "num_limit": 5,
        "num_limit_type": None,
        "filter_jurnal_file": None,
        "restrict": 1,
        "DOIs": None,
        "SciHub_URL": None,
        "chrome_version": None,
        "cites": None,
        "use_doi_as_filename": False,
        "SciDB_URL": None,
        "skip_words": None,
        "single_proxy": None,
        "doi_file": None,
        "description": "The document should focus on the processes of liquid intrusion and extrusion in confined media, either from a theoretical or experimental perspective. It may include analysis of physical properties such as wettability, hydrophobicity, surface tension, and bubble nucleation. The document should also discuss technological applications such as energy storage, liquid separation, or chromatography, as well as implications for biological or bioinspired systems. Relevant theoretical models could include confined classical nucleation theories (cCNT), experimental methods such as liquid porosimetry or calorimetry, and atomistic or DFT-based simulations. Keywords should include terms like 'intrusion-extrusion', 'wetting-drying', 'hydrophobicity-lyophobicity', 'nucleation', and 'nanoporous materials.'",
        "eliminate_false_values": False
    }
    
    get_pdfs(params)

