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

def run_search(params):
    get_pdfs(params)

# Remove the main execution block to allow importing in main.py
# if __name__ == "__main__":
#     run_search()

