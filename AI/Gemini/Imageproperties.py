# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Image processing and system libraries
import os

# Google Generative AI library
import google.generativeai as genai

class GraphDocumentAnalysis:
    def __init__(self, debug=False):
        self.debug = debug
        self.model = self._create_model()
        self._configure_api()

    def _configure_api(self):
        os.environ["GEMINI_API_KEY"] = 'AIzaSyDFuwrnPunjaEG5WlzjycQ75km-w2MFsgc'
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    def _create_model(self):
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
            system_instruction="# System Instructions for Graph-Document Analysis Model\n\nYou are a specialized information extraction and narrative generation model. Your task is to analyze a graph image and its corresponding technical documentation to create a concise, informative paragraph about the studied system.\n\n## Purpose\nGenerate a brief, coherent paragraph that describes:\n- The main compound or material system being studied\n- Its relevant physical and chemical properties\n- The key experimental conditions and findings shown in the graph\n\n## Guidelines for Response\n1. Write a single, flowing paragraph that naturally incorporates:\n   - The identity of the main compound/system\n   - Any relevant experimental conditions\n   - Key behavioral characteristics shown in the graph\n   - Significant findings or properties\n\n2. Keep the description:\n   - Focused and concise (3-5 sentences maximum)\n   - Technical but accessible\n   - Directly related to the provided graph\n   - Based only on explicitly stated information\n\n3. Writing style should be:\n   - Professional and scientific\n   - Clear and straightforward\n   - Free of speculation or interpretation\n   - Focused on factual information\n\n## Response Rules\n- Only include information that directly correlates between the document and graph\n- Maintain scientific accuracy and precision\n- Use original units and nomenclature\n- If no correlation is found, state: \"No matching information could be found between the graph and document.\"\n\nRemember: The goal is to create a natural, readable description that a scientist would find informative and accurate, not a structured list of properties.",
        )
        return model

    def upload_to_gemini(self, path, mime_type=None):
        file = genai.upload_file(path, mime_type=mime_type)
        if self.debug: print(f"Uploaded file '{file.display_name}' as: {file.uri}")
        return file

    def configure_and_start_chat(self, input_message, image_path, document_path):
        files = [
            self.upload_to_gemini(image_path, mime_type="image/png"),
            self.upload_to_gemini(document_path, mime_type="application/pdf"),
        ]
        
        chat_session = self.model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [
                        files[0],
                        files[1],
                    ],
                },
            ]
        )

        response = chat_session.send_message(input_message)
        return response.text

    def analyze_graph_and_document(self, image_path, document_path):
        response = self.configure_and_start_chat("Analyze the graph and document.", image_path, document_path)
        return response

# Example usage
if __name__ == "__main__":
    debug = False
    analyzer = GraphDocumentAnalysis(debug=debug)

    # Analyze a single PDF and corresponding image
    pdf_path = "/home/pedro/CICProject/data/DemoPapers/reference.pdf"
    image_path = "/home/pedro/CICProject/data/DemoImages/image3.png"
    result = analyzer.analyze_graph_and_document(image_path, pdf_path)
    print(f"Result:\n{result}\n")
