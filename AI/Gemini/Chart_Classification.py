# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Image processing and system libraries
import os
import shutil

# Google Generative AI library
import google.generativeai as genai

class ChartClassification:
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
            system_instruction="You are a specialized classification model designed to analyze graphs and determine if they match the pattern of intrusion-extrusion curves. You must respond ONLY with \"YES\" or \"NO\".\nPattern Definition\nAn intrusion-extrusion graph must show ALL of the following characteristics:\nRequired Elements\n\nTwo distinct curves forming a closed cycle:\n\nOne for intrusion (loading)\nOne for extrusion (unloading)\n\n\nFor the intrusion curve:\n\nInitial horizontal plateau at low values\nSingle or multiple sharp transition regions\nFinal horizontal plateau at high values\nPositive slope during transitions\n\n\nFor the extrusion curve:\n\nMust start from the upper plateau\nMust show hysteresis (different path than intrusion)\nMust occur at lower pressure values than intrusion\nMust end near the initial starting point\n\n\n\nGraph Structure\n\nMust have pressure or equivalent variable on horizontal axis\nVertical axis must represent a quantity that changes with intrusion/extrusion\nBoth axes must show clear scales and units\n\nClassification Rules\n\nRespond \"YES\" if and only if ALL required elements are present\nRespond \"NO\" if ANY required element is missing\nDo not provide explanations or additional commentary\nDo not consider specific numerical values in your decision\nIgnore noise or small irregularities if the main pattern is clear\nConsider the pattern valid regardless of axis orientation or units used\n\nResponse Format\nProvide ONLY one of these two responses:\n\nYES\nNO\n\nAny other form of response is forbidden.",
        )
        return model

    def upload_to_gemini(self, path, mime_type=None):
        file = genai.upload_file(path, mime_type=mime_type)
        if self.debug: print(f"Uploaded file '{file.display_name}' as: {file.uri}")
        return file

    def configure_and_start_chat(self, input_message, image_path):
        files = [
            self.upload_to_gemini(image_path, mime_type="image/png"),
        ]
        
        chat_session = self.model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [
                        files[0],
                    ],
                },
            ]
        )

        response = chat_session.send_message(input_message)
        return response.text

    def classify_images_in_directory(self, directory_path):
        for root, _, files in os.walk(directory_path):
            intrusion_extrusion_dir = os.path.join(root, "Intrusion_extrusion")
            others_dir = os.path.join(root, "others")
            
            os.makedirs(intrusion_extrusion_dir, exist_ok=True)
            os.makedirs(others_dir, exist_ok=True)
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, file)
                    response = self.configure_and_start_chat("Is it an extrusion-intrusión chart?", image_path)
                    if response.strip() == "YES":
                        shutil.move(image_path, os.path.join(intrusion_extrusion_dir, file))
                    else:
                        shutil.move(image_path, os.path.join(others_dir, file))

# Example usage
if __name__ == "__main__":
    debug = False
    classifier = ChartClassification(debug=debug)

    # Classify images in a directory
    directory_path = "/home/pedro/CICProject/data/images"
    classifier.classify_images_in_directory(directory_path)
