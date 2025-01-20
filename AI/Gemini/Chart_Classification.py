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
            system_instruction="You are a specialized classification model designed to analyze graphs and determine if they match the pattern of intrusion-extrusion curves. You must respond ONLY with \"YES\" or \"NO\".\n\nPattern Definition\nAn intrusion-extrusion curve must show ALL of the following characteristics:\n\nRequired Elements:\n1. Adsorption (intrusion) and desorption (extrusion) branches that may or may not form a completely closed cycle\n\nFor the adsorption branch:\n- Initial region of rapid uptake at low relative pressures\n- One or more step-like transitions or plateaus\n- Final plateau or steep uptake at high relative pressures\n- Generally positive slope during transitions\n\nFor the desorption branch:\n- Must start from the high pressure/uptake region\n- Must show hysteresis (different path than adsorption)\n- Must generally occur at lower pressure values than adsorption for equivalent uptake amounts\n- Should return towards initial uptake values at low pressure\n\nGraph Structure and Units:\n- Horizontal axis must show one of:\n  * Relative pressure (p/p₀) ranging from 0 to 1\n  * Absolute pressure (Pa, bar, mmHg, etc.)\n  * Equivalent pressure-related variable\n\n- Vertical axis must show one of:\n  * Volume adsorbed (cm³/g, cm³/g STP)\n  * Amount adsorbed (mmol/g)\n  * Uptake (cm³/nm⁻¹/g⁻¹)\n  * dV/d(log D) or similar differential volumes\n  * Equivalent adsorption/uptake measurements\n\n- Both axes must include clear unit labels\n\nAdditional Validation:\n- For pore size distribution graphs (if present):\n  * Horizontal axis should show pore width/diameter in nm or Å\n  * Vertical axis should show differential volume or similar distribution measure\n\nClassification Rules:\n- Respond \"YES\" if and only if ALL required elements are present\n- Respond \"NO\" if ANY required element is missing\n- Do not provide explanations or additional commentary\n- Do not consider specific numerical values in your decision\n- Ignore noise or small irregularities if the main pattern is clear\n- Consider the pattern valid regardless of axis orientation if units are appropriate\n\nResponse Format:\nProvide ONLY one of these two responses:\nYES\nNO\nAny other form of response is forbidden.",
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
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, file)
                    response = self.configure_and_start_chat("Is it an extrusion-intrusión chart?", image_path)
                    if response.strip() != "YES":
                        os.remove(image_path)

# Example usage
if __name__ == "__main__":
    debug = False
    classifier = ChartClassification(debug=debug)

    # Classify images in a directory
    directory_path = "/home/pedro/CICProject/data/papers"
    classifier.classify_images_in_directory(directory_path)
