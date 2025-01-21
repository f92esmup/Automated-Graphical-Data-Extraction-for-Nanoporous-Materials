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
            system_instruction="""Binary Classification of Intrusion-Extrusion Graphs

    Answer YES if the graph shows ALL of these characteristics:

    ## Curve Structure
    - Clear separation between intrusion and extrusion paths
    - At least one complete cycle (intrusion + extrusion)
    - Pressure axis showing positive values
    - Volume/uptake axis showing measurable changes

    ## Essential Features
    - Hysteresis between loading and unloading curves
    - Defined onset pressures for intrusion
    - Return path (extrusion) differs from entry path (intrusion)
    - Reasonable pressure range for the studied system (typically 0-200 MPa)

    ## Data Quality
    - Clear axis labels with units
    - Distinguishable data points or curves
    - Consistent baseline
    - No unexplained discontinuities

    Answer NO if ANY of these conditions are present:

    ## Invalid Characteristics
    - Missing or incomplete curves
    - Negative pressure values (unless specifically studying negative pressure regions)
    - No clear hysteresis
    - Physically impossible features (e.g., volume increasing during extrusion above intrusion curve)
    - Missing or incorrect units
    - Discontinuities that cannot be explained by phase transitions or material behavior
    - Curves that cross in physically impossible ways
    - Data that violates conservation of mass/volume

    ## Technical Errors
    - Axes without labels or units
    - Missing legend when multiple curves are present
    - Inconsistent or incorrect scaling
    - Undefined experimental conditions when crucial
    - Mathematical impossibilities in the data

    ## Physical Impossibilities
    - Volume changes that exceed material limitations
    - Pressure ranges outside equipment capabilities
    - Instantaneous transitions that violate physical laws
    - Curves that violate thermodynamic principles
    - Response times faster than physical limitations

    ## Response Format
    Respond ONLY with:
    - "YES" if all positive criteria are met and no negative criteria are present
    - "NO" if any negative criterion is present or any positive criterion is missing

    Do not provide explanations, justifications, or additional commentary unless specifically requested."""
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
