# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Image processing and system libraries
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt

# Google Generative AI library
import google.generativeai as genai

class GeminiImageProcessor:
    def __init__(self, debug=False):
        self.debug = debug
        self.model = self._create_model()

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
            system_instruction="""You will be provided with a single image, from which you must extract the following information and format it as a JSON object:
                - title
                - x_label (use HTML structure for subscript, e.g., N<sub>L</sub>)
                - x_numbers
                - y_label (use HTML structure for subscript, e.g., Ω [k<sub>B</sub>T])
                - y_numbers
                - legend
                - x_units
                - y_units (use HTML structure for subscript, e.g., k<sub>B</sub>T)
                - lines:
                    - An array of objects, where each object represents a line in the graph. Each object should contain:
                        - line_number: Enumerate the lines detected in the image based on their vertical position. Start from the top of the image and assign numbers in order from top to bottom (e.g., "line1", "line2", etc.).
                        - legend_label: Assign the corresponding legend label to each line. If a line has no legend label, leave this field as an empty string.
                        - line_type: Classify the line as either "compression" or "decompression" based on the following:
                            - **Compression**: Lines where the volume decreases and the pressure increases (typically a downward curve in a P-V diagram).
                            - **Decompression**: Lines where the volume increases and the pressure decreases (typically an upward curve in a P-V diagram).

                The x_numbers and y_numbers should be arrays of numbers. 

                Do not include any additional text or explanations, only the JSON object."""
        )
        return model

    def upload_to_gemini(self, path, mime_type=None):
        file = genai.upload_file(path, mime_type=mime_type)
        if self.debug: print(f"Uploaded file '{file.display_name}' as: {file.uri}")
        return file

    def process_image_with_gemini(self, image_path):
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

        response = chat_session.send_message("Provide the requested data in JSON format.")
        
        try:
            response_json = response.text.strip('```json\n').strip('\n```')
            response_json = json.loads(response_json)
            if self.debug: print(f"JSON output completed")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

        return response_json

# Example usage:
# processor = GeminiImageProcessor(debug=True)
# result = processor.process_image_with_gemini('/path/to/image.png')
