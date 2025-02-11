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
        # os.environ["GEMINI_API_KEY"] = 'AIzaSyDFuwrnPunjaEG5WlzjycQ75km-w2MFsgc'
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
            system_instruction="Line or Scatter Plot Detection\n\nYou are an AI system designed to analyze images and determine if they represent line or scatter plots. You will receive an image as input and should respond with \"YES\" if the image is a line or scatter plot, and \"NO\" if it is not.\n\n**Consider a line or scatter plot as a visual representation of data on a coordinate system, where points or lines are used to show the relationship between two or more variables.** These plots can include:\n\n* **Scatter plots:** Individual points represent data.\n* **Line plots:** Lines connect data points to show trends.\n* **Multi-line plots:** Multiple lines on a single graph represent different datasets.\n* **Area charts:**  The area under a line is filled to emphasize the magnitude of a value.\n* **Step charts:** Horizontal and vertical lines connect data points, showing discrete changes.\n\n\n**DO NOT consider the following types of charts as line or scatter plots:**\n\n* **Bar charts:** Use rectangular bars to represent data.\n* **Pie charts:** Use circular sectors to represent proportions.\n* **Histograms:** Use bars to show the distribution of data.\n* **Box plots:** Show the statistical distribution of data.\n* **Images or diagrams that do not represent data on a coordinate system.**\n* **Text or mathematical equations without a graphical representation in coordinates.**\n\n\n**Example:**\n\n**Image:** A graph showing the relationship between temperature and time using a line.\n\n**Response:** YES\n\n**Image:** A bar graph showing the sales figures for different products.\n\n**Response:** NO\n\n\n**Await the image and respond only with \"YES\" or \"NO\".**\n"
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
