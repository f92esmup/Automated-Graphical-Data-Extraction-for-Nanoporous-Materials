# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Image processing and system libraries
import os

# Google Generative AI library
import google.generativeai as genai

# Set the API key in the environment variables
os.environ["GEMINI_API_KEY"] = 'AIzaSyDFuwrnPunjaEG5WlzjycQ75km-w2MFsgc'
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def upload_to_gemini(path, debug, mime_type=None):
    """Uploads the given file to Gemini.

    See https://ai.google.dev/gemini-api/docs/prompting_with_media
    """
    file = genai.upload_file(path, mime_type=mime_type)
    if debug: print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

def configure_and_start_chat(input_message, image_path, debug):
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
        system_instruction="RESPOND only with YES or NO.",
    )

    # Upload the image to Gemini
    files = [
        upload_to_gemini(image_path, debug, mime_type="image/png"),
    ]
    
    chat_session = model.start_chat(
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

# Example usage
input_message = "Is it an extrusion-intrusión chart are you sure?"
image_path = "/home/pedro/CICProject/data/DemoImages/image3.png"
debug = False
print(configure_and_start_chat(input_message, image_path, debug))
