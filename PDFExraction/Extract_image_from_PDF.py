import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import supervision as sv
import os
from pdf2image import convert_from_path, pdfinfo_from_path  # Updated import
from supervision import BoxAnnotator  # Updated import
from PyPDF2 import PdfReader
from MLmanager.Gemini.Chart_Classification import ChartClassification  # Import the ChartClassification class
from tqdm import tqdm  # Add tqdm import

# Set Hugging Face API key
#os.environ["HF_API_KEY"] = "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"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizers parallelism warning

class ImageInference:
    def __init__(self, model_path,classification=True, text_extraction=False):
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(self.DEVICE)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.text_extraction = text_extraction
        self.classification = classification
        if self.classification:
            self.chart_classifier = ChartClassification(debug=False)  # Initialize ChartClassification

    def inference_from_image(self, image_path, page_number, output_dir):
        try:
            # Load and preprocess the image
            image = Image.open(image_path)
            task = "<OD>"
            text = "<OD>"

            inputs = self.processor(text=text, images=image, return_tensors="pt").to(self.DEVICE)

            # Perform inference
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3
                )

            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            response = self.processor.post_process_generation(generated_text, task=task, image_size=(image.width, image.height))
            detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, response, resolution_wh=image.size)

            # Remove annotation code
            # bounding_box_annotator = BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)  # Updated class name
            # label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)

            # image = bounding_box_annotator.annotate(image, detections)
            # image = label_annotator.annotate(image, detections)

            # Save the cropped images
            for j, bbox in enumerate(response["<OD>"]["bboxes"]):
                x1, y1, x2, y2 = bbox
                try:
                    cropped_image = image.crop((x1, y1, x2, y2))
                    cropped_image_path = os.path.join(output_dir, f'image_{page_number}_{j + 1}.png')
                    cropped_image.save(cropped_image_path)
                except Exception as e:
                    print(f"Error cropping image: {e}")

            return response
        except Exception as e:
            print(f"An error occurred during inference: {e}")
            return None

    def inference_from_directory(self, directory_path, output_dir):
        for filename in os.listdir(directory_path):
            if filename.endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(directory_path, filename)
                response = self.inference_from_image(image_path)
                if response:
                    self.process_roi([Image.open(image_path)], response, output_dir)

    def process_roi(self, pages, roi_data, output_dir):
        for i, page in enumerate(pages):
            image_path = os.path.join(output_dir, f'page_{i + 1}.png')
            page.save(image_path, 'PNG')

            # Extract ROI from the image
            for j, bbox in enumerate(roi_data["<OD>"]["bboxes"]):  # Iterate through bounding boxes
                x1, y1, x2, y2 = bbox
                try:
                    # Crop the image using the coordinates from the ROI data
                    cropped_image = page.crop((x1, y1, x2, y2))

                    # Save the cropped image with the new naming convention
                    cropped_image_path = os.path.join(output_dir, f'image{j + 1}.png')
                    cropped_image.save(cropped_image_path)

                except Exception as e:
                    print(f"Error cropping image: {e}")

        print(f'The pages of the PDF and their ROIs have been saved in {output_dir}')

    def extract_text_from_pdf(self, pdf_path, output_dir):
        try:
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            text_output_path = os.path.join(output_dir, f"{pdf_name}.txt")

            with open(pdf_path, "rb") as file:
                reader = PdfReader(file)
                text = ""
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text += page.extract_text() + "\n"

            with open(text_output_path, "w") as text_file:
                text_file.write(text)

            print(f"Text extracted and saved to {text_output_path}")
        except Exception as e:
            print(f"An error occurred while extracting text: {e}")

    def detect_images_in_page(self, page):
        try:
            images = page['/Resources']['/XObject'].get_object()
            return any(images)
        except KeyError:
            return False

    def convert_pdf_to_images_and_infer(self, input_dir, output_dir):
        # Check if input and output directories exist, if not create them
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)
            print(f"Input directory {input_dir} created.")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Output directory {output_dir} created.")
        
        pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            pdf_path = os.path.join(input_dir, pdf_file)
            pdf_name = os.path.splitext(pdf_file)[0]
            pdf_output_dir = os.path.join(output_dir, pdf_name)
            os.makedirs(pdf_output_dir, exist_ok=True)
            
            # Convert PDF to images
            pages = convert_from_path(pdf_path)
            reader = PdfReader(pdf_path)
            
            for i, page in enumerate(tqdm(pages, desc=f"Processing pages of {pdf_file}", leave=False)):
                page_number = i + 1
                pdf_page = reader.pages[i]
                
                if self.detect_images_in_page(pdf_page):
                    image_path = os.path.join(pdf_output_dir, f'page_{page_number}.png')
                    page.save(image_path, 'PNG')
                    
                    # Perform inference on the saved image
                    response = self.inference_from_image(image_path, page_number, pdf_output_dir)
                    
                    # Delete the image after inference
                    os.remove(image_path)
            
            print(f'The pages of the PDF {pdf_file} have been processed and saved in {pdf_output_dir}')
            
            # Extract text from the PDF
            if self.text_extraction:
                self.extract_text_from_pdf(pdf_path, pdf_output_dir)
                
            if self.classification:
                self.chart_classifier.classify_images_in_directory(pdf_output_dir)
            
if __name__ == "__main__":
    # Example usage: Replace with your model path and directory path
    model_path = "./AI/FLorence-Demo/florence2-lora" # Replace with your model path
    pdf_input_dir = "/home/pedro/CICProject/data/papers" # Replace with your PDF input directory
    output_dir = "./data/papers" # Replace with your output directory
    
    # Load the model and perform inference
    inference = ImageInference(model_path)

    inference.convert_pdf_to_images_and_infer(pdf_input_dir, output_dir)
