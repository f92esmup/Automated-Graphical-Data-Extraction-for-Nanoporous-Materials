import fitz  # PyMuPDF
import io
from PIL import Image
import os

def clear_output_folder(output_folder):
    for filename in os.listdir(output_folder):
        file_path = os.path.join(output_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def extract_images_from_pdf(pdf_path, output_folder):
    pdf_document = fitz.open(pdf_path)
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image['image']
            image_ext = base_image['ext']
            image = Image.open(io.BytesIO(image_bytes))
            image.save(f"{output_folder}/image_{page_number+1}_{img_index+1}.{image_ext}")

# Ejemplo de uso
pdf_path = './data/papers/reference.pdf'
output_folder = './output'
clear_output_folder(output_folder)
extract_images_from_pdf(pdf_path, output_folder)