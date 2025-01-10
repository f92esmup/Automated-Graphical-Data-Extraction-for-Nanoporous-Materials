import fitz  # PyMuPDF
import io
from PIL import Image
import os

def clear_output_folder(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(output_folder):
        file_path = os.path.join(output_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def extract_images_from_pdf(pdf_path, output_folder):
    pdf_document = fitz.open(pdf_path)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    pdf_output_folder = os.path.join(output_folder, pdf_name)
    if not os.path.exists(pdf_output_folder):
        os.makedirs(pdf_output_folder)
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image['image']
            image_ext = base_image['ext']
            image = Image.open(io.BytesIO(image_bytes))
            image.save(f"{pdf_output_folder}/image_{page_number+1}_{img_index+1}.{image_ext}")

# Ejemplo de uso
papers_folder = './data/papers'
output_folder = './data/output'
clear_output_folder(output_folder)

for pdf_filename in os.listdir(papers_folder):
    if pdf_filename.endswith('.pdf'):
        pdf_path = os.path.join(papers_folder, pdf_filename)
        extract_images_from_pdf(pdf_path, output_folder)