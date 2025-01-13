import fitz  # PyMuPDF
import io
from PIL import Image
import os

def extract_text_from_pdf(pdf_path, output_folder):
    pdf_document = fitz.open(pdf_path)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    text_output_folder = os.path.join(output_folder, 'text')
    if not os.path.exists(text_output_folder):
        os.makedirs(text_output_folder)
    text_output_path = os.path.join(text_output_folder, f"{pdf_name}.txt")
    with open(text_output_path, 'w', encoding='utf-8') as text_file:
        for page_number in range(len(pdf_document)):
            page = pdf_document.load_page(page_number)
            text = page.get_text()
            text_file.write(text)

def extract_images_from_pdf(pdf_path, output_folder):
    pdf_document = fitz.open(pdf_path)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    pdf_output_folder = os.path.join(output_folder, 'images', pdf_name)
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
            
            # Save the image in the output folder
            image.save(f"{pdf_output_folder}/image_{page_number+1}_{img_index+1}.{image_ext}")

# Remove example usage to allow importing in main.py
# papers_folder = './data/papers'
# output_folder = './data'
# for pdf_filename in os.listdir(papers_folder):
#     if pdf_filename.endswith('.pdf'):
#         pdf_path = os.path.join(papers_folder, pdf_filename)
#         extract_images_from_pdf(pdf_path, output_folder)
#         extract_text_from_pdf(pdf_path, output_folder)