import os
from pdf2image import convert_from_path

# Ruta del archivo PDF
pdf_path = '/home/pedro/CICProject/PDFExraction/reference.pdf'

# Directorio de salida
output_dir = '/home/pedro/CICProject/PDFExraction/pdf2image'
os.makedirs(output_dir, exist_ok=True)

# Convertir PDF a imágenes
pages = convert_from_path(pdf_path)

# Guardar cada página como una imagen
for i, page in enumerate(pages):
    image_path = os.path.join(output_dir, f'page_{i + 1}.png')
    page.save(image_path, 'PNG')

print(f'Las páginas del PDF se han guardado en {output_dir}')