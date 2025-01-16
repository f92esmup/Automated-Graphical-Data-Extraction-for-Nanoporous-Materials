import os
from pdf2image import convert_from_path

class PDFToImageConverter:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def convert(self):
        for pdf_file in os.listdir(self.input_dir):
            if pdf_file.endswith('.pdf'):
                pdf_path = os.path.join(self.input_dir, pdf_file)
                pdf_name = os.path.splitext(pdf_file)[0]
                pdf_output_dir = os.path.join(self.output_dir, pdf_name)
                os.makedirs(pdf_output_dir, exist_ok=True)
                
                # Convertir PDF a imágenes
                pages = convert_from_path(pdf_path)
                
                # Guardar cada página como una imagen
                for i, page in enumerate(pages):
                    image_path = os.path.join(pdf_output_dir, f'page_{i + 1}.png')
                    page.save(image_path, 'PNG')
                
                print(f'Las páginas del PDF {pdf_file} se han guardado en {pdf_output_dir}')

# Uso de la clase
if __name__ == "__main__":
    # Directorio de entrada
    input_dir = '/home/pedro/CICProject/data/DemoPapers'
    
    # Directorio de salida
    output_dir = '/home/pedro/CICProject/data/images'
    
    converter = PDFToImageConverter(input_dir, output_dir)
    converter.convert()