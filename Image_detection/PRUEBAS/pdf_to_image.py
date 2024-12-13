from pdf2image import convert_from_path
import os

def extract_pages_as_images(pdf_path, output_folder, dpi=300):
    # Asegurarnos de que la carpeta de salida exista
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        # Eliminar todas las imágenes existentes en la carpeta de salida
        for file in os.listdir(output_folder):
            file_path = os.path.join(output_folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    
    try:
        # Convertir las páginas del PDF en imágenes
        pages = convert_from_path(pdf_path, dpi=dpi)
    except Exception as e:
        print(f"Error al convertir el PDF: {e}")
        return
    
    # Guardar cada página como una imagen
    for i, page in enumerate(pages):
        image_name = f"page_{i + 1}_high_quality.png"
        page.save(os.path.join(output_folder, image_name), 'PNG')
        print(f"Página renderizada como imagen: {image_name}")

if __name__ == "__main__":
    pdf_path = "/Users/pedroescudero/Desktop/FAM/boletin_2.pdf"
    output_folder = "/Users/pedroescudero/Desktop/FAM/boletin_2_images"
    extract_pages_as_images(pdf_path, output_folder)
