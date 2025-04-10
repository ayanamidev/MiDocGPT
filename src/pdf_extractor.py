from PyPDF2 import PdfReader
from utils import pdf_bytes_to_images, VisionModel
import io

def extract_text_from_pdf(pdf_path):
    #Abir PDF
    reader = PdfReader(pdf_path)
    
    full_text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            full_text += page_text + "\n"
        else:   
            print(f"No text found on page {page.number}")

    return full_text


if __name__ == "__main__":
    import os
    
    pdf_path = "data/pdf.pdf"
    
    if os.path.exists(pdf_path):
        extracted_text = extract_text_from_pdf(pdf_path)
        
    print(extracted_text)
        

def extract_text_with_vision_model(pdf_path):
    # Inicializar el modelo de visióncls
    
    vision_model = VisionModel()
    
    # Leer el archivo PDF como bytes
    with open(pdf_path, "rb") as file:
        pdf_bytes = io.BytesIO(file.read())
    
    markdown_content = ""
    
    # Procesar cada página con el modelo de visión
    for page_num, img_bytes in pdf_bytes_to_images(pdf_bytes):
        # Extraer texto básico para complementar
        reader = PdfReader(pdf_path)
        page_index = page_num - 1
        extracted_text = reader.pages[page_index].extract_text() if page_index < len(reader.pages) else ""
        
        # Procesar con el modelo de visión
        print(f"Procesando página {page_num}...")
        page_markdown = vision_model.send_image_and_text_to_gpt4_vision(
            img_bytes, extracted_text, page_num
        )
        markdown_content += f"## Página {page_num}\n\n{page_markdown}\n\n"
    
    return markdown_content

if __name__ == "__main__":
    pdf_path = "data/pdf.pdf"
    
    if os.path.exists(pdf_path):
        print("=== Extracción básica ===")
        extracted_text = extract_text_from_pdf(pdf_path)
        print(extracted_text)
        
        print("\n=== Extracción con modelo de visión ===")
        try:
            vision_text = extract_text_with_vision_model(pdf_path)
            print(vision_text)
        except Exception as e:
            print(f"Error en la extracción con visión: {e}")
    else:
        print(f"El archivo {pdf_path} no existe")