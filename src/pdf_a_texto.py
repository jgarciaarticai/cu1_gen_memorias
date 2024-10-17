from langchain_community.document_loaders import PyPDFLoader
import pytesseract
from pdf2image import convert_from_path
import logging
import pdfplumber

logger = logging.getLogger(__name__)


def extraer_texto_ocr(archivo, resolucion):
    # convert to image using resolution dpi 
    pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"

    logger.info(f"Procesando el archivo: {archivo} con resolución: {resolucion}")
    try:
        pages = convert_from_path(archivo, int(resolucion))
    except Exception as e:
        logger.error(f"Error al convertir el PDF en imágenes: {e}")
        return ""
    logger.info("PDF convertido a imagenes")

    # extract text
    text_data = ''
    for i, page in enumerate(pages):
        try:
            logger.info(f'Procesando página {i+1}...')
            text = pytesseract.image_to_string(page)
            logger.info(f'Texto extraído de la página {i+1}:\n{text}')
            text_data += text + '\n'
        except Exception as e:
            logger.error(f"Error al procesar la página {i+1} con OCR: {e}")
    return text_data


def extraer_texto(archivo):
    # Intenta cargar el PDF y extraer texto directamente
    try:
        loader = PyPDFLoader(archivo)
        data = loader.load()
        text = "\n".join(doc.page_content for doc in data)
        if text.strip():
            return text
        else:
            return None
    except Exception as e:
        logger.error(f"Error al extraer texto del PDF directamente: {e}")
        return None


def extraer_texto_plumber(archivo):
    # Abrir el archivo PDF en modo lectura binaria
    with pdfplumber.open(archivo) as pdf_file:
        # Inicializar una variable para acumular el texto
        texto_completo = ""
        
        # Iterar sobre todas las páginas del PDF
        for num_pagina in range(len(pdf_file.pages)):
            # Obtener el objeto de la página
            pagina = pdf_file.pages[num_pagina]
            # Extraer el texto de la página
            texto = pagina.extract_text()
            
            if texto:
                # Acumular el texto extraído
                texto_completo += texto
                # Añadir una nueva línea para separar las páginas
                texto_completo += '\n\n'
            else:
                logger.error(f"No se pudo extraer texto de la página {num_pagina + 1}")
                
    # Retornar el texto completo
    return texto_completo


def procesar_pdf(archivo, resolucion):
    logger.info(f"Procesando archivo: {archivo}")
    
    # Intentar extraer texto directamente del PDF
    text = extraer_texto(archivo)
    
    if text:
        logger.info("Texto extraído directamente del PDF.")
    else:
        logger.info("Texto no extraído directamente, se utilizará OCR.")
        text = extraer_texto_ocr(archivo, resolucion)
    
    return text