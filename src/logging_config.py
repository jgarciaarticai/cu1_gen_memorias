import os
import logging
from datetime import datetime

def setup_logging():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(BASE_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Crear un archivo de log con un nombre único basado en la fecha y hora de inicio
    log_file_path = os.path.join(log_dir, f"gen_memorias_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # Configuración del logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # File handler para guardar los logs en un archivo
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    # Stream handler para mostrar los logs en la consola
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)  # o DEBUG si prefieres más detalles en consola
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return log_file_path
