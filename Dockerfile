# Usar una imagen base de Python
FROM python:3.11.9
# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalar Poppler
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    && apt-get clean

# Copiar el archivo de dependencias (si usas un requirements.txt)
COPY requirements.txt .

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código de la aplicación al contenedor
COPY . .

# Comando por defecto para ejecutar tu aplicación (modifica según sea necesario)
CMD ["python", "src/CU1_GEN_MEMORIAS.py"]