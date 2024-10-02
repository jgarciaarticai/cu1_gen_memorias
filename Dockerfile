# Usar una imagen base de Python
FROM python:3.11.9
# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar el archivo de dependencias (si usas un requirements.txt)
COPY requirements.txt .

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código de la aplicación al contenedor
COPY . .

# Comando por defecto para ejecutar tu aplicación (modifica según sea necesario)
CMD ["python", "app.py"]
