# CU1_GEN_MEMORIAS

**CU1_GEN_MEMORIAS** es una aplicación diseñada para la generación automatizada de documentos a partir de consultas contextuales sobre archivos PDF, utilizando procesamiento de lenguaje natural y modelos de lenguaje avanzado (LLMs). La aplicación emplea técnicas de recuperación de información y generación de texto asistida por IA, permitiendo automatizar la creación de informes con una mínima intervención humana.

## Funcionalidades Principales

1. **Extracción de texto desde PDFs**:  
   La aplicación convierte documentos PDF en texto utilizando un sistema OCR cuando es necesario, para luego procesar y fragmentar los contenidos en partes manejables.

2. **Base de Datos Vectorial**:  
   Los textos extraídos de los documentos se convierten en vectores de embeddings y se almacenan en una base de datos FAISS, permitiendo realizar consultas eficientes basadas en similitud semántica.

3. **Modelo de Embeddings y Recuperación**:  
   Utiliza el modelo de embeddings *Ollama* para representar los textos y emplea técnicas avanzadas de recuperación de información como el *MultiQueryRetriever* para encontrar fragmentos relevantes de texto basados en las consultas del usuario.

4. **Generación de Textos (RAG - Retrieval-Augmented Generation)**:  
   Combina la recuperación de fragmentos relevantes con un proceso de generación de texto automatizado. El modelo de lenguaje (LLM) genera respuestas basadas en los textos recuperados, que luego son formateadas para su inclusión en un documento final.

5. **Integración con Plantillas Excel y Word**:  
   La aplicación utiliza plantillas predefinidas en Excel y Word para personalizar las consultas y los informes generados. Los resultados se insertan en las posiciones designadas en la plantilla Word, generando un documento final que reemplaza automáticamente los marcadores con las respuestas generadas por la IA.

6. **Registro y Monitorización**:  
   El sistema está configurado con una robusta infraestructura de logging, que permite un seguimiento detallado del procesamiento de documentos, generación de respuestas y la detección de errores.

## Componentes Clave

- **pdf_a_texto**: Módulo encargado de la conversión de archivos PDF en texto, incluyendo soporte para OCR.
- **LangChain y FAISS**: Tecnologías utilizadas para la creación de la base de datos vectorial y el procesamiento de consultas semánticas.
- **Ollama**: Modelo de lenguaje y de embeddings utilizado tanto para la creación de vectores como para la generación de respuestas.
- **PromptTemplate**: Plantilla utilizada para dar formato a las consultas enviadas al modelo LLM.
- **MultiQueryRetriever**: Componente que optimiza las búsquedas contextuales sobre el texto almacenado en la base de datos vectorial.
- **RAG (Retrieval-Augmented Generation)**: Pipeline de generación de respuestas basadas en el contenido recuperado y generado por el modelo LLM.

## Requisitos

- Python 3.11+
- Dependencias especificadas en `requirements.txt`
- FAISS para la indexación y búsqueda en la base de datos vectorial
- Ollama como modelo de lenguaje

## Configuración

La configuración de la aplicación se maneja a través de variables de entorno definidas en un archivo `.env`. Asegúrate de especificar correctamente las rutas a las plantillas, el modelo de embeddings, los parámetros del LLM y las carpetas de entrada y salida.

## Ejecución

1. Coloca los archivos PDF que deseas procesar en la carpeta especificada en la variable de entorno `INPUT_FOLDER`.
2. Define los prompts y marcadores en la plantilla Excel especificada por `CU_X_PROMPTS_TEMPLATE.xlsx`.
3. Define los marcadores en la plantilla Word de salida especificada por `CU_X_FORMAT_TEMPLATE.docx`.
4. Al ejecutar el script, la aplicación generará automáticamente un documento Word con las respuestas insertadas en los marcadores correspondientes.
5. Los documentos generados se guardarán en la carpeta especificada por `OUTPUT_FOLDER`.

## Licencia
Esta aplicación ha sido desarrollada por la empresa Artica+i
