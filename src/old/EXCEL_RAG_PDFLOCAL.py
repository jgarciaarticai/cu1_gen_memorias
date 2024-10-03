import pandas as pd
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain import PromptTemplate
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
import sys
import os
import json
from dotenv import load_dotenv
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Cargar configuración general desde config.json
config_path = os.path.join(BASE_DIR, 'config', 'config.json')
with open(config_path, 'r') as config_file:
    config = json.load(config_file)


archivo_entrada = config["input_file"]
carpeta_salida = config["output_folder"]
modelo = config["default_model"]
plantilla_excel = config["prompts_template"]
plantilla_contexto = config["context_template"]


class SuppressStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


# Ruta del PDF a leer
print(f"El archivo que se va a leer es: {archivo_entrada}")


# Cargar el PDF y dividirlo en chunks
loader = PyPDFLoader(archivo_entrada)
data = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)


# Sacar los embeddings
with SuppressStdout():
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())


# Leer los prompts de la plantilla    
queries_df = pd.read_excel(plantilla_excel)
print(f"La plantilla con los prompts es: {plantilla_excel}")

if 'PROMPT' not in queries_df.columns:
    raise ValueError("El archivo Excel debe contener una columna llamada 'PROMPT'.")


# Plantilla de prompt context
with open(plantilla_contexto, 'r') as contexto:
    template = contexto.read()

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

# Configuracion del LLM
llm = Ollama(model=modelo, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
)


# Preparar el fichero de salida
output_file_path = carpeta_salida + "/memoria_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"
print(f"El fichero de salida es: {output_file_path}")


# Abrir el archivo de salida para escritura
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for _, row in queries_df.iterrows():
        prompt = row['PROMPT']
        print(f"\n\nPROMPT: {prompt}")
        if pd.isna(prompt) or prompt.strip() == "":
            continue

        # Preguntar al modelo
        try:
            result = qa_chain({"query": prompt})
            answer = result.get('result', 'No se encontró una respuesta.')
   
        except Exception as e:
            answer = f'Error al procesar la consulta: {str(e)}'
            
        # Escribir pregunta y respuesta en el fichero de salida
        output_file.write(f"Pregunta: {prompt}\nRespuesta: {answer}\n\n")

print("Respuestas guardadas en el archivo de salida.")

