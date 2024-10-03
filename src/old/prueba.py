import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import sys
import os
import json
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Cargar configuración general desde config.json
config_path = os.path.join(BASE_DIR, 'config', 'config.json')
with open(config_path, 'r', encoding='utf-8') as config_file:
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

# Configuración del LLM
llm = Ollama(model=modelo, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

# Ruta del PDF a leer
print(f"El archivo que se va a leer es: {archivo_entrada}")

# Cargar el PDF y dividirlo en chunks
loader = PyPDFLoader(archivo_entrada)
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
all_splits = text_splitter.split_documents(data)

# Sacar los embeddings
with SuppressStdout():
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())
retriever = vectorstore.as_retriever()


# Crear las cadenas de preguntas y respuestas
system_prompt = (
    "Eres un asistente que responde preguntas sobre un documento."
    "Utiliza las siguientes piezas de contexto para responder la pregunta final sobre el documento de entrada al principio de la conversación."
    "Si no sabes la respuesta, simplemente di que no la sabes, no intentes inventar una respuesta."
    "Utiliza diez oraciones como máximo y mantén la respuesta lo más concisa posible."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Preparar el fichero de salida
txt_file_path = os.path.join(carpeta_salida, "memoriaRAG_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt")

# Leer los prompts de la plantilla    
queries_df = pd.read_excel(plantilla_excel)
print(f"La plantilla con los prompts es: {plantilla_excel}")

if 'PROMPT' not in queries_df.columns:
    raise ValueError("El archivo Excel debe contener una columna llamada 'PROMPT'.")

# Procesar las preguntas y escribir las respuestas en el documento
with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
    for _, row in queries_df.iterrows():
        prompt = row['PROMPT']
        print(f"\n\nPROMPT: {prompt}")
        if pd.isna(prompt) or prompt.strip() == "":
            continue

        # Crear una instancia de la cadena con el historial limpio
        conversational_rag_chain = RunnableWithMessageHistory(
            question_answer_chain,
            lambda _: ChatMessageHistory(),  # Inicializa un historial de mensajes vacío para cada pregunta
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        # Preguntar al modelo
        try:
            # Obtener la respuesta del modelo
            response = conversational_rag_chain.invoke(
                {"input": prompt, "context": all_splits},
                config={"configurable": {"session_id": "abc123"}},
            )
            answer = response
        except Exception as e:
            answer = f'Error al procesar la consulta: {str(e)}'
            
        # Escribir pregunta y respuesta en un fichero .txt
        txt_file.write(f"Pregunta: {prompt}\nRespuesta: {answer}\n\n")
