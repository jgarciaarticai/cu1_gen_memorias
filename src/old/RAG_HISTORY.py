
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
import sys
import os
import json
from dotenv import load_dotenv
from datetime import datetime
from docx import Document

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Cargar configuración general desde config.json
config_path = os.path.join(BASE_DIR, 'config', 'config.json')
with open(config_path, 'r', encoding='utf-8') as config_file:
    config = json.load(config_file)


archivo_entrada = config["input_file"]
carpeta_salida = config["output_folder"]
modelo = config["default_model"]
plantilla_excel = config["prompts_template"]
plantilla_word = config["format_template"]
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


# Configuracion del LLM
llm = Ollama(model=modelo, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

# Ruta del PDF a leer
print(f"El archivo que se va a leer es: {archivo_entrada}")


# Cargar el PDF y dividirlo en chunks
loader = PyPDFLoader(archivo_entrada)
data = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=4096 - len(plantilla_contexto), chunk_overlap=400)
all_splits = text_splitter.split_documents(data)


# Sacar los embeddings
with SuppressStdout():
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())
retriever = vectorstore.as_retriever()


### Contextualize question ###
contextualize_q_system_prompt = (
    "Dado un historial de chat y el contenido del documento de entrada que hace referencia al contexto en el historial de chat formular una pregunta independiente que no puede entenderse sin el historial de chat." 
    "NO respondas la pregunta, Simplemente reformúlelo si es necesario y, en caso contrario, devuélvalo tal como está"
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


### Answer question ###
system_prompt = (
    "Eres un asistente que responde preguntas sobre un documento."
    "Utiliza las siguientes piezas de contexto para responder la pregunta final sobre el documento de entrada al principio de la conversación"
    "Si no sabes la respuesta, simplemente di que no la sabes, no intentes inventar una respuesta."
    "Utiliza diez oraciones como máximo y mantén la respuesta lo más concisa posible."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


### Statefully manage chat history ###
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)



# Preparar el fichero de salida
txt_file_path = carpeta_salida + "/memoriaRAG_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"

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

        # Preguntar al modelo
        try:
            answer = conversational_rag_chain.invoke(
                {"input": prompt},
                config={"configurable": {"session_id": "abc123"}},
            )
        except Exception as e:
            answer = f'Error al procesar la consulta: {str(e)}'
            
        # Escribir pregunta y respuesta en un fichero .txt
        txt_file.write(f"Pregunta: {prompt}\nRespuesta: {answer}\n\n")