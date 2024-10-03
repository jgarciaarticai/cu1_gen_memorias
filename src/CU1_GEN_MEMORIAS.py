import pandas as pd
import os
import json
import logging
from datetime import datetime
from docx import Document

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore

import pdf_a_texto
from logging_config import setup_logging


# Definir el directorio base de la solución
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configuracion del logging
log_file_path = setup_logging()
logger = logging.getLogger(__name__)


try:
    # Cargar configuración general desde config.json
    config_path = os.path.join(BASE_DIR, 'config', 'config.json')
    with open(config_path, 'r', encoding='utf-8') as config_file:
        config = json.load(config_file)
    logger.info("Configuración cargada correctamente.")

    # Leer rutas a plantillas
    carpeta_entrada = config["input_folder"]
    carpeta_salida = config["output_folder"]
    plantilla_excel = config["prompts_template"]
    plantilla_word = config["format_template"]
    plantilla_contexto = config["context_template"]


    # Inicializar la base de datos vectorial
    embeddings_model = config["embedding_model"]
    local_embeddings = OllamaEmbeddings(model=embeddings_model, base_url="http://host.docker.internal:11434")
    logger.info(f"Modelo embeddings: {embeddings_model}")
    
    index = faiss.IndexFlatL2(len(local_embeddings.embed_query("hola mundo")))
    
    vectorstore = FAISS(
        embedding_function=local_embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    
    # Procesar cada documento en la carpeta de entrada
    for filename in os.listdir(carpeta_entrada):
        filepath = os.path.join(carpeta_entrada, filename)
        if filename.endswith(".pdf"):
            logger.info(f"Procesando archivo: {filepath}")
            data = pdf_a_texto.procesar_pdf(filepath, config["resolucion_ocr"]) # OCR si es necesario
            logger.info("Archivo PDF cargado correctamente.")

            # Dividir el archivo en chunks
            text_splitter = RecursiveCharacterTextSplitter(config["chunk_size"], config["chunk_overlap"])
            all_splits = text_splitter.split_text(data)
            logger.info(f"Documento {filename} dividido en chunks correctamente.")

            # Añadir los chunks al índice FAISS
            vectorstore.add_texts(texts=all_splits, embedding=local_embeddings, normalize_L2=True)
            logger.info(f"Fragmentos de {filename} añadidos a la base de datos vectorial.")

    logger.info("Base de datos vectorial completada correctamente.")
    
    retriever = vectorstore.as_retriever(
#        search_type="similarity_score_threshold", 
#        search_kwargs={"k": int(config["k"]), "score_threshold": float(config["score_threshold"])}
        search_type="mmr"
    )

    # Indicamos el modelo
    model = Ollama(
        base_url="http://host.docker.internal:11434",
        system=config["system_prompt"],
        model=config["default_model"], 
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        temperature=config["temperature"],
        top_p=config["top_p"],
    )
    logger.info(f"Modelo de LLM: {config['default_model']} con temperature: {config['temperature']} y top_p: {config['top_p']}")

    # RAG
    # Añadimos comprension contextual
    #compressor = LLMChainExtractor.from_llm(model)
    #compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    with open(plantilla_contexto, 'r', encoding='utf-8') as contexto:
        RAG_TEMPLATE = contexto.read()
   
    rag_prompt = PromptTemplate.from_template(RAG_TEMPLATE)
    
    qa_chain = (
        RunnablePassthrough.assign(context=lambda input: format_docs(input["context"]))
        | rag_prompt
        | model
        | StrOutputParser()
    )
    logger.info("Chain de RAG creada correctamente.")

    # LEER PROMPTS Y CARGAR PLANTILLAS
    queries_df = pd.read_excel(plantilla_excel)
    logger.info(f"La plantilla con los prompts es: {plantilla_excel}")

    # Cargar el documento .docx
    doc = Document(plantilla_word)
    logger.info("Plantilla Word cargada correctamente.")
    output_file_path_docx = os.path.join(carpeta_salida, f"CU1_MEMORIA_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx")

    # EJECUCION
    for _, row in queries_df.iterrows():
        question = row['PROMPT']
        marcador = row['MARCADOR']
        logger.info(f"Procesando PROMPT: {question}")

        if pd.isna(question) or question.strip() == "":
            logger.warning("Pregunta vacía o nula, se omite.")
            continue

        try:
            # Buscamos trozos relevantes en el contexto
            #docs = compression_retriever.invoke(question)
            docs = retriever.invoke(question)
            logger.info(f"RETRIEVED DOCS: {docs}")
            
            respuesta = qa_chain.invoke({"context": docs, "question": question})
            logger.info(f"RESPUESTA: {respuesta}")
        except Exception as e:
            logger.error(f"Error al procesar la consulta: {str(e)}")
            respuesta = f'Error al procesar la consulta: {str(e)}'

        # Buscar el marcador en el documento y reemplazarlo con la respuesta
        found = False
        for para in doc.paragraphs:
            if marcador in para.text:
                para.text = para.text.replace(marcador, respuesta)
                found = True
                break
        if not found:
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if marcador in cell.text:
                            cell.text = cell.text.replace(marcador, respuesta)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
        if not found:
            logger.warning(f"Marcador '{marcador}' no encontrado en el documento.")
        doc.save(output_file_path_docx)
    
    
    logger.info(f"Se ha generado la memoria: {output_file_path_docx}")

except Exception as e:
    logger.critical(f"Error crítico durante la ejecución del script: {str(e)}", exc_info=True)
    print(f"Ha ocurrido un error crítico. Revisa el log para más detalles: {log_file_path}")
