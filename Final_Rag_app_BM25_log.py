import logging
from fastapi import FastAPI, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from fastapi import FastAPI, Query
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain_community.retrievers import BM25Retriever
from pydantic import BaseModel
from langchain_nomic import NomicEmbeddings
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
from enum import Enum
import os
import PyPDF2
import psycopg2
import concurrent.futures
import time
from custom_logging import setup_custom_logger

# Initialize the custom logger
logger = setup_custom_logger('Rag_app_BM25_log')

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI()

class Options_Retriever_type(str, Enum):
    bm25 = "bm25"
    similarity = "similarity"
class Model_pick(str, Enum):
    llama = "llama3.2"
    qwen = "qwen:1.8b"
    mistral = "mistral"
# Configuration
DB_CONFIG = {
    "host": "localhost",
    "port": 6006,
    "user": "langchain",
    "password": "langchain",
    "dbname": "vector_store",
}
load_dotenv()

# Embedding Model Pick Function
def embedding_model_pick(type):
    logger.debug(f"Choosing embedding model type: {type}")
    #---- With API key -----
    if type == "api":
        os.environ["NOMIC_API_KEY"] = os.getenv("NOMIC_API_KEY")
        embeddings_api = NomicEmbeddings(model="nomic-embed-text-v1.5")
        logger.info("Using Nomic Embeddings API model.")
        return embeddings_api
    elif type == "local":
        #---- For local ollama nomic embedding setup ----
        embeddings_local = OllamaEmbeddings(model="nomic-embed-text")
        logger.info("Using Ollama Embeddings local model.")
        return embeddings_local

# Initialize Vector Store and LLM
def connect_to_pgvector():
    logger.debug("Connecting to PGVector.")
    connection_string = f"postgresql+psycopg://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
    vector_store = PGVector(
        embeddings=embedding_model_pick("api"),
        collection_name="embeddings",
        connection=connection_string,
        use_jsonb=True,
    )
    logger.info("Connected to PGVector successfully.")
    return vector_store

vector_store = connect_to_pgvector()


# Fetch documents from the vector store
def fetch_all_documents():
    logger.debug("Fetching all documents from the database.")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        query = "SELECT * FROM public.langchain_pg_embedding;"
        cur.execute(query)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        docs = [Document(page_content=row[3], metadata={"id": row[0], "embedding": row[2]}) for row in rows]
        logger.info(f"Fetched {len(docs)} documents from the database.")
        return docs
    except Exception as e:
        logger.error(f"Error fetching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching documents: {str(e)}")

# Utility Functions
def extract_text_from_pdf(file: UploadFile):
    logger.debug("Extracting text from PDF.")
    try:
        pdf_reader = PyPDF2.PdfReader(file.file)
        text = "".join([page.extract_text() for page in pdf_reader.pages])
        logger.info(f"Extracted text from PDF")
        return text
    except Exception:
        logger.error("Failed to extract text from PDF.")
        raise HTTPException(status_code=400, detail="Failed to extract text from PDF.")

def split_into_chunks(text: str, chunk_size: int):
    words = text.split()
    logger.debug(f"Splitting text into {chunk_size} size chunks on total chunk count - {len(words)}")
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    logger.info(f"Text split into {len(chunks)} chunks.")
    return chunks

def insert_embeddings(vector_store, text_chunks):
    logger.debug("Inserting embeddings into the vector store.")
    docs = [Document(page_content=chunk, metadata={"id": i}) for i, chunk in enumerate(text_chunks)]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(vector_store.add_documents, [doc], ids=[doc.metadata["id"]]) for doc in docs]
        concurrent.futures.wait(futures)
    logger.info("Embeddings inserted into the vector store.")

def format_docs(docs):
    logger.debug("Formatting documents.")
    return "\n\n".join(doc.page_content for doc in docs)

def retriever_eval(question: str, vector_store, similarity_search):  
    logger.debug(f"Performing similarity content check with search type: {similarity_search}.")
    chunks = {}
    if similarity_search == "similarity":
        # Retrieve top-k documents using vector-based similarity
        docs = vector_store.similarity_search(question, k=3)
        logger.info(f"Found {len(docs)} relevant documents.")
        for i, doc in enumerate(docs):
            logger.debug(f"{i} - Using chunk (ID: {doc.metadata['id']}): {doc.page_content}...")
            chunks.update({doc.metadata["id"]: doc.page_content})
    elif similarity_search == "bm25":
        docs = fetch_all_documents()
        bm25_retriever = BM25Retriever.from_documents(documents=docs)
        docs = bm25_retriever.invoke(question)
        for i, doc in enumerate(docs):
            logger.debug(f"{i} - Using chunk (ID: {doc.metadata['id']}): {doc.page_content}...")
            chunks.update({doc.metadata["id"]: doc.page_content})
    return docs[:3],chunks  # Return top 3 reranked documents

# Enhanced RAG Chain with Preprocessing and Optimization
def create_chain(llm, retriever_similarity):
    logger.debug("Creating RAG chain.")
    RAG_TEMPLATE = """You are a security assistant. Answer the question based strictly on the following SSP context:
    {context}
    Question: {question}

    Answer the question using only the information provided in the context. 
    Do not add any extra details, explanations, or generalizations. 
    Provide a direct answer based strictly on the context without any elaboration or assumptions. 
    Answer should be short and exact 90 percent from the context.
    Answer with max in 100 words.
    """

    rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    print(f"rag_prompt ---- {rag_prompt}")
    return (
        RunnablePassthrough.assign(context=lambda input: format_docs(input["context"]))
        | rag_prompt
        | llm
        | StrOutputParser()
    )

# FastAPI Endpoints
@app.get("/test-connection/")
def test_connection():
    logger.debug("Testing database connection.")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute('SELECT 1')
        result = cur.fetchone()
        if result and result[0] == 1:
            logger.info("Connection test successful.")
            return JSONResponse(content={"message": "Connection is valid."})
        else:
            logger.error("Test query did not return expected result.")
            raise HTTPException(status_code=500, detail="Test query did not return expected result.")
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile, chunk_size: int = 1000):
    logger.debug(f"Uploading PDF - {file.filename} | with chunk size - {chunk_size}")
    start_time = time.time()
    text = extract_text_from_pdf(file)
    chunks = split_into_chunks(text, chunk_size=chunk_size)
    insert_embeddings(vector_store, chunks)
    elapsed_time = time.time() - start_time
    logger.info(f"PDF uploaded and embeddings inserted in {elapsed_time:.2f} seconds.")
    return JSONResponse(content={"message": "PDF processed and embeddings inserted successfully.", "processing_time": elapsed_time})

@app.post("/query/")
async def query(question: str = Form(...),type: Options_Retriever_type = Form(...),model: Model_pick = Form(...)):
    logger.debug(f"Processing query: {question} | Retriever type: {type} | Model: {model}")
    print(f"Processing query: {question} | Retriever type: {type} | Model: {model}")
    try:
        # type = "bm25"
        # model = "qwen:1.8b"
        # question = "What is the AC-01 Control Policy?"
        start_time = time.time()
        llm = ChatOllama(model= model)
        # similarity for similarity_check and bm25 for bm25_retriever
        retriever,chunks_dict = retriever_eval(question, vector_store, type)  
        chain = create_chain(llm, retriever)
        response = chain.invoke({"context": retriever, "question": question})
        elapsed_time = time.time() - start_time
        logger.info(f"Query processed in {elapsed_time:.2f} seconds.")
        return JSONResponse(content={"response": response, "query_time": elapsed_time, "chunks_ID": chunks_dict})
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")