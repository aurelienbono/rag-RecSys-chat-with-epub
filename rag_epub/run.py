from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import UnstructuredEPubLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
from glob import glob

# Load environment variables from .env file
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_key, model_name="text-embedding-3-small"
)

# Initialize the Chroma client with persistence
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name, embedding_function=openai_ef
)

client = OpenAI(api_key=openai_key)

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    question: str

# Function to load EPUB documents from a directory
def load_documents_from_directory(directory_path):
    file_paths = glob(os.path.join(directory_path, "**", "*.epub"), recursive=True)
    
    documents = []
    for file_path in file_paths:
        loader = UnstructuredEPubLoader(file_path)
        doc = loader.load()
        text = "\n".join([d.page_content for d in doc])
        documents.append({"id": os.path.basename(file_path), "text": text})
    
    return documents


# Function to split text into chunks
def split_text(text, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to generate embeddings using OpenAI API
def get_openai_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    embedding = response.data[0].embedding
    return embedding

# Endpoint to load and process documents
@app.post("/load-documents/")
async def load_documents(directory_path: str):
    documents = load_documents_from_directory(directory_path)
    chunked_documents = []
    for doc in documents:
        chunks = split_text(doc["text"])
        for i, chunk in enumerate(chunks):
            chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})

    for doc in tqdm(chunked_documents):
        doc["embedding"] = get_openai_embedding(doc["text"])

    for doc in tqdm(chunked_documents):
        collection.upsert(
            ids=[doc["id"]], documents=[doc["text"]], embeddings=[doc["embedding"]]
        )

    return {"message": f"Loaded {len(documents)} documents"}

# Function to query documents
def query_documents(question, n_results=2):
    results = collection.query(query_texts=question, n_results=n_results)
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    return relevant_chunks

# Function to generate a response from OpenAI with stricter constraints
def generate_response(question, relevant_chunks):
    if not relevant_chunks:
        return "I don't have enough information in the provided documents to answer this question.", []

    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use only the retrieved context below "
        "to answer the question. If the answer is not found in the context, say: "
        "'I don't have enough information in the provided documents to answer this question.' "
        "Do not make up any information. Keep your answer concise, using three sentences maximum."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
    )

    answer = response.choices[0].message.content
    return answer, relevant_chunks   

# Endpoint to query the system
@app.post("/query/", response_model=QueryResponse)
async def query_endpoint(query_request: QueryRequest):
    relevant_chunks = query_documents(query_request.question)
    answer, sources = generate_response(query_request.question, relevant_chunks)
    return {"question": query_request.question, "answer": answer,   "sources": sources}

# To run the application, use the command: uvicorn your_filename:app --reload

@app.get("/")
async def home():
    return {"Message":"Api is running..."}

# To run the application, use the command: uvicorn your_filename:app --reload
