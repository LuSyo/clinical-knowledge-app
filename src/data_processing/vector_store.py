import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List

def create_vector_store(chunks: List[Document], persist_directory: str = "./data/chroma_db"):
    """
    Creates a Chroma vector store from document chunks and saves it to disk.
    """
    # initialise embeddings model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # 2. Create the Vector Store
    # This step sends chunks to the API for embedding and stores them in Chroma.
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    return vector_store