import os
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings

def initialize_vector_store(persist_directory, cohere_api_key, collection_name="jharkhand_policies"):
    """
    Initialize and return the Chroma vector store with Cohere embeddings.
    Creates a new store if one doesn't exist at the persist_directory.
    """
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory, exist_ok=True)
        
    embeddings = CohereEmbeddings(model="embed-v4.0", cohere_api_key=cohere_api_key)
    
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name
    )
    
    return vector_store