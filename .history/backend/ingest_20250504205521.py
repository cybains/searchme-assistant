from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# Initialize the Hugging Face model for embeddings
def load_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Example model
    return model

def embed_documents(documents):
    model = load_model()
    embeddings = model.encode(documents, convert_to_tensor=True)
    return embeddings

def save_embeddings(embeddings):
    # Ensure that the directory exists
    if not os.path.exists("vectorstore"):
        os.makedirs("vectorstore")
    
    # FAISS setup: create an index and add embeddings
    index = faiss.IndexFlatL2(embeddings.shape[1])  # Using L2 distance
    index.add(embeddings)
    
    # Save index to the 'vectorstore' directory
    faiss.write_index(index, "vectorstore/faiss.index")

def ingest():
    # Example: load documents from a data folder
    documents = []
    data_folder = "data/"
    for filename in os.listdir(data_folder):
        if filename.endswith(".txt"):  # Example file type
            with open(os.path.join(data_folder, filename), "r") as file:
                documents.append(file.read())

    embeddings = embed_documents(documents)
    save_embeddings(embeddings)

if __name__ == "__main__":
    ingest()
