from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def load_model():
    # Load pre-trained model from Hugging Face or Sentence-Transformers
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Example model
    return model

def embed_documents(documents):
    model = load_model()
    embeddings = model.encode(documents, convert_to_tensor=True)
    return embeddings

def save_embeddings(embeddings):
    # Save the FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])  # Create FAISS index
    index.add(embeddings)
    faiss.write_index(index, "vectorstore/faiss.index")

def ingest():
    # Load your documents
    documents = ["Document 1 text", "Document 2 text", "Document 3 text"]  # Example
    embeddings = embed_documents(documents)
    save_embeddings(embeddings)

if __name__ == "__main__":
    ingest()
